//! Trained-head 2-slot rotation: index-atomic 10-step publish
//! sequence.  The caller (a train or convert pipeline) has
//! already allocated `head_id`, produced the weights, streamed the
//! ACSTHEAD-wrapped `.mpk` to a tempfile under
//! `<workspace>/.tmp/`, fsynced it, and built the [`HeadManifest`]
//! with derived fields populated.
//!
//! `publish_trained_head` then runs the rotation under the
//! per-workspace mutation mutex (caller-held).  The [`HeadIndex`]
//! file is the publish point: a crash before the index commit
//! leaves unreferenced `<head_id>.{mpk,json}` files (orphan
//! residue swept by boot recovery); a crash after the commit
//! published the head durably.  Post-commit cleanup of displaced
//! bytes is best-effort.
//!
//! # Layout
//!
//! ```text
//! <workspace_dir>/
//!     workspace.json                 -- core (head_count derived)
//!     heads.json                     -- index (<= 2 entries)
//!     heads/
//!         <head_id>.mpk              -- caller-provided ACSTHEAD blob
//!         <head_id>.json             -- HeadManifest
//!     .tmp/                          -- caller staged the .mpk here
//! ```

use std::path::{Path, PathBuf};

use crate::common::ids::HeadId;
use crate::common::workspace::{HeadIndex, HeadManifest, HeadRecord, MAX_HEADS_PER_WORKSPACE};
use crate::file_mgr::cache::WorkspaceCacheCell;
use crate::file_mgr::error::{FileError, io_err};
use crate::file_mgr::fs_atomic::put_atomic;
use crate::file_mgr::schema::{
    head_artifact_path, head_manifest_path, heads_dir, read_workspace_core, write_head_index,
    write_workspace_core,
};
use crate::file_mgr::validate::fsync_dir;

/// Inputs the caller hands to [`publish_trained_head`].
///
/// The `.mpk` payload is referenced by path -- the caller streamed
/// it to a tempfile under `<workspace_dir>/.tmp/` (same filesystem,
/// so the rename in step 5 is intra-FS atomic) and fsynced it.  The
/// manifest's bytes are small (hundreds of bytes; labels + scalar
/// metadata) so they ride inline.
///
/// `manifest.head_id` MUST equal `head_id`; the rotation primitive
/// asserts the equality so a buggy caller does not race the
/// index-vs-file pair.
#[derive(Clone, Debug)]
pub struct PendingHead {
    /// Identity of the head being published.  Drives the
    /// `<head_id>.{mpk,json}` filenames + the [`HeadRecord`]
    /// entry the rotation appends to `heads.json`.
    pub head_id: HeadId,
    /// Path to the already-staged + fsynced `.mpk` tempfile.
    /// Lives under `<workspace_dir>/.tmp/<random>`; the
    /// rotation atomically renames it into
    /// `<workspace_dir>/heads/<head_id>.mpk`.
    pub mpk_tempfile: PathBuf,
    /// Per-head manifest body.  Fully populated by the caller
    /// (sha256 / n_classes / size_bytes / labels /
    /// workspace_revision).  The rotation serializes + atomically
    /// writes it to `<workspace_dir>/heads/<head_id>.json` in
    /// step 1 (after staging through `<workspace_dir>/.tmp/`).
    pub manifest: HeadManifest,
}

/// Outcome of a successful [`publish_trained_head`] call.
/// `displaced_head_id` is `Some(_)` exactly when the published
/// head pushed a previous-generation head out of the 2-slot
/// sliding window.
#[derive(Clone, Copy, Debug)]
pub struct HeadRotationResult {
    /// Head id that the rotation displaced from the index, or
    /// `None` if the workspace had < 2 heads before the publish.
    pub displaced_head_id: Option<HeadId>,
}

/// Publish a trained head into the workspace's 2-slot index.  The
/// caller MUST hold the per-workspace mutation mutex for the full
/// duration of this call.  The rotation is sync; never `.await`
/// inside.
///
/// 10 steps in order:
///
/// 1. Stage `<id>.json` to `.tmp/<random>` and atomically write
///    it; the caller already staged `<id>.mpk` to `.tmp/`.
/// 2. fsync the `.mpk` tempfile (caller did this before calling;
///    the manifest tempfile is fsynced by `put_atomic` in step
///    1).
/// 3. Read the current `heads.json` (from the cache snapshot;
///    the per-workspace mutex serializes against any concurrent
///    mutation so the cache's snapshot is current).
/// 4. Compute next `heads[]`: prepend the new entry; if
///    `len > 2`, mark the tail for deletion.
/// 5. Atomically rename the staged tempfiles into
///    `heads/<id>.{mpk,json}`.
/// 6. fsync `heads/`.
/// 7. Atomic-rewrite `heads.json` (this is the publish point;
///    `put_atomic` fsyncs the file + parent dir).
/// 8. Atomic-rewrite `workspace.json` with refreshed
///    `head_count`; publish the new core + head index to the
///    cache atomically from the reader's POV (two `ArcSwap`
///    stores, but readers observe each independently and never
///    see partial state).
/// 9. Best-effort delete the displaced head's `.mpk` + `.json`
///    from `heads/`.  A missing file is benign because the
///    publish point at step 7 already removed it from the
///    index.
/// 10. fsync `heads/` after the displaced-head removal.
///
/// # Crash recovery
///
/// `heads.json` is the publish point.  A file in `heads/` not
/// referenced by `heads.json.heads[]` is orphan residue swept by
/// boot recovery.  `heads.json` referencing a missing head file
/// is corruption because the index commit (step 7) lands after
/// the file renames (step 5).  `workspace.json.head_count` is
/// derived and boot-repairable.
pub fn publish_trained_head(
    workspace_dir: &Path,
    cache: &WorkspaceCacheCell,
    pending: PendingHead,
) -> Result<HeadRotationResult, FileError> {
    if pending.manifest.head_id != pending.head_id {
        return Err(io_err(
            workspace_dir.display(),
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "publish_trained_head: PendingHead.head_id != manifest.head_id",
            ),
        ));
    }
    if !pending.mpk_tempfile.is_file() {
        return Err(io_err(
            pending.mpk_tempfile.display(),
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "publish_trained_head: mpk_tempfile is missing or not a regular file",
            ),
        ));
    }
    let heads = heads_dir(workspace_dir);
    std::fs::create_dir_all(&heads).map_err(|e| io_err(heads.display(), e))?;

    // Step 1: Stage the manifest body in `.tmp/` and atomically
    // rename it into `heads/<id>.json`.  `put_atomic` covers
    // tempfile-create + fsync + rename + parent-dir fsync inside
    // `heads/`, so the manifest landing is durable before we
    // touch the index.  The `.mpk` tempfile (step 5) is renamed
    // separately under the caller's chosen filename.
    //
    // We choose to write the manifest bytes through
    // `put_atomic` directly to its final location (step 1
    // collapses with step 5 for the JSON half); the .mpk's
    // pre-existing tempfile gets renamed in step 5.
    let manifest_path = head_manifest_path(workspace_dir, pending.head_id);
    let manifest_bytes = serde_json::to_vec(&pending.manifest)?;
    put_atomic(&manifest_path, &manifest_bytes)?;

    // Step 5 (mpk half): atomic-rename caller's staged tempfile
    // into heads/<id>.mpk.  Caller fsynced the tempfile (step
    // 2) before invoking us; std::fs::rename is intra-FS
    // POSIX-atomic.
    let mpk_path = head_artifact_path(workspace_dir, pending.head_id);
    std::fs::rename(&pending.mpk_tempfile, &mpk_path).map_err(|e| io_err(mpk_path.display(), e))?;

    // Step 6: fsync heads/ so both new directory entries (the
    // .mpk rename + the .json rename inside `put_atomic`) reach
    // stable storage before the index commit.
    fsync_dir(&heads).map_err(|e| io_err(heads.display(), e))?;

    // Step 3 + 4: read current heads.json from the cache and
    // compute the next index.  The cache snapshot is canonical
    // because the caller holds the per-workspace mutation mutex
    // (the same lock that gates every publish); no concurrent
    // mutation can interleave.
    let prev_heads = cache.heads();
    let new_record = HeadRecord {
        head_id: pending.manifest.head_id,
        workspace_revision: pending.manifest.workspace_revision.clone(),
        sha256: pending.manifest.sha256.clone(),
        n_classes: pending.manifest.n_classes,
        size_bytes: pending.manifest.size_bytes,
        created_at: pending.manifest.created_at.clone(),
    };
    let mut next_records: Vec<HeadRecord> = Vec::with_capacity(prev_heads.heads.len() + 1);
    next_records.push(new_record);
    for rec in prev_heads.heads.iter() {
        next_records.push(rec.clone());
    }
    let displaced_head_id = if next_records.len() > MAX_HEADS_PER_WORKSPACE {
        // Sliding window: drop the oldest tail entry.  The
        // truncate-to-cap shape is deterministic so a future
        // change to MAX_HEADS_PER_WORKSPACE only requires the
        // common::workspace constant move.
        next_records
            .drain(MAX_HEADS_PER_WORKSPACE..)
            .next()
            .map(|r| r.head_id)
    } else {
        None
    };
    debug_assert!(next_records.len() <= MAX_HEADS_PER_WORKSPACE);
    let next_index = HeadIndex {
        heads: next_records,
    };

    // Step 7: atomic-rewrite heads.json.  This is the publish
    // point: any crash AFTER this returns produces a workspace
    // whose `heads.json` references the new head; any crash
    // BEFORE leaves the new files unreferenced (orphan residue).
    write_head_index(workspace_dir, &next_index)?;

    // Step 8: atomic-rewrite workspace.json with refreshed
    // head_count, then publish both caches.  We re-read the
    // core from disk rather than the cache because head_count
    // is derived from heads.json -- using the just-computed
    // next_index keeps the two files in lockstep at every
    // observable instant.
    //
    // Reading from disk guards against a torn cache snapshot
    // where another mutation path (a dataset-revision bump)
    // raced our publish; in practice the per-workspace mutex
    // we hold prevents that, but the disk read costs <1 ms and
    // keeps the head_count guarantee unconditional.
    let mut next_core = read_workspace_core(workspace_dir)?;
    next_core.head_count = next_index.heads.len() as u8;
    write_workspace_core(workspace_dir, &next_core)?;

    // Publish to cache: heads FIRST, then core.  The interleaved
    // observation window (heads = N+1 entries, core.head_count
    // still = N) is the safe direction: a reader's invariant
    // `core.head_count <= heads.len()` is preserved.  The
    // opposite order would briefly let `core.head_count = N+1`
    // promise an entry that the cached `heads.json` snapshot
    // doesn't yet hold.  The per-workspace mutex serializes
    // writers; readers consume each Arc snapshot independently.
    cache.publish_heads(next_index);
    cache.publish_core(next_core);

    // Step 9: best-effort delete the displaced head's bytes.
    // Failure here is logged but not propagated: the publish
    // point already moved, and boot recovery sweeps any
    // stragglers.
    if let Some(old) = displaced_head_id {
        let old_mpk = head_artifact_path(workspace_dir, old);
        let old_json = head_manifest_path(workspace_dir, old);
        if let Err(e) = std::fs::remove_file(&old_mpk)
            && e.kind() != std::io::ErrorKind::NotFound
        {
            tracing::warn!(
                target: "file_mgr",
                err = %e,
                path = %old_mpk.display(),
                "publish_trained_head: failed to remove displaced .mpk; boot recovery will sweep",
            );
        }
        if let Err(e) = std::fs::remove_file(&old_json)
            && e.kind() != std::io::ErrorKind::NotFound
        {
            tracing::warn!(
                target: "file_mgr",
                err = %e,
                path = %old_json.display(),
                "publish_trained_head: failed to remove displaced .json; boot recovery will sweep",
            );
        }
        // Step 10: fsync heads/ so the unlink directory-entry
        // updates reach stable storage.  Best-effort; a failure
        // here is logged but not propagated for the same reason
        // as the unlink failures above.
        if let Err(e) = fsync_dir(&heads) {
            tracing::warn!(
                target: "file_mgr",
                err = %e,
                path = %heads.display(),
                "publish_trained_head: failed to fsync heads/ after displaced removal",
            );
        }
    }

    Ok(HeadRotationResult { displaced_head_id })
}

// MARK: WorkspaceMgr::delete_head (single-head removal)

use crate::common::ids::WorkspaceId;
use crate::common::workspace::JobReference;
use crate::file_mgr::WorkspaceMgr;
use crate::file_mgr::schema::workspace_core_path;
use std::sync::Arc;

impl WorkspaceMgr {
    /// Remove a single completed head from a workspace.
    ///
    /// Behaviour:
    /// - 409 [`FileError::JobConflict`] when any running job
    ///   references this workspace (workspace-wide reference, or
    ///   any dataset reference -- the head deletion would race
    ///   the in-flight job).
    /// - 404 [`FileError::AssetNotFound`] when the head id is
    ///   not in the current `heads.json`.
    /// - On success: index-atomic removal of the entry from
    ///   `heads.json`, atomic-rewrite of `workspace.json` with
    ///   refreshed `head_count`, deletion of
    ///   `heads/<head_id>.{mpk,json}`, fsync of `heads/`, and
    ///   publish of the new core + head index to the cache.
    ///
    /// Holds the per-workspace mutation mutex throughout.
    /// `&Arc<Self>` because the conflict check goes through
    /// `JobRegistry::try_acquire_lease` which clones the
    /// registry Arc.
    pub fn delete_head(
        self: &Arc<Self>,
        ws: &WorkspaceId,
        head_id: HeadId,
    ) -> Result<(), FileError> {
        let workspace_dir = self.workspace_dir(ws);
        if !workspace_core_path(&workspace_dir).exists() {
            return Err(FileError::NotFound(ws.to_string()));
        }
        // Take a bare workspace-wide lease for the duration of
        // the delete.  Overlapping running jobs bump us out with
        // 409 `JobConflict`; the lease drops on return.
        let _ref_guard = self
            .jobs
            .try_acquire_lease(vec![JobReference::Workspace { workspace_id: *ws }])
            .map_err(FileError::from)?;

        // Per-workspace mutation mutex.  Sync; never `.await`.
        let lock = self.metadata_lock(ws);
        let _guard = lock.lock();

        // Resolve the cache cell (lazy-load on first touch).
        let cell = self.cache_cell_for_head_delete(ws)?;
        let prev_heads = cell.heads();
        let mut next_records = Vec::with_capacity(prev_heads.heads.len());
        let mut found = false;
        for rec in prev_heads.heads.iter() {
            if rec.head_id == head_id {
                found = true;
                continue;
            }
            next_records.push(rec.clone());
        }
        if !found {
            return Err(FileError::AssetNotFound {
                ws: ws.to_string(),
                kind: crate::file_mgr::AssetKind::HeadMpk,
                name: format!("{head_id}.mpk"),
            });
        }
        let next_index = HeadIndex {
            heads: next_records,
        };

        // Index-atomic publish point.
        write_head_index(&workspace_dir, &next_index)?;

        // Update workspace.json.head_count alongside the index.
        let mut next_core = read_workspace_core(&workspace_dir)?;
        next_core.head_count = next_index.heads.len() as u8;
        write_workspace_core(&workspace_dir, &next_core)?;

        // Publish to cache.
        cell.publish_heads(next_index);
        cell.publish_core(next_core);

        // Best-effort byte cleanup.  A missing file is benign;
        // boot recovery sweeps any stragglers.
        let mpk_path = head_artifact_path(&workspace_dir, head_id);
        let json_path = head_manifest_path(&workspace_dir, head_id);
        if let Err(e) = std::fs::remove_file(&mpk_path)
            && e.kind() != std::io::ErrorKind::NotFound
        {
            tracing::warn!(
                target: "file_mgr",
                err = %e,
                path = %mpk_path.display(),
                "delete_head: failed to remove .mpk; boot recovery will sweep",
            );
        }
        if let Err(e) = std::fs::remove_file(&json_path)
            && e.kind() != std::io::ErrorKind::NotFound
        {
            tracing::warn!(
                target: "file_mgr",
                err = %e,
                path = %json_path.display(),
                "delete_head: failed to remove .json; boot recovery will sweep",
            );
        }
        let heads = heads_dir(&workspace_dir);
        if let Err(e) = fsync_dir(&heads) {
            tracing::warn!(
                target: "file_mgr",
                err = %e,
                path = %heads.display(),
                "delete_head: failed to fsync heads/ after removal",
            );
        }
        Ok(())
    }

    /// Resolve (or lazy-load) the cache cell for `delete_head`.
    /// Mirrors `cache_cell_for_dataset` / `cache_cell` shape; we
    /// add a third copy here rather than promote one of those to
    /// `pub(crate)` because both are file-local helpers gated to
    /// their callers' invariants.
    fn cache_cell_for_head_delete(
        &self,
        ws: &WorkspaceId,
    ) -> Result<Arc<WorkspaceCacheCell>, FileError> {
        if let Some(cell) = self.caches.get(ws) {
            return Ok(cell.clone());
        }
        let workspace_dir = self.workspace_dir(ws);
        let cell = Arc::new(WorkspaceCacheCell::load_from_disk(&workspace_dir)?);
        Ok(self
            .caches
            .entry(*ws)
            .or_insert_with(|| cell.clone())
            .clone())
    }

    /// Index-atomic publish of a freshly trained or converted head
    /// into the workspace's 2-slot rotation.  Routes the producer
    /// through the same per-workspace mutex + cache cell that
    /// `delete_head` and the asset surface use, so the rotation
    /// primitive executes against the live `WorkspaceCacheCell`.
    /// Sync; never `.await` inside.
    ///
    /// Caller guarantees:
    /// - `pending.mpk_tempfile` lives under
    ///   `<workspace_dir>/.tmp/` and is fsynced.
    /// - `pending.manifest` is fully populated and the
    ///   manifest's `head_id` matches `pending.head_id`.
    pub fn publish_trained_head_for_workspace(
        self: &Arc<Self>,
        ws: &WorkspaceId,
        pending: PendingHead,
    ) -> Result<HeadRotationResult, FileError> {
        let workspace_dir = self.workspace_dir(ws);
        if !workspace_core_path(&workspace_dir).exists() {
            return Err(FileError::NotFound(ws.to_string()));
        }
        let lock = self.metadata_lock(ws);
        let _guard = lock.lock();
        let cell = self.cache_cell_for_head_delete(ws)?;
        publish_trained_head(&workspace_dir, &cell, pending)
    }
}

// MARK: Tests

#[cfg(test)]
mod tests {
    #![allow(clippy::disallowed_methods)]
    // Orphan-fixture setup intentionally bypasses atomic publish helpers.

    use super::*;
    use crate::common::ids::WorkspaceId;
    use crate::common::workspace::{WorkspaceCore, WorkspaceRevision};
    use crate::file_mgr::schema::{HEADS_DIR_NAME, write_head_index, write_workspace_core};
    use std::io::Write;

    fn ws_id() -> WorkspaceId {
        WorkspaceId::parse("11111111-2222-4333-8444-555555555550").unwrap()
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
            sha256: "def".to_string(),
            n_classes: 3,
            size_bytes: 1024,
            created_at: "2026-05-07T12:34:56Z".to_string(),
            labels: vec!["cat".to_string(), "dog".to_string(), "bird".to_string()],
        }
    }

    /// Stage a fake `.mpk` tempfile under `<workspace>/.tmp/`
    /// with deterministic-but-distinct bytes per head id so the
    /// post-publish read can verify it landed.
    fn stage_mpk_tempfile(workspace_dir: &Path, head_id: HeadId) -> PathBuf {
        let tmp_dir = workspace_dir.join(".tmp");
        std::fs::create_dir_all(&tmp_dir).unwrap();
        let path = tmp_dir.join(format!("staged-{head_id}.mpk"));
        let mut f = std::fs::File::create(&path).unwrap();
        // The rotation does not parse the .mpk; any opaque bytes
        // suffice.  Tag them with the head id so a swapped file
        // surfaces in the test assertion.
        f.write_all(format!("MPK-{head_id}").as_bytes()).unwrap();
        f.sync_all().unwrap();
        path
    }

    fn fresh_workspace() -> (tempfile::TempDir, WorkspaceCacheCell) {
        let tmp = tempfile::tempdir().unwrap();
        let core = sample_core(0, 0);
        write_workspace_core(tmp.path(), &core).unwrap();
        write_head_index(tmp.path(), &HeadIndex::default()).unwrap();
        std::fs::create_dir_all(tmp.path().join(HEADS_DIR_NAME)).unwrap();
        let cache = WorkspaceCacheCell::new(core, HeadIndex::default());
        (tmp, cache)
    }

    /// Test 1 (happy path): publish a single new head; the index
    /// reflects it; workspace.json.head_count == 1; cache holds
    /// the new state.
    #[test]
    fn publish_trained_head_happy_path() {
        let (tmp, cache) = fresh_workspace();
        let head_id = HeadId::new();
        let mpk = stage_mpk_tempfile(tmp.path(), head_id);
        let manifest = sample_manifest(head_id, 5);

        let result = publish_trained_head(
            tmp.path(),
            &cache,
            PendingHead {
                head_id,
                mpk_tempfile: mpk.clone(),
                manifest: manifest.clone(),
            },
        )
        .unwrap();

        assert!(
            result.displaced_head_id.is_none(),
            "first publish never displaces"
        );
        // .mpk and .json land under heads/.
        let mpk_final = head_artifact_path(tmp.path(), head_id);
        let json_final = head_manifest_path(tmp.path(), head_id);
        assert!(mpk_final.is_file(), "mpk landed at {}", mpk_final.display());
        assert!(json_final.is_file());
        assert!(!mpk.exists(), "tempfile was renamed away");
        // heads.json reflects exactly one head.
        let on_disk = crate::file_mgr::schema::read_head_index(tmp.path()).unwrap();
        assert_eq!(on_disk.heads.len(), 1);
        assert_eq!(on_disk.heads[0].head_id, head_id);
        // workspace.json head_count == 1.
        let core = crate::file_mgr::schema::read_workspace_core(tmp.path()).unwrap();
        assert_eq!(core.head_count, 1);
        // Cache observes the new state.
        assert_eq!(cache.heads().heads.len(), 1);
        assert_eq!(cache.core().head_count, 1);
    }

    /// Test 2 (sliding window): publish 3 heads; only the most-
    /// recent 2 stay in `heads.json`; the displaced head's files
    /// are removed from `heads/`.
    #[test]
    fn publish_three_heads_keeps_only_two_most_recent() {
        let (tmp, cache) = fresh_workspace();
        let h1 = HeadId::new();
        let h2 = HeadId::new();
        let h3 = HeadId::new();

        // Publish h1.
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
        // Publish h2.
        let mpk = stage_mpk_tempfile(tmp.path(), h2);
        publish_trained_head(
            tmp.path(),
            &cache,
            PendingHead {
                head_id: h2,
                mpk_tempfile: mpk,
                manifest: sample_manifest(h2, 2),
            },
        )
        .unwrap();
        assert_eq!(cache.heads().heads.len(), 2);
        assert_eq!(cache.core().head_count, 2);

        // Publish h3 -- displaces h1.
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
        assert_eq!(
            result.displaced_head_id,
            Some(h1),
            "h1 was the oldest, so it gets displaced",
        );
        let on_disk = crate::file_mgr::schema::read_head_index(tmp.path()).unwrap();
        assert_eq!(on_disk.heads.len(), 2);
        assert_eq!(on_disk.heads[0].head_id, h3, "newest first");
        assert_eq!(on_disk.heads[1].head_id, h2);
        // h1's files were removed.
        assert!(!head_artifact_path(tmp.path(), h1).exists());
        assert!(!head_manifest_path(tmp.path(), h1).exists());
        // h2's + h3's files remain.
        assert!(head_artifact_path(tmp.path(), h2).is_file());
        assert!(head_manifest_path(tmp.path(), h2).is_file());
        assert!(head_artifact_path(tmp.path(), h3).is_file());
        assert!(head_manifest_path(tmp.path(), h3).is_file());
        // Core's head_count is still 2 (cap, not 3).
        let core = crate::file_mgr::schema::read_workspace_core(tmp.path()).unwrap();
        assert_eq!(core.head_count, 2);
    }

    /// Orphan tolerance: a `<random>.mpk` under `heads/` not in
    /// the index is residue boot recovery sweeps; the rotation
    /// primitive must NOT touch orphans, only its own displaced
    /// entry.
    #[test]
    fn publish_does_not_disturb_unrelated_orphans() {
        let (tmp, cache) = fresh_workspace();
        let orphan_id = HeadId::new();
        // Stage an orphan directly under heads/, never reachable
        // from heads.json.heads[].
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
        assert!(orphan_json.exists(), "rotation must not touch orphans");
    }

    /// Corruption tolerance: `heads.json` references a phantom
    /// head_id whose `.mpk` never landed.  Cache load succeeds
    /// because the cache treats the index as truth and does not
    /// stat each `.mpk`; physical-file consistency is boot
    /// recovery's concern.  Later API calls that consult the
    /// index surface the corruption as a `NotFound` on the `.mpk`.
    #[test]
    fn cache_load_tolerates_index_referencing_missing_file() {
        let tmp = tempfile::tempdir().unwrap();
        let core = sample_core(0, 1);
        write_workspace_core(tmp.path(), &core).unwrap();
        let phantom = HeadId::new();
        // Hand-craft an index that points at a head whose .mpk
        // never landed.
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
        // Cache load succeeds (it doesn't stat heads/).
        let cache = WorkspaceCacheCell::load_from_disk(tmp.path()).unwrap();
        assert_eq!(cache.heads().heads.len(), 1);
        // The .mpk is genuinely missing -- a follow-up consumer
        // (e.g. delete_head) will observe NotFound on the
        // physical file.  Pin that fact here so the corruption
        // shape stays documented.
        assert!(!head_artifact_path(tmp.path(), phantom).exists());
        assert!(!head_manifest_path(tmp.path(), phantom).exists());
    }

    /// Test 5 (best-effort displaced cleanup): if step 9 fails
    /// (e.g. file already removed because of a filesystem race
    /// or the orphan-sweep fired), the rotation as a whole still
    /// succeeds because the index commit in step 7 is the source
    /// of truth.  Simulate by manually unlinking the displaced
    /// head's files between publishing slot 2 and slot 3.
    #[test]
    fn publish_succeeds_when_displaced_files_are_already_gone() {
        let (tmp, cache) = fresh_workspace();
        let h1 = HeadId::new();
        let h2 = HeadId::new();
        let h3 = HeadId::new();
        // Publish h1 + h2 normally.
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
        // Race: manually remove h1's files (simulate a
        // concurrent orphan sweep or filesystem error).
        std::fs::remove_file(head_artifact_path(tmp.path(), h1)).unwrap();
        std::fs::remove_file(head_manifest_path(tmp.path(), h1)).unwrap();
        // h3 publish still succeeds; the best-effort cleanup
        // observes the missing files and logs a warning but does
        // not propagate the error.
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
        let on_disk = crate::file_mgr::schema::read_head_index(tmp.path()).unwrap();
        assert_eq!(on_disk.heads.len(), 2);
        assert_eq!(on_disk.heads[0].head_id, h3);
        assert_eq!(on_disk.heads[1].head_id, h2);
    }

    /// Reject publishing when the manifest's head_id disagrees
    /// with the pending head_id -- defends against caller bugs
    /// that would otherwise produce a heads/<a>.mpk +
    /// heads/<a>.json containing head_id=<b> mismatch.
    #[test]
    fn publish_rejects_mismatched_head_id() {
        let (tmp, cache) = fresh_workspace();
        let h1 = HeadId::new();
        let h2 = HeadId::new();
        let mpk = stage_mpk_tempfile(tmp.path(), h1);
        // Manifest carries h2's id; PendingHead carries h1's.
        let bad_manifest = sample_manifest(h2, 1);
        let result = publish_trained_head(
            tmp.path(),
            &cache,
            PendingHead {
                head_id: h1,
                mpk_tempfile: mpk,
                manifest: bad_manifest,
            },
        );
        assert!(matches!(result, Err(FileError::Io { .. })));
        // Nothing landed.
        assert!(!head_artifact_path(tmp.path(), h1).exists());
        assert!(!head_manifest_path(tmp.path(), h1).exists());
        assert!(!head_artifact_path(tmp.path(), h2).exists());
        assert!(!head_manifest_path(tmp.path(), h2).exists());
    }

    /// Reject publishing when the staged .mpk tempfile does not
    /// exist -- defends against caller bugs that would otherwise
    /// commit an empty `heads/<id>.mpk` rename.
    #[test]
    fn publish_rejects_missing_mpk_tempfile() {
        let (tmp, cache) = fresh_workspace();
        let h1 = HeadId::new();
        let result = publish_trained_head(
            tmp.path(),
            &cache,
            PendingHead {
                head_id: h1,
                mpk_tempfile: tmp.path().join(".tmp/never-staged.mpk"),
                manifest: sample_manifest(h1, 1),
            },
        );
        assert!(matches!(result, Err(FileError::Io { .. })));
        assert!(!head_artifact_path(tmp.path(), h1).exists());
    }
}
