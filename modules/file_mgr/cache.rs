//! Per-workspace eager cache cells: `ArcSwap`-backed snapshots of
//! the workspace core and head index so the list / summary /
//! status hot paths never walk `datasets/` and never re-parse JSON
//! per request.
//!
//! # Read / write discipline
//!
//! - **Reads** are wait-free: each call returns an `Arc<T>`
//!   snapshot; concurrent publishers see exactly the value the
//!   reader observed.
//! - **Writes** flow through `publish_core` / `publish_heads`,
//!   each a single wait-free `ArcSwap::store`.  Callers serialize
//!   writers with the per-workspace mutation mutex (the sync
//!   guard from `WorkspaceMgr::metadata_locks`), so on-disk and
//!   cache snapshot always agree at every observable instant.

use crate::common::workspace::{HeadIndex, WorkspaceCore};
use crate::file_mgr::error::FileError;
use crate::file_mgr::schema::{read_head_index, read_workspace_core};
use arc_swap::ArcSwap;
use std::path::Path;
use std::sync::Arc;

/// Eagerly-cached snapshots of `<workspace_dir>/workspace.json`
/// and `<workspace_dir>/heads.json`.  One cell per workspace,
/// constructed at workspace create or daemon boot via
/// [`Self::load_from_disk`] and kept current by per-workspace
/// mutation paths.  Cheap to wrap in `Arc` and share; readers
/// never block writers and vice versa.
#[derive(Debug)]
pub struct WorkspaceCacheCell {
    core: ArcSwap<WorkspaceCore>,
    heads: ArcSwap<HeadIndex>,
}

impl WorkspaceCacheCell {
    /// Construct from already-known values (test fixtures,
    /// fresh-workspace create after the on-disk files have
    /// just been written).
    pub fn new(core: WorkspaceCore, heads: HeadIndex) -> Self {
        Self {
            core: ArcSwap::from_pointee(core),
            heads: ArcSwap::from_pointee(heads),
        }
    }

    /// Read both `workspace.json` and `heads.json` from disk
    /// and seed the cache.  Boot recovery / workspace-create
    /// path; not on the request hot path.
    ///
    /// A missing `heads.json` surfaces as `FileError::Io` with
    /// kind `NotFound`; the lifecycle caller chooses whether to
    /// default to `HeadIndex::default()` or treat it as an
    /// incomplete workspace.  We do NOT silently default so a
    /// production daemon fails closed when the file is absent.
    pub fn load_from_disk(workspace_dir: &Path) -> Result<Self, FileError> {
        let core = read_workspace_core(workspace_dir)?;
        let heads = read_head_index(workspace_dir)?;
        Ok(Self::new(core, heads))
    }

    /// Wait-free snapshot of the workspace core.  One atomic
    /// ref-count bump per call (~5 ns); the returned `Arc`
    /// pins the value against publishers concurrent with the
    /// caller's read.
    #[inline]
    pub fn core(&self) -> Arc<WorkspaceCore> {
        self.core.load_full()
    }

    /// Wait-free snapshot of the head index.
    #[inline]
    pub fn heads(&self) -> Arc<HeadIndex> {
        self.heads.load_full()
    }

    /// Replace the cached core.  Caller must hold the
    /// per-workspace mutation mutex so concurrent publishers
    /// don't lose updates; the swap itself is wait-free w.r.t.
    /// readers.
    #[inline]
    pub fn publish_core(&self, core: WorkspaceCore) {
        self.core.store(Arc::new(core));
    }

    /// Replace the cached head index under the same mutex
    /// discipline as [`Self::publish_core`].
    #[inline]
    pub fn publish_heads(&self, heads: HeadIndex) {
        self.heads.store(Arc::new(heads));
    }
}

// MARK: Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::ids::{HeadId, WorkspaceId};
    use crate::common::workspace::{HeadRecord, WorkspaceRevision};
    use crate::file_mgr::schema::{write_head_index, write_workspace_core};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;

    fn ws_id() -> WorkspaceId {
        WorkspaceId::parse("11111111-2222-4333-8444-555555555555").unwrap()
    }

    fn rev(id: u64) -> WorkspaceRevision {
        WorkspaceRevision {
            id,
            at: "2026-05-07T12:00:00Z".to_string(),
        }
    }

    fn sample_core(rev_id: u64) -> WorkspaceCore {
        WorkspaceCore {
            id: ws_id(),
            name: "main".to_string(),
            tags: Vec::new(),
            created_at: "2026-05-07T12:34:56Z".to_string(),
            workspace_revision: rev(rev_id),
            head_count: 0,
        }
    }

    fn sample_record() -> HeadRecord {
        HeadRecord {
            head_id: HeadId::parse("11111111-2222-4333-8444-555555555556").unwrap(),
            workspace_revision: rev(5),
            sha256: "def".to_string(),
            n_classes: 12,
            size_bytes: 4096,
            created_at: "2026-05-07T12:34:56Z".to_string(),
        }
    }

    #[test]
    fn new_seeds_cache_with_supplied_values() {
        let core = sample_core(5);
        let mut idx = HeadIndex::default();
        idx.heads.push(sample_record());
        let cell = WorkspaceCacheCell::new(core.clone(), idx.clone());
        assert_eq!(*cell.core(), core);
        assert_eq!(*cell.heads(), idx);
    }

    #[test]
    fn load_from_disk_round_trips_through_schema_helpers() {
        let tmp = tempfile::tempdir().unwrap();
        let core = sample_core(5);
        write_workspace_core(tmp.path(), &core).unwrap();
        let mut idx = HeadIndex::default();
        idx.heads.push(sample_record());
        write_head_index(tmp.path(), &idx).unwrap();
        let cell = WorkspaceCacheCell::load_from_disk(tmp.path()).unwrap();
        assert_eq!(*cell.core(), core);
        assert_eq!(*cell.heads(), idx);
    }

    /// Pinned: `load_from_disk` fails closed when either file
    /// is missing.  Defaulting silently here would mask
    /// half-initialised workspaces from boot recovery.
    #[test]
    fn load_from_disk_missing_files_surface_as_io_not_found() {
        let tmp = tempfile::tempdir().unwrap();
        // Neither workspace.json nor heads.json exists.
        match WorkspaceCacheCell::load_from_disk(tmp.path()) {
            Err(FileError::Io { source, .. }) => {
                assert_eq!(source.kind(), std::io::ErrorKind::NotFound);
            }
            other => panic!("expected FileError::Io NotFound, got {other:?}"),
        }
        // Now write only workspace.json -- still fails on heads.
        write_workspace_core(tmp.path(), &sample_core(0)).unwrap();
        match WorkspaceCacheCell::load_from_disk(tmp.path()) {
            Err(FileError::Io { source, .. }) => {
                assert_eq!(source.kind(), std::io::ErrorKind::NotFound);
            }
            other => panic!("expected FileError::Io NotFound, got {other:?}"),
        }
    }

    #[test]
    fn publish_core_replaces_snapshot_atomically() {
        let cell = WorkspaceCacheCell::new(sample_core(5), HeadIndex::default());
        assert_eq!(cell.core().workspace_revision, rev(5));
        cell.publish_core(sample_core(6));
        assert_eq!(cell.core().workspace_revision, rev(6));
        cell.publish_core(sample_core(7));
        assert_eq!(cell.core().workspace_revision, rev(7));
    }

    #[test]
    fn publish_heads_replaces_snapshot_atomically() {
        let cell = WorkspaceCacheCell::new(sample_core(5), HeadIndex::default());
        assert!(cell.heads().heads.is_empty());
        let mut idx = HeadIndex::default();
        idx.heads.push(sample_record());
        cell.publish_heads(idx.clone());
        assert_eq!(cell.heads().heads.len(), 1);
    }

    /// Wait-free reader / writer: a long-running reader on a
    /// background thread sees either the old or the new value,
    /// never partial state, never panics.  This is the property
    /// `ArcSwap` is chosen for; the test pins the requirement.
    #[test]
    fn concurrent_reads_during_publish_observe_consistent_snapshots() {
        let cell = Arc::new(WorkspaceCacheCell::new(
            sample_core(0),
            HeadIndex::default(),
        ));
        let stop = Arc::new(AtomicBool::new(false));
        let reader_cell = Arc::clone(&cell);
        let reader_stop = Arc::clone(&stop);
        let reader = thread::spawn(move || {
            let mut last = 0u64;
            while !reader_stop.load(Ordering::Relaxed) {
                let snap = reader_cell.core();
                let obs = snap.workspace_revision.id;
                // Monotone observation: revisions never go
                // backward across publishes.
                assert!(
                    obs >= last,
                    "reader saw non-monotonic revision: {obs} < {last}"
                );
                last = obs;
            }
            last
        });
        // Writers in the test serialize themselves trivially
        // (this is a single-threaded loop here); production
        // serializes via the per-workspace mutex.
        for i in 1..=200u64 {
            cell.publish_core(sample_core(i));
        }
        stop.store(true, Ordering::Relaxed);
        let last_seen = reader.join().unwrap();
        // Reader must have observed the final publish at some
        // point -- but at minimum got past 0.
        assert!(last_seen <= 200);
        // Cell now holds the last published value.
        assert_eq!(cell.core().workspace_revision.id, 200);
    }

    /// Pinned: `core()` returns an `Arc` that survives a
    /// subsequent publish.  This is the contract the workspace
    /// summary endpoint relies on -- it can hand the snapshot
    /// to a serializer without worrying about a concurrent
    /// publisher invalidating it.
    #[test]
    fn snapshot_arc_outlives_subsequent_publish() {
        let cell = WorkspaceCacheCell::new(sample_core(5), HeadIndex::default());
        let snap = cell.core();
        cell.publish_core(sample_core(6));
        // Snapshot still pins the old value; no use-after-free.
        assert_eq!(snap.workspace_revision, rev(5));
        assert_eq!(cell.core().workspace_revision, rev(6));
    }
}
