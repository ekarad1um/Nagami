//! Atomic delete-staging primitives for asynchronous asset and
//! workspace deletes.  Owns the four-step delete state machine
//! consumed by `WorkspaceMgr::delete_workspace_async`,
//! `WorkspaceMgr::delete_dataset_async`, and the boot-recovery
//! sweep.  Higher-level callers compose the granular helpers in
//! the documented order; the type system intentionally does NOT
//! encode the order via type-state -- the discipline lives in
//! function-doc preconditions and the `staging_publish_order`
//! test.
//!
//! # Ordering contract
//!
//! Under the per-workspace mutation mutex (or active-then-workspace
//! lock pair for whole-workspace delete):
//!
//! 1. `[write_tombstone]` -- atomically write the JSON marker
//!    that boot recovery uses to resume an interrupted delete.
//!    The tombstone is the durable "I intend to delete `target`"
//!    record.
//! 2. *Caller responsibility* -- atomically rewrite
//!    `workspace.json` to advance `dataset_revision` (only
//!    applies to dataset deletes; workspace deletes skip).
//!    Doing this BEFORE the byte rename below is the
//!    revision-before-dataset-mutation invariant: a crash here
//!    can mark heads stale conservatively but never leaves a
//!    head current after a dataset change.
//! 3. `[stage_payload]` -- atomic-rename `target` into
//!    `<staging>/<prefix>-<job_id>/payload`.  The old parent dir
//!    AND the new staging dir are fsynced so the unlink + new
//!    name both reach stable storage before the call returns.
//! 4. *Caller responsibility* -- publish the new core / head
//!    cache (so workspace summaries observe the post-delete
//!    revision) and unlock.
//! 5. *Off mutex* -- call `[drain_staged_payload]` repeatedly
//!    with a bounded budget until it returns
//!    `[DrainResult::Done]`.  Each call removes at most
//!    `budget` filesystem entries; intermediate failures are
//!    safe to retry.
//! 6. `[finalize_staged_delete]` -- remove the (now empty)
//!    payload, the stage directory, the tombstone, and fsync
//!    the staging parent.
//!
//! # Crash recovery
//!
//! At any step a crash leaves recoverable state:
//!
//! - between (1) and (2): tombstone exists, no revision bump,
//!   no byte mutation -- boot deletes the tombstone.
//! - between (2) and (3): tombstone + revision bump exist, no
//!   byte mutation -- boot deletes the tombstone (the bumped
//!   revision conservatively stales heads but the dataset is
//!   intact).
//! - between (3) and (5)/(6): tombstone + staged payload exist
//!   -- boot resumes drain + finalize.
//! - between (6) parts: tombstone or empty stage dir alone --
//!   boot finishes the cleanup.
//!
//! Boot recovery is the only consumer that walks existing
//! `.tmp/delete-*.json` files; this module supplies the reading +
//! completion primitives.

use crate::common::asset_path::AssetPath;
use crate::common::ids::{JobId, WorkspaceId};
use crate::file_mgr::error::{FileError, io_err, metadata_parse_err};
use crate::file_mgr::fs_atomic::put_atomic;
use crate::file_mgr::validate::fsync_dir;
use std::path::{Path, PathBuf};

/// Per-call entry budget for `drain_staged_payload`.  Operators
/// may override at runtime via the `[file]` TOML block.
pub const DEFAULT_DELETE_BATCH_ENTRIES: usize = 256;

/// Filename prefix for dataset-delete tombstone JSON files.
/// The full filename is `<DATASET_PREFIX><job_id>.json`.
pub const DATASET_TOMBSTONE_PREFIX: &str = "delete-assets-";
/// Filename prefix for converter-delete tombstone JSON files;
/// distinct from `DATASET_TOMBSTONE_PREFIX` so a boot scan can
/// dispatch by filename without opening each tombstone.
pub const CONVERTER_TOMBSTONE_PREFIX: &str = "delete-converters-";
/// Filename prefix for workspace-delete tombstone JSON files.
pub const WORKSPACE_TOMBSTONE_PREFIX: &str = "delete-workspace-";

/// Conventional name of the staged payload directory inside a
/// stage dir.  The actual deleted target -- a single file or a
/// directory tree -- is renamed to this path; `[drain_staged_payload]`
/// removes whatever shape it finds.
pub const STAGED_PAYLOAD_NAME: &str = "payload";

// MARK: DeleteTombstone JSON shape

/// Tombstone JSON written before staging an async delete
/// payload.  Boot recovery deserializes these to resume an
/// interrupted delete.  The discriminator (`Dataset` /
/// `Converter` / `Workspace`) is also encoded in the filename
/// prefix so a boot scan can dispatch without first opening
/// every file.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum DeleteTombstone {
    /// Async dataset file or subtree delete under `datasets/`.
    /// `path` is the asset target whose bytes have been (or will
    /// be) renamed into the workspace's staging dir; `None`
    /// records a whole-tree wipe (`DELETE /assets/datasets`).
    /// `workspace_revision_id` records the workspace revision the
    /// delete was published at, for boot-recovery diagnostics.
    Dataset {
        /// Owning delete job.
        job_id: JobId,
        /// Workspace whose `datasets/` tree was mutated.
        workspace_id: WorkspaceId,
        /// Asset path being deleted, relative to `datasets/`.
        /// `None` means the entire tree was renamed into staging
        /// (whole-tree wipe).  Old tombstones written before the
        /// whole-tree shape landed always carry `Some(path)`; the
        /// `serde(default)` keeps them deserializing.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        path: Option<AssetPath>,
        /// Workspace revision id at publish time (already
        /// bumped on disk before the tombstone was written).
        workspace_revision_id: u64,
        /// RFC3339 wall-clock at tombstone creation.
        created_at: String,
    },
    /// Async converter file or subtree delete under
    /// `converters/`.  Mirrors [`Self::Dataset`] in shape; the
    /// discriminator + filename prefix lets boot recovery treat
    /// the two trees independently.
    Converter {
        /// Owning delete job.
        job_id: JobId,
        /// Workspace whose `converters/` tree was mutated.
        workspace_id: WorkspaceId,
        /// Asset path being deleted, relative to `converters/`.
        /// `None` records a whole-tree wipe.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        path: Option<AssetPath>,
        /// Workspace revision id at publish time.
        workspace_revision_id: u64,
        /// RFC3339 wall-clock at tombstone creation.
        created_at: String,
    },
    /// Async whole-workspace delete.  No `path` -- the entire
    /// workspace directory tree was renamed under the root
    /// staging dir.
    Workspace {
        /// Owning delete job.
        job_id: JobId,
        /// Workspace being deleted.
        workspace_id: WorkspaceId,
        /// RFC3339 wall-clock at tombstone creation.
        created_at: String,
    },
}

impl DeleteTombstone {
    /// Owning delete job, regardless of variant.
    pub fn job_id(&self) -> JobId {
        match self {
            DeleteTombstone::Dataset { job_id, .. }
            | DeleteTombstone::Converter { job_id, .. }
            | DeleteTombstone::Workspace { job_id, .. } => *job_id,
        }
    }

    /// Workspace targeted by the delete, regardless of variant.
    pub fn workspace_id(&self) -> WorkspaceId {
        match self {
            DeleteTombstone::Dataset { workspace_id, .. }
            | DeleteTombstone::Converter { workspace_id, .. }
            | DeleteTombstone::Workspace { workspace_id, .. } => *workspace_id,
        }
    }

    /// Tombstone-name prefix for this variant (`delete-assets-`,
    /// `delete-converters-`, `delete-workspace-`).  Pinned per
    /// variant; boot recovery dispatches by prefix.
    fn prefix(&self) -> &'static str {
        match self {
            DeleteTombstone::Dataset { .. } => DATASET_TOMBSTONE_PREFIX,
            DeleteTombstone::Converter { .. } => CONVERTER_TOMBSTONE_PREFIX,
            DeleteTombstone::Workspace { .. } => WORKSPACE_TOMBSTONE_PREFIX,
        }
    }

    /// Conventional tombstone filename for this delete (without
    /// the staging directory prefix).  Boot recovery scans for
    /// files matching `delete-assets-*.json` /
    /// `delete-converters-*.json` / `delete-workspace-*.json`
    /// and dispatches by prefix.
    pub fn filename(&self) -> String {
        format!("{}{}.json", self.prefix(), self.job_id())
    }

    /// Conventional stage-directory name for this delete (the
    /// directory that holds the renamed payload).  Mirrors
    /// `[Self::filename]` minus the `.json` suffix.
    pub fn stage_dir_name(&self) -> String {
        format!("{}{}", self.prefix(), self.job_id())
    }
}

// MARK: StagedDelete path bundle

/// Resolved filesystem paths for one in-flight async delete.
/// Built from a `[DeleteTombstone]` and the staging-dir parent
/// (workspace `.tmp/` for dataset deletes, root `.tmp/` for
/// workspace deletes).  Carrying this struct beats re-deriving
/// the four paths at every step; it also gives tests a single
/// shape to assert against.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StagedDelete {
    /// `<staging>/delete-{assets,workspace}-<job_id>.json`.
    pub tombstone: PathBuf,
    /// `<staging>/delete-{assets,workspace}-<job_id>/`.
    pub stage_dir: PathBuf,
    /// `<staging>/delete-{assets,workspace}-<job_id>/payload`.
    pub payload: PathBuf,
}

impl StagedDelete {
    /// Resolve the bundle for a tombstone written under the
    /// given `staging_dir` (workspace `.tmp/` or root `.tmp/`).
    pub fn for_tombstone(staging_dir: &Path, tombstone: &DeleteTombstone) -> Self {
        let tombstone_path = staging_dir.join(tombstone.filename());
        let stage_dir = staging_dir.join(tombstone.stage_dir_name());
        let payload = stage_dir.join(STAGED_PAYLOAD_NAME);
        Self {
            tombstone: tombstone_path,
            stage_dir,
            payload,
        }
    }
}

// MARK: Step 1 -- write_tombstone

/// Atomically write a delete tombstone to its conventional
/// path under the staging directory.
///
/// Steps:
/// 1. Ensure `staging_dir` exists (`create_dir_all` is idempotent).
/// 2. Serialize `tombstone` to JSON.
/// 3. Hand off to `[crate::file_mgr::fs_atomic::put_atomic]`,
///    which writes a tempfile, fsyncs it, atomically renames
///    into place, and fsyncs the staging directory.
///
/// Returns the resolved `[StagedDelete]` so callers can chain
/// directly into `[stage_payload]` without re-deriving the
/// four paths.
pub fn write_tombstone(
    staging_dir: &Path,
    tombstone: &DeleteTombstone,
) -> Result<StagedDelete, FileError> {
    std::fs::create_dir_all(staging_dir).map_err(|e| io_err(staging_dir.display(), e))?;
    let staged = StagedDelete::for_tombstone(staging_dir, tombstone);
    let bytes = serde_json::to_vec(tombstone)?;
    put_atomic(&staged.tombstone, &bytes)?;
    Ok(staged)
}

// MARK: Step 3 -- stage_payload

/// Atomically rename `target` (file or directory) into
/// `staged.payload`.
///
/// Preconditions:
/// - `[write_tombstone]` has already produced the tombstone.
/// - The caller has advanced any associated revision counter
///   (e.g. `workspace.json.workspace_revision`) and synced it.
/// - `target` exists; the staging directory exists; the
///   eventual payload path does not yet exist.
///
/// Postconditions on success:
/// - `target` is gone from its original location (parent dir
///   fsynced).
/// - `staged.payload` is the renamed-in-place form (`stage_dir`
///   created and fsynced).
///
/// On any failure, the staged path may or may not exist; callers
/// must treat the operation as "in progress" and rely on boot
/// recovery to resume drain + finalize.
pub fn stage_payload(target: &Path, staged: &StagedDelete) -> Result<(), FileError> {
    let old_parent = target.parent().ok_or_else(|| {
        io_err(
            target.display(),
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "stage_payload: target has no parent directory",
            ),
        )
    })?;
    std::fs::create_dir_all(&staged.stage_dir)
        .map_err(|e| io_err(staged.stage_dir.display(), e))?;
    // Fsync the staging-dir parent so the freshly-created
    // `stage_dir` entry is durable BEFORE the rename below makes
    // its child the on-disk source of truth.  Without this, a
    // crash after the rename succeeds but before the parent
    // fsync below could leave the rename's destination dirent
    // visible while the stage_dir's own dirent (under the
    // staging parent) is not -- effectively orphaning the
    // payload from the operator's perspective on remount.
    if let Some(staging_parent) = staged.stage_dir.parent() {
        fsync_dir(staging_parent).map_err(|e| io_err(staging_parent.display(), e))?;
    }
    std::fs::rename(target, &staged.payload).map_err(|e| io_err(target.display(), e))?;
    // Old parent's directory entry update (the unlink) must
    // reach stable storage before we acknowledge the stage.
    fsync_dir(old_parent).map_err(|e| io_err(old_parent.display(), e))?;
    fsync_dir(&staged.stage_dir).map_err(|e| io_err(staged.stage_dir.display(), e))?;
    Ok(())
}

// MARK: Step 5 -- drain_staged_payload

/// Outcome of one `[drain_staged_payload]` call.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum DrainResult {
    /// All payload bytes have been removed.  The empty
    /// `staged.payload` (or its absence) remains; call
    /// `[finalize_staged_delete]` to clean up the tombstone +
    /// stage dir.
    Done,
    /// The budget was exhausted before the payload was fully
    /// drained.  Call again with a fresh budget to resume.
    More,
}

/// Remove up to `budget` filesystem entries from
/// `staged.payload`.  Returns `[DrainResult::Done]` once the
/// payload no longer holds any entries (the empty payload
/// directory itself is left in place; `[finalize_staged_delete]`
/// handles its removal alongside the tombstone + stage dir).
///
/// Idempotent: a missing `staged.payload` returns
/// `[DrainResult::Done]` immediately, so a partial crash + retry
/// converges.  Symlinks are unlinked (not followed), defending
/// against operator tampering even though `datasets/` is
/// daemon-owned.
///
/// Walk order is leaf-first via depth-first recursion so each
/// `remove_dir` sees an empty directory.  Each removed entry
/// (file or empty directory) costs one budget unit; descending
/// into a subdirectory does not.
pub fn drain_staged_payload(
    staged: &StagedDelete,
    budget: usize,
) -> Result<DrainResult, FileError> {
    // `Path::exists()` follows symlinks; a dangling symlink at
    // the payload position would make this branch return Done
    // and silently leak the symlink itself.  Use
    // `symlink_metadata` directly to detect both presence
    // (incl. broken symlinks) and type without following.
    let metadata = match std::fs::symlink_metadata(&staged.payload) {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Ok(DrainResult::Done);
        }
        Err(e) => {
            return Err(io_err(staged.payload.display(), e));
        }
    };
    if !metadata.is_dir() {
        // Single-file payload (or symlink) -- one unlink and
        // we're done.  The payload's lstat already excluded
        // following symlinks.
        std::fs::remove_file(&staged.payload).map_err(|e| io_err(staged.payload.display(), e))?;
        return Ok(DrainResult::Done);
    }
    let mut removed = 0usize;
    drain_dir(&staged.payload, budget, &mut removed)
}

/// Recursive helper for directory drain.  The payload root
/// itself is preserved when `Done`; callers rely on
/// `[finalize_staged_delete]` to remove it.
fn drain_dir(dir: &Path, budget: usize, removed: &mut usize) -> Result<DrainResult, FileError> {
    let entries = std::fs::read_dir(dir).map_err(|e| io_err(dir.display(), e))?;
    for entry_result in entries {
        if *removed >= budget {
            return Ok(DrainResult::More);
        }
        let entry = entry_result.map_err(|e| io_err(dir.display(), e))?;
        let path = entry.path();
        // `entry.file_type()` reuses the readdir-supplied
        // dirent type on filesystems that report it (ext4),
        // so this is one syscall per entry instead of two
        // (readdir + stat).  Critically it does NOT follow
        // symlinks: a symlinked-to-directory reports
        // `is_symlink()` and falls through to the unlink
        // branch rather than recursing across the link.
        let ft = entry.file_type().map_err(|e| io_err(path.display(), e))?;
        if ft.is_dir() {
            match drain_dir(&path, budget, removed)? {
                DrainResult::More => return Ok(DrainResult::More),
                DrainResult::Done => {
                    if *removed >= budget {
                        return Ok(DrainResult::More);
                    }
                    std::fs::remove_dir(&path).map_err(|e| io_err(path.display(), e))?;
                    *removed += 1;
                }
            }
        } else {
            // Regular file OR symlink: `remove_file` unlinks
            // either without following.
            std::fs::remove_file(&path).map_err(|e| io_err(path.display(), e))?;
            *removed += 1;
        }
    }
    Ok(DrainResult::Done)
}

// MARK: Step 6 -- finalize_staged_delete

/// Remove the (now empty) payload, the stage directory, and
/// the tombstone, then fsync the staging parent so the
/// cleanup is durable.  Idempotent: missing entries are
/// treated as "already gone" so a crash mid-finalize
/// converges on retry.
///
/// Precondition: `[drain_staged_payload]` has returned
/// `[DrainResult::Done]` for `staged`.  If the payload still
/// has contents the directory removal will surface as
/// `FileError::Io` (`ENOTEMPTY`).
pub fn finalize_staged_delete(staged: &StagedDelete) -> Result<(), FileError> {
    // Remove the payload directory if it still exists; a
    // single-file payload is already gone after `drain`.
    match std::fs::symlink_metadata(&staged.payload) {
        Ok(md) if md.is_dir() => {
            std::fs::remove_dir(&staged.payload)
                .map_err(|e| io_err(staged.payload.display(), e))?;
        }
        Ok(_) => {
            // Stale file or symlink -- unlink it for safety.
            std::fs::remove_file(&staged.payload)
                .map_err(|e| io_err(staged.payload.display(), e))?;
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => {
            return Err(io_err(staged.payload.display(), e));
        }
    }
    // Remove the stage dir.
    match std::fs::remove_dir(&staged.stage_dir) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => {
            return Err(io_err(staged.stage_dir.display(), e));
        }
    }
    // Remove the tombstone.
    match std::fs::remove_file(&staged.tombstone) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => {
            return Err(io_err(staged.tombstone.display(), e));
        }
    }
    // fsync the staging parent so the three removals reach
    // stable storage.  The directory is the same parent for
    // tombstone + stage_dir (they live side by side); using
    // `tombstone.parent()` is canonical.
    if let Some(staging_dir) = staged.tombstone.parent() {
        // Best-effort if the staging dir was itself removed
        // by a parallel cleaner (shouldn't happen in nominal
        // operation, but harmless).
        if staging_dir.exists() {
            fsync_dir(staging_dir).map_err(|e| io_err(staging_dir.display(), e))?;
        }
    }
    Ok(())
}

// MARK: Boot-recovery scan helper

/// Read a tombstone JSON from disk.  Wraps the deserializer so
/// boot recovery can dispatch by `[DeleteTombstone]` shape
/// without writing the parse path twice.  A malformed tombstone
/// surfaces as `FileError::MetadataParse`; the recovery caller
/// chooses whether to delete and proceed or abort.
pub fn read_tombstone(path: &Path) -> Result<DeleteTombstone, FileError> {
    let bytes = std::fs::read(path).map_err(|e| io_err(path.display(), e))?;
    serde_json::from_slice(&bytes).map_err(|source| metadata_parse_err(path.display(), source))
}

// MARK: Tests

#[cfg(test)]
mod tests {
    #![allow(clippy::disallowed_methods)]
    // Fixture setup intentionally writes staged payload bytes directly.

    use super::*;
    use std::fs;
    use std::io::Write;

    fn ws_id() -> WorkspaceId {
        WorkspaceId::parse("11111111-2222-4333-8444-555555555555").unwrap()
    }

    fn job_id() -> JobId {
        JobId::parse("22222222-3333-4444-8555-666666666666").unwrap()
    }

    fn dataset_tombstone() -> DeleteTombstone {
        DeleteTombstone::Dataset {
            job_id: job_id(),
            workspace_id: ws_id(),
            path: Some(AssetPath::parse("audio_dataset/cat").unwrap()),
            workspace_revision_id: 6,
            created_at: "2026-05-07T13:00:00Z".to_string(),
        }
    }

    fn converter_tombstone() -> DeleteTombstone {
        DeleteTombstone::Converter {
            job_id: job_id(),
            workspace_id: ws_id(),
            path: Some(AssetPath::parse("tfjs/model.json").unwrap()),
            workspace_revision_id: 7,
            created_at: "2026-05-07T13:00:00Z".to_string(),
        }
    }

    fn workspace_tombstone() -> DeleteTombstone {
        DeleteTombstone::Workspace {
            job_id: job_id(),
            workspace_id: ws_id(),
            created_at: "2026-05-07T13:00:00Z".to_string(),
        }
    }

    fn make_target_dir(root: &Path, layout: &[(&str, &[u8])]) -> PathBuf {
        let target = root.join("target");
        for (rel, bytes) in layout {
            let full = target.join(rel);
            if let Some(parent) = full.parent() {
                fs::create_dir_all(parent).unwrap();
            }
            let mut f = fs::File::create(&full).unwrap();
            f.write_all(bytes).unwrap();
        }
        target
    }

    fn count_filesystem_entries(root: &Path) -> usize {
        if !root.exists() {
            return 0;
        }
        let mut total = 0usize;
        for entry in fs::read_dir(root).unwrap() {
            let entry = entry.unwrap();
            total += 1;
            if entry.file_type().unwrap().is_dir() {
                total += count_filesystem_entries(&entry.path());
            }
        }
        total
    }

    // MARK: DeleteTombstone serde + filename derivation

    #[test]
    fn dataset_tombstone_round_trips_and_names_files() {
        let t = dataset_tombstone();
        let json = serde_json::to_string(&t).unwrap();
        let parsed: DeleteTombstone = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, t);
        assert!(t.filename().starts_with(DATASET_TOMBSTONE_PREFIX));
        assert!(t.filename().ends_with(".json"));
        // Filename / stage-dir-name must agree on the job id.
        assert_eq!(t.filename(), format!("{}.json", t.stage_dir_name()));
    }

    #[test]
    fn workspace_tombstone_round_trips_and_names_files() {
        let t = workspace_tombstone();
        let json = serde_json::to_string(&t).unwrap();
        let parsed: DeleteTombstone = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, t);
        assert!(t.filename().starts_with(WORKSPACE_TOMBSTONE_PREFIX));
    }

    #[test]
    fn converter_tombstone_round_trips_and_names_files() {
        let t = converter_tombstone();
        let json = serde_json::to_string(&t).unwrap();
        let parsed: DeleteTombstone = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, t);
        // Discriminator + filename prefix are distinct from the
        // dataset variant so a boot scan can dispatch by name.
        assert!(json.contains("\"kind\":\"converter\""));
        assert!(t.filename().starts_with(CONVERTER_TOMBSTONE_PREFIX));
        assert!(!t.filename().starts_with(DATASET_TOMBSTONE_PREFIX));
        assert_eq!(t.filename(), format!("{}.json", t.stage_dir_name()));
    }

    #[test]
    fn tombstone_rejects_unknown_fields() {
        let bad = r#"{
            "kind": "dataset",
            "job_id": "22222222-3333-4444-8555-666666666666",
            "workspace_id": "11111111-2222-4333-8444-555555555555",
            "path": "audio_dataset",
            "workspace_revision_id": 1,
            "created_at": "2026-05-07T13:00:00Z",
            "extra": "no"
        }"#;
        assert!(serde_json::from_str::<DeleteTombstone>(bad).is_err());
    }

    // MARK: write_tombstone

    #[test]
    fn write_tombstone_creates_staging_dir_and_atomic_file() {
        let tmp = tempfile::tempdir().unwrap();
        let staging = tmp.path().join(".tmp");
        // Staging dir does not yet exist.
        assert!(!staging.exists());
        let t = dataset_tombstone();
        let staged = write_tombstone(&staging, &t).unwrap();
        assert!(staging.is_dir(), "staging dir was created");
        assert!(staged.tombstone.is_file(), "tombstone written");
        // The tombstone parses back to the same shape.
        let recovered = read_tombstone(&staged.tombstone).unwrap();
        assert_eq!(recovered, t);
    }

    #[test]
    fn write_tombstone_is_idempotent_on_existing_staging() {
        let tmp = tempfile::tempdir().unwrap();
        let staging = tmp.path().join(".tmp");
        fs::create_dir_all(&staging).unwrap();
        let t = dataset_tombstone();
        // Writing twice should overwrite atomically.  Boot
        // recovery may rewrite the same tombstone if the daemon
        // restarts mid-stage; that is allowed.
        write_tombstone(&staging, &t).unwrap();
        write_tombstone(&staging, &t).unwrap();
        let staged = StagedDelete::for_tombstone(&staging, &t);
        assert!(staged.tombstone.exists());
    }

    // MARK: stage_payload

    #[test]
    fn stage_payload_renames_file_target_into_staging() {
        let tmp = tempfile::tempdir().unwrap();
        let target = tmp.path().join("dataset.wav");
        fs::write(&target, b"audio bytes").unwrap();
        let staging = tmp.path().join(".tmp");
        let t = dataset_tombstone();
        let staged = write_tombstone(&staging, &t).unwrap();
        stage_payload(&target, &staged).unwrap();
        assert!(!target.exists(), "target removed from original");
        assert!(staged.payload.is_file(), "payload rename in place");
        assert_eq!(fs::read(&staged.payload).unwrap(), b"audio bytes");
    }

    #[test]
    fn stage_payload_renames_directory_target() {
        let tmp = tempfile::tempdir().unwrap();
        let target = make_target_dir(
            tmp.path(),
            &[
                ("a.wav", b"a"),
                ("nested/b.wav", b"b"),
                ("nested/c.wav", b"c"),
            ],
        );
        let staging = tmp.path().join(".tmp");
        let t = dataset_tombstone();
        let staged = write_tombstone(&staging, &t).unwrap();
        stage_payload(&target, &staged).unwrap();
        assert!(!target.exists());
        assert!(staged.payload.is_dir());
        assert!(staged.payload.join("a.wav").is_file());
        assert!(staged.payload.join("nested/c.wav").is_file());
    }

    // MARK: drain_staged_payload

    #[test]
    fn drain_staged_payload_done_on_missing_payload() {
        let tmp = tempfile::tempdir().unwrap();
        let staging = tmp.path().join(".tmp");
        let t = dataset_tombstone();
        let staged = write_tombstone(&staging, &t).unwrap();
        // No stage_payload call -- payload doesn't exist.
        let res = drain_staged_payload(&staged, 100).unwrap();
        assert_eq!(res, DrainResult::Done);
    }

    #[test]
    fn drain_staged_payload_removes_single_file() {
        let tmp = tempfile::tempdir().unwrap();
        let target = tmp.path().join("dataset.wav");
        fs::write(&target, b"audio").unwrap();
        let staging = tmp.path().join(".tmp");
        let staged = write_tombstone(&staging, &dataset_tombstone()).unwrap();
        stage_payload(&target, &staged).unwrap();
        let res = drain_staged_payload(&staged, 1).unwrap();
        assert_eq!(res, DrainResult::Done);
        assert!(!staged.payload.exists());
    }

    #[test]
    fn drain_staged_payload_recursively_empties_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let target = make_target_dir(
            tmp.path(),
            &[
                ("a.wav", b"a"),
                ("nested/b.wav", b"b"),
                ("nested/deeper/c.wav", b"c"),
                ("nested/deeper/d.wav", b"d"),
            ],
        );
        let staging = tmp.path().join(".tmp");
        let staged = write_tombstone(&staging, &dataset_tombstone()).unwrap();
        stage_payload(&target, &staged).unwrap();
        // Generous budget -- one call clears everything.
        let res = drain_staged_payload(&staged, 1024).unwrap();
        assert_eq!(res, DrainResult::Done);
        // Payload root remains as an empty directory; finalize
        // is the step that removes it.
        assert!(staged.payload.is_dir());
        assert_eq!(count_filesystem_entries(&staged.payload), 0);
    }

    /// Pinned: the per-call budget is honoured, multi-call
    /// drain converges, and partial state between calls is
    /// always recoverable (idempotent on retry).
    #[test]
    fn drain_staged_payload_respects_budget_and_resumes() {
        let tmp = tempfile::tempdir().unwrap();
        let target = make_target_dir(
            tmp.path(),
            &[
                ("a", b"1"),
                ("b", b"2"),
                ("c", b"3"),
                ("d", b"4"),
                ("e", b"5"),
            ],
        );
        let staging = tmp.path().join(".tmp");
        let staged = write_tombstone(&staging, &dataset_tombstone()).unwrap();
        stage_payload(&target, &staged).unwrap();
        // Budget = 2: must report `More` until everything's gone.
        let mut iterations = 0usize;
        loop {
            iterations += 1;
            let res = drain_staged_payload(&staged, 2).unwrap();
            if res == DrainResult::Done {
                break;
            }
            // Safety net: 5 entries / 2 per batch = at most 3
            // iterations; bound the loop to surface infinite
            // recursion immediately.
            assert!(iterations <= 5, "drain failed to converge");
        }
        assert!(staged.payload.exists()); // root remains
        assert_eq!(count_filesystem_entries(&staged.payload), 0);
    }

    /// Symlinks under the payload tree are unlinked, not
    /// followed.  Defensive against operator tampering even
    /// though `datasets/` is daemon-owned.
    #[cfg(unix)]
    #[test]
    fn drain_staged_payload_unlinks_symlinks_without_following() {
        use std::os::unix::fs::symlink;
        let tmp = tempfile::tempdir().unwrap();
        // External target the symlink would point at.
        let outside = tmp.path().join("outside.txt");
        fs::write(&outside, b"do not delete").unwrap();
        let target = make_target_dir(tmp.path(), &[("real.wav", b"real")]);
        // Add a symlink inside the target subtree.
        symlink(&outside, target.join("link")).unwrap();
        let staging = tmp.path().join(".tmp");
        let staged = write_tombstone(&staging, &dataset_tombstone()).unwrap();
        stage_payload(&target, &staged).unwrap();
        let res = drain_staged_payload(&staged, 1024).unwrap();
        assert_eq!(res, DrainResult::Done);
        // Symlink target survives.
        assert!(outside.is_file());
        assert_eq!(fs::read(&outside).unwrap(), b"do not delete");
    }

    // MARK: finalize_staged_delete

    #[test]
    fn finalize_staged_delete_clears_tombstone_and_stage_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let target = tmp.path().join("dataset.wav");
        fs::write(&target, b"audio").unwrap();
        let staging = tmp.path().join(".tmp");
        let staged = write_tombstone(&staging, &dataset_tombstone()).unwrap();
        stage_payload(&target, &staged).unwrap();
        drain_staged_payload(&staged, 1024).unwrap();
        finalize_staged_delete(&staged).unwrap();
        assert!(!staged.tombstone.exists());
        assert!(!staged.stage_dir.exists());
        assert!(!staged.payload.exists());
        // Staging dir itself remains -- it's a daemon-managed
        // location reused by future deletes.
        assert!(staging.is_dir());
    }

    #[test]
    fn finalize_staged_delete_is_idempotent_on_missing_pieces() {
        let tmp = tempfile::tempdir().unwrap();
        let staging = tmp.path().join(".tmp");
        fs::create_dir_all(&staging).unwrap();
        let staged = StagedDelete::for_tombstone(&staging, &dataset_tombstone());
        // Nothing exists; finalize must not error.
        finalize_staged_delete(&staged).unwrap();
    }

    /// `finalize` must REFUSE to remove a non-empty payload --
    /// the precondition is that drain returned `Done`.  Catches
    /// any caller mistake of skipping drain.
    #[test]
    fn finalize_staged_delete_rejects_nonempty_payload() {
        let tmp = tempfile::tempdir().unwrap();
        let target = make_target_dir(tmp.path(), &[("a.wav", b"a")]);
        let staging = tmp.path().join(".tmp");
        let staged = write_tombstone(&staging, &dataset_tombstone()).unwrap();
        stage_payload(&target, &staged).unwrap();
        // Did NOT drain.  Finalize should fail because payload
        // is a non-empty directory.
        let res = finalize_staged_delete(&staged);
        assert!(
            res.is_err(),
            "non-empty payload must not be silently removed"
        );
    }

    // MARK: Publish-order invariants

    /// Pinned end-to-end ordering test: the four steps
    /// (tombstone -> stage -> drain -> finalize) leave the
    /// filesystem in a consistent state at every observable
    /// boundary.  Any reordering of these calls would break a
    /// boot-recovery invariant; this test is the canonical
    /// guard against that.
    #[test]
    fn staging_publish_order_end_to_end() {
        let tmp = tempfile::tempdir().unwrap();
        let target = make_target_dir(tmp.path(), &[("dataset.wav", b"a"), ("nested/b.wav", b"b")]);
        let staging = tmp.path().join(".tmp");

        // Step 1: tombstone exists; target intact.
        let t = dataset_tombstone();
        let staged = write_tombstone(&staging, &t).unwrap();
        assert!(staged.tombstone.is_file(), "step 1: tombstone present");
        assert!(target.exists(), "step 1: target unchanged");
        assert!(!staged.payload.exists(), "step 1: payload not yet staged");

        // Step 3: target moved into staging payload.
        stage_payload(&target, &staged).unwrap();
        assert!(
            staged.tombstone.is_file(),
            "step 3: tombstone still present"
        );
        assert!(!target.exists(), "step 3: target gone from original");
        assert!(staged.payload.is_dir(), "step 3: payload renamed in place");

        // Step 5: payload contents drained but root preserved.
        let res = drain_staged_payload(&staged, 1024).unwrap();
        assert_eq!(res, DrainResult::Done);
        assert!(
            staged.payload.is_dir(),
            "step 5: empty payload root remains"
        );
        assert_eq!(count_filesystem_entries(&staged.payload), 0);

        // Step 6: tombstone, stage dir, payload, all gone.
        finalize_staged_delete(&staged).unwrap();
        assert!(!staged.tombstone.exists());
        assert!(!staged.stage_dir.exists());
        assert!(staging.is_dir(), "staging dir survives for reuse");
    }

    /// Crash-resume property: after step 3 the filesystem holds
    /// (tombstone + staged payload).  A "crashed and restarted"
    /// daemon resumes by calling drain + finalize -- those must
    /// converge from the staged state without further input.
    /// Boot recovery exercises this end-to-end; here we pin the
    /// primitives' resume contract.
    #[test]
    fn drain_finalize_resumes_from_post_stage_state() {
        let tmp = tempfile::tempdir().unwrap();
        let target = make_target_dir(
            tmp.path(),
            &[
                ("a.wav", b"a"),
                ("nested/b.wav", b"b"),
                ("nested/deeper/c.wav", b"c"),
            ],
        );
        let staging = tmp.path().join(".tmp");
        let staged = write_tombstone(&staging, &dataset_tombstone()).unwrap();
        stage_payload(&target, &staged).unwrap();

        // Simulate: daemon dies here.  A fresh boot reads the
        // tombstone and re-derives the staged paths.
        let recovered_tombstone = read_tombstone(&staged.tombstone).unwrap();
        let recovered_staged = StagedDelete::for_tombstone(&staging, &recovered_tombstone);
        assert_eq!(recovered_staged, staged);

        // Drive the pipeline forward without re-staging.
        loop {
            let res = drain_staged_payload(&recovered_staged, 1).unwrap();
            if res == DrainResult::Done {
                break;
            }
        }
        finalize_staged_delete(&recovered_staged).unwrap();
        assert!(!staged.tombstone.exists());
    }

    /// Boot-scan dispatch: the filename prefix classifies a
    /// tombstone before the file is opened.  Pinned because boot
    /// recovery reads directory listings without parsing every
    /// tombstone JSON.
    #[test]
    fn tombstone_filename_prefix_dispatches_kind() {
        let dataset = dataset_tombstone();
        let converter = converter_tombstone();
        let workspace = workspace_tombstone();
        // Each variant's filename prefix is unique so a boot scan
        // can classify by name before opening the JSON body.
        assert!(dataset.filename().starts_with(DATASET_TOMBSTONE_PREFIX));
        assert!(!dataset.filename().starts_with(CONVERTER_TOMBSTONE_PREFIX));
        assert!(!dataset.filename().starts_with(WORKSPACE_TOMBSTONE_PREFIX));
        assert!(converter.filename().starts_with(CONVERTER_TOMBSTONE_PREFIX));
        assert!(!converter.filename().starts_with(DATASET_TOMBSTONE_PREFIX));
        assert!(!converter.filename().starts_with(WORKSPACE_TOMBSTONE_PREFIX));
        assert!(workspace.filename().starts_with(WORKSPACE_TOMBSTONE_PREFIX));
        assert!(!workspace.filename().starts_with(DATASET_TOMBSTONE_PREFIX));
        assert!(!workspace.filename().starts_with(CONVERTER_TOMBSTONE_PREFIX));
    }

    // MARK: Defaults / constants pinned

    #[test]
    fn default_delete_batch_entries_matches_storage_table() {
        // Pinned at 256; bumping requires touching the docs.
        assert_eq!(DEFAULT_DELETE_BATCH_ENTRIES, 256);
    }
}
