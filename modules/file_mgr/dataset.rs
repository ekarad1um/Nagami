//! Workspace asset-tree surface: the `AssetPath`-shaped methods
//! on [`WorkspaceMgr`] that back the `GET /assets` /
//! `GET /assets/{*path}` / `PUT /assets/{*path}` /
//! `DELETE /assets/{*path}` routes.
//!
//! # Method taxonomy
//!
//! - [`WorkspaceMgr::workspace_asset_path`] -- pure path join
//!   `<workspace_dir>/<asset_path>`.  No I/O beyond a workspace
//!   existence check; rejects `.tmp/`.
//! - [`WorkspaceMgr::list_workspace_children`] -- bounded
//!   direct-child listing for `GET /assets`; read-only, never
//!   takes the per-workspace mutation mutex, never recursive.
//! - [`WorkspaceMgr::open_workspace_file`] -- resolve + stat a
//!   regular file; returns path + metadata so the route streams
//!   via `tokio::fs::File` WITHOUT holding the workspace mutex.
//! - [`WorkspaceMgr::upload_workspace_file`] -- atomic single-file
//!   commit.  Holds the per-workspace mutation mutex across
//!   job-conflict check, sibling-collision check,
//!   revision-before-bytes write, atomic rename + parent fsync,
//!   and cache publish.
//! - [`WorkspaceMgr::start_workspace_asset_delete`] -- begin async
//!   asset-tree delete (all four mutable trees).  Holds the
//!   per-workspace mutex across tombstone + (datasets/converters
//!   only) revision-bump + stage-payload, then drains off-mutex.
//!   Boot recovery resumes interrupted drains.
//!
//! Conflict admission consumes the cross-cutting
//! [`crate::file_mgr::job_registry::JobRegistry`]; HTTP 409
//! `JobConflict` is the wire contract.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::common::asset_path::AssetPath;
use crate::common::ids::{JobId, WorkspaceId};
use crate::common::workspace::{JobReference, JobType, WorkspaceCore, WorkspaceRevision};
use crate::file_mgr::WorkspaceMgr;
use crate::file_mgr::error::{FileError, io_err};
use crate::file_mgr::job_registry::JobHandle;
use crate::file_mgr::schema::{workspace_core_path, write_workspace_core};
use crate::file_mgr::staging::{
    DEFAULT_DELETE_BATCH_ENTRIES, DeleteTombstone, DrainResult, StagedDelete, drain_staged_payload,
    finalize_staged_delete, stage_payload, write_tombstone,
};
use crate::file_mgr::time_util::{now_rfc3339, rfc3339_from};
use crate::file_mgr::validate::fsync_dir;

/// Subdirectory holding daemon-owned dataset bytes.  Defined here
/// because this module owns the dataset surface; `schema.rs` froze
/// the layout earlier and remains the source of truth for paths.
pub const DATASETS_DIR_NAME: &str = "datasets";

/// Subdirectory holding daemon-owned converter input bytes.
pub const CONVERTERS_DIR_NAME: &str = "converters";

/// Subdirectory holding the trainer's per-job JSONL log backstop.
/// Producer: `TrainJobLog` in [`crate::training`].  Deletes drain
/// through the unified async asset surface (with a producer-active
/// pre-check that returns 409 while a Train job for the workspace
/// is running).
pub const TRAINING_LOGS_DIR_NAME: &str = "training_logs";

/// Subdirectory holding the converter's per-job JSONL log
/// backstop.  Producer: `ConvertJobLog` in
/// [`crate::converter`].  (Linked here as plain text rather
/// than an intra-doc link because the helper is private to the
/// converter crate; rustdoc cannot resolve a `pub(crate)` item
/// from a sibling module.)
pub const CONVERTER_LOGS_DIR_NAME: &str = "converter_logs";

/// Discriminator for which workspace tree a workspace-rooted
/// asset path targets.  Drives the tombstone variant + `JobType`
/// for async deletes; for log trees it also selects the
/// producer-conflict pre-check (Train for `training_logs`,
/// Convert for `converter_logs`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AssetTree {
    /// Datasets tree (`<workspace_dir>/datasets/`).  Operator
    /// uploads here; deletes are async (JobType::DatasetDelete).
    Datasets,
    /// Converters tree (`<workspace_dir>/converters/`).  Operator
    /// uploads here; deletes are async (JobType::ConverterDelete).
    Converters,
    /// Trainer's per-job JSONL backstop
    /// (`<workspace_dir>/training_logs/`).  Daemon-only producer;
    /// no operator uploads.  Deletes are async
    /// (JobType::TrainingLogsDelete) but pre-checked against the
    /// workspace's active Train job (returns 409 if running).
    TrainingLogs,
    /// Converter's per-job JSONL backstop
    /// (`<workspace_dir>/converter_logs/`).  Daemon-only producer;
    /// no operator uploads.  Deletes are async
    /// (JobType::ConverterLogsDelete) but pre-checked against the
    /// workspace's active Convert job (returns 409 if running).
    ConverterLogs,
}

impl AssetTree {
    /// On-disk subdirectory name.
    pub fn dir_name(self) -> &'static str {
        match self {
            AssetTree::Datasets => DATASETS_DIR_NAME,
            AssetTree::Converters => CONVERTERS_DIR_NAME,
            AssetTree::TrainingLogs => TRAINING_LOGS_DIR_NAME,
            AssetTree::ConverterLogs => CONVERTER_LOGS_DIR_NAME,
        }
    }

    /// Whether operator-driven uploads target this tree.  Logs are
    /// daemon-only producers; rejecting uploads to log dirs at
    /// the validator keeps the upload route honest after the
    /// "any mutable tree" relaxation in `parse_mutable_path`.
    pub fn accepts_uploads(self) -> bool {
        matches!(self, AssetTree::Datasets | AssetTree::Converters)
    }

    /// `JobType` for the async-delete job that wipes this tree.
    /// Total over the four variants; both `start_async_tree_delete`
    /// (datasets/converters) and `start_async_log_delete`
    /// (training/converter logs) admit through this mapping so the
    /// per-tree job-type discriminator lives in one place.
    fn async_delete_job_type(self) -> JobType {
        match self {
            AssetTree::Datasets => JobType::DatasetDelete,
            AssetTree::Converters => JobType::ConverterDelete,
            AssetTree::TrainingLogs => JobType::TrainingLogsDelete,
            AssetTree::ConverterLogs => JobType::ConverterLogsDelete,
        }
    }
}

/// Clone `prev_core`, advance its `workspace_revision` (id +1,
/// timestamp = now), and return the bumped core alongside the new
/// revision id.  Used by `upload_workspace_file` and
/// `start_async_tree_delete` (datasets/converters), both of which
/// write the bumped `workspace.json` revision-before-bytes and
/// need the same `id.saturating_add(1)` policy on overflow.  The
/// caller still owns the cache publish and the `workspace.json`
/// write.  Log-tree deletes do NOT call this -- logs aren't
/// workspace state in the §9 sense.
fn bump_workspace_revision(prev_core: &WorkspaceCore) -> (u64, WorkspaceCore) {
    let next_revision_id = prev_core.workspace_revision.id.saturating_add(1);
    let mut next_core = prev_core.clone();
    next_core.workspace_revision = WorkspaceRevision {
        id: next_revision_id,
        at: now_rfc3339(),
    };
    (next_revision_id, next_core)
}

/// Parse a workspace-rooted path into a `(tree, sub)` pair where
/// `sub` is `None` for the bare-tree form (e.g. `"datasets"`,
/// `"training_logs"`) and `Some(rel)` for a child-rooted path
/// (e.g. `"datasets/cls/sample.wav"`).  The bare-tree form is
/// used by `DELETE /assets/<tree>` whole-tree wipes; uploads and
/// single-file / sub-tree deletes always carry a `Some(rel)`.
fn parse_mutable_path(path: &AssetPath) -> Result<(AssetTree, Option<AssetPath>), FileError> {
    let mut comps = path.components();
    let first = comps.next().ok_or_else(|| {
        FileError::InvalidName(format!("workspace path empty: {}", path.as_str()))
    })?;
    let tree = match first {
        DATASETS_DIR_NAME => AssetTree::Datasets,
        CONVERTERS_DIR_NAME => AssetTree::Converters,
        TRAINING_LOGS_DIR_NAME => AssetTree::TrainingLogs,
        CONVERTER_LOGS_DIR_NAME => AssetTree::ConverterLogs,
        other => {
            return Err(FileError::InvalidName(format!(
                "workspace mutation path top-level must be one of \
                 `datasets` / `converters` / `training_logs` / `converter_logs`; got {other:?}",
            )));
        }
    };
    // The bare-tree form (`<tree>` with no separator after) means
    // the caller is asking to address the tree as a whole; deletes
    // dispatch to a whole-tree wipe.
    let Some(sub_raw) = path
        .as_str()
        .strip_prefix(tree.dir_name())
        .and_then(|tail| tail.strip_prefix('/'))
    else {
        return Ok((tree, None));
    };
    let sub = AssetPath::parse(sub_raw).map_err(|e| {
        FileError::InvalidName(format!(
            "internal: failed to re-parse sub-path {sub_raw:?} from {}: {e}",
            path.as_str()
        ))
    })?;
    Ok((tree, Some(sub)))
}

/// Datasets-only upload gate: require `datasets/<class>/<file>`
/// at minimum so the trainer's first-subdir-as-class convention
/// finds the bytes.  Deletes don't call this -- removing an
/// entire class folder via `DELETE datasets/<class>` is
/// supported.
fn validate_dataset_upload_depth(tree: AssetTree, path: &AssetPath) -> Result<(), FileError> {
    if tree != AssetTree::Datasets {
        return Ok(());
    }
    if path.components().count() < 3 {
        return Err(FileError::InvalidName(format!(
            "dataset upload requires `datasets/<class>/<file>` minimum; got `{}` -- \
             the trainer treats the first non-hidden subdirectory of `datasets/` as the class label",
            path.as_str(),
        )));
    }
    Ok(())
}

/// Dispatch mutation-rejection emission by `AssetTree`.  Dataset
/// and converter rejections land on separate counters; log-tree
/// rejections (which only occur on a hostile or buggy upload
/// attempt — uploads are rejected before this counter) bucketize
/// on the closest tree so an operator dashboard still sees them.
fn emit_mutation_rejected(tree: AssetTree) {
    match tree {
        AssetTree::Datasets | AssetTree::TrainingLogs => {
            crate::file_mgr::metrics_hooks::emit_dataset_mutation_rejected()
        }
        AssetTree::Converters | AssetTree::ConverterLogs => {
            crate::file_mgr::metrics_hooks::emit_converter_mutation_rejected()
        }
    }
}

/// Per-call default for [`WorkspaceMgr::list_workspace_children`];
/// operators override per request via `?limit=`.
pub const DEFAULT_DATASET_LIST_LIMIT: usize = 100;
/// Hard ceiling on `?limit=`.
pub const MAX_DATASET_LIST_LIMIT: usize = 1000;

/// Per-log-tree shape constraint: the only addressable shapes
/// are the bare tree (`training_logs` / `converter_logs` —
/// whole-tree wipe) and a single `.jsonl` filename
/// (`training_logs/<id>.jsonl`).  Nested sub-paths and
/// non-`.jsonl` extensions are rejected at the dispatcher
/// before any state mutation.
fn validate_log_subpath(sub: &AssetPath) -> Result<(), FileError> {
    let comp_count = sub.components().count();
    if comp_count != 1 {
        return Err(FileError::InvalidName(format!(
            "log file path must be a single component (e.g. `<job_id>.jsonl`); \
             got {comp_count} components in {:?}",
            sub.as_str(),
        )));
    }
    let name = sub.as_str();
    if !name.ends_with(".jsonl") {
        return Err(FileError::InvalidName(format!(
            "log file path must end in `.jsonl`; got {name:?}",
        )));
    }
    Ok(())
}

/// Workspace-relative on-disk target for an async asset delete:
/// `<workspace_dir>/<tree>/<sub>` for sub-path wipes,
/// `<workspace_dir>/<tree>` for whole-tree wipes.  Shared by
/// the dataset/converter and log-tree async dispatchers; the
/// per-tree differences are handled by the caller.
fn build_delete_target(workspace_dir: &Path, tree: AssetTree, sub: Option<&AssetPath>) -> PathBuf {
    let mut p = workspace_dir.join(tree.dir_name());
    if let Some(rel) = sub {
        for component in rel.components() {
            p.push(component);
        }
    }
    p
}

/// Snapshot's display target_path for an async asset delete:
/// the bare tree name for whole-tree wipes, otherwise
/// `<tree>/<sub>`.  Shared by the dataset/converter and
/// log-tree async dispatchers.
///
/// Precondition (held by both call sites):
///
/// * `sub` was produced by [`parse_mutable_path`] stripping a
///   tree-name prefix off a parsed [`AssetPath`].  Therefore
///   `tree.dir_name() + "/" + sub.as_str()` reconstructs the
///   original validated path's bytes byte-for-byte; depth, length,
///   per-component, and allowed-byte invariants all round-trip.
///   `AssetPath::parse` cannot fail under this precondition; an
///   arbitrary `sub` (e.g. one already at `MAX_DEPTH`) would
///   overflow when joined and is NOT a supported input.
fn build_display_path(tree: AssetTree, sub: Option<&AssetPath>) -> AssetPath {
    let raw = match sub {
        Some(rel) => format!("{}/{}", tree.dir_name(), rel.as_str()),
        None => tree.dir_name().to_string(),
    };
    AssetPath::parse(&raw).expect(
        "build_display_path precondition violated: \
         join exceeds AssetPath limits (sub must come from a tree-prefix strip)",
    )
}

// MARK: DatasetEntry / DatasetListing

/// Filesystem entry kind reported by
/// [`WorkspaceMgr::list_workspace_children`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum EntryKind {
    /// Regular file.
    File,
    /// Subdirectory.
    Directory,
}

/// One direct child of a workspace asset directory.
/// `size_bytes` is `Some(_)` for files, `None` for directories;
/// recursive size aggregation is intentionally not exposed.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub struct DatasetEntry {
    /// Filename component (no path separators).
    pub name: String,
    /// File or directory.
    #[serde(flatten)]
    pub kind: EntryKind,
    /// Byte size for files; absent for directories.
    pub size_bytes: Option<u64>,
    /// RFC3339 last-modification timestamp (UTC) reported by the
    /// filesystem.  Always present so clients can sort by recency
    /// uniformly across files and directories.  Birth time is
    /// **not** exposed here -- POSIX status-change time is not
    /// the intended semantic, and Linux btime requires
    /// `statx(STATX_BTIME)` plumbing that has not landed yet;
    /// adding a `ctime` field later is non-breaking, so the
    /// minimum bar this PR ships is mtime alone.
    pub mtime: String,
}

/// Paginated direct-child listing under a workspace asset directory.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub struct DatasetListing {
    /// Sorted (by name) page of entries.
    pub entries: Vec<DatasetEntry>,
    /// Total entry count under the parent (pre-pagination) so
    /// clients can size scrollbars without re-listing.
    pub total: usize,
    /// Echoed offset (for client convenience).
    pub offset: usize,
    /// Echoed limit (after clamping to [`MAX_DATASET_LIST_LIMIT`]).
    pub limit: usize,
}

/// Receipt returned by [`WorkspaceMgr::upload_workspace_file`].
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub struct DatasetUploadReceipt {
    /// Validated workspace-rooted asset path (including the
    /// `datasets/` or `converters/` tree-root component).
    pub path: AssetPath,
    /// Lowercase-hex sha256 the caller observed during streaming.
    pub sha256: String,
    /// Body byte count the caller observed during streaming.
    pub size_bytes: u64,
    /// `workspace_revision.id` AFTER the bump.
    pub workspace_revision_id: u64,
}

// MARK: WorkspaceMgr methods

impl WorkspaceMgr {
    /// Pure path join `<workspace_dir>/<asset_path>`; the path
    /// already carries the `datasets/` or `converters/` tree-root
    /// component.  Upload + delete dispatchers call this after
    /// `validate_mutable_subpath` has cleared the top-level.
    pub(crate) fn workspace_asset_path_join(&self, ws: &WorkspaceId, path: &AssetPath) -> PathBuf {
        let mut p = self.workspace_dir(ws);
        for component in path.components() {
            p.push(component);
        }
        p
    }

    /// Resolve a workspace-rooted asset path to its absolute disk
    /// path after validating the workspace exists; does not stat
    /// the file.  `.tmp/` paths are rejected so the staging
    /// surface stays unaddressable (defense in depth -- the
    /// `AssetPath` parser already disallows leading dots).
    pub fn workspace_asset_path(
        &self,
        ws: &WorkspaceId,
        path: &AssetPath,
    ) -> Result<PathBuf, FileError> {
        let dir = self.workspace_dir(ws);
        if !workspace_core_path(&dir).exists() {
            return Err(FileError::NotFound(ws.to_string()));
        }
        if let Some(first) = path.components().next()
            && first == ".tmp"
        {
            return Err(FileError::InvalidName(format!(
                "internal staging path is not externally addressable: {}",
                path.as_str()
            )));
        }
        Ok(self.workspace_asset_path_join(ws, path))
    }

    /// Validate the file exists, is a regular file (not directory
    /// or symlink), and return its path + metadata.  Streaming
    /// happens at the route layer via `tokio::fs::File::open` on
    /// the returned `PathBuf` AFTER the workspace mutation mutex
    /// (if any) is released.
    pub fn open_workspace_file(
        &self,
        ws: &WorkspaceId,
        path: &AssetPath,
    ) -> Result<(PathBuf, std::fs::Metadata), FileError> {
        let target = self.workspace_asset_path(ws, path)?;
        // `symlink_metadata` does not follow symlinks; a
        // symlinked-to-file fails the regular-file gate below
        // even if the link target is a regular file.  Defends
        // against operator tampering even though `datasets/`
        // and `converters/` are daemon-owned.
        let md = std::fs::symlink_metadata(&target)
            .map_err(|source| io_err(target.display(), source))?;
        if !md.is_file() {
            return Err(io_err(
                target.display(),
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "open_workspace_file: target is not a regular file",
                ),
            ));
        }
        Ok((target, md))
    }

    /// Bounded direct-child listing rooted at the workspace dir.
    /// `relative = None` lists the workspace top-level (excluding
    /// `.tmp/`); `relative = Some(p)` lists `<workspace_dir>/<p>/`.
    /// Sorted by name; `limit` is clamped to
    /// [`MAX_DATASET_LIST_LIMIT`].  Never holds the workspace
    /// mutation mutex.
    pub fn list_workspace_children(
        &self,
        ws: &WorkspaceId,
        relative: Option<&AssetPath>,
        offset: usize,
        limit: usize,
    ) -> Result<DatasetListing, FileError> {
        let workspace_dir = self.workspace_dir(ws);
        if !workspace_core_path(&workspace_dir).exists() {
            return Err(FileError::NotFound(ws.to_string()));
        }
        let target = match relative {
            Some(p) => self.workspace_asset_path_join(ws, p),
            None => workspace_dir.clone(),
        };
        // Defensive check: if a workspace-rooted relative path
        // started with `.tmp`, the listing surface must NOT
        // expose internal staging entries.  `AssetPath`'s
        // leading-dot rule already blocks this, but the explicit
        // guard documents the contract.
        if let Some(p) = relative
            && let Some(first) = p.components().next()
            && first == ".tmp"
        {
            return Err(FileError::InvalidName(format!(
                "internal staging path is not externally addressable: {}",
                p.as_str()
            )));
        }
        if !target.exists() {
            return Ok(DatasetListing {
                entries: Vec::new(),
                total: 0,
                offset,
                limit: limit.min(MAX_DATASET_LIST_LIMIT),
            });
        }
        let md = std::fs::symlink_metadata(&target)
            .map_err(|source| io_err(target.display(), source))?;
        if !md.is_dir() {
            return Err(io_err(
                target.display(),
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "list_workspace_children: target is not a directory",
                ),
            ));
        }
        // Workspace-root listing excludes `.tmp/` so operators see
        // the workspace structure without navigating into
        // delete-staging payloads.
        let exclude_tmp = relative.is_none();
        let read = std::fs::read_dir(&target).map_err(|source| io_err(target.display(), source))?;
        let mut all: Vec<DatasetEntry> = Vec::new();
        for entry in read {
            let entry = entry.map_err(|source| io_err(target.display(), source))?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if exclude_tmp && name == ".tmp" {
                continue;
            }
            // One stat per entry: serves the file-type
            // discriminator, the byte size on files, and the
            // mtime stamp the listing surfaces.  `entry.metadata()`
            // is the same `lstat` syscall `entry.file_type()` would
            // issue on most filesystems, so collapsing to a single
            // call keeps the listing's syscall budget unchanged.
            let md = entry
                .metadata()
                .map_err(|source| io_err(entry.path().display(), source))?;
            let mtime = md
                .modified()
                .map(rfc3339_from)
                .unwrap_or_else(|_| String::from("1970-01-01T00:00:00Z"));
            let ft = md.file_type();
            if ft.is_dir() {
                all.push(DatasetEntry {
                    name,
                    kind: EntryKind::Directory,
                    size_bytes: None,
                    mtime,
                });
            } else if ft.is_file() {
                all.push(DatasetEntry {
                    name,
                    kind: EntryKind::File,
                    size_bytes: Some(md.len()),
                    mtime,
                });
            }
        }
        all.sort_by(|a, b| a.name.cmp(&b.name));
        let total = all.len();
        let limit = limit.min(MAX_DATASET_LIST_LIMIT);
        let entries = all.into_iter().skip(offset).take(limit).collect::<Vec<_>>();
        Ok(DatasetListing {
            entries,
            total,
            offset,
            limit,
        })
    }

    /// Commit a single-file workspace asset upload.
    ///
    /// Preconditions enforced by the caller (the API route):
    /// - `src_tmpfile` is on the same filesystem as
    ///   the workspace root (typically a tempfile
    ///   under `<workspace_dir>/.tmp/`).
    /// - `observed_sha256` / `observed_size` were computed
    ///   incrementally during streaming and accurately reflect
    ///   `src_tmpfile`'s contents.
    ///
    /// Holds the per-workspace mutation mutex across:
    /// 1. job-conflict gate (`DatasetRefRegistry`),
    /// 2. case-insensitive sibling collision check,
    /// 3. `workspace.json` revision-bump (atomic rewrite),
    /// 4. tempfile rename + parent fsync,
    /// 5. cache publish.
    ///
    /// Revision-before-bytes is the publish-order invariant: a
    /// crash between (3) and (4) leaves a workspace whose
    /// `workspace_revision` conservatively stales heads but the
    /// asset bytes are unchanged; a crash AFTER (4) leaves the
    /// new bytes referenced by an already-bumped revision.
    /// Neither shape leaves a head current after a mutation.
    pub fn upload_workspace_file(
        self: &Arc<Self>,
        ws: &WorkspaceId,
        path: &AssetPath,
        src_tmpfile: &Path,
        observed_sha256: &str,
        observed_size: u64,
    ) -> Result<DatasetUploadReceipt, FileError> {
        // Validate before the lease so a malformed path returns 400
        // without contending the registry.  The tree drives the
        // rejection-counter dispatch (dataset vs converter).
        let (tree, sub) = parse_mutable_path(path)?;
        // Uploads target only datasets/ and converters/; the log
        // trees are daemon-produced.  Reject up front so a hostile
        // or buggy client cannot smuggle bytes into
        // `training_logs/` or `converter_logs/` via the unified
        // path validator.
        if !tree.accepts_uploads() {
            emit_mutation_rejected(tree);
            return Err(FileError::InvalidName(format!(
                "uploads to {} are reserved for the daemon producer",
                tree.dir_name(),
            )));
        }
        // Uploads always carry a sub-path; the bare `<tree>` form
        // is a delete-only shape.
        let Some(_sub) = sub else {
            emit_mutation_rejected(tree);
            return Err(FileError::InvalidName(format!(
                "upload path requires at least one child component below `{}/`: {}",
                tree.dir_name(),
                path.as_str(),
            )));
        };
        // Datasets-only depth gate; emit on the dataset bucket
        // for symmetry with the lease-conflict path below.
        if let Err(e) = validate_dataset_upload_depth(tree, path) {
            emit_mutation_rejected(tree);
            return Err(e);
        }

        // Bare workspace-scoped lease blocks only an active
        // `WorkspaceDelete`; uploads and file-deletes overlap each
        // other.  Rejections land on the per-tree counter so
        // operators see contention separately from successful
        // uploads.
        let _ref_guard = self
            .jobs
            .try_acquire_lease(vec![JobReference::Workspace { workspace_id: *ws }])
            .map_err(|e| {
                emit_mutation_rejected(tree);
                FileError::from(e)
            })?;

        let workspace_dir = self.workspace_dir(ws);
        if !workspace_core_path(&workspace_dir).exists() {
            return Err(FileError::NotFound(ws.to_string()));
        }
        // Workspace-rooted join: the path already carries the tree
        // component; the `create_dir_all` later in this function
        // covers workspaces created before the converter directory
        // layout was introduced.
        let final_path = self.workspace_asset_path_join(ws, path);

        // Per-workspace mutation mutex.  Sync; never `.await`
        // inside.  Reuse the existing `metadata_locks` map so
        // the dataset surface contends with the legacy
        // upload / metadata write path on the same key.
        let lock = self.metadata_lock(ws);
        let _guard = lock.lock();

        // Sibling collision check: case-insensitive comparison
        // against every entry in the same parent directory.
        // Held INSIDE the lock so two concurrent uploads of
        // `Foo.mpk` and `foo.mpk` on case-insensitive FS
        // (macOS/HFS+, Windows/NTFS) cannot both pass the scan
        // before either commits.  The parent may not exist yet
        // for a deeper subtree -- a missing parent means no
        // siblings, no collision.
        if let Some(parent) = final_path.parent()
            && parent.exists()
        {
            let target_name = final_path
                .file_name()
                .and_then(|s| s.to_str())
                .ok_or_else(|| FileError::InvalidName(path.as_str().to_string()))?;
            for entry in
                std::fs::read_dir(parent).map_err(|source| io_err(parent.display(), source))?
            {
                let entry = entry.map_err(|source| io_err(parent.display(), source))?;
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if name_str.eq_ignore_ascii_case(target_name) && name_str != target_name {
                    emit_mutation_rejected(tree);
                    return Err(FileError::NameConflict(format!(
                        "workspace upload {target_name:?} collides case-insensitively with \
                         existing {name_str:?}",
                    )));
                }
            }
        }

        let cell = self.cache_cell_for_dataset(ws)?;
        let (next_revision_id, next_core) = bump_workspace_revision(&cell.core());

        // (3) Revision-before-bytes: rewrite `workspace.json`
        // atomically before renaming the staged tempfile into
        // place.  `put_atomic` fsyncs the file + parent.
        write_workspace_core(&workspace_dir, &next_core)?;

        // (4) Atomic rename + parent fsync.  Create any
        // missing intermediate components first so a deeply
        // nested upload (`a/b/c/d.wav`) doesn't fail on
        // ENOENT.
        if let Some(parent) = final_path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| io_err(parent.display(), source))?;
        }
        std::fs::rename(src_tmpfile, &final_path)
            .map_err(|source| io_err(final_path.display(), source))?;
        if let Some(parent) = final_path.parent() {
            fsync_dir(parent).map_err(|source| io_err(parent.display(), source))?;
        }

        // (5) Publish the new core to the cache.
        cell.publish_core(next_core);

        // Count the successful upload + size via the file_mgr
        // metrics hook (which the daemon installs at boot to
        // forward into `WorkspaceMetrics::record_upload`).
        // Hooked after
        // the cache publish so a mid-flight failure (cache
        // primitive panic, etc.) does not over-count.
        crate::file_mgr::metrics_hooks::emit_upload(observed_size);

        Ok(DatasetUploadReceipt {
            path: path.clone(),
            sha256: observed_sha256.to_string(),
            size_bytes: observed_size,
            workspace_revision_id: next_revision_id,
        })
    }

    /// Workspace-asset delete dispatcher.
    ///
    /// All four trees share one async tombstone+stage+drain
    /// shape: the rename + tombstone land durably under the
    /// per-workspace mutex, then the staged payload drains
    /// off-mutex on the blocking pool.  Boot recovery resumes
    /// any drain interrupted by a daemon crash.
    ///
    /// Per-tree differences:
    ///
    /// - `datasets/...` / `converters/...` bump
    ///   `workspace_revision` (revision-before-bytes invariant)
    ///   and use `JobType::DatasetDelete` / `ConverterDelete` +
    ///   the matching tombstone variant.
    /// - `training_logs/...` / `converter_logs/...` skip the
    ///   revision bump (logs aren't workspace state in the §9
    ///   sense) and use `JobType::TrainingLogsDelete` /
    ///   `ConverterLogsDelete` + the log tombstone variant.
    ///   The dispatcher pre-checks for an active producer
    ///   (`Train` for `training_logs`, `Convert` for
    ///   `converter_logs`) in the same workspace and refuses
    ///   with `JobConflict`; the check is best-effort (the
    ///   registry doesn't synchronise with the workspace mutex,
    ///   so a producer can still start between the check and
    ///   stage_payload).  Sub-paths under log trees must be a
    ///   single `.jsonl` filename; nested sub-paths or other
    ///   extensions are rejected at the dispatcher.
    ///
    /// Returns the owning [`JobId`] in every case; the route
    /// layer maps to `202 Accepted` + `{ job_id }`.  Clients
    /// poll `GET /jobs/{job_id}` or stream
    /// `GET /jobs/{job_id}/events` for terminal state.
    pub fn start_workspace_asset_delete(
        self: &Arc<Self>,
        ws: &WorkspaceId,
        path: &AssetPath,
    ) -> Result<JobId, FileError> {
        let (tree, sub) = parse_mutable_path(path)?;
        let workspace_dir = self.workspace_dir(ws);
        if !workspace_core_path(&workspace_dir).exists() {
            return Err(FileError::NotFound(ws.to_string()));
        }
        match tree {
            AssetTree::Datasets | AssetTree::Converters => {
                self.start_async_tree_delete(ws, &workspace_dir, tree, sub)
            }
            AssetTree::TrainingLogs | AssetTree::ConverterLogs => {
                self.start_async_log_delete(ws, &workspace_dir, tree, sub)
            }
        }
    }

    /// Async-delete branch shared by datasets and converters.
    /// `sub = None` means whole-tree wipe: stage the entire tree
    /// dir under `<workspace>/.tmp/delete-<kind>-<job_id>/payload`
    /// and drain.  `sub = Some(rel)` stages only `<workspace>/<tree>/<rel>`.
    fn start_async_tree_delete(
        self: &Arc<Self>,
        ws: &WorkspaceId,
        workspace_dir: &Path,
        tree: AssetTree,
        sub: Option<AssetPath>,
    ) -> Result<JobId, FileError> {
        debug_assert!(
            matches!(tree, AssetTree::Datasets | AssetTree::Converters),
            "start_async_tree_delete called with non-async tree {tree:?}",
        );
        let target = build_delete_target(workspace_dir, tree, sub.as_ref());
        // Symlink-safe metadata read; missing target surfaces as
        // `Io { kind: NotFound }` so the API route can promote
        // to 404.
        std::fs::symlink_metadata(&target).map_err(|source| io_err(target.display(), source))?;
        // Asset deletes register a workspace-scoped reference
        // that gates only a `WorkspaceDelete`.  The shared
        // `max_delete_jobs` slot serializes simultaneous deletes
        // across the dataset / converter / workspace-delete
        // family.
        let job_type = tree.async_delete_job_type();
        // The snapshot's display target_path mirrors what the
        // operator asked for: a sub-path for sub-tree deletes,
        // the bare tree name for whole-tree wipes.
        let display_path = build_display_path(tree, sub.as_ref());
        let job_handle = self
            .jobs
            .try_acquire(
                job_type,
                vec![JobReference::Workspace { workspace_id: *ws }],
                Some(display_path),
            )
            .map_err(|e| {
                emit_mutation_rejected(tree);
                FileError::from(e)
            })?;
        let job_id = job_handle.job_id();

        let lock = self.metadata_lock(ws);
        let _guard = lock.lock();

        let cell = self.cache_cell_for_dataset(ws)?;
        let (next_revision_id, next_core) = bump_workspace_revision(&cell.core());

        // Tombstone variant matches the tree.  `path` on the
        // tombstone is the tree-relative sub-path or `None` for
        // a whole-tree wipe; recovery only consumes the
        // staging-dir location (derived from the filename) so
        // the field is diagnostic.
        let tombstone = match tree {
            AssetTree::Datasets => DeleteTombstone::Dataset {
                job_id,
                workspace_id: *ws,
                path: sub.clone(),
                workspace_revision_id: next_revision_id,
                created_at: now_rfc3339(),
            },
            AssetTree::Converters => DeleteTombstone::Converter {
                job_id,
                workspace_id: *ws,
                path: sub.clone(),
                workspace_revision_id: next_revision_id,
                created_at: now_rfc3339(),
            },
            AssetTree::TrainingLogs | AssetTree::ConverterLogs => {
                unreachable!("debug_assert above forbids log trees in this branch")
            }
        };
        let staging_dir = workspace_dir.join(".tmp");
        let staged = write_tombstone(&staging_dir, &tombstone)?;
        // Revision-before-bytes: bump `workspace.json` BEFORE
        // renaming the target into staging.
        write_workspace_core(workspace_dir, &next_core)?;
        stage_payload(&target, &staged)?;
        // Whole-tree wipe leaves the workspace without the tree
        // dir; recreate it empty so subsequent listings see the
        // canonical structural shape (`datasets/`, `converters/`,
        // `heads/`, ...) the workspace creation laid down.
        if sub.is_none() {
            std::fs::create_dir_all(&target).map_err(|source| io_err(target.display(), source))?;
            fsync_dir(workspace_dir).map_err(|source| io_err(workspace_dir.display(), source))?;
        }
        cell.publish_core(next_core);

        // Drop the per-workspace lock before draining; the drain
        // runs off-mutex.
        drop(_guard);

        spawn_asset_drain(staged, job_handle);

        Ok(job_id)
    }

    /// Async-delete branch for log trees (`training_logs/`,
    /// `converter_logs/`).  Same shape as
    /// [`Self::start_async_tree_delete`] minus the
    /// `workspace_revision` bump (logs aren't workspace state).
    /// Pre-checks for an active producer (Train / Convert) in
    /// the same workspace and returns `JobConflict` before any
    /// state mutation; the check is best-effort.
    fn start_async_log_delete(
        self: &Arc<Self>,
        ws: &WorkspaceId,
        workspace_dir: &Path,
        tree: AssetTree,
        sub: Option<AssetPath>,
    ) -> Result<JobId, FileError> {
        debug_assert!(
            matches!(tree, AssetTree::TrainingLogs | AssetTree::ConverterLogs),
            "start_async_log_delete called with non-log tree {tree:?}",
        );

        // Per-log-tree shape constraint comes BEFORE the
        // producer-active check: a malformed sub-path is a
        // client-side input error and should surface as 400
        // regardless of whether a producer is currently running.
        // Only `<id>.jsonl` single-file deletes and the bare-tree
        // wipe are addressable.
        if let Some(rel) = &sub {
            validate_log_subpath(rel)?;
        }

        let (job_kind, conflict_label, producer_active) = match tree {
            AssetTree::TrainingLogs => (
                "train",
                "training_logs",
                self.jobs.has_active_train_for(*ws),
            ),
            AssetTree::ConverterLogs => (
                "convert",
                "converter_logs",
                self.jobs.has_active_convert_for(*ws),
            ),
            AssetTree::Datasets | AssetTree::Converters => {
                unreachable!("start_async_log_delete called with non-log tree {tree:?}")
            }
        };
        if producer_active {
            return Err(FileError::JobConflict {
                message: format!(
                    "{conflict_label} cannot be cleared while a {job_kind} job for {ws} is active",
                ),
            });
        }

        let target = build_delete_target(workspace_dir, tree, sub.as_ref());
        // Symlink-safe metadata read; missing target surfaces as
        // `Io { kind: NotFound }` -> 404.  Whole-tree wipe of a
        // log dir that the producer never created (operator
        // never ran train/convert) lands here as 404; the
        // operator's idempotent "clear logs" pattern checks
        // for 404 explicitly.
        std::fs::symlink_metadata(&target).map_err(|source| io_err(target.display(), source))?;

        let job_type = tree.async_delete_job_type();
        let display_path = build_display_path(tree, sub.as_ref());
        let job_handle = self
            .jobs
            .try_acquire(
                job_type,
                vec![JobReference::Workspace { workspace_id: *ws }],
                Some(display_path),
            )
            .map_err(|e| {
                emit_mutation_rejected(tree);
                FileError::from(e)
            })?;
        let job_id = job_handle.job_id();

        let lock = self.metadata_lock(ws);
        let _guard = lock.lock();

        // Tombstone variant matches the tree.  Logs skip
        // `workspace_revision_id` (logs aren't workspace state).
        // Recovery diagnostic only consumes the staging-dir
        // location (derived from the filename) so the `path`
        // field is purely informational.
        let tombstone = match tree {
            AssetTree::TrainingLogs => DeleteTombstone::TrainingLogs {
                job_id,
                workspace_id: *ws,
                path: sub.clone(),
                created_at: now_rfc3339(),
            },
            AssetTree::ConverterLogs => DeleteTombstone::ConverterLogs {
                job_id,
                workspace_id: *ws,
                path: sub.clone(),
                created_at: now_rfc3339(),
            },
            AssetTree::Datasets | AssetTree::Converters => {
                unreachable!("debug_assert above forbids non-log trees in this branch")
            }
        };
        let staging_dir = workspace_dir.join(".tmp");
        let staged = write_tombstone(&staging_dir, &tombstone)?;
        // No revision bump; tombstone-then-stage is the durable
        // pair for log async-deletes.
        stage_payload(&target, &staged)?;
        // Whole-tree wipe leaves the workspace without the log
        // tree dir; recreate it empty so subsequent producer
        // runs find the canonical structural shape.
        if sub.is_none() {
            std::fs::create_dir_all(&target).map_err(|source| io_err(target.display(), source))?;
            fsync_dir(workspace_dir).map_err(|source| io_err(workspace_dir.display(), source))?;
        }

        // Drop the per-workspace lock before draining; the drain
        // runs off-mutex.
        drop(_guard);

        spawn_asset_drain(staged, job_handle);

        Ok(job_id)
    }

    /// Internal helper: resolve the cache cell for a dataset
    /// op.  Lazy-loads on first touch so workspaces created
    /// before the cache integrator landed (or recovered from
    /// boot) still hit the eager-cache path on the second
    /// access.
    fn cache_cell_for_dataset(
        &self,
        ws: &WorkspaceId,
    ) -> Result<Arc<crate::file_mgr::cache::WorkspaceCacheCell>, FileError> {
        if let Some(cell) = self.caches.get(ws) {
            return Ok(cell.clone());
        }
        let workspace_dir = self.workspace_dir(ws);
        let cell = Arc::new(crate::file_mgr::cache::WorkspaceCacheCell::load_from_disk(
            &workspace_dir,
        )?);
        Ok(self
            .caches
            .entry(*ws)
            .or_insert_with(|| cell.clone())
            .clone())
    }

    /// Workspace-wide [`JobType::WorkspaceDelete`] handle for
    /// `WorkspaceMgr::start_delete_workspace`.  Holds the entry's
    /// references (a single [`JobReference::Workspace`]) until
    /// the drain finishes; follow-up upload / delete requests
    /// observe the conflict before staging any bytes.
    pub(crate) fn register_workspace_delete(
        self: &Arc<Self>,
        workspace_id: WorkspaceId,
    ) -> Result<JobHandle, FileError> {
        self.jobs
            .try_acquire(
                JobType::WorkspaceDelete,
                vec![JobReference::Workspace { workspace_id }],
                None,
            )
            .map_err(FileError::from)
    }

    /// Bare workspace-scoped reference lease for in-flight
    /// non-job operations (convert input leases, head delete,
    /// etc.).  Blocks only an active `WorkspaceDelete` in the
    /// same workspace; does NOT conflict with train, convert, or
    /// file-delete jobs, and other bare leases coexist with it.
    ///
    /// Returns an opaque [`JobRefHandle`] that drops the lease on
    /// scope exit.  `Send + 'static` so the convert producer can
    /// move the guard into a `tokio::task::spawn_blocking` closure.
    pub fn register_job_reference(
        self: &Arc<Self>,
        reference: JobReference,
    ) -> Result<JobRefHandle, FileError> {
        let lease = self
            .jobs
            .try_acquire_lease(vec![reference])
            .map_err(FileError::from)?;
        Ok(JobRefHandle {
            _lease: Some(lease),
            _job: None,
        })
    }

    /// Register a typed convert job in the registry.  Acquires
    /// the global `JobType::Convert` slot (bounded by
    /// `max_convert_jobs = 1`) and stamps a workspace-scoped
    /// reference for `WorkspaceDelete` exclusion.  A second
    /// concurrent convert observes `RegistryConflict::JobConflict`
    /// and surfaces 409.
    pub fn register_convert_job(
        self: &Arc<Self>,
        workspace_id: WorkspaceId,
    ) -> Result<JobHandle, FileError> {
        self.jobs
            .try_acquire(
                JobType::Convert,
                vec![JobReference::Workspace { workspace_id }],
                None,
            )
            .map_err(FileError::from)
    }

    /// Register a typed train job in the registry.  Acquires the
    /// global `JobType::Train` slot (bounded by `max_train_jobs
    /// = 1`) and stamps a workspace-scoped reference for
    /// `WorkspaceDelete` exclusion.  A second train globally
    /// returns `RegistryConflict::AnotherTrainRunning`.
    pub fn register_train_job(
        self: &Arc<Self>,
        workspace_id: WorkspaceId,
    ) -> Result<JobHandle, FileError> {
        self.jobs
            .try_acquire(
                JobType::Train,
                vec![JobReference::Workspace { workspace_id }],
                None,
            )
            .map_err(FileError::from)
    }
}

/// Opaque RAII guard for a job-reference lease (bare lease or
/// typed job handle).  `Send + 'static` so the convert producer
/// can move it into a `tokio::task::spawn_blocking` closure.
/// Drop releases whichever inner the wrapper carries.
#[derive(Debug)]
pub struct JobRefHandle {
    _lease: Option<crate::file_mgr::job_registry::LeaseGuard>,
    _job: Option<JobHandle>,
}

impl JobRefHandle {
    /// Construct from a typed [`JobHandle`].  Used by the
    /// convert producer to wrap its full job admission as the
    /// same opaque shape the api crate already imports.
    pub fn from_job_handle(handle: JobHandle) -> Self {
        Self {
            _lease: None,
            _job: Some(handle),
        }
    }
}

#[allow(unused_imports)]
pub(crate) use crate::file_mgr::job_registry::JobRegistry as _JobRegistry;

// MARK: Drain dispatch

/// Spawn the off-mutex drain for an asset-tree delete, holding
/// `job_handle` for the lifetime of the drain so a follow-up
/// upload / delete request observes the conflict.  Mirrors
/// `WorkspaceMgr::start_delete_workspace`'s shape (tokio
/// blocking pool when a runtime is available, inline otherwise).
/// On drain success the handle is `succeed`-ed; on failure
/// `fail`-ed -- both transitions emit a terminal event and
/// release the registry references.
/// Drain `staged` to completion, finalize, and terminate
/// `job_handle` (`succeed(None)` on success; `fail(...)` on
/// drain or finalize error).  `log_suffix` is appended to the
/// "asset delete ... failed" tracing messages so operators can
/// distinguish the async (`""`) and sync-test (`" (sync path)"`)
/// callers in logs.  `max_iters` caps the drain loop on the sync
/// path so a hung drain in a synchronous test fails loudly
/// instead of hanging the test runner; the async path passes
/// `None` because a stuck `spawn_blocking` worker is safely
/// abandoned by tokio at runtime drop.
fn drain_to_completion(
    staged: &StagedDelete,
    job_handle: JobHandle,
    log_suffix: &str,
    max_iters: Option<usize>,
) {
    let mut iter = 0usize;
    loop {
        if let Some(max) = max_iters {
            iter += 1;
            if iter > max {
                tracing::error!(
                    target: "file_mgr",
                    "asset delete drain failed to converge after {max} iterations",
                );
                break;
            }
        }
        match drain_staged_payload(staged, DEFAULT_DELETE_BATCH_ENTRIES) {
            Ok(DrainResult::Done) => break,
            Ok(DrainResult::More) => continue,
            Err(e) => {
                tracing::warn!(
                    target: "file_mgr",
                    err = %e,
                    "asset delete drain failed{log_suffix}; boot recovery will resume",
                );
                job_handle.fail(format!("asset delete drain failed: {e}"));
                return;
            }
        }
    }
    if let Err(e) = finalize_staged_delete(staged) {
        tracing::warn!(
            target: "file_mgr",
            err = %e,
            "asset delete finalize failed{log_suffix}; boot recovery will resume",
        );
        job_handle.fail(format!("asset delete finalize failed: {e}"));
        return;
    }
    job_handle.succeed(None);
}

fn spawn_asset_drain(staged: StagedDelete, job_handle: JobHandle) {
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        handle.spawn_blocking(move || {
            drain_to_completion(&staged, job_handle, "", None);
        });
    } else {
        // Sync test path: drain inline.  The 1M-iteration cap is
        // a hung-drain guard so a regression in the staging
        // payload reader fails the test instead of hanging the
        // runner.  Failure does not surface beyond a tracing
        // line: the staged state is recoverable via boot.
        drain_to_completion(&staged, job_handle, " (sync path)", Some(1_000_000));
    }
}

// MARK: Helper -- write a tempfile in a workspace `.tmp/` dir.
// Used by the API route's streaming-then-commit dance; exposed
// here so test code can exercise the upload-commit method
// without redoing the body-streaming half.

/// Stage `bytes` to a tempfile under `<workspace_dir>/.tmp/` and
/// return the path.  Test-only convenience around
/// [`tempfile::NamedTempFile`].
#[cfg(test)]
pub(crate) fn stage_test_tempfile(
    workspace_dir: &Path,
    bytes: &[u8],
) -> Result<tempfile::NamedTempFile, FileError> {
    use std::io::Write;
    let tmp_dir = workspace_dir.join(".tmp");
    std::fs::create_dir_all(&tmp_dir).map_err(|source| io_err(tmp_dir.display(), source))?;
    let mut tmp = tempfile::NamedTempFile::new_in(&tmp_dir)
        .map_err(|source| io_err(tmp_dir.display(), source))?;
    tmp.write_all(bytes)
        .map_err(|source| io_err(tmp.path().display(), source))?;
    tmp.as_file()
        .sync_all()
        .map_err(|source| io_err(tmp.path().display(), source))?;
    Ok(tmp)
}

// MARK: Tests

#[cfg(test)]
#[allow(clippy::disallowed_methods)]
// Tests stage fixtures with `std::fs::write`; the production
// constraint in `clippy.toml` (writes through file_mgr's atomic
// writer) does not apply to test setup helpers.
mod tests {
    use super::*;
    use crate::file_mgr::WorkspaceMgr;

    fn fresh_mgr(root: PathBuf) -> Arc<WorkspaceMgr> {
        let mgr = Arc::new(WorkspaceMgr::new(root));
        mgr.ensure_root_layout().expect("layout");
        mgr
    }

    fn new_workspace(mgr: &Arc<WorkspaceMgr>, name: &str) -> WorkspaceId {
        mgr.create(name).expect("create workspace")
    }

    #[test]
    fn workspace_asset_path_join_resolves_under_workspace_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let mgr = fresh_mgr(tmp.path().to_path_buf());
        let ws = new_workspace(&mgr, "main");
        let p = AssetPath::parse("datasets/audio/cat/sample.wav").unwrap();
        let resolved = mgr.workspace_asset_path_join(&ws, &p);
        assert_eq!(
            resolved,
            mgr.workspace_dir(&ws)
                .join("datasets")
                .join("audio")
                .join("cat")
                .join("sample.wav")
        );
    }

    #[test]
    fn upload_workspace_file_round_trip_datasets() {
        let tmp = tempfile::tempdir().unwrap();
        let mgr = fresh_mgr(tmp.path().to_path_buf());
        let ws = new_workspace(&mgr, "main");
        let p = AssetPath::parse("datasets/audio_dataset/cat/sample.wav").unwrap();
        let bytes = b"hello world";
        let staged = stage_test_tempfile(&mgr.workspace_dir(&ws), bytes).unwrap();
        let receipt = mgr
            .upload_workspace_file(&ws, &p, staged.path(), "deadbeef", bytes.len() as u64)
            .expect("upload");
        assert_eq!(receipt.workspace_revision_id, 1);
        assert_eq!(receipt.size_bytes, bytes.len() as u64);
        // File landed under `<ws>/datasets/audio_dataset/cat/sample.wav`.
        let final_path = mgr.workspace_asset_path_join(&ws, &p);
        let read = std::fs::read(&final_path).expect("read final");
        assert_eq!(read, bytes);
        // Workspace core cache reflects the bumped revision.
        let summary = mgr.summary(&ws).expect("summary");
        assert_eq!(summary.core.workspace_revision.id, 1);
    }

    /// The `converters/` tree accepts uploads under the same
    /// protocol; only the on-disk subdirectory differs.
    #[test]
    fn upload_workspace_file_round_trip_converters() {
        let tmp = tempfile::tempdir().unwrap();
        let mgr = fresh_mgr(tmp.path().to_path_buf());
        let ws = new_workspace(&mgr, "main");
        let p = AssetPath::parse("converters/tfjs/model.json").unwrap();
        let bytes = br#"{"format":"tfjs"}"#;
        let staged = stage_test_tempfile(&mgr.workspace_dir(&ws), bytes).unwrap();
        let receipt = mgr
            .upload_workspace_file(&ws, &p, staged.path(), "deadbeef", bytes.len() as u64)
            .expect("upload");
        assert_eq!(receipt.workspace_revision_id, 1);
        // File landed under `<ws>/converters/tfjs/model.json`.
        let final_path = mgr.workspace_asset_path_join(&ws, &p);
        assert!(final_path.is_file());
        let read = std::fs::read(&final_path).expect("read final");
        assert_eq!(read, bytes);
    }

    /// Paths that don't start with `datasets/` or `converters/`
    /// reject at the file_mgr boundary even when the bytes have
    /// already been streamed to a tempfile.
    /// The route layer normally rejects these earlier, but the
    /// lib gate is the canonical guard.
    #[test]
    fn upload_workspace_file_rejects_non_mutable_top_level() {
        let tmp = tempfile::tempdir().unwrap();
        let mgr = fresh_mgr(tmp.path().to_path_buf());
        let ws = new_workspace(&mgr, "main");
        for bad in [
            "heads/x.mpk",
            "training_logs/foo.jsonl",
            "workspace.json",
            "scratch/file.bin",
        ] {
            let p = AssetPath::parse(bad).unwrap();
            let staged = stage_test_tempfile(&mgr.workspace_dir(&ws), b"x").unwrap();
            let err = mgr
                .upload_workspace_file(&ws, &p, staged.path(), "deadbeef", 1)
                .unwrap_err();
            assert!(
                matches!(err, FileError::InvalidName(_)),
                "expected InvalidName for `{bad}`; got {err:?}"
            );
        }
    }

    /// Bare `datasets/` and `converters/` reject; the mutation
    /// contract requires at least one child component.
    #[test]
    fn upload_workspace_file_rejects_tree_root_without_child() {
        let tmp = tempfile::tempdir().unwrap();
        let mgr = fresh_mgr(tmp.path().to_path_buf());
        let ws = new_workspace(&mgr, "main");
        for bare in ["datasets", "converters"] {
            let p = AssetPath::parse(bare).unwrap();
            let staged = stage_test_tempfile(&mgr.workspace_dir(&ws), b"x").unwrap();
            let err = mgr
                .upload_workspace_file(&ws, &p, staged.path(), "deadbeef", 1)
                .unwrap_err();
            assert!(
                matches!(err, FileError::InvalidName(_)),
                "expected InvalidName for bare `{bare}`; got {err:?}"
            );
        }
    }

    #[test]
    fn upload_bumps_revision_each_time() {
        let tmp = tempfile::tempdir().unwrap();
        let mgr = fresh_mgr(tmp.path().to_path_buf());
        let ws = new_workspace(&mgr, "main");
        for (i, name) in ["a.json", "b.json", "c.json"].iter().enumerate() {
            let p = AssetPath::parse(&format!("datasets/cls/{name}")).unwrap();
            let bytes = format!("{i}").into_bytes();
            let staged = stage_test_tempfile(&mgr.workspace_dir(&ws), &bytes).unwrap();
            let receipt = mgr
                .upload_workspace_file(&ws, &p, staged.path(), "deadbeef", bytes.len() as u64)
                .expect("upload");
            assert_eq!(receipt.workspace_revision_id, (i as u64) + 1);
        }
    }

    /// Datasets-only depth gate: `datasets/<file>` (no class
    /// folder) rejects on upload because the trainer keys class
    /// labels off the first subdirectory of `datasets/`.  The
    /// converter tree is exempt -- `converters/<file>` uploads.
    #[test]
    fn upload_rejects_dataset_without_class_folder() {
        let tmp = tempfile::tempdir().unwrap();
        let mgr = fresh_mgr(tmp.path().to_path_buf());
        let ws = new_workspace(&mgr, "main");
        let p = AssetPath::parse("datasets/sample.wav").unwrap();
        let staged = stage_test_tempfile(&mgr.workspace_dir(&ws), b"x").unwrap();
        let err = mgr
            .upload_workspace_file(&ws, &p, staged.path(), "deadbeef", 1)
            .unwrap_err();
        assert!(
            matches!(err, FileError::InvalidName(_)),
            "expected InvalidName for datasets/<file>; got {err:?}",
        );
        // Converters at depth 2 are still accepted.
        let p = AssetPath::parse("converters/loose.json").unwrap();
        let staged = stage_test_tempfile(&mgr.workspace_dir(&ws), b"x").unwrap();
        mgr.upload_workspace_file(&ws, &p, staged.path(), "deadbeef", 1)
            .expect("converters/<file> uploads");
    }

    /// The class-folder delete (`DELETE datasets/<class>`) keeps
    /// working under the new upload gate -- the depth check
    /// applies only to uploads.
    #[test]
    fn delete_dataset_class_folder_remains_supported() {
        let tmp = tempfile::tempdir().unwrap();
        let mgr = fresh_mgr(tmp.path().to_path_buf());
        let ws = new_workspace(&mgr, "main");
        let p = AssetPath::parse("datasets/cat/sample.wav").unwrap();
        let staged = stage_test_tempfile(&mgr.workspace_dir(&ws), b"x").unwrap();
        mgr.upload_workspace_file(&ws, &p, staged.path(), "deadbeef", 1)
            .expect("upload");
        let class_root = AssetPath::parse("datasets/cat").unwrap();
        mgr.start_workspace_asset_delete(&ws, &class_root)
            .expect("class folder delete admits at depth 2");
    }

    #[test]
    fn upload_then_delete_round_trip_sync() {
        let tmp = tempfile::tempdir().unwrap();
        let mgr = fresh_mgr(tmp.path().to_path_buf());
        let ws = new_workspace(&mgr, "main");
        let p = AssetPath::parse("datasets/audio/sample.wav").unwrap();
        let staged_tmp = stage_test_tempfile(&mgr.workspace_dir(&ws), b"x").unwrap();
        mgr.upload_workspace_file(&ws, &p, staged_tmp.path(), "deadbeef", 1)
            .unwrap();
        let pre = mgr.summary(&ws).unwrap().core.workspace_revision.id;
        let _job = mgr.start_workspace_asset_delete(&ws, &p).expect("delete");
        let post = mgr.summary(&ws).unwrap().core.workspace_revision.id;
        assert_eq!(post, pre + 1, "delete bumps revision");
        // File gone.
        let final_path = mgr.workspace_asset_path_join(&ws, &p);
        assert!(!final_path.exists(), "{final_path:?} still present");
    }

    /// A converter-tree delete writes a `Converter` tombstone
    /// (`delete-converters-*.json`) and admits as
    /// `JobType::ConverterDelete` -- distinct from the dataset
    /// delete path.  Bytes drain through the same staging
    /// machinery and the post-delete revision still bumps.
    #[test]
    fn converter_upload_then_delete_round_trip_sync() {
        let tmp = tempfile::tempdir().unwrap();
        let mgr = fresh_mgr(tmp.path().to_path_buf());
        let ws = new_workspace(&mgr, "main");
        let p = AssetPath::parse("converters/tfjs/model.json").unwrap();
        let staged_tmp = stage_test_tempfile(&mgr.workspace_dir(&ws), b"manifest").unwrap();
        mgr.upload_workspace_file(&ws, &p, staged_tmp.path(), "deadbeef", 8)
            .unwrap();
        let pre = mgr.summary(&ws).unwrap().core.workspace_revision.id;
        let _job = mgr
            .start_workspace_asset_delete(&ws, &p)
            .expect("converter delete");
        let post = mgr.summary(&ws).unwrap().core.workspace_revision.id;
        assert_eq!(post, pre + 1, "converter delete bumps revision");
        let final_path = mgr.workspace_asset_path_join(&ws, &p);
        assert!(!final_path.exists());
    }

    /// `start_workspace_asset_delete` rejects any non-mutable
    /// top-level (heads, training_logs, workspace.json, ...).
    /// The route layer normally rejects earlier; this is the
    /// canonical lib-boundary guard.
    #[test]
    fn start_workspace_asset_delete_rejects_non_mutable_top_level() {
        let tmp = tempfile::tempdir().unwrap();
        let mgr = fresh_mgr(tmp.path().to_path_buf());
        let ws = new_workspace(&mgr, "main");
        // `training_logs` and `converter_logs` are now mutable
        // top-levels (async log-delete branch via `parse_mutable_path`);
        // the canonical-rejection set is everything else.  Single
        // tree-name components stay rejected because they aren't
        // one of the four mutable trees.
        for bad in [
            "heads/x.mpk",
            "heads",
            "workspace.json",
            "scratch/file.bin",
            "scratch",
        ] {
            let p = AssetPath::parse(bad).unwrap();
            let err = mgr.start_workspace_asset_delete(&ws, &p).unwrap_err();
            assert!(
                matches!(err, FileError::InvalidName(_)),
                "expected InvalidName for `{bad}`; got {err:?}"
            );
        }
    }

    /// Single-file `training_logs/<job>.jsonl` /
    /// `converter_logs/<job>.jsonl` deletes for a non-existent
    /// file surface as `NotFound` -- the async path stats the
    /// target before staging, matching the dataset/converter
    /// sub-path semantic.  Operators that need idempotent
    /// "remove this log file" check 404 explicitly.
    ///
    /// The bare-tree paths (`training_logs`, `converter_logs`)
    /// are NOT covered by this test: workspace creation lays the
    /// log dirs down empty, so the whole-tree wipe always finds
    /// the target and proceeds (drain is just a no-op).  That
    /// keeps the operator's "clear all logs" pattern idempotent
    /// without any 404 special-case.
    #[test]
    fn start_workspace_asset_delete_log_file_returns_not_found_when_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let mgr = fresh_mgr(tmp.path().to_path_buf());
        let ws = new_workspace(&mgr, "main");
        for path in [
            "training_logs/00000000-0000-4000-8000-000000000001.jsonl",
            "converter_logs/00000000-0000-4000-8000-000000000002.jsonl",
        ] {
            let p = AssetPath::parse(path).unwrap();
            let err = mgr.start_workspace_asset_delete(&ws, &p).unwrap_err();
            assert!(
                matches!(
                    &err,
                    FileError::Io { source, .. }
                        if source.kind() == std::io::ErrorKind::NotFound
                ),
                "expected NotFound for missing `{path}`; got {err:?}",
            );
        }
    }

    /// Whole-tree wipe of empty `training_logs/` / `converter_logs/`
    /// (the freshly-created workspace shape) returns a successful
    /// async outcome -- the staging machinery handles the empty
    /// payload gracefully (drain is a no-op, finalize cleans up
    /// the tombstone).  This keeps the operator's "clear all
    /// logs" pattern idempotent across fresh-and-aged workspaces.
    #[test]
    fn start_workspace_asset_delete_log_whole_tree_succeeds_on_empty_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let mgr = fresh_mgr(tmp.path().to_path_buf());
        let ws = new_workspace(&mgr, "main");
        for path in ["training_logs", "converter_logs"] {
            let p = AssetPath::parse(path).unwrap();
            let _job = mgr
                .start_workspace_asset_delete(&ws, &p)
                .unwrap_or_else(|e| panic!("whole-tree empty wipe of `{path}` failed: {e:?}"));
            // Recreated empty dir survives.
            let dir = mgr.workspace_dir(&ws).join(path);
            assert!(dir.exists(), "{path}/ recreated after whole-tree wipe");
        }
    }

    /// Single-file log delete and whole-dir wipe both go through
    /// the async tombstone+stage+drain path; both return a
    /// `JobId`, both stage the target into the workspace `.tmp/`
    /// staging dir, both let the inline drain (sync test path)
    /// finish before the call returns.  Whole-dir wipe of a log
    /// tree recreates the empty dir so subsequent producer runs
    /// find the canonical structural shape; non-`.jsonl`
    /// siblings (forbidden by `validate_log_subpath` for
    /// single-file deletes, but a stray operator-pasted file
    /// could exist) are renamed away with the rest of the dir
    /// in the whole-tree wipe.
    #[test]
    fn start_workspace_asset_delete_log_paths_async_and_drain_into_staging() {
        let tmp = tempfile::tempdir().unwrap();
        let mgr = fresh_mgr(tmp.path().to_path_buf());
        let ws = new_workspace(&mgr, "main");
        let ws_dir = mgr.workspace_dir(&ws);
        let log_dir = ws_dir.join("training_logs");
        std::fs::create_dir_all(&log_dir).unwrap();
        let a = log_dir.join("00000000-0000-4000-8000-000000000001.jsonl");
        let b = log_dir.join("00000000-0000-4000-8000-000000000002.jsonl");
        std::fs::write(&a, b"{}").unwrap();
        std::fs::write(&b, b"{}").unwrap();

        // Single-file wipe: stages + drains the named jsonl;
        // sibling jsonl is untouched.
        let p =
            AssetPath::parse("training_logs/00000000-0000-4000-8000-000000000001.jsonl").unwrap();
        let _job = mgr
            .start_workspace_asset_delete(&ws, &p)
            .expect("single-file log delete");
        // Sync-test drain runs inline; the staged payload should
        // be gone after the call returns.
        assert!(!a.exists(), "single-file log delete drained `a`");
        assert!(b.exists(), "sibling jsonl untouched by single-file delete");

        // Whole-dir wipe: stages the directory; the empty
        // `training_logs/` dir is recreated for the canonical
        // structural shape.
        let p = AssetPath::parse("training_logs").unwrap();
        let _job = mgr
            .start_workspace_asset_delete(&ws, &p)
            .expect("whole-tree log delete");
        assert!(!b.exists(), "whole-dir log wipe drained `b`");
        let recreated = ws_dir.join("training_logs");
        assert!(
            recreated.exists() && recreated.is_dir(),
            "empty training_logs/ recreated after whole-tree wipe",
        );
    }

    /// Log sub-paths that don't fit `<dir>/<id>.jsonl` are
    /// rejected at the dispatcher before any state mutation.
    /// Pinned because the dispatcher must reject malformed sub-paths
    /// up-front so a typo doesn't proceed past the producer-active
    /// gate into the staging machinery.
    #[test]
    fn start_workspace_asset_delete_log_subpath_shape_constraints() {
        let tmp = tempfile::tempdir().unwrap();
        let mgr = fresh_mgr(tmp.path().to_path_buf());
        let ws = new_workspace(&mgr, "main");
        for bad in [
            "training_logs/sub/x.jsonl",  // nested
            "training_logs/note.txt",     // wrong extension
            "converter_logs/sub/y.jsonl", // nested
            "converter_logs/keep.txt",    // wrong extension
        ] {
            let p = AssetPath::parse(bad).unwrap();
            let err = mgr.start_workspace_asset_delete(&ws, &p).unwrap_err();
            assert!(
                matches!(err, FileError::InvalidName(_)),
                "expected InvalidName for `{bad}`; got {err:?}",
            );
        }
    }

    /// Ordering pin: log-tree dispatcher must surface the shape
    /// `InvalidName` BEFORE the producer-active `JobConflict`.  A
    /// malformed sub-path is a client-side input error and should
    /// reject regardless of whether the producer is currently
    /// running; an active producer must not mask the diagnostic.
    /// Without this pin the two checks could be silently re-ordered
    /// in a future refactor and a malformed-path-during-train would
    /// 409 instead of 400.
    #[test]
    fn start_workspace_asset_delete_log_shape_check_runs_before_producer_check() {
        let tmp = tempfile::tempdir().unwrap();
        let mgr = fresh_mgr(tmp.path().to_path_buf());
        let ws = new_workspace(&mgr, "main");
        // Acquire a Train job so the producer-active check would
        // fire if reached.  Held for the duration of the test.
        let _train = mgr
            .jobs
            .try_acquire(
                JobType::Train,
                vec![JobReference::Workspace { workspace_id: ws }],
                None,
            )
            .expect("train admission");
        let bad = AssetPath::parse("training_logs/nested/job.jsonl").unwrap();
        let err = mgr.start_workspace_asset_delete(&ws, &bad).unwrap_err();
        assert!(
            matches!(err, FileError::InvalidName(_)),
            "shape error must surface before producer-active 409; got {err:?}",
        );
    }

    /// Whole-tree wipe of `datasets/` and `converters/` returns
    /// the async outcome and stages the entire tree dir under
    /// `.tmp/`; the empty tree dir is recreated so subsequent
    /// uploads / lists see the canonical workspace shape.
    #[test]
    fn start_workspace_asset_delete_datasets_whole_tree_returns_async_and_recreates_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let mgr = fresh_mgr(tmp.path().to_path_buf());
        let ws = new_workspace(&mgr, "main");
        // Stage some content so the rename has bytes to drain.
        let p = AssetPath::parse("datasets/cls/sample.bin").unwrap();
        let scratch = tmp.path().join("scratch.bin");
        std::fs::write(&scratch, b"x").unwrap();
        mgr.upload_workspace_file(&ws, &p, &scratch, "00", 1)
            .unwrap();

        // Whole-tree wipe.
        let p = AssetPath::parse("datasets").unwrap();
        let _job = mgr
            .start_workspace_asset_delete(&ws, &p)
            .expect("whole-tree datasets delete");
        // The empty `datasets/` dir survived the rename.
        let datasets_dir = mgr.workspace_dir(&ws).join("datasets");
        assert!(datasets_dir.exists(), "empty datasets/ recreated");
        assert!(datasets_dir.is_dir());
        assert_eq!(
            std::fs::read_dir(&datasets_dir).unwrap().count(),
            0,
            "datasets/ is empty post-wipe",
        );
    }

    #[test]
    fn upload_blocked_by_active_workspace_delete() {
        // Uploads are gated only by an active `WorkspaceDelete`
        // in the same workspace; an in-flight train / convert /
        // dataset-delete does NOT block uploads.
        let tmp = tempfile::tempdir().unwrap();
        let mgr = fresh_mgr(tmp.path().to_path_buf());
        let ws = new_workspace(&mgr, "main");
        let _del = mgr
            .jobs
            .try_acquire(
                JobType::WorkspaceDelete,
                vec![JobReference::Workspace { workspace_id: ws }],
                None,
            )
            .expect("workspace-delete admitted");
        let p = AssetPath::parse("datasets/audio/cat/sample.wav").unwrap();
        let staged = stage_test_tempfile(&mgr.workspace_dir(&ws), b"x").unwrap();
        let err = mgr
            .upload_workspace_file(&ws, &p, staged.path(), "deadbeef", 1)
            .unwrap_err();
        assert!(matches!(err, FileError::JobConflict { .. }));
    }

    #[test]
    fn upload_coexists_with_active_train_in_same_workspace() {
        // Train + upload in the same workspace are explicitly
        // allowed to overlap; no path-overlap conflict.
        let tmp = tempfile::tempdir().unwrap();
        let mgr = fresh_mgr(tmp.path().to_path_buf());
        let ws = new_workspace(&mgr, "main");
        let _train = mgr
            .jobs
            .try_acquire(
                JobType::Train,
                vec![JobReference::Workspace { workspace_id: ws }],
                None,
            )
            .expect("train admitted");
        let p = AssetPath::parse("datasets/audio/cat/sample.wav").unwrap();
        let staged = stage_test_tempfile(&mgr.workspace_dir(&ws), b"x").unwrap();
        let receipt = mgr
            .upload_workspace_file(&ws, &p, staged.path(), "deadbeef", 1)
            .expect("upload during train succeeds");
        assert_eq!(receipt.workspace_revision_id, 1);
    }
}
