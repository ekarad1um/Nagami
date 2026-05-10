//! `WorkspaceRegistry` aggregate of [`WorkspaceMgr`]: the
//! workspace-lifecycle surface and shared `workspace_dir` helper.
//!
//! Workspaces nest under `<root>/workspaces/<id>/`.  Creation
//! writes `workspace.json` + `heads.json` via
//! `file_mgr::schema::*` and seeds the eager cache; deletion
//! stages the tree under
//! `<root>/.tmp/delete-workspace-<job_id>/payload` and drains it
//! off-mutex via [`crate::file_mgr::staging`].

use std::path::PathBuf;
use std::sync::Arc;

use crate::common::ids::{JobId, WorkspaceId};
use crate::common::workspace::{HeadIndex, WorkspaceCore, WorkspaceRevision};

use crate::file_mgr::WorkspaceMgr;
use crate::file_mgr::cache::WorkspaceCacheCell;
use crate::file_mgr::error::{FileError, io_err};
use crate::file_mgr::schema::{
    ACTIVE_DIR_NAME, BACKBONE_DIR_NAME, ROOT_TMP_DIR_NAME, WORKSPACES_DIR_NAME, root_tmp_dir,
    workspace_dir_for, workspaces_dir, write_head_index, write_workspace_core,
};
use crate::file_mgr::staging::{
    DEFAULT_DELETE_BATCH_ENTRIES, DeleteTombstone, DrainResult, StagedDelete, drain_staged_payload,
    finalize_staged_delete, stage_payload, write_tombstone,
};
use crate::file_mgr::time_util::now_rfc3339;
use crate::file_mgr::validate::fsync_dir;

/// Validate a workspace name: non-empty, <=128 bytes, no NUL,
/// controls, path separators, or leading / trailing whitespace.
fn validate_workspace_name(name: &str) -> Result<(), FileError> {
    if name.is_empty() {
        return Err(FileError::InvalidName(name.to_string()));
    }
    if name.len() > 128 {
        return Err(FileError::InvalidName(name.to_string()));
    }
    if name.contains('\0') || name.contains('/') || name.contains('\\') {
        return Err(FileError::InvalidName(name.to_string()));
    }
    if name.chars().next().is_some_and(char::is_whitespace) {
        return Err(FileError::InvalidName(name.to_string()));
    }
    if name.chars().last().is_some_and(char::is_whitespace) {
        return Err(FileError::InvalidName(name.to_string()));
    }
    if name.chars().any(char::is_control) {
        return Err(FileError::InvalidName(name.to_string()));
    }
    Ok(())
}

/// Maximum number of tags retained on a workspace.
const MAX_WORKSPACE_TAGS: usize = 32;
/// Maximum byte length of a single tag (post ASCII trim).
const MAX_TAG_BYTES: usize = 64;

/// Per-tag normalization + validation.
///
/// Each tag is trimmed of leading / trailing ASCII whitespace
/// before validation; the trimmed form is what gets stored.
/// Constraints (post-trim):
/// 1. non-empty,
/// 2. <=64 bytes (UTF-8 byte count),
/// 3. no NUL,
/// 4. no path separators (`/` or `\`),
/// 5. no control characters,
/// 6. no duplicate under Unicode case-insensitive comparison
///    (`str::to_lowercase`; simple case folding, no NFC),
/// 7. caller-side: <=32 entries total.
///
/// Returns the trimmed, validated tag set so the caller can
/// stamp it onto `WorkspaceCore.tags` directly.  Order is
/// preserved from the input.
fn validate_workspace_tags(tags: &[String]) -> Result<Vec<String>, FileError> {
    if tags.len() > MAX_WORKSPACE_TAGS {
        return Err(FileError::InvalidName(format!(
            "tags exceed max {MAX_WORKSPACE_TAGS}: got {}",
            tags.len()
        )));
    }
    let mut out: Vec<String> = Vec::with_capacity(tags.len());
    let mut seen_lower: std::collections::HashSet<String> =
        std::collections::HashSet::with_capacity(tags.len());
    for raw in tags {
        let trimmed = raw.trim_matches(|c: char| c.is_ascii_whitespace());
        if trimmed.is_empty() {
            return Err(FileError::InvalidName(format!(
                "tag empty after ASCII whitespace trim: {raw:?}"
            )));
        }
        if trimmed.len() > MAX_TAG_BYTES {
            return Err(FileError::InvalidName(format!(
                "tag exceeds {MAX_TAG_BYTES} bytes: {trimmed:?}"
            )));
        }
        if trimmed.contains('\0') || trimmed.contains('/') || trimmed.contains('\\') {
            return Err(FileError::InvalidName(format!(
                "tag contains NUL or path separator: {trimmed:?}"
            )));
        }
        if trimmed.chars().any(char::is_control) {
            return Err(FileError::InvalidName(format!(
                "tag contains control character: {trimmed:?}"
            )));
        }
        // Unicode-aware case fold: `str::to_lowercase` handles
        // Latin-extended, Cyrillic, Greek, etc. via the Unicode
        // `Lowercase` core property.  Simple (not full) case folding
        // -- German `ß` stays as `ß`, Turkish dotted/dotless I uses
        // root locale.  No NFC normalization, so decomposed and
        // composed forms of the same visual string are treated as
        // distinct (acceptable trade-off; adding NFC requires the
        // `unicode-normalization` crate).
        let lower = trimmed.to_lowercase();
        if !seen_lower.insert(lower) {
            return Err(FileError::InvalidName(format!(
                "duplicate tag (case-insensitive): {trimmed:?}"
            )));
        }
        out.push(trimmed.to_string());
    }
    Ok(out)
}

impl WorkspaceMgr {
    /// Idempotently create the root layout
    /// (`workspaces/`, `.tmp/`, `active/`, `backbone/`) under
    /// `self.root`.  Daemon first-boot calls this before any
    /// other lifecycle work; subsequent calls are no-ops.
    pub fn ensure_root_layout(&self) -> Result<(), FileError> {
        std::fs::create_dir_all(&self.root).map_err(|e| io_err(self.root.display(), e))?;
        for sub in [
            WORKSPACES_DIR_NAME,
            ROOT_TMP_DIR_NAME,
            ACTIVE_DIR_NAME,
            BACKBONE_DIR_NAME,
        ] {
            let p = self.root.join(sub);
            std::fs::create_dir_all(&p).map_err(|e| io_err(p.display(), e))?;
        }
        Ok(())
    }

    /// Create a new workspace under `<root>/workspaces/<id>/`.
    /// Generates a fresh UUID, builds the directory tree, writes
    /// `workspace.json` + `heads.json`, and seeds the eager
    /// cache.
    ///
    /// # Concurrency
    ///
    /// `registry_lock` (sync `parking_lot::Mutex`) serializes
    /// the list-check-create sequence so two concurrent
    /// `create("main")` calls cannot both win with distinct
    /// UUIDs.  No `.await` inside the critical section.
    ///
    /// # Crash consistency
    ///
    /// Every subdirectory + `heads.json` lands BEFORE
    /// `workspace.json`.  A crash before the final
    /// `write_workspace_core` leaves a directory without a
    /// `workspace.json` -- boot recovery treats that as an
    /// incomplete create and `list_workspaces` skips it.
    /// Workspace dir, `<root>/workspaces/`, and `<root>/` are
    /// fsynced in turn so the new directory entry reaches
    /// stable storage.
    pub fn create(&self, name: &str) -> Result<WorkspaceId, FileError> {
        self.create_with_tags(name, &[])
    }

    /// Create a new workspace with optional operator-supplied tags.
    /// Tags are trimmed and validated; an empty slice yields
    /// `tags = []` on the persisted core.
    pub fn create_with_tags(&self, name: &str, tags: &[String]) -> Result<WorkspaceId, FileError> {
        validate_workspace_name(name)?;
        let normalized_tags = validate_workspace_tags(tags)?;
        // Hold the registry lock across list -> uniqueness check
        // -> commit.  `parking_lot::Mutex` is sync; no `.await`
        // inside this function, so cancellation can't strand
        // the guard.
        let _registry_guard = self.registry_lock.lock();
        self.ensure_workspace_name_available(name, None)?;
        // Make sure the parent `<root>/workspaces/` exists.
        // The daemon's first-boot path calls `ensure_root_layout`
        // before any `create`, but tests / direct callers may
        // skip it; `create_dir_all` is idempotent.
        let workspaces_root = workspaces_dir(&self.root);
        std::fs::create_dir_all(&workspaces_root)
            .map_err(|e| io_err(workspaces_root.display(), e))?;
        let id = WorkspaceId::new();
        let ws = self.workspace_dir(&id);
        // The legacy `weights/` / `labels/` / `metadata.json`
        // surface is retired; both producers (train + convert)
        // publish only into `heads/` + `heads.json`.  Workspace
        // creation lays down only the current on-disk shape.
        for sub in [
            "datasets",
            "converters",
            "heads",
            "training_logs",
            "converter_logs",
            ".tmp",
        ] {
            std::fs::create_dir_all(ws.join(sub)).map_err(|e| io_err(ws.join(sub).display(), e))?;
        }
        let now = now_rfc3339();
        // Empty head index lands BEFORE the workspace core so a
        // crash between the two leaves a directory without a
        // `workspace.json` -- and `list_workspaces` filters
        // those out.
        write_head_index(&ws, &HeadIndex::default())?;
        let core = WorkspaceCore {
            id,
            name: name.to_string(),
            tags: normalized_tags,
            created_at: now.clone(),
            workspace_revision: WorkspaceRevision { id: 0, at: now },
            head_count: 0,
        };
        write_workspace_core(&ws, &core)?;
        // Three-level fsync so the workspace's directory entry
        // reaches stable storage.  `write_workspace_core` already
        // fsynced `ws` (via `put_atomic`); we additionally fsync
        // the `<root>/workspaces/` and `<root>/` parents so a
        // crash + remount sees the new entry.
        if workspaces_root.exists() {
            fsync_dir(&workspaces_root).map_err(|e| io_err(workspaces_root.display(), e))?;
        }
        if self.root.exists() {
            fsync_dir(&self.root).map_err(|e| io_err(self.root.display(), e))?;
        }
        // Seed the eager cache with the just-written values.
        self.caches.insert(
            id,
            Arc::new(WorkspaceCacheCell::new(core, HeadIndex::default())),
        );
        Ok(id)
    }

    /// Atomically update one or both of `workspace.json`'s
    /// mutable metadata fields.  At least one of
    /// `name` / `tags` must be `Some(_)`; both `None` is rejected
    /// at the route boundary, so this method panics in that case
    /// (debug build only) and is a no-op in release.
    ///
    /// Validation:
    /// - `name`: same rules as `create` (length, charset, no
    ///   whitespace edges, Unicode case-insensitive uniqueness
    ///   across the workspace registry **excluding the current
    ///   workspace**).
    /// - `tags`: trim + per-tag rules + within-workspace Unicode
    ///   case-insensitive uniqueness + max 32 entries.
    ///
    /// Crash consistency:
    /// 1. Read the current core (cached or on-disk).
    /// 2. Apply the typed updates -- preserves
    ///    `workspace_revision`, `head_count`, `created_at`.
    /// 3. Atomic-rewrite `workspace.json` via `put_atomic` (file
    ///    fsync + parent fsync).
    /// 4. Publish the new core to the cache.
    ///
    /// `workspace_revision`, `head_count`, and head freshness are
    /// NOT touched -- name / tag edits are operator metadata, not
    /// workspace mutations.
    pub fn patch_workspace(
        &self,
        id: &WorkspaceId,
        name: Option<&str>,
        tags: Option<&[String]>,
    ) -> Result<Arc<crate::common::workspace::WorkspaceCore>, FileError> {
        if name.is_none() && tags.is_none() {
            return Err(FileError::InvalidName(
                "patch_workspace requires at least one of `name` or `tags`".into(),
            ));
        }
        // Validate the inputs BEFORE acquiring the registry /
        // per-workspace mutex so a malformed request returns
        // 400 / 409 cheaply.
        if let Some(n) = name {
            validate_workspace_name(n)?;
        }
        let normalized_tags = match tags {
            Some(t) => Some(validate_workspace_tags(t)?),
            None => None,
        };

        // Hold the registry-level mutex across the uniqueness
        // scan + per-workspace publish so a concurrent
        // `create("foo")` cannot slip a name in between our scan
        // and our commit.  `parking_lot::Mutex` is sync; no
        // `.await` inside.
        let _registry_guard = self.registry_lock.lock();

        if let Some(new_name) = name {
            self.ensure_workspace_name_available(new_name, Some(id))?;
        }

        let workspace_dir = self.workspace_dir(id);
        if !crate::file_mgr::schema::workspace_core_path(&workspace_dir).exists() {
            return Err(FileError::NotFound(id.to_string()));
        }

        // Per-workspace mutation mutex serializes against the
        // dataset / head publish paths.  Sync; never `.await`.
        let lock = self.metadata_lock(id);
        let _guard = lock.lock();

        // Resolve the cache cell (lazy-load on first touch); the
        // patch publishes back through this cell so subsequent
        // summary reads observe the new metadata.
        let cell = self.cache_cell_for_patch(id)?;
        let prev_core = cell.core();
        let mut next_core = (*prev_core).clone();
        if let Some(new_name) = name {
            next_core.name = new_name.to_string();
        }
        if let Some(new_tags) = normalized_tags {
            next_core.tags = new_tags;
        }
        // Workspace revision / head count / created_at are
        // intentionally preserved: name + tag edits are operator
        // metadata, not workspace mutations.

        crate::file_mgr::schema::write_workspace_core(&workspace_dir, &next_core)?;
        cell.publish_core(next_core.clone());
        Ok(Arc::new(next_core))
    }

    /// Resolve (or lazy-load) the cache cell for `patch_workspace`.
    /// Mirrors the lookup pattern in `dataset.rs::cache_cell_for_dataset`
    /// and `head_rotation.rs::cache_cell_for_head_delete`; we keep
    /// per-caller copies rather than centralising because each
    /// pins different surface invariants.
    fn cache_cell_for_patch(&self, id: &WorkspaceId) -> Result<Arc<WorkspaceCacheCell>, FileError> {
        if let Some(cell) = self.caches.get(id) {
            return Ok(cell.clone());
        }
        let workspace_dir = self.workspace_dir(id);
        let cell = Arc::new(WorkspaceCacheCell::load_from_disk(&workspace_dir)?);
        Ok(self
            .caches
            .entry(*id)
            .or_insert_with(|| cell.clone())
            .clone())
    }

    /// Ensure `name` is Unicode-case-insensitively unique across
    /// published workspaces.  Callers hold `registry_lock` across
    /// this scan and their subsequent commit so create/rename races
    /// cannot slip through the gap.
    ///
    /// Case fold uses `str::to_lowercase` (simple Unicode case
    /// folding via the `Lowercase` core property).  This catches
    /// `"Café"` ↔ `"café"` collisions but not exotic edge cases:
    /// German `ß` stays as `ß`, Turkish dotted/dotless I uses root
    /// locale, and decomposed vs composed forms (NFC vs NFD) are
    /// treated as distinct.
    fn ensure_workspace_name_available(
        &self,
        name: &str,
        except: Option<&WorkspaceId>,
    ) -> Result<(), FileError> {
        let lower = name.to_lowercase();
        for existing in self.list_workspaces()? {
            if except.is_some_and(|id| existing == *id) {
                continue;
            }
            let core = match self.read_cached_core(&existing) {
                Ok(c) => c,
                // Half-completed creates surface as `NotFound`;
                // skip rather than abort the whole scan.
                Err(FileError::Io { source, .. })
                    if source.kind() == std::io::ErrorKind::NotFound =>
                {
                    continue;
                }
                Err(e) => return Err(e),
            };
            if core.name.to_lowercase() == lower {
                return Err(FileError::NameConflict(name.to_string()));
            }
        }
        Ok(())
    }

    /// Synchronous fallback for [`Self::start_delete_workspace`].
    /// Stages the workspace, drains it, and finalizes inline so
    /// existing test surfaces that expect "delete returns when the
    /// bytes are gone" continue to pass.  Production code should
    /// call [`Self::start_delete_workspace`] directly and observe
    /// the job through the `JobRegistry`.
    pub fn delete(&self, id: &WorkspaceId) -> Result<(), FileError> {
        let (staged, _job) = self.start_delete_workspace_inner(id)?;
        // Drive the drain to completion in bounded batches; this
        // matches the off-mutex worker the async path runs.
        loop {
            match drain_staged_payload(&staged, DEFAULT_DELETE_BATCH_ENTRIES)? {
                DrainResult::Done => break,
                DrainResult::More => continue,
            }
        }
        finalize_staged_delete(&staged)?;
        Ok(())
    }

    /// Begin an asynchronous workspace delete.  Stages the
    /// workspace tree under `<root>/.tmp/delete-workspace-<job_id>/payload`
    /// and returns the `JobId` immediately; the caller is
    /// responsible for spawning a drain task that calls
    /// [`crate::file_mgr::staging::drain_staged_payload`] +
    /// [`crate::file_mgr::staging::finalize_staged_delete`] to
    /// completion (or relying on boot recovery if the daemon
    /// crashes).
    ///
    /// The synchronous fallback [`Self::delete`] calls this then
    /// drains inline so existing tests stay green.
    ///
    /// Admission flows through
    /// [`crate::file_mgr::job_registry::JobRegistry`] as a
    /// [`crate::common::workspace::JobType::WorkspaceDelete`]; the
    /// handle holds a single
    /// [`crate::common::workspace::JobReference::Workspace`]
    /// reference.  Follow-up upload / file-delete / workspace-
    /// delete requests observe HTTP 409 `JobConflict` until the
    /// drain completes and the handle drops.  `&Arc<Self>`
    /// because the registry's admission path bumps the registry's
    /// `Arc` for the resulting handle.
    pub fn start_delete_workspace(self: &Arc<Self>, id: &WorkspaceId) -> Result<JobId, FileError> {
        // Existence gate runs FIRST so the second delete on an
        // already-deleted workspace returns `NotFound` (HTTP 404)
        // even when an earlier drain is still in flight.
        let ws = self.workspace_dir(id);
        let core_path = crate::file_mgr::schema::workspace_core_path(&ws);
        if !core_path.exists() {
            return Err(FileError::NotFound(id.to_string()));
        }
        // Acquire the JobHandle BEFORE any disk mutation so
        // overlap conflicts fire before staging.  The registry-
        // allocated job_id supersedes any tombstone-side
        // identifier so the JSONL log + `/jobs/{job_id}` snapshot
        // share one identity.
        let job_handle = self.register_workspace_delete(*id)?;
        let job_id = job_handle.job_id();
        let staged = self.start_delete_workspace_inner_with_id(id, job_id)?;
        // Drain happens off the request path; a tokio runtime
        // may not exist (sync test contexts).  Spawn a blocking
        // task only when a runtime handle is available.
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            let staged_for_task = staged.clone();
            handle.spawn_blocking(move || {
                let budget = DEFAULT_DELETE_BATCH_ENTRIES;
                loop {
                    match drain_staged_payload(&staged_for_task, budget) {
                        Ok(DrainResult::Done) => break,
                        Ok(DrainResult::More) => continue,
                        Err(e) => {
                            tracing::warn!(
                                target: "file_mgr",
                                err = %e,
                                "workspace delete drain failed; boot recovery will resume",
                            );
                            job_handle.fail(format!("workspace delete drain failed: {e}"));
                            return;
                        }
                    }
                }
                if let Err(e) = finalize_staged_delete(&staged_for_task) {
                    tracing::warn!(
                        target: "file_mgr",
                        err = %e,
                        "workspace delete finalize failed; boot recovery will resume",
                    );
                    job_handle.fail(format!("workspace delete finalize failed: {e}"));
                    return;
                }
                job_handle.succeed(Some(
                    crate::file_mgr::job_registry::JobResult::WorkspaceDelete {
                        active_source_deleted: false,
                    },
                ));
            });
        } else {
            // No tokio runtime -- run inline.  This branch is
            // hit by sync tests; the cost is minor (the drain
            // is small).  Failure logs but does not surface to
            // the caller because the staged state is recoverable
            // via boot.
            let mut iter = 0usize;
            loop {
                iter += 1;
                if iter > 1_000_000 {
                    // Safety bound: pathological cases (>256M
                    // entries) deserve operator attention.
                    tracing::error!(
                        target: "file_mgr",
                        "workspace delete drain failed to converge after 1M iterations",
                    );
                    break;
                }
                match drain_staged_payload(&staged, DEFAULT_DELETE_BATCH_ENTRIES)? {
                    DrainResult::Done => break,
                    DrainResult::More => continue,
                }
            }
            finalize_staged_delete(&staged)?;
            job_handle.succeed(Some(
                crate::file_mgr::job_registry::JobResult::WorkspaceDelete {
                    active_source_deleted: false,
                },
            ));
        }
        Ok(job_id)
    }

    /// Stage a workspace delete: writes the tombstone, renames
    /// the workspace directory under root `.tmp/`, fsyncs the
    /// parents, ejects the cache + per-id metadata lock entry.
    /// Returns the `StagedDelete` bundle so the caller (sync
    /// fallback or async worker) can drive drain + finalize.
    fn start_delete_workspace_inner(
        &self,
        id: &WorkspaceId,
    ) -> Result<(StagedDelete, JobId), FileError> {
        let job_id = JobId::new();
        let staged = self.start_delete_workspace_inner_with_id(id, job_id)?;
        Ok((staged, job_id))
    }

    /// Same as [`Self::start_delete_workspace_inner`], but reuses
    /// an externally-supplied `job_id` so the registry-allocated
    /// id can flow through the tombstone + JSONL log + `/jobs`
    /// snapshot under one identity.
    fn start_delete_workspace_inner_with_id(
        &self,
        id: &WorkspaceId,
        job_id: JobId,
    ) -> Result<StagedDelete, FileError> {
        let ws = self.workspace_dir(id);
        // Reject when `workspace.json` is missing -- a directory
        // without it is an incomplete create, not a workspace.
        let core_path = crate::file_mgr::schema::workspace_core_path(&ws);
        if !core_path.exists() {
            return Err(FileError::NotFound(id.to_string()));
        }
        let tombstone = DeleteTombstone::Workspace {
            job_id,
            workspace_id: *id,
            created_at: now_rfc3339(),
        };
        let staging_dir = root_tmp_dir(&self.root);
        let staged = write_tombstone(&staging_dir, &tombstone)?;
        stage_payload(&ws, &staged)?;
        // Old parent `<root>/workspaces/` was fsynced inside
        // `stage_payload`; the staging dir parent (`<root>/.tmp/`)
        // is also covered there.  Eject runtime references so
        // subsequent reads fail closed.
        self.caches.remove(id);
        self.metadata_locks.remove(id);
        Ok(staged)
    }

    /// List all workspaces under `<root>/workspaces/`, returning
    /// their UUIDs.  Sub-entries that aren't valid workspace
    /// dirs (no `workspace.json`, or directory name isn't a
    /// UUID-v4 string) are silently skipped: those are
    /// half-finished creates or unrelated files an operator
    /// dropped into the workspaces root.
    pub fn list_workspaces(&self) -> Result<Vec<WorkspaceId>, FileError> {
        let workspaces_root = workspaces_dir(&self.root);
        if !workspaces_root.exists() {
            return Ok(Vec::new());
        }
        let entries = std::fs::read_dir(&workspaces_root)
            .map_err(|e| io_err(workspaces_root.display(), e))?;
        let mut out = Vec::new();
        for e in entries.flatten() {
            if !e.file_type().is_ok_and(|t| t.is_dir()) {
                continue;
            }
            let id_str = e.file_name().to_string_lossy().into_owned();
            // Two gates: the dir name must parse as a strict
            // `WorkspaceId` (UUID-v4) AND the directory must
            // contain a `workspace.json`.  The crash-recovery
            // contract treats `workspace.json` as the publish
            // point, so a directory without one is an incomplete
            // create.
            let id = match WorkspaceId::parse(&id_str) {
                Ok(id) => id,
                Err(_) => continue,
            };
            let core_path = crate::file_mgr::schema::workspace_core_path(&e.path());
            if core_path.exists() {
                out.push(id);
            }
        }
        Ok(out)
    }

    /// Read the eagerly-cached workspace core, lazily loading
    /// from disk if absent.  Used by [`Self::create`]'s
    /// uniqueness check against pre-existing workspaces and by
    /// [`Self::summary`].
    pub(crate) fn read_cached_core(
        &self,
        id: &WorkspaceId,
    ) -> Result<Arc<WorkspaceCore>, FileError> {
        let cell = self.cache_cell(id)?;
        Ok(cell.core())
    }

    /// Resolve (or lazily load) the per-workspace cache cell.
    fn cache_cell(&self, id: &WorkspaceId) -> Result<Arc<WorkspaceCacheCell>, FileError> {
        if let Some(cell) = self.caches.get(id) {
            return Ok(cell.clone());
        }
        let ws = self.workspace_dir(id);
        let cell = Arc::new(WorkspaceCacheCell::load_from_disk(&ws)?);
        // `entry().or_insert_with` is the dashmap idiom for the
        // race where two callers both observed an absent slot.
        let inserted = self
            .caches
            .entry(*id)
            .or_insert_with(|| cell.clone())
            .clone();
        Ok(inserted)
    }

    /// Hot-path summary read.  Returns the cached
    /// `WorkspaceCore` + `HeadIndex` snapshots plus the derived
    /// per-head [`crate::common::workspace::HeadStatus`] list.
    /// Never walks `datasets/`.
    pub fn summary(
        &self,
        id: &WorkspaceId,
    ) -> Result<crate::file_mgr::WorkspaceSummary, FileError> {
        let cell = self.cache_cell(id)?;
        let core = cell.core();
        let heads = cell.heads();
        let head_statuses = heads
            .heads
            .iter()
            .map(|h| {
                crate::common::workspace::HeadStatus::from_revisions(
                    &h.workspace_revision,
                    &core.workspace_revision,
                )
            })
            .collect();
        Ok(crate::file_mgr::WorkspaceSummary {
            core,
            heads,
            head_statuses,
        })
    }

    /// Path helper resolving `<root>/workspaces/<workspace_id>/`.
    pub(crate) fn workspace_dir(&self, id: &WorkspaceId) -> PathBuf {
        workspace_dir_for(&self.root, id)
    }
}
