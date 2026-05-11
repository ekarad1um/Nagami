//! Workspace + asset filesystem trait surface.
//!
//! Held by the api as `Arc<dyn FsService>`.  Production impl
//! [`FsServiceImpl`] is a thin facade over [`WorkspaceMgr`];
//! tests substitute an in-memory mock that satisfies the
//! trait without depending on the on-disk layout.
//!
//! # Layering
//!
//! The trait + its DTO references all live in `file_mgr`
//! rather than [`crate::common::traits`].  Pulling
//! [`WorkspaceMetadata`], [`AssetKind`], etc. into `common`
//! would force `common` to depend on `time` (for
//! [`WorkspaceMetadata::new`]), violating the contract
//! module's zero-workspace-dep rule.  Consumers (api,
//! training) already import `file_mgr`'s DTO surface; adding
//! the trait there is zero new coupling.
//!
//! # Object safety
//!
//! Every method is dyn-compatible:
//!
//! - No generic parameters in any method signature.
//! - [`MetadataGuard`] is itself a trait (returned as
//!   `Box<dyn MetadataGuard + '_>`) so the per-impl guard
//!   state stays private.
//! - [`FsService::install_from_path`] takes `&Path` (a
//!   pre-staged tempfile) instead of an `AsyncRead`; that
//!   keeps the trait tokio-free and lets api callers stage
//!   request bodies in their own `spawn_blocking`-friendly
//!   form.  See [`FsService::acquire_upload_permit`] for the
//!   admission gate that the api wraps around the
//!   streaming-then-install pair.
//!
//! # Sync-only surface
//!
//! Every trait method is sync; none returns a `Future`.
//! Callers in async contexts wrap the entire critical
//! section in [`tokio::task::spawn_blocking`].  The
//! operations are short blocking I/O (atomic rename +
//! per-workspace lock + metadata write); pushing them into
//! a `Box<dyn Future>` would add allocation + dynamic
//! dispatch with no win, since callers will
//! `spawn_blocking` either way.

use crate::common::asset_path::AssetPath;
use crate::common::error::{Categorized, ErrorKind};
use crate::common::ids::{HeadId, JobId, WorkspaceId};
use crate::common::workspace::JobReference;
use crate::file_mgr::dataset::{DatasetListing, DatasetUploadReceipt, JobRefHandle};
use crate::file_mgr::error::{FileError, io_err};
use crate::file_mgr::metadata::{AssetKind, AssetRecord, WorkspaceMetadata};
use crate::file_mgr::{AssetReceipt, WorkspaceReport, WorkspaceSummary};
use std::path::{Path, PathBuf};

/// Type-erased error returned by every [`FsService`] method.
///
/// Concrete error variants stay in [`FileError`] (the impl
/// side); the trait boundary boxes them so api handlers can
/// hold `Arc<dyn FsService>` without naming the concrete
/// error type.  The `kind()` classifier is preserved across
/// the boundary so HTTP status mapping (via [`Categorized`])
/// still works without per-variant matches.
#[derive(Debug)]
pub struct FsError {
    inner: Box<dyn std::error::Error + Send + Sync + 'static>,
    kind: ErrorKind,
}

impl FsError {
    /// Box any `Categorized + Error + Send + Sync + 'static`
    /// into an [`FsError`], capturing its [`ErrorKind`] so
    /// downstream classifiers don't need to downcast.
    pub fn new<E>(e: E) -> Self
    where
        E: std::error::Error + Send + Sync + Categorized + 'static,
    {
        let kind = e.kind();
        Self {
            inner: Box::new(e),
            kind,
        }
    }
}

impl std::fmt::Display for FsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.inner, f)
    }
}

impl std::error::Error for FsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.inner.as_ref())
    }
}

impl Categorized for FsError {
    fn kind(&self) -> ErrorKind {
        self.kind
    }
}

impl From<FileError> for FsError {
    fn from(e: FileError) -> Self {
        Self::new(e)
    }
}

/// RAII permit gating concurrent uploads.
///
/// Held across the api's streaming staging pass and the
/// eventual [`FsService::install_from_path`] call.  The
/// opaque `Box<dyn Send>` lets the impl side stash an
/// [`tokio::sync::OwnedSemaphorePermit`] (or any other
/// guard) without forcing tokio into the trait surface; the
/// trait stays std-only.  Drop releases the slot.
pub struct UploadPermit {
    _inner: Box<dyn Send + 'static>,
}

impl UploadPermit {
    /// Construct from any `Send + 'static` guard.  Used by
    /// [`FsServiceImpl::acquire_upload_permit`] to wrap an
    /// [`tokio::sync::OwnedSemaphorePermit`].
    pub fn from_guard<T: Send + 'static>(guard: T) -> Self {
        Self {
            _inner: Box::new(guard),
        }
    }
}

impl std::fmt::Debug for UploadPermit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UploadPermit").finish_non_exhaustive()
    }
}

/// Per-workspace metadata read-modify-write critical
/// section.
///
/// Returned by [`FsService::metadata_mut`].  Holds the
/// per-workspace metadata lock for its entire lifetime.
/// Mutate via [`Self::metadata_mut`]; persist + release via
/// [`Self::commit`].  Dropping without committing releases
/// the lock and discards changes (rollback by design).
///
/// # Concurrency contract
///
/// Trait objects of this trait are `!Send` by default -- the
/// underlying `parking_lot::Mutex` guard does not yield.
/// Callers in async contexts must hold the guard inside a
/// single [`tokio::task::spawn_blocking`] closure.  The
/// compiler enforces this via the `!Send` future bound:
/// holding the guard across an `.await` in an async
/// function makes the surrounding `Future` `!Send` and the
/// runtime rejects it.
///
/// # Canonical usage
///
/// ```text
/// let fs: Arc<dyn FsService> = state.files.clone();
/// task::spawn_blocking(move || -> Result<(), FsError> {
///     let mut g = fs.metadata_mut(&ws)?;
///     g.metadata_mut().assets.push(record);
///     g.commit()
/// }).await??;
/// ```
pub trait MetadataGuard {
    /// Read-only borrow of the locked metadata.
    fn metadata(&self) -> &WorkspaceMetadata;

    /// Mutable borrow.  The next [`Self::commit`] persists
    /// these changes; a drop without commit discards them.
    fn metadata_mut(&mut self) -> &mut WorkspaceMetadata;

    /// Atomic write + release.  Consumes the guard (via
    /// `Box<Self>` per the dyn-trait calling convention) so
    /// callers can't keep mutating after persist.
    fn commit(self: Box<Self>) -> Result<(), FsError>;
}

/// Object-safe filesystem service.  Held by api as
/// `Arc<dyn FsService>`.  Production impl:
/// [`FsServiceImpl`] (a thin facade over [`WorkspaceMgr`]).
///
/// # Method taxonomy
///
/// - **Path readers** ([`Self::root`],
///   [`Self::workspace_tmpdir`], [`Self::asset_path`]) are
///   pure path joins; no I/O.  Safe to call from any
///   context.
/// - **Workspace lifecycle + metadata reads**
///   ([`Self::create`], [`Self::delete`],
///   [`Self::list_workspaces`], [`Self::read_metadata`],
///   [`Self::list_assets`], [`Self::validate`]) do blocking
///   I/O.  Async callers wrap in
///   [`tokio::task::spawn_blocking`].
/// - **Atomic RMW** ([`Self::metadata_mut`] +
///   [`MetadataGuard::commit`]) holds the per-workspace
///   metadata mutex.  Strict `spawn_blocking`-only.
/// - **Upload** ([`Self::acquire_upload_permit`] +
///   [`Self::install_from_path`]): the api streams the
///   request body into a tempfile under
///   [`Self::workspace_tmpdir`], then calls
///   [`Self::install_from_path`] to atomic-rename + commit
///   the metadata record.  The permit gates the global
///   concurrency cap across both phases.
pub trait FsService: Send + Sync + 'static {
    // MARK: Pure path readers

    /// Workspace root directory.  Used by api callers that
    /// need to construct sibling tempdirs on the same
    /// filesystem (e.g. the converter's staging dir).
    fn root(&self) -> &Path;

    /// `<workspace>/.tmp/`, guaranteed to exist (the
    /// directory is laid down at workspace-creation time).
    /// Used by api callers that stage tempfiles for
    /// [`Self::install_from_path`].
    fn workspace_tmpdir(&self, ws: &WorkspaceId) -> PathBuf;

    /// On-disk path for an asset under
    /// `<workspace>/<subdir>/`.  Pure path join; trusts the
    /// caller to have run
    /// [`crate::file_mgr::validate_asset_name`] on `name`.
    fn asset_path(&self, ws: &WorkspaceId, kind: AssetKind, name: &str) -> PathBuf;

    // MARK: Workspace lifecycle + read paths

    fn create(&self, name: &str) -> Result<WorkspaceId, FsError>;
    /// Create a workspace with optional operator-supplied tags.
    /// Tags are trimmed and validated; an empty slice yields
    /// `tags = []` on the persisted core.
    fn create_with_tags(&self, name: &str, tags: &[String]) -> Result<WorkspaceId, FsError>;
    /// Atomic name + tag edit; at least one of `name` / `tags`
    /// must be `Some(_)`.  Returns the freshly published core so
    /// the api can echo updated metadata without a follow-up
    /// summary read.
    fn patch_workspace(
        &self,
        ws: &WorkspaceId,
        name: Option<&str>,
        tags: Option<&[String]>,
    ) -> Result<Arc<crate::common::workspace::WorkspaceCore>, FsError>;
    fn delete(&self, ws: &WorkspaceId) -> Result<(), FsError>;
    /// Begin an asynchronous workspace delete and return the
    /// owning `JobId`.  The workspace tree moves under root
    /// `.tmp/delete-workspace-<job_id>/payload`; the off-mutex
    /// drain runs on the tokio blocking pool.  Boot recovery
    /// resumes any drain interrupted by a daemon crash.
    fn start_delete_workspace(&self, ws: &WorkspaceId) -> Result<JobId, FsError>;
    /// Idempotently create the root layout (`workspaces/`,
    /// `.tmp/`, `active/`, `backbone/`).  Daemon first-boot calls
    /// this before any other lifecycle work.
    fn ensure_root_layout(&self) -> Result<(), FsError>;
    fn list_workspaces(&self) -> Result<Vec<WorkspaceId>, FsError>;
    fn read_metadata(&self, ws: &WorkspaceId) -> Result<WorkspaceMetadata, FsError>;
    /// Hot-path summary read backed by the per-workspace eager
    /// cache; never walks `datasets/`.
    fn summary(&self, ws: &WorkspaceId) -> Result<WorkspaceSummary, FsError>;
    fn list_assets(&self, ws: &WorkspaceId, kind: AssetKind) -> Result<Vec<AssetRecord>, FsError>;
    fn validate(&self, ws: &WorkspaceId) -> Result<WorkspaceReport, FsError>;

    // MARK: Atomic per-workspace metadata RMW

    /// Acquire the per-workspace metadata lock and load the
    /// current value.  Mutate the returned guard, then call
    /// [`MetadataGuard::commit`] to atomically write +
    /// release.  Drop without commit = rollback.
    fn metadata_mut(&self, ws: &WorkspaceId) -> Result<Box<dyn MetadataGuard + '_>, FsError>;

    // MARK: Upload + install

    /// Acquire an upload concurrency permit.  Fail-fast (no
    /// blocking); returns [`FsError`] with kind
    /// [`ErrorKind::Conflict`] when the global in-flight
    /// cap is reached.  The permit's `Drop` releases the
    /// slot; api callers should hold it across the
    /// staging-stream + the follow-up
    /// [`Self::install_from_path`] call.
    fn acquire_upload_permit(&self) -> Result<UploadPermit, FsError>;

    /// Per-request upload byte cap from
    /// [`crate::file_mgr::AdmissionCfg`].  Used by the api's
    /// streaming staging pass to enforce the limit without
    /// re-reading config.  Returns [`u64::MAX`] if no cap
    /// is configured.
    fn max_upload_bytes(&self) -> u64;

    /// Install a pre-staged asset by atomic rename + metadata
    /// commit.
    ///
    /// `src` must be on the same filesystem as the workspace
    /// root (caller's responsibility -- typically a
    /// `NamedTempFile::new_in(fs.workspace_tmpdir(ws))`).
    /// Holds the per-workspace metadata lock across the
    /// rename + commit.  Validates name + extension; rejects
    /// case-insensitive collisions per the casing policy in
    /// [`crate::file_mgr`]; computes sha256 from `src`
    /// before rename.
    fn install_from_path(
        &self,
        ws: &WorkspaceId,
        kind: AssetKind,
        name: &str,
        src: &Path,
    ) -> Result<AssetReceipt, FsError>;

    // MARK: Workspace asset surface (AssetPath-shaped)

    /// Workspace-rooted asset path resolution + listings + regular
    /// file open.  `path` carries the tree-root component
    /// (`datasets/...` or `converters/...`); `.tmp/` is rejected.
    fn workspace_asset_path(&self, ws: &WorkspaceId, path: &AssetPath) -> Result<PathBuf, FsError>;
    fn open_workspace_file(
        &self,
        ws: &WorkspaceId,
        path: &AssetPath,
    ) -> Result<(PathBuf, std::fs::Metadata), FsError>;
    fn list_workspace_children(
        &self,
        ws: &WorkspaceId,
        relative: Option<&AssetPath>,
        offset: usize,
        limit: usize,
    ) -> Result<DatasetListing, FsError>;

    /// Commit a single-file workspace upload from a pre-staged
    /// tempfile.  `path` must start with `datasets/` or
    /// `converters/` and carry at least one child.  See
    /// [`crate::file_mgr::WorkspaceMgr::upload_workspace_file`]
    /// for the full ordering contract.
    fn upload_workspace_file(
        &self,
        ws: &WorkspaceId,
        path: &AssetPath,
        src_tmpfile: &Path,
        observed_sha256: &str,
        observed_size: u64,
    ) -> Result<DatasetUploadReceipt, FsError>;

    /// Begin an asynchronous workspace asset delete; returns the
    /// owning [`JobId`] immediately.  All four mutable trees
    /// (`datasets/`, `converters/`, `training_logs/`,
    /// `converter_logs/`) share the same async tombstone+stage+drain
    /// shape — datasets/converters bump `workspace_revision`,
    /// log trees skip the bump but go through identical staging +
    /// drain machinery + recovery scan.  See
    /// [`crate::file_mgr::WorkspaceMgr::start_workspace_asset_delete`]
    /// for the per-tree details.
    fn start_workspace_asset_delete(
        &self,
        ws: &WorkspaceId,
        path: &AssetPath,
    ) -> Result<JobId, FsError>;

    /// Synchronously remove a single trained head from the
    /// workspace's 2-slot index.  Holds the per-workspace
    /// mutation mutex; rewrites `heads.json` and `workspace.json`
    /// atomically, then unlinks `heads/<head_id>.{mpk,json}`
    /// best-effort.
    fn delete_head(&self, ws: &WorkspaceId, head_id: HeadId) -> Result<(), FsError>;

    /// Publish a freshly produced trained head into the
    /// workspace's 2-slot index.  Routes through the
    /// per-workspace mutation mutex + cache cell so the rotation
    /// primitive runs against the live `WorkspaceCacheCell`.
    fn publish_trained_head(
        &self,
        ws: &WorkspaceId,
        pending: crate::file_mgr::PendingHead,
    ) -> Result<crate::file_mgr::HeadRotationResult, FsError>;

    /// Register an arbitrary [`JobReference`] against the
    /// in-memory job-conflict shim.  Returns an opaque RAII guard
    /// that releases the lease on drop.  Used by the convert
    /// producer to claim per-input-file leases before spawning
    /// the worker; conflicting requests observe `JobConflict`.
    ///
    /// The default impl panics; impls that own the registry
    /// (production [`FsServiceImpl`]) override this.  Mock impls
    /// in tests that do not exercise the conflict surface keep
    /// the default and never call into it.
    fn register_job_reference(&self, _reference: JobReference) -> Result<JobRefHandle, FsError> {
        Err(FsError::new(FileError::JobConflict {
            message: "register_job_reference is unimplemented for this FsService impl".to_string(),
        }))
    }

    /// Atomically write `bytes` to `path` via the canonical
    /// tempfile + sync_all + rename + dir-fsync primitive
    /// (see [`crate::file_mgr::put_atomic`]).  The path's
    /// parent directory must already exist; the staging
    /// tempfile lands as a sibling (intra-FS rename
    /// guarantee).
    ///
    /// Surfaced on the trait so the converter's pipeline can
    /// drive its head/labels/metadata writes through
    /// `Arc<dyn FsService>` instead of carrying its own
    /// atomic-write helper.  The default impl is the only
    /// impl -- concrete [`FsService`] types share one
    /// durability discipline.
    ///
    /// Distinct from [`Self::install_bytes`]: that one is
    /// workspace-scoped (writes a named asset under a
    /// [`WorkspaceId`] + [`AssetKind`] subdir, with admission
    /// checks + metadata commit).  This one is path-scoped
    /// -- used for writes outside the workspace asset tree
    /// (e.g., the converter's sibling tempdir staging area,
    /// which the api then [`Self::install_from_path`]s into
    /// the dst workspace).
    fn put_atomic(&self, path: &Path, bytes: &[u8]) -> Result<(), FsError> {
        crate::file_mgr::fs_atomic::put_atomic(path, bytes).map_err(FsError::from)
    }

    /// Convenience: stage `bytes` to a tempfile in
    /// [`Self::workspace_tmpdir`], then
    /// [`Self::install_from_path`].  Used by the training
    /// crate's checkpoint-emit path; the api's streaming
    /// upload uses the explicit two-phase shape so it can
    /// enforce [`Self::max_upload_bytes`] mid-stream.
    fn install_bytes(
        &self,
        ws: &WorkspaceId,
        kind: AssetKind,
        name: &str,
        bytes: &[u8],
    ) -> Result<AssetReceipt, FsError> {
        use std::io::Write;
        let tmp_dir = self.workspace_tmpdir(ws);
        let mut tmp = tempfile::NamedTempFile::new_in(&tmp_dir)
            .map_err(|e| FsError::new(io_err(tmp_dir.display(), e)))?;
        tmp.write_all(bytes)
            .map_err(|e| FsError::new(io_err(tmp.path().display(), e)))?;
        tmp.as_file()
            .sync_all()
            .map_err(|e| FsError::new(io_err(tmp.path().display(), e)))?;
        self.install_from_path(ws, kind, name, tmp.path())
        // `tmp` drops on success (`NamedTempFile::Drop` is a
        // no-op when the file no longer exists post-rename).
    }
}

// MARK: Production impl

use std::sync::Arc;

use crate::file_mgr::{AdmissionCfg, WorkspaceMgr};

/// Production impl of [`FsService`].  Thin facade over
/// [`WorkspaceMgr`]; constructed once at daemon boot, held
/// by the api as `Arc<dyn FsService>`.
///
/// The facade design (vs moving every method body onto
/// `Self`) preserves the in-crate test surface:
/// [`WorkspaceMgr`]'s ~20 unit tests continue to exercise
/// the concrete type directly, and external consumers reach
/// the same behaviour via the trait object so production
/// wiring matches the test fixtures exactly.
#[derive(Debug, Clone)]
pub struct FsServiceImpl {
    mgr: Arc<WorkspaceMgr>,
}

impl FsServiceImpl {
    /// Construct without admission control.  Mirrors
    /// [`WorkspaceMgr::new`].
    pub fn new(root: PathBuf) -> Self {
        Self {
            mgr: Arc::new(WorkspaceMgr::new(root)),
        }
    }

    /// Construct with admission caps enforced.  Daemon code
    /// uses this; tests + bootstrap paths that exercise
    /// unbounded uploads keep using [`Self::new`].
    pub fn with_admission(root: PathBuf, cfg: AdmissionCfg) -> Self {
        Self {
            mgr: Arc::new(WorkspaceMgr::with_admission(root, cfg)),
        }
    }

    /// Construct with admission caps + a shared
    /// [`crate::file_mgr::JobRegistry`].  Daemon boot uses this
    /// so `AppState::jobs` and the workspace-side admission paths
    /// register against one instance.
    pub fn with_admission_and_jobs(
        root: PathBuf,
        cfg: AdmissionCfg,
        jobs: Arc<crate::file_mgr::JobRegistry>,
    ) -> Self {
        Self {
            mgr: Arc::new(WorkspaceMgr::with_admission_and_jobs(root, cfg, jobs)),
        }
    }
}

/// Production impl of [`MetadataGuard`].
///
/// Holds an owned per-workspace metadata mutex (via
/// `parking_lot`'s `lock_arc`, not a borrow) so the guard is
/// detached from the `&self` lifetime.  That lets it cross
/// the [`tokio::task::spawn_blocking`] boundary as a
/// `Box<dyn ...>` without lifetime gymnastics.
///
/// The mutex guard is NOT `Send` (parking_lot default), so
/// the api's async future stays `!Send` if a caller
/// mistakenly holds one across an `.await`; the runtime
/// rejects it at compile time.
pub struct FileMetadataGuard {
    // `_arc_guard` keeps the per-workspace mutex held for
    // the guard's lifetime.  Field order matters for Drop:
    // the guard drops AFTER `meta` / `committed`, releasing
    // the lock last.
    mgr: Arc<WorkspaceMgr>,
    ws: WorkspaceId,
    meta: WorkspaceMetadata,
    committed: bool,
    dirty: bool,
    _arc_guard: parking_lot::ArcMutexGuard<parking_lot::RawMutex, ()>,
}

impl std::fmt::Debug for FileMetadataGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileMetadataGuard")
            .field("ws", &self.ws)
            .field("dirty", &self.dirty)
            .field("committed", &self.committed)
            .finish_non_exhaustive()
    }
}

impl MetadataGuard for FileMetadataGuard {
    fn metadata(&self) -> &WorkspaceMetadata {
        &self.meta
    }

    fn metadata_mut(&mut self) -> &mut WorkspaceMetadata {
        self.dirty = true;
        &mut self.meta
    }

    fn commit(mut self: Box<Self>) -> Result<(), FsError> {
        // Persist via the manager's existing write path
        // (backed by `fs_atomic::put_atomic`); set
        // `committed` BEFORE returning Ok so `Drop` doesn't
        // log a spurious "dropped without commit" warning.
        // The `_arc_guard` releases the lock when the Box
        // drops after this returns.
        self.mgr
            .write_metadata(&self.ws, &self.meta)
            .map_err(FsError::new)?;
        self.committed = true;
        Ok(())
    }
}

impl Drop for FileMetadataGuard {
    fn drop(&mut self) {
        if self.dirty && !self.committed {
            tracing::warn!(
                target: "file_mgr",
                ws = %self.ws,
                "MetadataGuard dropped without commit; mutations rolled back",
            );
        }
    }
}

impl FsService for FsServiceImpl {
    fn root(&self) -> &Path {
        self.mgr.root()
    }

    fn workspace_tmpdir(&self, ws: &WorkspaceId) -> PathBuf {
        // `<root>/workspaces/<id>/.tmp/`.  The `.tmp/` subdir is
        // laid down at workspace creation; api callers get a
        // guaranteed existing same-FS staging dir for
        // `tempfile::NamedTempFile::new_in`.
        crate::file_mgr::schema::workspace_dir_for(self.mgr.root(), ws).join(".tmp")
    }

    fn asset_path(&self, ws: &WorkspaceId, kind: AssetKind, name: &str) -> PathBuf {
        self.mgr.asset_path(ws, kind, name)
    }

    fn create(&self, name: &str) -> Result<WorkspaceId, FsError> {
        self.mgr.create(name).map_err(FsError::new)
    }

    fn create_with_tags(&self, name: &str, tags: &[String]) -> Result<WorkspaceId, FsError> {
        self.mgr.create_with_tags(name, tags).map_err(FsError::new)
    }

    fn patch_workspace(
        &self,
        ws: &WorkspaceId,
        name: Option<&str>,
        tags: Option<&[String]>,
    ) -> Result<Arc<crate::common::workspace::WorkspaceCore>, FsError> {
        self.mgr
            .patch_workspace(ws, name, tags)
            .map_err(FsError::new)
    }

    fn delete(&self, ws: &WorkspaceId) -> Result<(), FsError> {
        self.mgr.delete(ws).map_err(FsError::new)
    }

    fn start_delete_workspace(&self, ws: &WorkspaceId) -> Result<JobId, FsError> {
        self.mgr.start_delete_workspace(ws).map_err(FsError::new)
    }

    fn ensure_root_layout(&self) -> Result<(), FsError> {
        self.mgr.ensure_root_layout().map_err(FsError::new)
    }

    fn list_workspaces(&self) -> Result<Vec<WorkspaceId>, FsError> {
        self.mgr.list_workspaces().map_err(FsError::new)
    }

    fn read_metadata(&self, ws: &WorkspaceId) -> Result<WorkspaceMetadata, FsError> {
        self.mgr.read_metadata(ws).map_err(FsError::new)
    }

    fn summary(&self, ws: &WorkspaceId) -> Result<WorkspaceSummary, FsError> {
        self.mgr.summary(ws).map_err(FsError::new)
    }

    fn list_assets(&self, ws: &WorkspaceId, kind: AssetKind) -> Result<Vec<AssetRecord>, FsError> {
        self.mgr.list_assets(ws, kind).map_err(FsError::new)
    }

    fn validate(&self, ws: &WorkspaceId) -> Result<WorkspaceReport, FsError> {
        self.mgr.validate(ws).map_err(FsError::new)
    }

    fn metadata_mut(&self, ws: &WorkspaceId) -> Result<Box<dyn MetadataGuard + '_>, FsError> {
        let lock = self.mgr.metadata_lock(ws);
        // `lock_arc` returns an owned `ArcMutexGuard` (no
        // lifetime tied to `self`).  That detaches the
        // guard from the trait method's `&self` borrow so
        // the returned `Box<dyn MetadataGuard + '_>` is
        // free to be moved into a `spawn_blocking` closure
        // that captures only `Arc<dyn FsService>` by value.
        let arc_guard = lock.lock_arc();
        let meta = self.mgr.read_metadata(ws).map_err(FsError::new)?;
        Ok(Box::new(FileMetadataGuard {
            mgr: self.mgr.clone(),
            ws: *ws,
            meta,
            committed: false,
            dirty: false,
            _arc_guard: arc_guard,
        }))
    }

    fn acquire_upload_permit(&self) -> Result<UploadPermit, FsError> {
        match self.mgr.try_acquire_upload_permit() {
            Ok(opt) => match opt {
                Some(p) => Ok(UploadPermit::from_guard(p)),
                None => Ok(UploadPermit::from_guard(())),
            },
            Err(e) => Err(FsError::new(e)),
        }
    }

    fn max_upload_bytes(&self) -> u64 {
        self.mgr.max_upload_bytes()
    }

    fn install_from_path(
        &self,
        ws: &WorkspaceId,
        kind: AssetKind,
        name: &str,
        src: &Path,
    ) -> Result<AssetReceipt, FsError> {
        self.mgr
            .install_from_path(ws, kind, name, src)
            .map_err(FsError::new)
    }

    fn workspace_asset_path(&self, ws: &WorkspaceId, path: &AssetPath) -> Result<PathBuf, FsError> {
        self.mgr
            .workspace_asset_path(ws, path)
            .map_err(FsError::new)
    }

    fn open_workspace_file(
        &self,
        ws: &WorkspaceId,
        path: &AssetPath,
    ) -> Result<(PathBuf, std::fs::Metadata), FsError> {
        self.mgr.open_workspace_file(ws, path).map_err(FsError::new)
    }

    fn list_workspace_children(
        &self,
        ws: &WorkspaceId,
        relative: Option<&AssetPath>,
        offset: usize,
        limit: usize,
    ) -> Result<DatasetListing, FsError> {
        self.mgr
            .list_workspace_children(ws, relative, offset, limit)
            .map_err(FsError::new)
    }

    fn upload_workspace_file(
        &self,
        ws: &WorkspaceId,
        path: &AssetPath,
        src_tmpfile: &Path,
        observed_sha256: &str,
        observed_size: u64,
    ) -> Result<DatasetUploadReceipt, FsError> {
        self.mgr
            .upload_workspace_file(ws, path, src_tmpfile, observed_sha256, observed_size)
            .map_err(FsError::new)
    }

    fn start_workspace_asset_delete(
        &self,
        ws: &WorkspaceId,
        path: &AssetPath,
    ) -> Result<JobId, FsError> {
        self.mgr
            .start_workspace_asset_delete(ws, path)
            .map_err(FsError::new)
    }

    fn delete_head(&self, ws: &WorkspaceId, head_id: HeadId) -> Result<(), FsError> {
        self.mgr.delete_head(ws, head_id).map_err(FsError::new)
    }

    fn publish_trained_head(
        &self,
        ws: &WorkspaceId,
        pending: crate::file_mgr::PendingHead,
    ) -> Result<crate::file_mgr::HeadRotationResult, FsError> {
        self.mgr
            .publish_trained_head_for_workspace(ws, pending)
            .map_err(FsError::new)
    }

    fn register_job_reference(&self, reference: JobReference) -> Result<JobRefHandle, FsError> {
        self.mgr
            .register_job_reference(reference)
            .map_err(FsError::new)
    }
}
