//! `Uploader` aggregate of [`WorkspaceMgr`].
//!
//! Owns the admission cap surface ([`AdmissionCfg`] + the
//! global semaphore) and the two upload paths:
//!
//! - [`WorkspaceMgr::upload`] -- async streaming
//!   [`tokio::io::AsyncRead`] into a tempfile, sha256
//!   incrementally, atomic rename + metadata commit under
//!   the per-workspace lock.
//! - [`WorkspaceMgr::install_from_path`] -- sync atomic
//!   install of an already-staged tempfile (the api uses
//!   this after streaming the request body itself).
//!
//! # Admission caps
//!
//! Two-axis cap: [`AdmissionCfg::max_upload_bytes`]
//! (per-request size ceiling) +
//! [`AdmissionCfg::max_concurrent_uploads`] (global
//! in-flight count).  The semaphore is
//! [`tokio::sync::Semaphore`] driven via `try_acquire_owned`
//! semantics -- an over-cap upload fails fast with
//! [`FileError::TooManyConcurrentUploads`] rather than
//! blocking, matching the operator-facing contract that "the
//! request returns immediately; retry later".

use std::path::Path;
use std::sync::Arc;

use crate::common::ids::{AssetId, WorkspaceId};
use sha2::{Digest, Sha256};

use crate::file_mgr::error::{FileError, io_err, metadata_parse_err};
use crate::file_mgr::metadata::{AssetKind, AssetRecord};
use crate::file_mgr::validate::{
    fsync_dir, hex_lowercase, sha256_file_streaming, validate_extension,
};
use crate::file_mgr::{AssetReceipt, WorkspaceMgr, validate_asset_name};

/// Admission caps for [`WorkspaceMgr::upload`].  Defaults
/// are conservative for a single-tenant on-device
/// deployment; operators with larger workloads tune via
/// [`WorkspaceMgr::with_admission`].
///
/// Today only `max_upload_bytes` and
/// `max_concurrent_uploads` are enforced.  The full
/// admission story (free-disk pre-check via statvfs,
/// per-workspace size quotas) is platform-specific
/// (statvfs is Linux/macOS-only via `nix`) and an
/// accumulation surface across uploads -- both deferred.
#[derive(Clone, Copy, Debug)]
pub struct AdmissionCfg {
    /// Per-request hard ceiling on uncompressed upload bytes.
    /// Default 256 MiB.  Operator-tunable via the `[file]`
    /// TOML block alongside the free-RAM and disk-pressure
    /// prechecks.
    pub max_upload_bytes: u64,
    /// Maximum number of [`WorkspaceMgr::upload`] calls
    /// allowed in flight at once across the whole
    /// [`WorkspaceMgr`].  Default 2.  The cap is global
    /// (not per-workspace): concurrent uploads to ANY
    /// workspace count against the same semaphore so a
    /// single hostile workspace can't drain the daemon's
    /// file-handle budget by saturating the cap with itself.
    pub max_concurrent_uploads: u32,
}

impl Default for AdmissionCfg {
    fn default() -> Self {
        // Defaults match the workspace-redesign §9 storage table
        // (`max_concurrent_uploads = 4`, raised from 2 for better
        // bulk-load throughput on bounded SBC deployments).
        Self {
            max_upload_bytes: 256 * 1024 * 1024,
            max_concurrent_uploads: 4,
        }
    }
}

/// Shared admission state held inside [`WorkspaceMgr`].
/// `pub(crate)` so the parent module can store an
/// `Option<Arc<AdmissionState>>` field without re-exporting
/// the type.  The semaphore is shared via [`Arc`] so all
/// clones of [`WorkspaceMgr`] count against the same global
/// cap.
#[derive(Debug)]
pub(crate) struct AdmissionState {
    pub(crate) cfg: AdmissionCfg,
    /// [`tokio::sync::Semaphore`] is async-aware
    /// (`acquire_owned` returns a future); we use
    /// `try_acquire_owned()` to fail fast.  An upload that
    /// has to wait for a slot doesn't fit the
    /// operator-facing contract (the request would block
    /// until some other client finished, with no
    /// client-visible signal).  `try_acquire_owned()`
    /// returns `Err(TryAcquireError)` when no permits are
    /// available; we surface that as
    /// [`FileError::TooManyConcurrentUploads`].
    pub(crate) semaphore: Arc<tokio::sync::Semaphore>,
}

impl AdmissionState {
    pub(crate) fn new(cfg: AdmissionCfg) -> Self {
        Self {
            cfg,
            semaphore: Arc::new(tokio::sync::Semaphore::new(
                cfg.max_concurrent_uploads as usize,
            )),
        }
    }
}

impl WorkspaceMgr {
    /// Upload a single asset by streaming bytes from a
    /// reader.  Writes to a tempfile, sha256s as it goes,
    /// then atomically renames into place and updates
    /// `metadata.json`.  Returns the [`AssetReceipt`].
    ///
    /// Validates name + extension.  Rejects relative path
    /// components (`..`, `/`) to prevent escape.
    pub async fn upload<R>(
        &self,
        id: &WorkspaceId,
        kind: AssetKind,
        name: &str,
        mut body: R,
    ) -> Result<AssetReceipt, FileError>
    where
        R: tokio::io::AsyncRead + Unpin,
    {
        // Admission gate.  `try_acquire_owned()` is
        // fail-fast (no waiting): if the semaphore is
        // exhausted, reject with `TooManyConcurrentUploads`
        // so the operator gets an immediate signal rather
        // than a request that blocks indefinitely.  The
        // owned permit is held for the lifetime of this
        // function and dropped on return (success or
        // failure); Drop releases the slot so future uploads
        // observe an updated count.  `_admission_permit`
        // binds the permit to a name so it isn't dropped
        // immediately.
        let _admission_permit = if let Some(state) = &self.admission {
            match state.semaphore.clone().try_acquire_owned() {
                Ok(p) => Some(p),
                Err(_) => {
                    // `available_permits` is the live count;
                    // subtract from the cap to get "active"
                    // without an extra counter.
                    let available = state.semaphore.available_permits() as u32;
                    let active = state.cfg.max_concurrent_uploads.saturating_sub(available);
                    return Err(FileError::TooManyConcurrentUploads {
                        active,
                        max: state.cfg.max_concurrent_uploads,
                    });
                }
            }
        } else {
            None
        };

        validate_asset_name(name)?;
        validate_extension(name, kind.allowed_ext())?;

        // Every blocking syscall in the upload prelude
        // (`PathBuf::exists` -> `stat(2)`, two
        // `create_dir_all` calls,
        // `tempfile::NamedTempFile::new_in` -> `mkstemp(2)`,
        // `tmp.reopen` -> `open(2)`, the metadata read for
        // the preflight collision check) runs inside
        // `spawn_blocking` so a disk-pressure stall on the
        // dataset eMMC can't park a tokio worker thread for
        // hundreds of ms.
        //
        // The closure also performs a best-effort preflight
        // collision check: if the metadata already records
        // an asset whose name collides case-insensitively
        // with `name`, reject BEFORE consuming the upload
        // body.  The authoritative check still runs under
        // the per-workspace lock at commit time below; the
        // preflight just saves up to `max_upload_bytes` of
        // wasted disk write under contention.  Best-effort
        // because no lock is held -- a concurrent upload
        // can land between this check and our commit; the
        // commit-time check catches that race.
        //
        // Returns the `NamedTempFile` (owns the eventual
        // rename target) plus a sibling `std::fs::File` for
        // the body-streaming writer.
        let workspace_dir = self.workspace_dir(id);
        let final_path = self.asset_path(id, kind, name);
        let id_for_err = *id;
        let name_for_check = name.to_string();
        let tmp_dir_for_closure = workspace_dir.join(".tmp");
        let final_parent_for_closure = final_path.parent().map(std::path::Path::to_path_buf);
        let metadata_path_for_closure = workspace_dir.join("metadata.json");
        let (tmp, sync_fd) = tokio::task::spawn_blocking(
            move || -> Result<(tempfile::NamedTempFile, std::fs::File), FileError> {
                if !workspace_dir.exists() {
                    return Err(FileError::NotFound(id_for_err.to_string()));
                }
                // Preflight collision check (see comment
                // block above).  Read+parse the metadata
                // file directly without routing through
                // [`WorkspaceMgr::read_metadata`] so the
                // closure stays Send-clean (`&self` can't
                // cross the spawn_blocking boundary).
                //
                // Schema-version gate runs here too: on a
                // too-new metadata.json, the body would
                // otherwise stream all the way to disk
                // before commit-time rejected with
                // `SchemaTooNew`, voiding the preflight's
                // whole purpose (avoiding wasted disk write
                // under operator-supplied error).  One u32
                // comparison is cheap; gate here too with
                // the same error variants the &self method
                // produces.
                let meta_bytes = std::fs::read(&metadata_path_for_closure)
                    .map_err(|e| io_err(metadata_path_for_closure.display(), e))?;
                let meta_preflight: crate::file_mgr::WorkspaceMetadata =
                    serde_json::from_slice(&meta_bytes)
                        .map_err(|e| metadata_parse_err(metadata_path_for_closure.display(), e))?;
                if meta_preflight.schema_version > crate::file_mgr::WorkspaceMetadata::CURRENT {
                    return Err(FileError::SchemaTooNew {
                        path: metadata_path_for_closure.display().to_string(),
                        found: meta_preflight.schema_version,
                        max: crate::file_mgr::WorkspaceMetadata::CURRENT,
                    });
                }
                if meta_preflight.schema_version
                    < crate::file_mgr::WorkspaceMetadata::MIN_COMPATIBLE
                {
                    return Err(FileError::SchemaTooOld {
                        path: metadata_path_for_closure.display().to_string(),
                        found: meta_preflight.schema_version,
                        min: crate::file_mgr::WorkspaceMetadata::MIN_COMPATIBLE,
                    });
                }
                if let Some(existing) = meta_preflight.find_case_insensitive(kind, &name_for_check)
                    && existing.name != name_for_check.as_str()
                {
                    return Err(FileError::NameConflict(format!(
                        "{kind:?} asset {name_for_check:?} collides case-insensitively with \
                         existing {:?}; case-sensitive identifiers + case-sensitive disk \
                         paths is the workspace policy",
                        existing.name
                    )));
                }
                std::fs::create_dir_all(&tmp_dir_for_closure)
                    .map_err(|e| io_err(tmp_dir_for_closure.display(), e))?;
                if let Some(parent) = &final_parent_for_closure {
                    std::fs::create_dir_all(parent).map_err(|e| io_err(parent.display(), e))?;
                }
                let tmp = tempfile::NamedTempFile::new_in(&tmp_dir_for_closure)
                    .map_err(|e| io_err(tmp_dir_for_closure.display(), e))?;
                // `reopen` returns a sibling fd backed by
                // the SAME tempfile inode; needed because
                // we hand the writer off to `tokio::fs`
                // while `NamedTempFile` keeps its own fd
                // alive for the eventual `persist` rename.
                let sync_fd = tmp.reopen().map_err(|e| io_err(tmp.path().display(), e))?;
                Ok((tmp, sync_fd))
            },
        )
        .await
        // `JoinError` surfaces only on task panic / cancellation;
        // wrap as `io::Error::other` so the [`FileError::Io`]
        // shape is preserved without growing a new error variant
        // for a case that should never reach production.
        .map_err(|je| io_err("<upload-prelude-spawn-blocking>", std::io::Error::other(je)))??;
        let tmp_path = tmp.path().to_path_buf();
        // Re-attach the sibling fd as a `tokio::fs::File`
        // for async streaming.  The original
        // `NamedTempFile` fd stays alive on `tmp` for
        // `persist` later.
        let mut writer = tokio::fs::File::from_std(sync_fd);

        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 64 * 1024];
        let mut total: u64 = 0;
        // Per-request size cap.  Read from `admission.cfg`;
        // when admission isn't configured, `max` stays at
        // [`u64::MAX`] (no cap).  The check fires after
        // `total` is updated, so we reject the chunk that
        // crosses the threshold; the tempfile drops on Err
        // return (`NamedTempFile::Drop` unlinks), so no
        // partial commit reaches the asset path.
        let max_upload_bytes = self.max_upload_bytes();
        loop {
            let n = body
                .read(&mut buf)
                .await
                .map_err(|e| io_err("<upload-stream>", e))?;
            if n == 0 {
                break;
            }
            total = total.saturating_add(n as u64);
            if total > max_upload_bytes {
                return Err(FileError::PayloadTooLarge {
                    observed: total,
                    max: max_upload_bytes,
                });
            }
            hasher.update(&buf[..n]);
            writer
                .write_all(&buf[..n])
                .await
                .map_err(|e| io_err(tmp_path.display(), e))?;
        }
        writer
            .flush()
            .await
            .map_err(|e| io_err(tmp_path.display(), e))?;
        writer
            .sync_all()
            .await
            .map_err(|e| io_err(tmp_path.display(), e))?;
        drop(writer);

        let digest = hex_lowercase(&hasher.finalize());

        // The rename + collision-check + metadata commit
        // run under a single per-workspace lock so
        // concurrent uploads of `Foo.mpk` and `foo.mpk`
        // can't both rename before either checks: holding
        // the lock across `tmp.persist` serializes them,
        // the second one observes the first's metadata
        // record, and rejects with `NameConflict`.  On
        // case-sensitive filesystems the collision check
        // still matters: it makes the casing policy
        // uniform across host platforms (an asset
        // uploaded as `Foo.mpk` on Linux can't be
        // re-uploaded as `foo.mpk` later, sidestepping a
        // diagnose-once-per-host bug).  The lock is
        // `parking_lot` sync; we never await while holding
        // it (`tmp.persist` and `fsync_dir` are sync I/O,
        // and the metadata read/write is sync), so async
        // cancellation can't strand the lock.
        let lock = self.metadata_lock(id);
        let _guard = lock.lock();
        let mut meta = self.read_metadata(id)?;
        if let Some(existing) = meta.find_case_insensitive(kind, name)
            && existing.name != name
        {
            return Err(FileError::NameConflict(format!(
                "{kind:?} asset {name:?} collides case-insensitively with existing {:?}; \
                 case-sensitive identifiers + case-sensitive disk paths is the workspace policy",
                existing.name
            )));
        }
        // `tempfile::NamedTempFile::persist` requires sync I/O.
        tmp.persist(&final_path)?;
        // fsync the asset's parent directory so the rename
        // itself is durable.  `writer.sync_all()` above
        // made the content durable, but the
        // directory-entry update from rename can still be
        // lost on a power-cycle.  One sync syscall on a
        // directory inode -- microseconds.
        if let Some(parent) = final_path.parent() {
            fsync_dir(parent).map_err(|e| io_err(parent.display(), e))?;
        }

        // `validate_asset_name` already routed this name
        // through [`AssetId::parse`]; re-parse here is the
        // single point that upgrades the validated `&str`
        // into the typed id stored in the metadata record.
        // Error mapping via `?` covers the (unreachable)
        // case of validator drift without leaving an
        // `expect` panic in the upload commit path.
        let asset_id = AssetId::parse(name)?;
        let record = AssetRecord {
            kind,
            name: asset_id,
            sha256: digest.clone(),
            size_bytes: total,
        };
        let upsert_record = record.clone();
        if let Some(idx) = meta.find_index(kind, name) {
            meta.assets[idx] = upsert_record;
        } else {
            meta.assets.push(upsert_record);
        }
        self.write_metadata(id, &meta)?;
        drop(_guard);

        Ok(AssetReceipt {
            kind: record.kind,
            name: record.name.into(),
            sha256: digest,
            size_bytes: total,
            path: final_path,
        })
    }

    /// Sync version of [`Self::upload`]'s rename + commit
    /// for callers that have already staged the bytes to a
    /// tempfile on the workspace's filesystem.  Used by
    /// [`crate::file_mgr::FsService::install_from_path`]
    /// and indirectly by
    /// [`crate::file_mgr::FsService::install_bytes`].
    ///
    /// `src` MUST be on the same filesystem as the
    /// workspace root so the rename is atomic.  Callers
    /// guarantee this by staging via
    /// `tempfile::NamedTempFile::new_in(fs.workspace_tmpdir(ws))`.
    /// On `EXDEV` (cross-device link) the rename fails
    /// loudly with [`FileError::Persist`], surfacing as a
    /// Conflict to the operator.
    ///
    /// Mirrors the streaming [`Self::upload`]:
    ///
    /// - validates name + extension
    /// - checks the workspace exists
    /// - holds the per-workspace metadata lock across the
    ///   case-insensitive collision check + the rename +
    ///   the metadata commit
    /// - computes sha256 by streaming `src` (no in-memory
    ///   copy)
    /// - fsyncs the asset's parent dir for rename
    ///   durability
    pub fn install_from_path(
        &self,
        id: &WorkspaceId,
        kind: AssetKind,
        name: &str,
        src: &Path,
    ) -> Result<AssetReceipt, FileError> {
        validate_asset_name(name)?;
        let ws = self.workspace_dir(id);
        if !ws.exists() {
            return Err(FileError::NotFound(id.to_string()));
        }
        validate_extension(name, kind.allowed_ext())?;

        let final_path = self.asset_path(id, kind, name);
        if let Some(parent) = final_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| io_err(parent.display(), e))?;
        }

        // Stream-hash + size from the staged tempfile.
        // Same 64 KiB chunk size as [`Self::upload`] so
        // multi-GB asset installs don't pull the whole file
        // into RAM.
        let metadata = std::fs::metadata(src).map_err(|e| io_err(src.display(), e))?;
        let total = metadata.len();
        let digest = sha256_file_streaming(src)?;

        // Per-workspace lock spans the collision check + the
        // rename + the metadata commit, identical to
        // [`Self::upload`]'s critical section.
        let lock = self.metadata_lock(id);
        let _guard = lock.lock();
        let mut meta = self.read_metadata(id)?;
        if let Some(existing) = meta.find_case_insensitive(kind, name)
            && existing.name != name
        {
            return Err(FileError::NameConflict(format!(
                "{kind:?} asset {name:?} collides case-insensitively with existing {:?}; \
                 case-sensitive identifiers + case-sensitive disk paths is the workspace policy",
                existing.name
            )));
        }
        std::fs::rename(src, &final_path).map_err(|e| io_err(final_path.display(), e))?;
        if let Some(parent) = final_path.parent() {
            fsync_dir(parent).map_err(|e| io_err(parent.display(), e))?;
        }

        // Same invariant as [`Self::upload`]: validator
        // already enforced [`AssetId::parse`]'s allowlist;
        // `?` propagates without panic.
        let asset_id = AssetId::parse(name)?;
        let record = AssetRecord {
            kind,
            name: asset_id,
            sha256: digest.clone(),
            size_bytes: total,
        };
        let upsert_record = record.clone();
        if let Some(idx) = meta.find_index(kind, name) {
            meta.assets[idx] = upsert_record;
        } else {
            meta.assets.push(upsert_record);
        }
        self.write_metadata(id, &meta)?;
        drop(_guard);

        Ok(AssetReceipt {
            kind: record.kind,
            name: record.name.into(),
            sha256: digest,
            size_bytes: total,
            path: final_path,
        })
    }

    /// Fail-fast acquisition of the global concurrent-upload
    /// permit.  Returns:
    ///
    /// - `Ok(Some(permit))` when admission is engaged and a
    ///   slot was free,
    /// - `Ok(None)` when admission is not engaged (the
    ///   legacy [`WorkspaceMgr::new`] path; uncapped),
    /// - `Err(TooManyConcurrentUploads)` when the cap is
    ///   full.
    ///
    /// The permit is owned (drop releases the slot) and is
    /// stashed inside
    /// [`crate::file_mgr::fs_service::UploadPermit`] for
    /// the api's staging-then-install dance.
    pub fn try_acquire_upload_permit(
        &self,
    ) -> Result<Option<tokio::sync::OwnedSemaphorePermit>, FileError> {
        let Some(state) = &self.admission else {
            return Ok(None);
        };
        match state.semaphore.clone().try_acquire_owned() {
            Ok(p) => Ok(Some(p)),
            Err(_) => {
                let available = state.semaphore.available_permits() as u32;
                let active = state.cfg.max_concurrent_uploads.saturating_sub(available);
                Err(FileError::TooManyConcurrentUploads {
                    active,
                    max: state.cfg.max_concurrent_uploads,
                })
            }
        }
    }

    /// Per-request upload byte cap.  [`u64::MAX`] if
    /// admission is not engaged.
    pub fn max_upload_bytes(&self) -> u64 {
        self.admission
            .as_ref()
            .map(|s| s.cfg.max_upload_bytes)
            .unwrap_or(u64::MAX)
    }
}
