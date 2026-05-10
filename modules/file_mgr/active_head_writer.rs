//! Active-head activation pipeline shared between daemon boot and
//! the `POST /active` handler.  Three sync helpers compose:
//!
//! 1. [`stage_and_validate_activation`] copies source bytes into
//!    `<root>/active/.tmp/<activation_id>/`, computes
//!    `sha256` / `labels_sha256`, builds + validates the
//!    [`ActiveHeadManifest`], and pre-loads a `HeadInner`
//!    candidate via the caller-supplied loader (kept out of
//!    `file_mgr` so the primitive does not depend on
//!    `inference`).
//! 2. [`publish_active_generation`] atomically renames the staged
//!    dir into `<root>/active/generations/<activation_id>/` and
//!    rewrites `current.json`.
//! 3. [`prune_old_generations`] retains only the current +
//!    previous generation directories.
//!
//! Lock order: the caller takes the global `active/` mutex first
//! (then, for Head origin, the per-workspace mutation mutex)
//! and runs all three helpers on the blocking pool.  The
//! prevalidated runtime candidate installs only after
//! `publish_active_generation` succeeds so the on-disk state is
//! durable before `HotHead` rotates.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::common::ids::{HeadId, WorkspaceId, default_runtime_head_id};
use crate::common::workspace::{
    ActiveHeadManifest, ActiveHeadValidationError, ActiveOrigin, HeadValidationError,
    WorkspaceRevision,
};
use crate::file_mgr::error::{FileError, io_err, metadata_parse_err};
use crate::file_mgr::fs_atomic::put_atomic;
use crate::file_mgr::schema::{
    ACTIVE_HEAD_FILENAME, ACTIVE_LABELS_FILENAME, ActiveCurrentPointer, active_dir,
    active_generation_dir, active_generations_dir, active_staging_dir, head_artifact_path,
    read_head_manifest, write_active_current,
};
use crate::file_mgr::validate::{fsync_dir, hex_lowercase};

/// Origin descriptor for a pending activation; carries the
/// sources the staging step needs to copy.  Mirrors the request
/// surface (`{workspace_id, head_id}` or `{default: true}`).
#[derive(Clone, Debug)]
pub enum ActivationOriginInput<'a> {
    /// Activate from a workspace's trained head.  The mpk +
    /// labels are sourced from `<workspace_dir>/heads/<head_id>.{mpk,json}`.
    Head {
        /// Per-workspace dir resolved by the caller via
        /// `WorkspaceMgr::workspace_dir(id)`.
        workspace_dir: &'a Path,
        /// Source workspace id stamped on the active manifest.
        workspace_id: WorkspaceId,
        /// Source head id; the activation's `runtime_head_id`
        /// equals this for `Head`-origin manifests.
        head_id: HeadId,
    },
    /// Activate the bundled default.  Sources from
    /// `<bundled_default_dir>/{head.mpk,labels.txt}`.
    Default,
}

/// Inputs for [`stage_and_validate_activation`].
///
/// Owned with explicit lifetimes so the caller can hand in
/// per-request paths without `'static` constraints.
#[derive(Debug)]
pub struct PendingActivation<'a> {
    /// Daemon `WORKSPACE_ROOT`; activation paths derived via
    /// the `file_mgr::schema::active_*` helpers.
    pub root: &'a Path,
    /// Whether to source from a workspace head or the bundled
    /// default fixture.
    pub origin_input: ActivationOriginInput<'a>,
    /// Path to the deployment-bundled default head dir
    /// (`misc/heads/00000000-default/`).  Read only when
    /// `origin_input` is [`ActivationOriginInput::Default`];
    /// supplied unconditionally so the call sites stay uniform.
    pub bundled_default_dir: &'a Path,
    /// RFC3339 wall-clock used for `manifest.activated_at`.
    /// Caller-supplied so tests can pin a deterministic value.
    pub now_rfc3339: String,
}

/// Successful result of [`stage_and_validate_activation`].
///
/// `candidate` is opaque to this crate (the primitive must not
/// depend on `inference`); the caller downcasts via
/// [`crate::common::traits::head_store::HeadStore::install_prevalidated`]
/// AFTER [`publish_active_generation`] has made `current.json`
/// durable.
pub struct ActivationResult {
    /// Validated manifest body; `manifest.json` was written to
    /// the staging directory before this returns.
    pub manifest: ActiveHeadManifest,
    /// Generation directory name (UUID-v4 string).  The staging
    /// directory lives at `active/.tmp/<activation_id>/`; the
    /// publish step renames it to
    /// `active/generations/<activation_id>/`.
    pub activation_id: String,
    /// Boxed prevalidated runtime candidate; the production
    /// loader returns a `Box<inference::HeadInner>` which
    /// `HotHead::install_prevalidated` downcasts.  Boxing keeps
    /// the primitive in the `file_mgr` layer without inverting
    /// the dep graph.
    pub candidate: Box<dyn std::any::Any + Send>,
}

impl std::fmt::Debug for ActivationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ActivationResult")
            .field("manifest", &self.manifest)
            .field("activation_id", &self.activation_id)
            .field("candidate", &"Box<dyn Any + Send>")
            .finish()
    }
}

/// Failure shapes from the activation primitive.  Mapped to
/// HTTP statuses via [`crate::common::error::Categorized`].
#[derive(Debug, thiserror::Error)]
pub enum ActivationError {
    /// Source artifact (workspace head, labels file, bundled
    /// default fixture, head manifest) is missing on disk.
    #[error("activation source not found: {what}")]
    NotFound {
        /// Operator-readable description of the missing
        /// resource (path or descriptor).
        what: String,
    },
    /// Streaming-hash of a staged source did not match the
    /// recorded `sha256` from the per-head manifest.  Always a
    /// content-validation failure (operator-visible 400) so the
    /// daemon refuses to publish a generation that disagrees
    /// with the source manifest.
    #[error("hash mismatch for {what}: expected {expected}, observed {observed}")]
    HashMismatch {
        /// Which artifact was hashed.
        what: String,
        /// Recorded sha256 from the source manifest.
        expected: String,
        /// Computed sha256 of the staged bytes.
        observed: String,
    },
    /// [`ActiveHeadManifest::validate`] rejected the constructed
    /// manifest.  Daemon-internal because the writer
    /// constructed the manifest itself; surfaces if a future
    /// revision drifts the discriminator without updating
    /// `runtime_head_id`.
    #[error("active head manifest validation: {0}")]
    Validation(#[from] ActiveHeadValidationError),
    /// Source per-head manifest failed
    /// [`crate::common::workspace::HeadManifest::validate`] --
    /// refuse to publish a generation from corrupt metadata.
    #[error("source head manifest validation: {0}")]
    SourceManifestInvalid(#[from] HeadValidationError),
    /// Per-head manifest read / parse failure during
    /// `Head`-origin staging (head id not in workspace, manifest
    /// JSON corrupt, etc.).  Wraps the file-mgr error so the
    /// HTTP classification stays uniform with the rest of the
    /// daemon's filesystem path.
    #[error("file: {0}")]
    File(#[from] FileError),
    /// Pre-load of the staged head into the runtime-shape
    /// candidate failed.  The candidate factory is opaque to
    /// `file_mgr` (it lives in `inference`); the caller
    /// classifies the inner error and surfaces the
    /// operator-visible message here.
    #[error("preload candidate: {message}")]
    Preload {
        /// Operator-readable diagnostic.
        message: String,
    },
}

impl crate::common::error::Categorized for ActivationError {
    fn kind(&self) -> crate::common::error::ErrorKind {
        use crate::common::error::ErrorKind::*;
        match self {
            ActivationError::NotFound { .. } => NotFound,
            // Bytes failed validation; operator-visible 400.
            ActivationError::HashMismatch { .. } | ActivationError::Preload { .. } => UserInput,
            // The writer built the manifest itself; a validation
            // failure here is daemon-internal.
            ActivationError::Validation(_) => Internal,
            ActivationError::SourceManifestInvalid(e) => e.kind(),
            // Delegate; preserves NotFound / UserInput / Internal
            // routing for IO + parse subvariants.
            ActivationError::File(e) => e.kind(),
        }
    }
}

/// Closure type for the runtime-shape pre-load.  The closure
/// receives the staged `head.mpk` + `labels.txt` paths plus the
/// `runtime_head_id`; on success it returns a `Box<dyn Any +
/// Send>` carrying the impl-specific candidate
/// (production: `inference::HeadInner`).  `String` failure
/// messages keep the layering clean -- the closure converts
/// `inference::HeadError` to a string at the boundary.
pub type HeadInnerLoader =
    dyn Fn(&Path, &Path, HeadId) -> Result<Box<dyn std::any::Any + Send>, String> + Send + Sync;

/// Stage the activation source into `<root>/active/.tmp/<activation_id>/`,
/// compute hashes, build + validate the manifest, and pre-load
/// the runtime candidate.
///
/// `head_inner_loader` is the candidate factory; production
/// wires `inference::HotHead::load` (sync; called from within
/// `spawn_blocking`).  Tests can substitute a synthetic
/// candidate to exercise the activation pipeline without a
/// real `.mpk`.
///
/// Sequence:
///
/// 1. Allocate `activation_id` (UUID-v4 string).
/// 2. Build `<root>/active/.tmp/<activation_id>/` (idempotent).
/// 3. For `Head` origin: read the per-head manifest + `.mpk`,
///    hash both, compare against the manifest's `sha256`.
/// 4. For `Default` origin: read the bundled fixture files.
/// 5. Stage bytes into `<staging>/{head.mpk,labels.txt}`.
/// 6. Materialize `labels.txt` from `manifest.labels[]` (Head
///    origin) or preserve the bundled bytes (Default origin).
/// 7. Compute `labels_sha256`.
/// 8. Build [`ActiveHeadManifest`].
/// 9. Run [`ActiveHeadManifest::validate`].
/// 10. Pre-load via `head_inner_loader`.
/// 11. Write `<staging>/manifest.json` via `put_atomic`.
pub fn stage_and_validate_activation(
    pending: PendingActivation<'_>,
    head_inner_loader: &HeadInnerLoader,
) -> Result<ActivationResult, ActivationError> {
    let activation_id = uuid::Uuid::new_v4().hyphenated().to_string();
    let staging_root = active_staging_dir(pending.root);
    std::fs::create_dir_all(&staging_root)
        .map_err(|e| ActivationError::File(io_err(staging_root.display(), e)))?;
    let staging_dir = staging_root.join(&activation_id);
    if staging_dir.exists() {
        // UUID-v4 reuse on a single-process call is statistically
        // impossible; treat the collision as fatal so the writer
        // does not silently overlay a partial earlier generation.
        return Err(ActivationError::File(io_err(
            staging_dir.display(),
            std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                "activation staging dir already exists",
            ),
        )));
    }
    std::fs::create_dir_all(&staging_dir)
        .map_err(|e| ActivationError::File(io_err(staging_dir.display(), e)))?;

    // Source the bytes + the canonical labels list per origin.
    let staged = match &pending.origin_input {
        ActivationOriginInput::Head {
            workspace_dir,
            workspace_id,
            head_id,
        } => stage_head_origin(workspace_dir, *workspace_id, *head_id, &staging_dir)?,
        ActivationOriginInput::Default => {
            stage_default_origin(pending.bundled_default_dir, &staging_dir)?
        }
    };

    // Build the manifest.  `validate()` fails closed on a
    // discriminator-vs-runtime_head_id drift.
    let runtime_head_id = match &pending.origin_input {
        ActivationOriginInput::Head { head_id, .. } => *head_id,
        ActivationOriginInput::Default => default_runtime_head_id(),
    };
    let manifest = ActiveHeadManifest {
        origin: staged.origin,
        runtime_head_id,
        sha256: staged.head_sha256,
        labels_sha256: staged.labels_sha256,
        n_classes: staged.n_classes,
        labels: staged.labels,
        activated_at: pending.now_rfc3339,
    };
    manifest.validate()?;

    // Pre-load the runtime candidate.  The loader runs sync
    // (caller wraps the whole helper in `spawn_blocking`); a
    // failure here surfaces as `Preload` so the activation is
    // refused before any on-disk publish.
    let head_mpk_staged = staging_dir.join(ACTIVE_HEAD_FILENAME);
    let labels_staged = staging_dir.join(ACTIVE_LABELS_FILENAME);
    let candidate = head_inner_loader(&head_mpk_staged, &labels_staged, runtime_head_id)
        .map_err(|message| ActivationError::Preload { message })?;

    // Persist the manifest into the staging directory.  The
    // schema helper [`write_active_manifest`] targets the
    // already-published generation path; staging is a separate
    // pre-publish location, so we go through `put_atomic`
    // directly with the staging-relative path.  The publish
    // step atomic-renames the staging dir into
    // `generations/<id>/`, taking the manifest with it.
    let manifest_bytes = serde_json::to_vec(&manifest).map_err(FileError::MetadataSerialize)?;
    put_atomic(
        &staging_dir.join(crate::file_mgr::schema::ACTIVE_MANIFEST_FILENAME),
        &manifest_bytes,
    )?;

    Ok(ActivationResult {
        manifest,
        activation_id,
        candidate,
    })
}

/// Internal carrier for the source-specific staging output.
struct StagedSource {
    origin: ActiveOrigin,
    head_sha256: String,
    labels_sha256: String,
    n_classes: u32,
    labels: Vec<String>,
}

/// Stage a `Head`-origin activation: copy the trained head's
/// `.mpk` + materialized `labels.txt` (rendered from the
/// per-head manifest's `labels[]`), verify the head bytes' hash
/// against the per-head manifest's recorded `sha256`, and emit
/// the active-head provenance triple.
fn stage_head_origin(
    workspace_dir: &Path,
    workspace_id: WorkspaceId,
    head_id: HeadId,
    staging_dir: &Path,
) -> Result<StagedSource, ActivationError> {
    let manifest = read_head_manifest(workspace_dir, head_id).map_err(|e| match e {
        FileError::Io { ref source, .. } if source.kind() == std::io::ErrorKind::NotFound => {
            ActivationError::NotFound {
                what: format!("head {head_id} in workspace {workspace_id}"),
            }
        }
        other => ActivationError::File(other),
    })?;
    // Fail closed on a hand-tampered manifest before any heavy IO.
    manifest.validate()?;

    let mpk_path = head_artifact_path(workspace_dir, head_id);
    if !mpk_path.is_file() {
        return Err(ActivationError::NotFound {
            what: format!("head {head_id} mpk in workspace {workspace_id}"),
        });
    }

    // Stream the mpk into staging while hashing in one pass;
    // avoids the ~80 MiB heap a full `fs::read` would need at
    // the convert cap.  Partial bytes inside an unpublished
    // staging dir get swept by boot recovery on next start.
    let staged_mpk = staging_dir.join(ACTIVE_HEAD_FILENAME);
    let head_sha = copy_and_hash(&mpk_path, &staged_mpk).map_err(|e| match e {
        FileError::Io { ref source, .. } if source.kind() == std::io::ErrorKind::NotFound => {
            ActivationError::NotFound {
                what: format!("head {head_id} mpk in workspace {workspace_id}"),
            }
        }
        other => ActivationError::File(other),
    })?;
    if head_sha != manifest.sha256 {
        return Err(ActivationError::HashMismatch {
            what: format!("head {head_id} mpk"),
            expected: manifest.sha256.clone(),
            observed: head_sha,
        });
    }

    // Render `labels.txt` deterministically from the manifest's
    // `labels[]` so the generation's labels file is the single
    // canonical source -- a future operator-side edit of the
    // workspace's manifest cannot drift the active labels off.
    let labels_text = labels_to_text(&manifest.labels);
    let labels_bytes = labels_text.as_bytes().to_vec();
    let labels_sha = sha256_hex_of(&labels_bytes);
    put_atomic(&staging_dir.join(ACTIVE_LABELS_FILENAME), &labels_bytes)?;

    let n_classes = u32::try_from(manifest.labels.len()).map_err(|_| {
        ActivationError::File(metadata_parse_err(
            staging_dir.display(),
            serde_json::Error::io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "labels count {} overflows u32 in workspace {workspace_id} head {head_id}",
                    manifest.labels.len()
                ),
            )),
        ))
    })?;

    Ok(StagedSource {
        origin: ActiveOrigin::Head {
            source_workspace_id: workspace_id,
            source_head_id: head_id,
            workspace_revision: WorkspaceRevision {
                id: manifest.workspace_revision.id,
                at: manifest.workspace_revision.at.clone(),
            },
        },
        head_sha256: head_sha,
        labels_sha256: labels_sha,
        n_classes,
        labels: manifest.labels,
    })
}

/// Stage a `Default`-origin activation: copy the bundled
/// `head.mpk` + `labels.txt`, hash both, and emit the
/// `Default`-origin provenance.  The bundled `labels.txt` is
/// preserved verbatim so a deployment-managed file is not
/// rewritten by the writer.
fn stage_default_origin(
    bundled_dir: &Path,
    staging_dir: &Path,
) -> Result<StagedSource, ActivationError> {
    let src_mpk = bundled_dir.join("head.mpk");
    let src_labels = bundled_dir.join("labels.txt");
    let mpk_bytes = std::fs::read(&src_mpk).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            ActivationError::NotFound {
                what: format!("bundled default head.mpk at {}", src_mpk.display()),
            }
        } else {
            ActivationError::File(io_err(src_mpk.display(), e))
        }
    })?;
    let labels_bytes = std::fs::read(&src_labels).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            ActivationError::NotFound {
                what: format!("bundled default labels.txt at {}", src_labels.display()),
            }
        } else {
            ActivationError::File(io_err(src_labels.display(), e))
        }
    })?;

    let labels_text = std::str::from_utf8(&labels_bytes).map_err(|e| {
        ActivationError::File(io_err(
            src_labels.display(),
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("labels utf-8: {e}"),
            ),
        ))
    })?;
    let labels: Vec<String> = labels_text
        .lines()
        .map(|s| s.trim_end_matches('\r'))
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect();
    if labels.is_empty() {
        return Err(ActivationError::File(io_err(
            src_labels.display(),
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "bundled labels.txt is empty",
            ),
        )));
    }
    let n_classes = u32::try_from(labels.len()).map_err(|_| {
        ActivationError::File(io_err(
            src_labels.display(),
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("labels count {} overflows u32", labels.len()),
            ),
        ))
    })?;

    let head_sha = sha256_hex_of(&mpk_bytes);
    let labels_sha = sha256_hex_of(&labels_bytes);

    put_atomic(&staging_dir.join(ACTIVE_HEAD_FILENAME), &mpk_bytes)?;
    put_atomic(&staging_dir.join(ACTIVE_LABELS_FILENAME), &labels_bytes)?;

    Ok(StagedSource {
        origin: ActiveOrigin::Default,
        head_sha256: head_sha,
        labels_sha256: labels_sha,
        n_classes,
        labels,
    })
}

/// Atomic-rename the staged generation into
/// `<root>/active/generations/<activation_id>/`, fsync the
/// generations parent, atomic-rewrite `<root>/active/current.json`,
/// fsync the active dir.  Caller MUST call this AFTER
/// [`stage_and_validate_activation`] returns Ok and BEFORE
/// installing the runtime candidate.
pub fn publish_active_generation(
    root: &Path,
    staging: &Path,
    _manifest: &ActiveHeadManifest,
    activation_id: &str,
) -> Result<(), ActivationError> {
    let generations_root = active_generations_dir(root);
    std::fs::create_dir_all(&generations_root)
        .map_err(|e| ActivationError::File(io_err(generations_root.display(), e)))?;
    let final_dir = active_generation_dir(root, activation_id);
    std::fs::rename(staging, &final_dir)
        .map_err(|e| ActivationError::File(io_err(final_dir.display(), e)))?;
    // fsync the generations parent so the new directory entry
    // reaches stable storage before the pointer flip.  Best-
    // effort on platforms that don't support directory fsync.
    fsync_dir(&generations_root)
        .map_err(|e| ActivationError::File(io_err(generations_root.display(), e)))?;
    write_active_current(
        root,
        &ActiveCurrentPointer {
            activation_id: activation_id.to_string(),
        },
    )?;
    // fsync the active dir so the `current.json` rename's
    // directory-entry update is durable.
    fsync_dir(&active_dir(root))
        .map_err(|e| ActivationError::File(io_err(active_dir(root).display(), e)))?;
    Ok(())
}

/// Retain only the generations whose directory name appears in
/// `keep`; remove every other entry under
/// `<root>/active/generations/`.  Returns the count of removed
/// directories so callers (status counters, tests) can verify
/// the expected pruning happened.  Best-effort fsync of the
/// generations parent so the unlinks reach stable storage.
///
/// **Caller MUST hold the activation serialization lock** (today:
/// `api::AppState::active_mutex`) for the whole `read_dir` →
/// `remove_dir_all` window.  Otherwise a concurrent publish can
/// land a generation outside this caller's `keep` list and have
/// its directory deleted, leaving `current.json` pointing at a
/// missing dir.
pub fn prune_old_generations<S: AsRef<str>>(
    root: &Path,
    keep: &[S],
) -> Result<usize, ActivationError> {
    let generations_root = active_generations_dir(root);
    if !generations_root.exists() {
        return Ok(0);
    }
    let entries = std::fs::read_dir(&generations_root)
        .map_err(|e| ActivationError::File(io_err(generations_root.display(), e)))?;
    let mut removed = 0usize;
    for entry in entries {
        let entry =
            entry.map_err(|e| ActivationError::File(io_err(generations_root.display(), e)))?;
        let name = match entry.file_name().into_string() {
            Ok(s) => s,
            // Non-UTF-8 entry names cannot match a UUID-v4 keep
            // list; sweep them as orphan residue.
            Err(_) => {
                let path = entry.path();
                if entry.path().is_dir() {
                    if let Err(e) = std::fs::remove_dir_all(&path) {
                        return Err(ActivationError::File(io_err(path.display(), e)));
                    }
                    removed += 1;
                }
                continue;
            }
        };
        if keep.iter().any(|k| k.as_ref() == name) {
            continue;
        }
        let path = entry.path();
        let metadata = entry
            .metadata()
            .map_err(|e| ActivationError::File(io_err(path.display(), e)))?;
        if metadata.is_dir() {
            std::fs::remove_dir_all(&path)
                .map_err(|e| ActivationError::File(io_err(path.display(), e)))?;
            removed += 1;
        }
    }
    if removed > 0 {
        let _ = fsync_dir(&generations_root);
    }
    Ok(removed)
}

/// Return the staging directory path for an in-flight
/// activation.  Pure path-helper; the directory itself is
/// created lazily by [`stage_and_validate_activation`].
pub fn staging_path_for(root: &Path, activation_id: &str) -> PathBuf {
    active_staging_dir(root).join(activation_id)
}

/// Render an active-head `labels.txt` body from the manifest's
/// `labels[]` list.  Newline-separated UTF-8; empty labels are
/// preserved (the per-head schema validates non-empty strings
/// upstream).
fn labels_to_text(labels: &[String]) -> String {
    labels.join("\n")
}

/// Lowercase-hex SHA-256 of a byte slice.  Centralized so the
/// active-head writer and per-head manifest writer produce the
/// same hex shape.
fn sha256_hex_of(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex_lowercase(&hasher.finalize())
}

/// Stream-copy `src` to `dst` and SHA-256 the bytes in one
/// pass; returns lowercase-hex digest.  64 KiB buffer; constant
/// memory.  `dst.parent()` must exist; the file is `sync_all`'d
/// before return but the parent dir is the caller's
/// responsibility (typically via the staging-dir rename).
fn copy_and_hash(src: &Path, dst: &Path) -> Result<String, FileError> {
    use std::io::{Read, Write};
    let mut reader = std::fs::File::open(src).map_err(|source| io_err(src.display(), source))?;
    let mut writer = std::fs::File::create(dst).map_err(|source| io_err(dst.display(), source))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let n = reader
            .read(&mut buf)
            .map_err(|source| io_err(src.display(), source))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        writer
            .write_all(&buf[..n])
            .map_err(|source| io_err(dst.display(), source))?;
    }
    writer
        .sync_all()
        .map_err(|source| io_err(dst.display(), source))?;
    Ok(hex_lowercase(&hasher.finalize()))
}

// MARK: Tests

#[cfg(test)]
mod tests {
    #![allow(clippy::disallowed_methods)]
    // Active-generation fixtures intentionally materialize files directly.

    use super::*;
    use crate::common::ids::{HeadId, WorkspaceId};
    use crate::common::workspace::{HeadIndex, HeadManifest, HeadRecord, WorkspaceCore};
    use crate::file_mgr::schema::{write_head_index, write_head_manifest, write_workspace_core};

    fn ws_id() -> WorkspaceId {
        WorkspaceId::parse("11111111-2222-4333-8444-555555555555").unwrap()
    }

    fn rev(id: u64) -> WorkspaceRevision {
        WorkspaceRevision {
            id,
            at: "2026-05-07T12:00:00Z".to_string(),
        }
    }

    fn synth_head_manifest(head_id: HeadId, mpk_bytes: &[u8]) -> HeadManifest {
        HeadManifest {
            head_id,
            workspace_id: ws_id(),
            workspace_revision: rev(5),
            sha256: sha256_hex_of(mpk_bytes),
            n_classes: 2,
            size_bytes: mpk_bytes.len() as u64,
            created_at: "2026-05-07T12:34:56Z".to_string(),
            labels: vec!["alpha".to_string(), "beta".to_string()],
        }
    }

    fn synth_workspace_core() -> WorkspaceCore {
        WorkspaceCore {
            id: ws_id(),
            name: "main".to_string(),
            tags: Vec::new(),
            created_at: "2026-05-07T12:34:56Z".to_string(),
            workspace_revision: rev(5),
            head_count: 1,
        }
    }

    /// Stages a workspace dir + a single trained head's bytes
    /// so the head-origin activation can find them.  Returns
    /// the workspace dir + the head id.
    fn fresh_workspace_with_head(root: &Path, mpk_bytes: &[u8]) -> (PathBuf, HeadId) {
        let ws_dir = crate::file_mgr::schema::workspace_dir_for(root, &ws_id());
        std::fs::create_dir_all(&ws_dir).unwrap();
        std::fs::create_dir_all(crate::file_mgr::schema::heads_dir(&ws_dir)).unwrap();
        write_workspace_core(&ws_dir, &synth_workspace_core()).unwrap();
        let mut idx = HeadIndex::default();
        let head_id = HeadId::parse("11111111-2222-4333-8444-555555555556").unwrap();
        let manifest = synth_head_manifest(head_id, mpk_bytes);
        idx.heads.push(HeadRecord {
            head_id,
            workspace_revision: manifest.workspace_revision.clone(),
            sha256: manifest.sha256.clone(),
            n_classes: manifest.n_classes,
            size_bytes: manifest.size_bytes,
            created_at: manifest.created_at.clone(),
        });
        write_head_index(&ws_dir, &idx).unwrap();
        write_head_manifest(&ws_dir, &manifest).unwrap();
        // Write the .mpk bytes opaquely; the activation
        // primitive does not parse them, the loader closure
        // (mocked here) does.
        std::fs::write(
            crate::file_mgr::schema::head_artifact_path(&ws_dir, head_id),
            mpk_bytes,
        )
        .unwrap();
        (ws_dir, head_id)
    }

    /// Synthetic loader for tests.  Returns a `()` candidate so
    /// the activation pipeline runs without depending on
    /// `inference::HeadInner`.
    fn synth_loader_ok() -> Box<HeadInnerLoader> {
        Box::new(|_mpk: &Path, _labels: &Path, _id: HeadId| {
            Ok(Box::new(()) as Box<dyn std::any::Any + Send>)
        })
    }

    /// Synthetic loader that always fails -- exercises the
    /// `Preload` failure path.
    fn synth_loader_fail() -> Box<HeadInnerLoader> {
        Box::new(|_mpk: &Path, _labels: &Path, _id: HeadId| {
            Err("synthetic preload failure".to_string())
        })
    }

    /// Bundled-default fixture builder for tests.  Writes a
    /// `head.mpk` + `labels.txt` under a tempdir-rooted
    /// `bundled_default_dir/`.
    fn fresh_bundled_default(root: &Path, mpk: &[u8], labels_text: &str) -> PathBuf {
        let dir = root.join("bundled_default");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("head.mpk"), mpk).unwrap();
        std::fs::write(dir.join("labels.txt"), labels_text).unwrap();
        dir
    }

    // MARK: stage_and_validate_activation

    #[test]
    fn head_origin_stages_and_validates() {
        let tmp = tempfile::tempdir().unwrap();
        let mpk = b"MPK-CONTENT-AAA";
        let (ws_dir, head_id) = fresh_workspace_with_head(tmp.path(), mpk);
        let bundled = fresh_bundled_default(tmp.path(), b"unused", "x\n");
        let pending = PendingActivation {
            root: tmp.path(),
            origin_input: ActivationOriginInput::Head {
                workspace_dir: &ws_dir,
                workspace_id: ws_id(),
                head_id,
            },
            bundled_default_dir: &bundled,
            now_rfc3339: "2026-05-07T12:34:56Z".to_string(),
        };
        let result = stage_and_validate_activation(pending, &*synth_loader_ok()).unwrap();

        // Manifest reflects the head provenance + runtime id ==
        // source head id.
        match &result.manifest.origin {
            ActiveOrigin::Head {
                source_workspace_id,
                source_head_id,
                ..
            } => {
                assert_eq!(*source_workspace_id, ws_id());
                assert_eq!(*source_head_id, head_id);
            }
            other => panic!("expected Head origin, got {other:?}"),
        }
        assert_eq!(result.manifest.runtime_head_id, head_id);
        assert_eq!(result.manifest.n_classes, 2);
        assert_eq!(
            result.manifest.labels,
            vec!["alpha".to_string(), "beta".to_string()]
        );

        // Staging dir contains head.mpk + labels.txt + manifest.json.
        let staging = staging_path_for(tmp.path(), &result.activation_id);
        assert!(staging.join(ACTIVE_HEAD_FILENAME).is_file());
        assert!(staging.join(ACTIVE_LABELS_FILENAME).is_file());
        assert!(
            staging
                .join(crate::file_mgr::schema::ACTIVE_MANIFEST_FILENAME)
                .is_file()
        );

        // Labels file is rendered from manifest.labels[] (newline-joined).
        let labels_disk = std::fs::read_to_string(staging.join(ACTIVE_LABELS_FILENAME)).unwrap();
        assert_eq!(labels_disk, "alpha\nbeta");
    }

    #[test]
    fn head_origin_rejects_hash_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let mpk = b"MPK-CONTENT-BBB";
        let (ws_dir, head_id) = fresh_workspace_with_head(tmp.path(), mpk);

        // Tamper with the head .mpk on disk after the manifest's
        // sha256 was recorded.  Activation must refuse with
        // HashMismatch rather than publish the drifted bytes.
        std::fs::write(
            crate::file_mgr::schema::head_artifact_path(&ws_dir, head_id),
            b"TAMPERED",
        )
        .unwrap();

        let bundled = fresh_bundled_default(tmp.path(), b"unused", "x\n");
        let pending = PendingActivation {
            root: tmp.path(),
            origin_input: ActivationOriginInput::Head {
                workspace_dir: &ws_dir,
                workspace_id: ws_id(),
                head_id,
            },
            bundled_default_dir: &bundled,
            now_rfc3339: "2026-05-07T12:34:56Z".to_string(),
        };
        let err = stage_and_validate_activation(pending, &*synth_loader_ok())
            .expect_err("hash mismatch must reject");
        assert!(matches!(err, ActivationError::HashMismatch { .. }));
    }

    #[test]
    fn head_origin_missing_head_id_surfaces_not_found() {
        let tmp = tempfile::tempdir().unwrap();
        let ws_dir = crate::file_mgr::schema::workspace_dir_for(tmp.path(), &ws_id());
        std::fs::create_dir_all(&ws_dir).unwrap();
        std::fs::create_dir_all(crate::file_mgr::schema::heads_dir(&ws_dir)).unwrap();
        write_workspace_core(&ws_dir, &synth_workspace_core()).unwrap();
        write_head_index(&ws_dir, &HeadIndex::default()).unwrap();
        let unknown = HeadId::parse("aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee").unwrap();
        let bundled = fresh_bundled_default(tmp.path(), b"unused", "x\n");
        let pending = PendingActivation {
            root: tmp.path(),
            origin_input: ActivationOriginInput::Head {
                workspace_dir: &ws_dir,
                workspace_id: ws_id(),
                head_id: unknown,
            },
            bundled_default_dir: &bundled,
            now_rfc3339: "2026-05-07T12:34:56Z".to_string(),
        };
        let err = stage_and_validate_activation(pending, &*synth_loader_ok())
            .expect_err("missing head id must reject");
        assert!(matches!(err, ActivationError::NotFound { .. }));
    }

    #[test]
    fn default_origin_stages_and_validates() {
        let tmp = tempfile::tempdir().unwrap();
        let bundled = fresh_bundled_default(tmp.path(), b"DEFAULT-MPK", "cat\ndog\nbird\n");
        let pending = PendingActivation {
            root: tmp.path(),
            origin_input: ActivationOriginInput::Default,
            bundled_default_dir: &bundled,
            now_rfc3339: "2026-05-07T12:34:56Z".to_string(),
        };
        let result = stage_and_validate_activation(pending, &*synth_loader_ok()).unwrap();
        assert!(matches!(result.manifest.origin, ActiveOrigin::Default));
        assert_eq!(result.manifest.runtime_head_id, default_runtime_head_id());
        assert_eq!(result.manifest.n_classes, 3);
        assert_eq!(
            result.manifest.labels,
            vec!["cat".to_string(), "dog".to_string(), "bird".to_string()]
        );

        let staging = staging_path_for(tmp.path(), &result.activation_id);
        assert!(staging.join(ACTIVE_HEAD_FILENAME).is_file());
        assert!(staging.join(ACTIVE_LABELS_FILENAME).is_file());
    }

    #[test]
    fn default_origin_missing_fixture_surfaces_not_found() {
        let tmp = tempfile::tempdir().unwrap();
        let pending = PendingActivation {
            root: tmp.path(),
            origin_input: ActivationOriginInput::Default,
            bundled_default_dir: &tmp.path().join("does_not_exist"),
            now_rfc3339: "2026-05-07T12:34:56Z".to_string(),
        };
        let err = stage_and_validate_activation(pending, &*synth_loader_ok())
            .expect_err("missing fixture must reject");
        assert!(matches!(err, ActivationError::NotFound { .. }));
    }

    #[test]
    fn preload_failure_propagates() {
        let tmp = tempfile::tempdir().unwrap();
        let bundled = fresh_bundled_default(tmp.path(), b"DEFAULT-MPK", "cat\n");
        let pending = PendingActivation {
            root: tmp.path(),
            origin_input: ActivationOriginInput::Default,
            bundled_default_dir: &bundled,
            now_rfc3339: "2026-05-07T12:34:56Z".to_string(),
        };
        let err = stage_and_validate_activation(pending, &*synth_loader_fail())
            .expect_err("preload failure must reject");
        assert!(matches!(err, ActivationError::Preload { .. }));
    }

    // MARK: publish_active_generation

    #[test]
    fn publish_renames_and_writes_pointer() {
        let tmp = tempfile::tempdir().unwrap();
        let bundled = fresh_bundled_default(tmp.path(), b"MPK", "x\n");
        let pending = PendingActivation {
            root: tmp.path(),
            origin_input: ActivationOriginInput::Default,
            bundled_default_dir: &bundled,
            now_rfc3339: "2026-05-07T12:34:56Z".to_string(),
        };
        let result = stage_and_validate_activation(pending, &*synth_loader_ok()).unwrap();
        let staging = staging_path_for(tmp.path(), &result.activation_id);

        publish_active_generation(
            tmp.path(),
            &staging,
            &result.manifest,
            &result.activation_id,
        )
        .unwrap();

        // Generation directory is in place.
        assert!(active_generation_dir(tmp.path(), &result.activation_id).is_dir());
        // current.json points at it.
        let pointer = crate::file_mgr::schema::read_active_current(tmp.path()).unwrap();
        assert_eq!(pointer.activation_id, result.activation_id);
        // Staging directory was consumed.
        assert!(!staging.exists());
    }

    // MARK: prune_old_generations

    #[test]
    fn prune_keeps_only_listed_generations() {
        let tmp = tempfile::tempdir().unwrap();
        let bundled = fresh_bundled_default(tmp.path(), b"MPK", "x\n");
        // Run three activations end-to-end.
        let mut ids = Vec::new();
        for _ in 0..3 {
            let pending = PendingActivation {
                root: tmp.path(),
                origin_input: ActivationOriginInput::Default,
                bundled_default_dir: &bundled,
                now_rfc3339: "2026-05-07T12:34:56Z".to_string(),
            };
            let result = stage_and_validate_activation(pending, &*synth_loader_ok()).unwrap();
            let staging = staging_path_for(tmp.path(), &result.activation_id);
            publish_active_generation(
                tmp.path(),
                &staging,
                &result.manifest,
                &result.activation_id,
            )
            .unwrap();
            ids.push(result.activation_id);
        }
        // Keep the last two; first should be pruned.
        let keep = vec![ids[1].clone(), ids[2].clone()];
        let removed = prune_old_generations(tmp.path(), &keep).unwrap();
        assert_eq!(removed, 1);
        assert!(!active_generation_dir(tmp.path(), &ids[0]).exists());
        assert!(active_generation_dir(tmp.path(), &ids[1]).is_dir());
        assert!(active_generation_dir(tmp.path(), &ids[2]).is_dir());
    }

    #[test]
    fn prune_on_missing_generations_dir_is_ok() {
        let tmp = tempfile::tempdir().unwrap();
        // No `<root>/active/generations/` exists; prune should
        // succeed with 0 removed.
        let removed = prune_old_generations::<&str>(tmp.path(), &[]).unwrap();
        assert_eq!(removed, 0);
    }

    // MARK: helpers

    #[test]
    fn labels_to_text_is_newline_joined() {
        assert_eq!(labels_to_text(&[]), "");
        assert_eq!(labels_to_text(&["a".into()]), "a");
        assert_eq!(
            labels_to_text(&["a".into(), "b".into(), "c".into()]),
            "a\nb\nc"
        );
    }
}
