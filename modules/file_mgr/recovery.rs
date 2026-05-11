//! Boot-time recovery sweeps.
//!
//! Runs once on every daemon start, AFTER
//! `WorkspaceMgr::ensure_root_layout` and BEFORE any API route or
//! inference engine goes live.  Three sweeps in fixed order
//! (root staging -> per-workspace -> active head) drain the
//! crash-recovery state:
//!
//! - root `.tmp/delete-workspace-*` tombstones + payloads -> drain + finalize.
//! - workspace `<id>/.tmp/delete-{assets,converters,training-logs,converter-logs}-*`
//!   tombstones + payloads -> drain + finalize (one pass per prefix).
//! - per-workspace daemon-owned head orphans (`<head_id>.{mpk,json}`
//!   not in `heads.json`) -> sweep.
//! - `workspace.json.head_count` <- `heads.json.heads.len()` (repair).
//! - active-head verify: streaming-hash `head.mpk` + `labels.txt`
//!   against the manifest; on `labels.txt` mismatch only,
//!   regenerate from `manifest.labels[]`; on `head.mpk` mismatch /
//!   load fail try the most recently published previous
//!   generation; on no valid generation activate the bundled
//!   default; on bundled-default missing return
//!   [`RecoveryActiveResult::Unhealthy`].
//!
//! Active recovery is last because activation MAY reference a
//! head in a workspace whose tombstone was just resolved; running
//! the active sweep first could observe a workspace that the
//! root-staging step is about to delete.
//!
//! # Returned report
//!
//! Each sweep returns a small report struct counting the work it
//! did so the daemon (and the status surface) can log + publish
//! boot-recovery progress.  The active sweep returns one of four
//! typed outcomes carrying the resolved manifest the caller must
//! install + publish.
//!
//! # Stop rule
//!
//! [`recover_active_head`] does NOT instantiate a runtime
//! `HotHead`.  It pre-validates the on-disk active generation,
//! returns the manifest for promoted/defaulted variants, and
//! lets the daemon do the runtime install via the existing
//! `HotHead::load` path.  This keeps `file_mgr` independent of
//! `inference`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use sha2::{Digest, Sha256};

use crate::common::ids::{HeadId, WorkspaceId};
use crate::common::workspace::{ActiveHeadManifest, ActiveHeadValidationError, HeadIndex};
use crate::file_mgr::active_head_writer::{
    ActivationError, ActivationOriginInput, DefaultHeadSource, HeadInnerLoader, PendingActivation,
    publish_active_generation, stage_and_validate_activation, staging_path_for,
};
use crate::file_mgr::cache::WorkspaceCacheCell;
use crate::file_mgr::error::{FileError, io_err};
use crate::file_mgr::fs_atomic::put_atomic;
use crate::file_mgr::schema::{
    ACTIVE_HEAD_FILENAME, ACTIVE_LABELS_FILENAME, ActiveCurrentPointer, active_current_path,
    active_dir, active_generation_dir, active_generations_dir, heads_dir, read_active_current,
    read_active_manifest, read_head_index, read_workspace_core, workspace_core_path,
    workspaces_dir, write_active_current, write_head_index, write_workspace_core,
};
use crate::file_mgr::staging::{
    CONVERTER_LOGS_TOMBSTONE_PREFIX, CONVERTER_TOMBSTONE_PREFIX, DATASET_TOMBSTONE_PREFIX,
    DEFAULT_DELETE_BATCH_ENTRIES, DrainResult, StagedDelete, TRAINING_LOGS_TOMBSTONE_PREFIX,
    WORKSPACE_TOMBSTONE_PREFIX, drain_staged_payload, finalize_staged_delete, read_tombstone,
};
use crate::file_mgr::time_util::now_rfc3339;
use crate::file_mgr::validate::{fsync_dir, hex_lowercase};

/// 32 KiB chunk size for streaming-hashing `head.mpk` and
/// `labels.txt` against the manifest's recorded sha256.
const STREAM_HASH_CHUNK: usize = 32 * 1024;

// MARK: RecoveryError

/// Failure shapes from the boot-recovery sweeps.  Mapped to
/// HTTP `Internal` -- recovery runs before the api is live, so
/// the categorisation only matters for callers that re-export
/// the error through their own surface (none today).
#[derive(Debug, thiserror::Error)]
pub enum RecoveryError {
    /// Underlying file-mgr error (io, parse, persist).  Surfaced
    /// from one of the schema readers, the staging primitives,
    /// or the recovery's own walker.
    #[error("file: {0}")]
    File(#[from] FileError),
    /// The activation pipeline rejected the bundled-default
    /// fallback.  Recovery propagates this into
    /// [`RecoveryActiveResult::Unhealthy`] with the formatted
    /// reason; the variant itself is reachable via the
    /// `recover_all` path when a caller surfaces an error
    /// instead of an unhealthy outcome.
    #[error("activation: {0}")]
    Activation(#[from] ActivationError),
    /// `read_active_manifest` returned an `ActiveHeadManifest`
    /// that failed [`ActiveHeadManifest::validate`].  Treated as
    /// a recoverable failure of the current generation; the
    /// caller falls through to the previous-generation
    /// candidate.
    #[error("active manifest validation: {0}")]
    Validation(#[from] ActiveHeadValidationError),
}

impl crate::common::error::Categorized for RecoveryError {
    fn kind(&self) -> crate::common::error::ErrorKind {
        // Recovery runs before any API surface is live; every
        // failure is daemon-internal.
        crate::common::error::ErrorKind::Internal
    }
}

// MARK: RecoveryActiveResult / RecoveryWorkspaceReport / RecoveryRootReport / RecoveryReport

/// Outcome of [`recover_active_head`].  Caller installs the
/// resolved manifest into the runtime `HotHead` and, for the
/// `Promoted*` / `Defaulted*` variants, has already had
/// `current.json` rewritten to point at the new generation.
#[derive(Debug)]
pub enum RecoveryActiveResult {
    /// Current generation passed `head.mpk` + `labels.txt`
    /// streaming-hash.  No on-disk pointer change; caller loads
    /// the manifest into the runtime.
    Current {
        /// Generation directory name (`<root>/active/generations/<id>/`).
        activation_id: String,
        /// Validated manifest body the caller installs.
        manifest: ActiveHeadManifest,
    },
    /// Current generation failed verify; the previous
    /// generation was promoted to current.  `current.json` was
    /// rewritten before this returns.
    PromotedPrevious {
        /// Generation directory name now referenced by `current.json`.
        activation_id: String,
        /// Validated manifest body the caller installs.
        manifest: ActiveHeadManifest,
    },
    /// No retained generation passed verify (or `current.json`
    /// was absent on a non-first-boot daemon); the bundled
    /// default was activated through the standard pipeline.
    /// `current.json` was rewritten before this returns.
    DefaultedFromBundle {
        /// Generation directory name of the freshly published default.
        activation_id: String,
        /// Validated manifest body the caller installs.
        manifest: ActiveHeadManifest,
    },
    /// Bundled default was missing or its activation pipeline
    /// failed.  Daemon boots without inference; caller marks
    /// the inference subsystem unhealthy with the variant's
    /// `reason` field in the operator-visible
    /// `degraded_reason`.
    Unhealthy {
        /// Operator-readable diagnostic.
        reason: String,
    },
}

impl RecoveryActiveResult {
    /// `reason` of the [`Self::Unhealthy`] variant; `None` for
    /// healthy outcomes.  Convenience for daemon log glue.
    pub fn unhealthy_reason(&self) -> Option<&str> {
        match self {
            RecoveryActiveResult::Unhealthy { reason } => Some(reason.as_str()),
            _ => None,
        }
    }
}

/// Counters returned by [`recover_workspaces`].  Operator-facing
/// diagnostics; the daemon publishes these through the
/// `boot_orphans_swept_total` status family.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RecoveryWorkspaceReport {
    /// Workspaces whose `workspace.json` parsed; incomplete
    /// creates (no `workspace.json`) are NOT counted here.  They
    /// are removed and counted in [`Self::incomplete_creates_removed`].
    pub workspaces_scanned: usize,
    /// `<head_id>.{mpk,json}` files dropped because the id was
    /// not in `heads.json.heads[]`.
    pub head_orphans_swept: usize,
    /// Workspaces whose `workspace.json.head_count` disagreed
    /// with `heads.json.heads.len()`; rewritten to match.
    pub head_count_repaired: usize,
    /// Dataset-delete tombstones drained + finalized to
    /// completion.
    pub dataset_tombstones_completed: usize,
    /// Dataset-delete stage directories without a matching
    /// tombstone removed.
    pub dataset_stage_orphans_swept: usize,
    /// Converter-delete tombstones drained + finalized.  The
    /// per-workspace `.tmp/` sweep walks dataset and converter
    /// prefixes alike.
    pub converter_tombstones_completed: usize,
    /// Converter-delete stage directories without a matching
    /// tombstone removed.
    pub converter_stage_orphans_swept: usize,
    /// Training-logs-delete tombstones drained + finalized.  Like
    /// [`Self::dataset_tombstones_completed`] but for the
    /// per-workspace `training_logs/` async-wipe path.
    pub training_logs_tombstones_completed: usize,
    /// Training-logs-delete stage directories without a matching
    /// tombstone removed.
    pub training_logs_stage_orphans_swept: usize,
    /// Converter-logs-delete tombstones drained + finalized.
    /// Mirror of [`Self::training_logs_tombstones_completed`] for
    /// the converter producer.
    pub converter_logs_tombstones_completed: usize,
    /// Converter-logs-delete stage directories without a matching
    /// tombstone removed.
    pub converter_logs_stage_orphans_swept: usize,
    /// Workspace directories without `workspace.json` (incomplete
    /// creates) removed; `<root>/workspaces/` is fsynced so the
    /// unlink is durable.
    pub incomplete_creates_removed: usize,
    /// Per-workspace recovery failures: `recover_one_workspace`
    /// returned `Err`, the orchestrator logged it and continued
    /// per the failure-tolerance contract.  Failed workspaces are
    /// NOT counted in [`Self::workspaces_scanned`].
    pub workspace_recovery_failures: usize,
}

/// Counters returned by [`recover_root_staging`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RecoveryRootReport {
    /// Workspace-delete tombstones drained + finalized to
    /// completion.
    pub workspace_tombstones_completed: usize,
    /// Workspace-delete stage directories without a matching
    /// tombstone removed.
    pub workspace_stage_orphans_swept: usize,
}

/// Aggregate report from [`recover_all`].
#[derive(Debug)]
pub struct RecoveryReport {
    /// Outcome of the active-head verify pass.
    pub active: RecoveryActiveResult,
    /// Per-workspace sweep counters.
    pub workspaces: RecoveryWorkspaceReport,
    /// Root staging sweep counters.
    pub root_staging: RecoveryRootReport,
}

// MARK: recover_all

/// Run every boot-recovery sweep in dependency order.  Returns
/// the aggregate report so the daemon can log counters + drive
/// the runtime active-head install.
///
/// # Order
///
/// 1. Root staging sweep -- complete pending workspace deletes;
///    eject those workspaces from `caches`.
/// 2. Per-workspace sweep -- complete pending dataset deletes,
///    repair `head_count`, drop daemon-owned head orphans.
///    Skips directories without `workspace.json` (incomplete
///    creates).
/// 3. Active-head verify -- streaming-hash + previous-generation
///    fallback + bundled-default activation.
///
/// The active sweep is intentionally LAST because activation may
/// reference a head in a workspace whose tombstone was just
/// resolved by step 1; running the active sweep first could
/// observe a workspace that root staging is about to delete.
pub fn recover_all(
    root: &Path,
    default_head: Option<DefaultHeadSource<'_>>,
    caches: &dashmap::DashMap<WorkspaceId, Arc<WorkspaceCacheCell>>,
    head_inner_loader: &HeadInnerLoader,
) -> Result<RecoveryReport, RecoveryError> {
    let root_staging = recover_root_staging(root, caches)?;
    let workspaces = recover_workspaces(root)?;
    let active = recover_active_head(root, default_head, head_inner_loader)?;
    Ok(RecoveryReport {
        active,
        workspaces,
        root_staging,
    })
}

// MARK: 1A -- recover_active_head

/// Verify the active-head generation and pick the runtime
/// candidate.
///
/// Sequence:
///
/// 1. Read `<root>/active/current.json`.  If absent -> activate
///    bundled default.
/// 2. Read pointed manifest, run [`ActiveHeadManifest::validate`].
/// 3. Streaming-hash `head.mpk` (32 KiB chunks) vs
///    `manifest.sha256`.  Mismatch -> try previous generation.
/// 4. Streaming-hash `labels.txt` vs `manifest.labels_sha256`.
///    Mismatch -> regenerate `labels.txt` from
///    `manifest.labels[]`, fsync, accept the generation.
/// 5. Previous-generation fallback: scan
///    `<root>/active/generations/`, sort by mtime descending,
///    skip the current id, take the next that passes verify
///    (with the same labels.txt regen rule).  Promoted
///    generation gets a fresh `current.json` write so the
///    on-disk pointer matches the runtime install.
/// 6. Bundled-default fallback: stage + publish via
///    [`stage_and_validate_activation`] + [`publish_active_generation`].
///    On failure -> [`RecoveryActiveResult::Unhealthy`].
///
/// `head_inner_loader` is plumbed through to the bundled-default
/// fallback's [`stage_and_validate_activation`] call; the
/// in-place "current passes" / "promote previous" paths do NOT
/// call the loader (the daemon's existing boot path re-loads via
/// `HotHead::load` after this returns).
pub fn recover_active_head(
    root: &Path,
    default_head: Option<DefaultHeadSource<'_>>,
    head_inner_loader: &HeadInnerLoader,
) -> Result<RecoveryActiveResult, RecoveryError> {
    let pointer_path = active_current_path(root);
    if !pointer_path.exists() {
        return activate_bundled_default(root, default_head, head_inner_loader);
    }

    let pointer = match read_active_current(root) {
        Ok(p) => p,
        Err(e) => {
            // Corrupt pointer -> treat the current generation as
            // missing; fall through to the previous-generation
            // / default fallback.
            tracing::warn!(
                target: "file_mgr::recovery",
                err = %e,
                "active current.json read/parse failed; falling back",
            );
            return promote_or_default(root, None, default_head, head_inner_loader);
        }
    };

    match verify_generation(root, &pointer.activation_id)? {
        VerifyOutcome::Ok { manifest } => Ok(RecoveryActiveResult::Current {
            activation_id: pointer.activation_id,
            manifest,
        }),
        VerifyOutcome::Failed => promote_or_default(
            root,
            Some(&pointer.activation_id),
            default_head,
            head_inner_loader,
        ),
    }
}

/// Outcome of [`verify_generation`].  `Ok` carries the validated
/// manifest the caller will install; `Failed` is unstructured
/// because the diagnostic is already logged at the failure site.
enum VerifyOutcome {
    Ok { manifest: ActiveHeadManifest },
    Failed,
}

/// Verify one generation in place: read manifest, validate,
/// streaming-hash `head.mpk`, then streaming-hash `labels.txt`
/// (regenerating from `manifest.labels[]` on mismatch).
///
/// On success the on-disk generation is in a state where
/// `head.mpk` matches `manifest.sha256` and `labels.txt` matches
/// `manifest.labels_sha256`.
fn verify_generation(root: &Path, activation_id: &str) -> Result<VerifyOutcome, RecoveryError> {
    let manifest = match read_active_manifest(root, activation_id) {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!(
                target: "file_mgr::recovery",
                activation_id = %activation_id,
                err = %e,
                "active manifest read/parse failed",
            );
            return Ok(VerifyOutcome::Failed);
        }
    };
    if let Err(e) = manifest.validate() {
        tracing::warn!(
            target: "file_mgr::recovery",
            activation_id = %activation_id,
            err = %e,
            "active manifest validation failed",
        );
        return Ok(VerifyOutcome::Failed);
    }
    let gen_dir = active_generation_dir(root, activation_id);
    let head_mpk = gen_dir.join(ACTIVE_HEAD_FILENAME);
    match sha256_stream(&head_mpk) {
        Ok(observed) if observed == manifest.sha256 => {}
        Ok(observed) => {
            tracing::warn!(
                target: "file_mgr::recovery",
                activation_id = %activation_id,
                expected = %manifest.sha256,
                observed = %observed,
                "active head.mpk hash mismatch",
            );
            return Ok(VerifyOutcome::Failed);
        }
        Err(e) => {
            tracing::warn!(
                target: "file_mgr::recovery",
                activation_id = %activation_id,
                path = %head_mpk.display(),
                err = %e,
                "active head.mpk read failed",
            );
            return Ok(VerifyOutcome::Failed);
        }
    }
    // labels.txt: hash, regen on mismatch only.
    let labels_path = gen_dir.join(ACTIVE_LABELS_FILENAME);
    let labels_ok = match sha256_stream(&labels_path) {
        Ok(observed) => observed == manifest.labels_sha256,
        Err(e) => {
            tracing::warn!(
                target: "file_mgr::recovery",
                activation_id = %activation_id,
                path = %labels_path.display(),
                err = %e,
                "active labels.txt read failed; regenerating from manifest",
            );
            false
        }
    };
    if !labels_ok {
        regenerate_labels_from_manifest(&labels_path, &manifest)?;
        tracing::info!(
            target: "file_mgr::recovery",
            activation_id = %activation_id,
            "regenerated active labels.txt from manifest.labels[]",
        );
    }
    Ok(VerifyOutcome::Ok { manifest })
}

/// Promote the most recently published valid generation other
/// than `current_id`, falling back to the bundled default.
fn promote_or_default(
    root: &Path,
    current_id: Option<&str>,
    default_head: Option<DefaultHeadSource<'_>>,
    head_inner_loader: &HeadInnerLoader,
) -> Result<RecoveryActiveResult, RecoveryError> {
    let generations_root = active_generations_dir(root);
    if !generations_root.is_dir() {
        return activate_bundled_default(root, default_head, head_inner_loader);
    }
    // Collect candidates, newest first by manifest-recorded
    // `activated_at` (RFC3339; lexicographic sort is correct for
    // RFC3339 timestamps and is robust to the coarse-mtime
    // semantics on macOS HFS+).  Tie-break by directory name
    // (lexicographic) so the order is deterministic on identical
    // timestamps.  Falls back to mtime when a manifest is
    // unparseable.  A directory we cannot stat is logged +
    // skipped rather than aborting the whole recovery (the
    // operator's other generations may still be intact).
    let mut candidates: Vec<(String, std::time::SystemTime, String)> = Vec::new();
    let entries =
        std::fs::read_dir(&generations_root).map_err(|e| io_err(generations_root.display(), e))?;
    for entry in entries {
        let entry = entry.map_err(|e| io_err(generations_root.display(), e))?;
        let Ok(name) = entry.file_name().into_string() else {
            continue;
        };
        if Some(name.as_str()) == current_id {
            continue;
        }
        let metadata = match entry.metadata() {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!(
                    target: "file_mgr::recovery",
                    path = %entry.path().display(),
                    err = %e,
                    "skipping unreadable generation entry during fallback",
                );
                continue;
            }
        };
        if !metadata.is_dir() {
            continue;
        }
        // Prefer manifest-recorded `activated_at` for sort key;
        // fall back to empty string when the manifest is
        // unparseable so mtime breaks the tie below.
        let activated_at = read_active_manifest(root, &name)
            .map(|m| m.activated_at)
            .unwrap_or_default();
        let mtime = metadata
            .modified()
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        candidates.push((activated_at, mtime, name));
    }
    // Sort: newest activated_at first; mtime breaks ties; name
    // breaks final ties (deterministic across `read_dir` order).
    candidates.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)).then(a.2.cmp(&b.2)));
    for (_, _, candidate_id) in candidates {
        if let VerifyOutcome::Ok { manifest } = verify_generation(root, &candidate_id)? {
            // Promote: rewrite `current.json` so the on-disk
            // pointer matches the runtime install.
            write_active_current(
                root,
                &ActiveCurrentPointer {
                    activation_id: candidate_id.clone(),
                },
            )?;
            // Best-effort fsync of the active dir so the pointer
            // rewrite reaches stable storage.
            if let Err(e) = fsync_dir(&active_dir(root)) {
                tracing::warn!(
                    target: "file_mgr::recovery",
                    err = %e,
                    "fsync active/ after promote_previous failed (best-effort)",
                );
            }
            tracing::warn!(
                target: "file_mgr::recovery",
                activation_id = %candidate_id,
                "promoted previous active generation; current.json rewritten",
            );
            return Ok(RecoveryActiveResult::PromotedPrevious {
                activation_id: candidate_id,
                manifest,
            });
        }
    }
    activate_bundled_default(root, default_head, head_inner_loader)
}

/// Activate the deployment-bundled default through the standard
/// `stage_and_validate_activation` + `publish_active_generation`
/// pipeline.  Failure (missing fixture, hash mismatch, preload
/// failure) becomes [`RecoveryActiveResult::Unhealthy`] so the
/// daemon can boot without inference for operator triage.
///
/// `default_head: None` means the launch config did not configure a
/// bundled default at all; surface `Unhealthy` directly without
/// staging anything.  Recovery's other sweeps (root staging,
/// per-workspace) still ran in `recover_all` -- this only suppresses
/// the active-head materialization step.
fn activate_bundled_default(
    root: &Path,
    default_head: Option<DefaultHeadSource<'_>>,
    head_inner_loader: &HeadInnerLoader,
) -> Result<RecoveryActiveResult, RecoveryError> {
    let Some(default_head) = default_head else {
        return Ok(RecoveryActiveResult::Unhealthy {
            reason: "head.default not configured in launch config".into(),
        });
    };
    let pending = PendingActivation {
        root,
        origin_input: ActivationOriginInput::Default {
            source: default_head,
        },
        now_rfc3339: now_rfc3339(),
    };
    let result = match stage_and_validate_activation(pending, head_inner_loader) {
        Ok(r) => r,
        Err(e) => {
            return Ok(RecoveryActiveResult::Unhealthy {
                reason: format!("bundled default activation failed: {e}"),
            });
        }
    };
    let staging = staging_path_for(root, &result.activation_id);
    if let Err(e) =
        publish_active_generation(root, &staging, &result.manifest, &result.activation_id)
    {
        return Ok(RecoveryActiveResult::Unhealthy {
            reason: format!("bundled default publish failed: {e}"),
        });
    }
    tracing::warn!(
        target: "file_mgr::recovery",
        activation_id = %result.activation_id,
        "boot recovery activated bundled default",
    );
    Ok(RecoveryActiveResult::DefaultedFromBundle {
        activation_id: result.activation_id,
        manifest: result.manifest,
    })
}

/// Render `manifest.labels[]` to `labels.txt` (newline-joined
/// UTF-8) atomically + fsync the parent so the regen reaches
/// stable storage.  Mirrors the active-head writer's
/// `labels_to_text` shape so the regenerated bytes match what a
/// fresh activation would have produced.
fn regenerate_labels_from_manifest(
    labels_path: &Path,
    manifest: &ActiveHeadManifest,
) -> Result<(), FileError> {
    let mut bytes: Vec<u8> = Vec::with_capacity(manifest.labels.iter().map(|s| s.len() + 1).sum());
    for (i, label) in manifest.labels.iter().enumerate() {
        if i > 0 {
            bytes.push(b'\n');
        }
        bytes.extend_from_slice(label.as_bytes());
    }
    put_atomic(labels_path, &bytes)
}

// MARK: 1B -- recover_workspaces

/// Per-workspace sweep: complete dataset-delete tombstones,
/// drop daemon-owned head orphans, repair derived `head_count`,
/// remove incomplete-create directories.  See module docs for
/// the full ordering.
pub fn recover_workspaces(root: &Path) -> Result<RecoveryWorkspaceReport, RecoveryError> {
    let workspaces_root = workspaces_dir(root);
    if !workspaces_root.is_dir() {
        return Ok(RecoveryWorkspaceReport::default());
    }
    let mut report = RecoveryWorkspaceReport::default();
    let mut incomplete_dirs: Vec<PathBuf> = Vec::new();
    let entries =
        std::fs::read_dir(&workspaces_root).map_err(|e| io_err(workspaces_root.display(), e))?;
    for entry in entries {
        let entry = entry.map_err(|e| io_err(workspaces_root.display(), e))?;
        let path = entry.path();
        let file_type = match entry.file_type() {
            Ok(ft) => ft,
            Err(e) => {
                tracing::warn!(
                    target: "file_mgr::recovery",
                    path = %path.display(),
                    err = %e,
                    "skipping unreadable workspaces entry",
                );
                continue;
            }
        };
        if !file_type.is_dir() {
            continue;
        }
        // Incomplete create: directory without `workspace.json`
        // (the publish point).  Defer the `remove_dir_all` until
        // after the outer iterator finishes so we don't perturb
        // its view.
        if !workspace_core_path(&path).exists() {
            incomplete_dirs.push(path);
            continue;
        }
        // Sweep this workspace.  Per-workspace sweep failures
        // are logged but do not abort the boot recovery; the
        // operator's other workspaces may still be intact.
        match recover_one_workspace(&path) {
            Ok(per) => {
                report.workspaces_scanned += 1;
                report.head_orphans_swept += per.head_orphans_swept;
                report.head_count_repaired += per.head_count_repaired;
                report.dataset_tombstones_completed += per.dataset_tombstones_completed;
                report.dataset_stage_orphans_swept += per.dataset_stage_orphans_swept;
                report.converter_tombstones_completed += per.converter_tombstones_completed;
                report.converter_stage_orphans_swept += per.converter_stage_orphans_swept;
                report.training_logs_tombstones_completed += per.training_logs_tombstones_completed;
                report.training_logs_stage_orphans_swept += per.training_logs_stage_orphans_swept;
                report.converter_logs_tombstones_completed +=
                    per.converter_logs_tombstones_completed;
                report.converter_logs_stage_orphans_swept += per.converter_logs_stage_orphans_swept;
            }
            Err(e) => {
                tracing::warn!(
                    target: "file_mgr::recovery",
                    workspace = %path.display(),
                    err = %e,
                    "per-workspace recovery failed; leaving for next boot",
                );
                report.workspace_recovery_failures += 1;
            }
        }
    }
    // Remove every incomplete-create directory; fsync the
    // workspaces parent once at the end so the unlinks reach
    // stable storage in one step rather than per-entry.
    let mut removed_any = false;
    for path in incomplete_dirs {
        match std::fs::remove_dir_all(&path) {
            Ok(()) => {
                report.incomplete_creates_removed += 1;
                removed_any = true;
                tracing::warn!(
                    target: "file_mgr::recovery",
                    path = %path.display(),
                    "removed incomplete-create workspace directory",
                );
            }
            Err(e) => {
                tracing::warn!(
                    target: "file_mgr::recovery",
                    path = %path.display(),
                    err = %e,
                    "incomplete-create cleanup failed; leaving for next boot",
                );
            }
        }
    }
    if removed_any && let Err(e) = fsync_dir(&workspaces_root) {
        tracing::warn!(
            target: "file_mgr::recovery",
            err = %e,
            "fsync workspaces/ after incomplete-create sweep failed (best-effort)",
        );
    }
    Ok(report)
}

/// Sweep counters local to one workspace; folded into the
/// outer [`RecoveryWorkspaceReport`].
#[derive(Default)]
struct PerWorkspaceCounts {
    head_orphans_swept: usize,
    head_count_repaired: usize,
    dataset_tombstones_completed: usize,
    dataset_stage_orphans_swept: usize,
    converter_tombstones_completed: usize,
    converter_stage_orphans_swept: usize,
    training_logs_tombstones_completed: usize,
    training_logs_stage_orphans_swept: usize,
    converter_logs_tombstones_completed: usize,
    converter_logs_stage_orphans_swept: usize,
}

/// Recover one workspace dir.  Order:
///
/// 1. Read `heads.json`.  Used by both the head-orphan sweep
///    AND the head-count repair.
/// 2. Sweep `<workspace>/heads/<head_id>.{mpk,json}` whose
///    `head_id` is not in `heads.json.heads[]`.
/// 3. Repair `workspace.json.head_count` <- `heads.json.heads.len()`.
/// 4. Complete `<workspace>/.tmp/delete-{assets,converters,training-logs,converter-logs}-*.json`
///    tombstones (drain + finalize) — one pass per prefix.
/// 5. Sweep `<workspace>/.tmp/delete-{assets,converters,training-logs,converter-logs}-*`
///    directories without a matching tombstone.
fn recover_one_workspace(workspace_dir: &Path) -> Result<PerWorkspaceCounts, FileError> {
    let mut counts = PerWorkspaceCounts::default();

    // 1+2. Head orphans sweep.
    let heads = match read_head_index(workspace_dir) {
        Ok(idx) => idx,
        Err(FileError::Io { ref source, .. }) if source.kind() == std::io::ErrorKind::NotFound => {
            // Workspace.json present but heads.json missing.
            // Treat as empty head index for the sweep AND
            // repair the index back to `[]` so subsequent
            // boots converge.  Boot recovery is the only place
            // this state is permitted (production rotation
            // always writes heads.json + workspace.json under
            // the per-workspace mutex).
            tracing::warn!(
                target: "file_mgr::recovery",
                workspace = %workspace_dir.display(),
                "heads.json missing under valid workspace.json; rewriting empty index",
            );
            write_head_index(workspace_dir, &HeadIndex::default())?;
            HeadIndex::default()
        }
        Err(e) => return Err(e),
    };
    counts.head_orphans_swept += sweep_head_orphans(workspace_dir, &heads)?;

    // 3. Head-count repair.
    counts.head_count_repaired += repair_head_count(workspace_dir, &heads)?;

    // 4+5. Per-workspace tombstones + orphans.  The per-workspace
    // `.tmp/` carries four prefixes (`delete-assets-*` dataset,
    // `delete-converters-*` converter, `delete-training-logs-*`
    // training logs, `delete-converter-logs-*` converter logs);
    // each gets its own pass.  The sweep is bounded per pass so a
    // malformed listing in one tree does not block the others.
    let staging_dir = workspace_dir.join(".tmp");
    if staging_dir.is_dir() {
        let (completed, orphan_swept) = drain_staging_dir(&staging_dir, StagingScope::Dataset)?;
        counts.dataset_tombstones_completed += completed;
        counts.dataset_stage_orphans_swept += orphan_swept;
        let (completed, orphan_swept) = drain_staging_dir(&staging_dir, StagingScope::Converter)?;
        counts.converter_tombstones_completed += completed;
        counts.converter_stage_orphans_swept += orphan_swept;
        let (completed, orphan_swept) =
            drain_staging_dir(&staging_dir, StagingScope::TrainingLogs)?;
        counts.training_logs_tombstones_completed += completed;
        counts.training_logs_stage_orphans_swept += orphan_swept;
        let (completed, orphan_swept) =
            drain_staging_dir(&staging_dir, StagingScope::ConverterLogs)?;
        counts.converter_logs_tombstones_completed += completed;
        counts.converter_logs_stage_orphans_swept += orphan_swept;
    }
    Ok(counts)
}

/// Walk `<workspace>/heads/` and unlink any
/// `<head_id>.{mpk,json}` whose stem does not appear in
/// `heads.json.heads[]`.  Returns the count of files removed
/// (each `.mpk` and `.json` is counted independently so a
/// fully-orphaned pair shows up as 2).
fn sweep_head_orphans(workspace_dir: &Path, heads: &HeadIndex) -> Result<usize, FileError> {
    let dir = heads_dir(workspace_dir);
    if !dir.is_dir() {
        return Ok(0);
    }
    let known: std::collections::HashSet<HeadId> = heads.heads.iter().map(|h| h.head_id).collect();
    let mut removed = 0usize;
    let entries = std::fs::read_dir(&dir).map_err(|e| io_err(dir.display(), e))?;
    for entry in entries {
        let entry = entry.map_err(|e| io_err(dir.display(), e))?;
        let path = entry.path();
        let ft = match entry.file_type() {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!(
                    target: "file_mgr::recovery",
                    path = %path.display(),
                    err = %e,
                    "skipping unreadable heads/ entry",
                );
                continue;
            }
        };
        if !ft.is_file() {
            continue;
        }
        // Only `.mpk` / `.json` files are daemon-owned in
        // `heads/`; an unrecognised extension is left alone so
        // an operator-pasted file is not silently deleted.
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let Some(stem_ext) = name.rsplit_once('.') else {
            continue;
        };
        if !matches!(stem_ext.1, "mpk" | "json") {
            continue;
        }
        // Non-UUID filename: not produced by the rotation
        // primitive; leave it alone for operator triage.
        let Ok(id) = HeadId::parse(stem_ext.0) else {
            continue;
        };
        if known.contains(&id) {
            continue;
        }
        match std::fs::remove_file(&path) {
            Ok(()) => {
                removed += 1;
                tracing::warn!(
                    target: "file_mgr::recovery",
                    path = %path.display(),
                    "removed daemon-owned head orphan",
                );
            }
            Err(e) => {
                tracing::warn!(
                    target: "file_mgr::recovery",
                    path = %path.display(),
                    err = %e,
                    "failed to remove head orphan; leaving for next boot",
                );
            }
        }
    }
    if removed > 0
        && let Err(e) = fsync_dir(&dir)
    {
        tracing::warn!(
            target: "file_mgr::recovery",
            err = %e,
            "fsync heads/ after orphan sweep failed (best-effort)",
        );
    }
    Ok(removed)
}

/// Repair `workspace.json.head_count` if it disagrees with
/// `heads.json.heads.len()`.  Returns `1` on a write, `0` on a
/// no-op so callers can sum across workspaces.
fn repair_head_count(workspace_dir: &Path, heads: &HeadIndex) -> Result<usize, FileError> {
    let core_path = workspace_core_path(workspace_dir);
    let core = match read_workspace_core(workspace_dir) {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(
                target: "file_mgr::recovery",
                path = %core_path.display(),
                err = %e,
                "read workspace.json failed during head_count repair; skipping",
            );
            return Ok(0);
        }
    };
    let expected = u8::try_from(heads.heads.len()).unwrap_or(u8::MAX);
    if core.head_count == expected {
        return Ok(0);
    }
    let mut updated = core.clone();
    updated.head_count = expected;
    write_workspace_core(workspace_dir, &updated)?;
    tracing::warn!(
        target: "file_mgr::recovery",
        workspace = %workspace_dir.display(),
        observed = core.head_count,
        repaired = expected,
        "repaired workspace.json.head_count",
    );
    Ok(1)
}

// MARK: 1C -- recover_root_staging

/// Root `.tmp/` sweep: complete pending workspace-delete
/// tombstones (drain + finalize), eject the targeted workspace
/// from `caches`, and remove orphan stage directories with no
/// matching tombstone.
pub fn recover_root_staging(
    root: &Path,
    caches: &dashmap::DashMap<WorkspaceId, Arc<WorkspaceCacheCell>>,
) -> Result<RecoveryRootReport, RecoveryError> {
    let staging_dir = root.join(".tmp");
    if !staging_dir.is_dir() {
        return Ok(RecoveryRootReport::default());
    }
    let mut report = RecoveryRootReport::default();
    let (completed, orphan_swept) = drain_workspace_staging_dir(&staging_dir, |ws_id| {
        caches.remove(&ws_id);
    })?;
    report.workspace_tombstones_completed += completed;
    report.workspace_stage_orphans_swept += orphan_swept;
    Ok(report)
}

/// Discriminator for [`drain_staging_dir`] / [`drain_workspace_staging_dir`].
/// Drives which tombstone prefix the sweep considers and whether
/// the cache-eviction hook fires.
#[derive(Clone, Copy)]
enum StagingScope {
    /// Workspace-scoped `.tmp/`: `delete-assets-*` files (dataset
    /// deletes).  Workspace-delete tombstones at this level
    /// would be a layout bug (workspace-deletes live at the
    /// root staging dir).
    Dataset,
    /// Workspace-scoped `.tmp/`: `delete-converters-*` files for
    /// converter-tree deletes (mirror of the `delete-assets-*`
    /// prefix used by dataset deletes).
    Converter,
    /// Workspace-scoped `.tmp/`: `delete-training-logs-*` files
    /// for training-logs async wipes.
    TrainingLogs,
    /// Workspace-scoped `.tmp/`: `delete-converter-logs-*` files
    /// for converter-logs async wipes.
    ConverterLogs,
}

/// Walk `staging_dir` for tombstones matching the given scope's
/// filename prefix + their orphan directories.  Mirrors
/// [`drain_workspace_staging_dir`] for the workspace-scoped
/// surface and does not run the cache-eviction hook.  Returns
/// `(completed, orphan_swept)`.
fn drain_staging_dir(staging_dir: &Path, scope: StagingScope) -> Result<(usize, usize), FileError> {
    let prefix = match scope {
        StagingScope::Dataset => DATASET_TOMBSTONE_PREFIX,
        StagingScope::Converter => CONVERTER_TOMBSTONE_PREFIX,
        StagingScope::TrainingLogs => TRAINING_LOGS_TOMBSTONE_PREFIX,
        StagingScope::ConverterLogs => CONVERTER_LOGS_TOMBSTONE_PREFIX,
    };
    walk_staging(staging_dir, prefix, |_| {})
}

/// Workspace-delete variant: complete tombstones + sweep
/// orphans, calling `evict` with the `WorkspaceId` from each
/// completed tombstone so the daemon's cache map drops the
/// vacated workspace before any API consumer can observe it.
fn drain_workspace_staging_dir<F>(staging_dir: &Path, evict: F) -> Result<(usize, usize), FileError>
where
    F: FnMut(WorkspaceId),
{
    walk_staging(staging_dir, WORKSPACE_TOMBSTONE_PREFIX, evict)
}

/// Generic two-pass sweep:
///
/// 1. Read every entry under `staging_dir`; classify each as
///    "tombstone JSON file matching `prefix`" or "stage directory
///    matching `prefix`".
/// 2. Pair each tombstone with its stage directory; if the
///    tombstone parses + the stage directory exists, drain +
///    finalize via the staging primitives.  Bare tombstones
///    (no stage dir) are unlinked + the staging-dir parent
///    fsynced.
/// 3. Stage directories with no matching tombstone are removed
///    via `remove_dir_all`.
///
/// The `evict` closure fires after each successful tombstone
/// completion with the tombstone's `workspace_id` so root-staging
/// recovery can drop the cache cell.  Dataset-scope sweeps pass
/// a no-op closure.
fn walk_staging<F>(
    staging_dir: &Path,
    prefix: &str,
    mut evict: F,
) -> Result<(usize, usize), FileError>
where
    F: FnMut(WorkspaceId),
{
    let mut tombstone_files: Vec<PathBuf> = Vec::new();
    let mut stage_dirs: std::collections::HashMap<String, PathBuf> =
        std::collections::HashMap::new();
    let entries = std::fs::read_dir(staging_dir).map_err(|e| io_err(staging_dir.display(), e))?;
    for entry in entries {
        let entry = entry.map_err(|e| io_err(staging_dir.display(), e))?;
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let ft = match entry.file_type() {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!(
                    target: "file_mgr::recovery",
                    path = %path.display(),
                    err = %e,
                    "skipping unreadable staging entry",
                );
                continue;
            }
        };
        if ft.is_file() {
            // Only consider files matching `<prefix><id>.json`.
            if name.starts_with(prefix) && name.ends_with(".json") {
                tombstone_files.push(path);
            }
        } else if ft.is_dir() && name.starts_with(prefix) {
            // Stage dir; key by the bare directory name so the
            // tombstone pairing below can derive the same key
            // from `tombstone.stage_dir_name()`.
            stage_dirs.insert(name.to_string(), path);
        }
    }

    let mut completed = 0usize;
    let mut orphan_swept = 0usize;

    for tombstone_path in tombstone_files {
        let tombstone = match read_tombstone(&tombstone_path) {
            Ok(t) => t,
            Err(e) => {
                // Corrupt tombstone: best-effort unlink + log.
                tracing::warn!(
                    target: "file_mgr::recovery",
                    path = %tombstone_path.display(),
                    err = %e,
                    "removing corrupt tombstone",
                );
                let _ = std::fs::remove_file(&tombstone_path);
                continue;
            }
        };
        let staged = StagedDelete::for_tombstone(staging_dir, &tombstone);
        // Drive drain + finalize.  An idempotent finalize
        // handles the "no stage dir" case (tombstone-only) by
        // unlinking just the tombstone.  We mark the stage
        // directory as paired so the orphan sweep below ignores
        // it.
        stage_dirs.remove(&tombstone.stage_dir_name());
        // Drain in bounded batches; production usage caps via
        // `DEFAULT_DELETE_BATCH_ENTRIES` to bound stack +
        // syscall pressure.  Boot recovery has a much larger
        // budget envelope (the daemon is otherwise idle), but
        // keeping the batch shape lets the same primitives
        // service a future "resume mid-boot" cancellation
        // without a divergent code path.  Bound the loop with
        // a safety counter so a pathological staged tree
        // surfaces as a recovery error rather than an infinite
        // loop.
        let mut iters = 0usize;
        let drain_ok = loop {
            iters += 1;
            if iters > 1_000_000 {
                tracing::error!(
                    target: "file_mgr::recovery",
                    tombstone = %tombstone_path.display(),
                    "staged delete drain failed to converge after 1M iterations",
                );
                break false;
            }
            match drain_staged_payload(&staged, DEFAULT_DELETE_BATCH_ENTRIES) {
                Ok(DrainResult::Done) => break true,
                Ok(DrainResult::More) => continue,
                Err(e) => {
                    tracing::warn!(
                        target: "file_mgr::recovery",
                        tombstone = %tombstone_path.display(),
                        err = %e,
                        "staged delete drain failed; leaving for next boot",
                    );
                    break false;
                }
            }
        };
        if !drain_ok {
            continue;
        }
        if let Err(e) = finalize_staged_delete(&staged) {
            tracing::warn!(
                target: "file_mgr::recovery",
                tombstone = %tombstone_path.display(),
                err = %e,
                "staged delete finalize failed; leaving for next boot",
            );
            continue;
        }
        completed += 1;
        evict(tombstone.workspace_id());
    }

    // Orphan stage directories with no matching tombstone:
    // remove via `remove_dir_all`.  The tombstone-paired entries
    // were drained from the map above, so anything left here is
    // unreferenced residue.
    for (name, path) in stage_dirs {
        match std::fs::remove_dir_all(&path) {
            Ok(()) => {
                orphan_swept += 1;
                tracing::warn!(
                    target: "file_mgr::recovery",
                    path = %path.display(),
                    "removed orphan stage directory without tombstone",
                );
            }
            Err(e) => {
                tracing::warn!(
                    target: "file_mgr::recovery",
                    name = %name,
                    err = %e,
                    "orphan stage cleanup failed; leaving for next boot",
                );
            }
        }
    }

    if (completed > 0 || orphan_swept > 0)
        && let Err(e) = fsync_dir(staging_dir)
    {
        tracing::warn!(
            target: "file_mgr::recovery",
            err = %e,
            "fsync staging dir after sweep failed (best-effort)",
        );
    }

    Ok((completed, orphan_swept))
}

// MARK: streaming sha256 helper

/// Lowercase-hex SHA-256 of a file, read in
/// [`STREAM_HASH_CHUNK`]-sized chunks.  Mirrors the encoding the
/// active-head writer produces; an empty file hashes to the
/// well-known zero-length digest.  Caller treats a `NotFound`
/// like any other read failure (the on-disk generation is
/// considered failed).
fn sha256_stream(path: &Path) -> Result<String, FileError> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| io_err(path.display(), e))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; STREAM_HASH_CHUNK];
    loop {
        let n = f.read(&mut buf).map_err(|e| io_err(path.display(), e))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex_lowercase(&hasher.finalize()))
}

// `write_head_index` is reachable through the `pub use` in
// `file_mgr.rs`; the inner re-import here keeps the module file
// self-contained.

// MARK: Tests

#[cfg(test)]
mod tests {
    #![allow(clippy::disallowed_methods)]
    // Recovery fixtures intentionally create corrupt/orphaned files directly.

    use super::*;
    use crate::common::asset_path::AssetPath;
    use crate::common::ids::JobId;
    use crate::common::workspace::{
        HeadIndex, HeadManifest, HeadRecord, WorkspaceCore, WorkspaceRevision,
    };
    use crate::file_mgr::schema::{
        head_artifact_path, head_manifest_path, write_head_index, write_head_manifest,
        write_workspace_core,
    };
    use crate::file_mgr::staging::{DeleteTombstone, stage_payload, write_tombstone};
    use std::path::PathBuf;

    // MARK: shared fixtures

    fn ws_id(byte: u8) -> WorkspaceId {
        let s = format!("11111111-2222-4333-8444-5555555555{byte:02x}");
        WorkspaceId::parse(&s).unwrap()
    }

    fn head_id(byte: u8) -> HeadId {
        let s = format!("11111111-2222-4333-8444-5555555555{byte:02x}");
        HeadId::parse(&s).unwrap()
    }

    fn job_id() -> JobId {
        JobId::parse("22222222-3333-4444-8555-666666666666").unwrap()
    }

    fn rev(id: u64) -> WorkspaceRevision {
        WorkspaceRevision {
            id,
            at: "2026-05-07T12:00:00Z".to_string(),
        }
    }

    fn synth_workspace_core(id: WorkspaceId, head_count: u8) -> WorkspaceCore {
        WorkspaceCore {
            id,
            name: "main".to_string(),
            tags: Vec::new(),
            created_at: "2026-05-07T12:34:56Z".to_string(),
            workspace_revision: rev(5),
            head_count,
        }
    }

    fn synth_head_manifest(workspace: WorkspaceId, hid: HeadId, mpk: &[u8]) -> HeadManifest {
        HeadManifest {
            head_id: hid,
            workspace_id: workspace,
            workspace_revision: rev(5),
            sha256: hex_sha256(mpk),
            n_classes: 2,
            size_bytes: mpk.len() as u64,
            created_at: "2026-05-07T12:34:56Z".to_string(),
            labels: vec!["alpha".to_string(), "beta".to_string()],
        }
    }

    fn hex_sha256(bytes: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        let digest = hasher.finalize();
        static HEX: &[u8; 16] = b"0123456789abcdef";
        let mut out = vec![0u8; digest.len() * 2];
        for (i, &b) in digest.iter().enumerate() {
            out[2 * i] = HEX[(b >> 4) as usize];
            out[2 * i + 1] = HEX[(b & 0x0f) as usize];
        }
        String::from_utf8(out).unwrap()
    }

    /// Synthetic loader for tests; returns a `()` candidate so
    /// the bundled-default activation pipeline runs without the
    /// `inference` crate.
    fn synth_loader_ok() -> Box<HeadInnerLoader> {
        Box::new(|_mpk: &Path, _labels: &Path, _id: HeadId| {
            Ok(Box::new(()) as Box<dyn std::any::Any + Send>)
        })
    }

    fn fresh_bundled_default(root: &Path, mpk: &[u8], labels_text: &str) -> (PathBuf, PathBuf) {
        let dir = root.join("bundled_default");
        std::fs::create_dir_all(&dir).unwrap();
        let head = dir.join("head.mpk");
        let labels = dir.join("labels.txt");
        std::fs::write(&head, mpk).unwrap();
        std::fs::write(&labels, labels_text).unwrap();
        (head, labels)
    }

    fn default_source<'a>(path: &'a Path, labels_path: &'a Path) -> DefaultHeadSource<'a> {
        DefaultHeadSource { path, labels_path }
    }

    fn default_origin<'a>(path: &'a Path, labels_path: &'a Path) -> ActivationOriginInput<'a> {
        ActivationOriginInput::Default {
            source: default_source(path, labels_path),
        }
    }

    /// Stage one workspace dir with the canonical files + a
    /// single trained head.  Returns the workspace id used.
    fn fresh_workspace_with_head(root: &Path, ws: WorkspaceId, head: HeadId) -> PathBuf {
        let ws_dir = crate::file_mgr::schema::workspace_dir_for(root, &ws);
        std::fs::create_dir_all(crate::file_mgr::schema::heads_dir(&ws_dir)).unwrap();
        std::fs::create_dir_all(ws_dir.join(".tmp")).unwrap();
        let mpk = b"MPK-CONTENT";
        let manifest = synth_head_manifest(ws, head, mpk);
        let mut idx = HeadIndex::default();
        idx.heads.push(HeadRecord {
            head_id: head,
            workspace_revision: manifest.workspace_revision.clone(),
            sha256: manifest.sha256.clone(),
            n_classes: manifest.n_classes,
            size_bytes: manifest.size_bytes,
            created_at: manifest.created_at.clone(),
        });
        write_head_index(&ws_dir, &idx).unwrap();
        write_head_manifest(&ws_dir, &manifest).unwrap();
        std::fs::write(head_artifact_path(&ws_dir, head), mpk).unwrap();
        write_workspace_core(&ws_dir, &synth_workspace_core(ws, 1)).unwrap();
        ws_dir
    }

    /// Run a full bundled-default activation against `root` so
    /// the per-test setup has a known-good current generation
    /// before each test mutates it.
    fn seed_default_active_generation(root: &Path) -> (PathBuf, PathBuf, String) {
        let (head, labels) = fresh_bundled_default(root, b"DEFAULT-MPK", "cat\ndog\n");
        let pending = PendingActivation {
            root,
            origin_input: default_origin(&head, &labels),
            now_rfc3339: now_rfc3339(),
        };
        let result = stage_and_validate_activation(pending, &*synth_loader_ok()).unwrap();
        let staging = staging_path_for(root, &result.activation_id);
        publish_active_generation(root, &staging, &result.manifest, &result.activation_id).unwrap();
        (head, labels, result.activation_id)
    }

    // MARK: 1A active-head tests

    #[test]
    fn recover_active_current_passes() {
        let tmp = tempfile::tempdir().unwrap();
        let (head, labels, current_id) = seed_default_active_generation(tmp.path());
        let result = recover_active_head(
            tmp.path(),
            Some(default_source(&head, &labels)),
            &*synth_loader_ok(),
        )
        .unwrap();
        match result {
            RecoveryActiveResult::Current { activation_id, .. } => {
                assert_eq!(activation_id, current_id);
            }
            other => panic!("expected Current, got {other:?}"),
        }
    }

    #[test]
    fn recover_active_corrupt_head_falls_back_to_previous() {
        let tmp = tempfile::tempdir().unwrap();
        // Seed two generations: the first becomes "previous"
        // after the second activation rewrites `current.json`.
        let (head1, labels1) = fresh_bundled_default(tmp.path(), b"DEFAULT-MPK-A", "cat\ndog\n");
        let pending1 = PendingActivation {
            root: tmp.path(),
            origin_input: default_origin(&head1, &labels1),
            now_rfc3339: now_rfc3339(),
        };
        let r1 = stage_and_validate_activation(pending1, &*synth_loader_ok()).unwrap();
        publish_active_generation(
            tmp.path(),
            &staging_path_for(tmp.path(), &r1.activation_id),
            &r1.manifest,
            &r1.activation_id,
        )
        .unwrap();
        // Sleep a moment so the two generations have distinct mtimes.
        std::thread::sleep(std::time::Duration::from_millis(50));
        let (head2, labels2) =
            fresh_bundled_default(tmp.path(), b"DEFAULT-MPK-B", "cat\ndog\nbird\n");
        let pending2 = PendingActivation {
            root: tmp.path(),
            origin_input: default_origin(&head2, &labels2),
            now_rfc3339: now_rfc3339(),
        };
        let r2 = stage_and_validate_activation(pending2, &*synth_loader_ok()).unwrap();
        publish_active_generation(
            tmp.path(),
            &staging_path_for(tmp.path(), &r2.activation_id),
            &r2.manifest,
            &r2.activation_id,
        )
        .unwrap();
        // Tamper the CURRENT generation's head.mpk so verify
        // fails; the previous generation should be promoted.
        let current_head =
            active_generation_dir(tmp.path(), &r2.activation_id).join(ACTIVE_HEAD_FILENAME);
        std::fs::write(&current_head, b"TAMPERED").unwrap();
        // bundled2 still exists for the test; the recovery
        // should NOT need to fall back to it because r1 is
        // valid.
        let result = recover_active_head(
            tmp.path(),
            Some(default_source(&head2, &labels2)),
            &*synth_loader_ok(),
        )
        .unwrap();
        match result {
            RecoveryActiveResult::PromotedPrevious { activation_id, .. } => {
                assert_eq!(activation_id, r1.activation_id);
            }
            other => panic!("expected PromotedPrevious, got {other:?}"),
        }
        // current.json points at the promoted previous
        // generation now.
        let pointer = read_active_current(tmp.path()).unwrap();
        assert_eq!(pointer.activation_id, r1.activation_id);
    }

    #[test]
    fn recover_active_corrupt_labels_regenerates_from_manifest() {
        let tmp = tempfile::tempdir().unwrap();
        let (head, labels, current_id) = seed_default_active_generation(tmp.path());
        // Tamper labels.txt so its hash mismatches; recovery
        // should regenerate from manifest.labels[].
        let labels_path =
            active_generation_dir(tmp.path(), &current_id).join(ACTIVE_LABELS_FILENAME);
        std::fs::write(&labels_path, b"TAMPERED LABELS").unwrap();
        let result = recover_active_head(
            tmp.path(),
            Some(default_source(&head, &labels)),
            &*synth_loader_ok(),
        )
        .unwrap();
        // The current generation passes after labels regen.
        match result {
            RecoveryActiveResult::Current { activation_id, .. } => {
                assert_eq!(activation_id, current_id);
            }
            other => panic!("expected Current after labels regen, got {other:?}"),
        }
        // labels.txt now holds the manifest's canonical bytes.
        let regen = std::fs::read_to_string(&labels_path).unwrap();
        // Bundled-default fixture produced labels ["cat", "dog"].
        assert_eq!(regen, "cat\ndog");
    }

    #[test]
    fn recover_active_no_valid_generation_falls_back_to_default() {
        let tmp = tempfile::tempdir().unwrap();
        let (head, labels, current_id) = seed_default_active_generation(tmp.path());
        // Tamper the only generation's head.mpk; no previous
        // generation exists; recovery activates the bundled
        // default afresh.
        let head_path = active_generation_dir(tmp.path(), &current_id).join(ACTIVE_HEAD_FILENAME);
        std::fs::write(&head_path, b"TAMPERED").unwrap();
        let result = recover_active_head(
            tmp.path(),
            Some(default_source(&head, &labels)),
            &*synth_loader_ok(),
        )
        .unwrap();
        match result {
            RecoveryActiveResult::DefaultedFromBundle { activation_id, .. } => {
                // The new id is freshly minted, distinct from the
                // tampered generation.
                assert_ne!(activation_id, current_id);
            }
            other => panic!("expected DefaultedFromBundle, got {other:?}"),
        }
    }

    #[test]
    fn recover_active_bundled_default_missing_returns_unhealthy() {
        let tmp = tempfile::tempdir().unwrap();
        // No `current.json` and the bundled fixture path doesn't exist.
        let missing_head = tmp.path().join("does_not_exist.mpk");
        let missing_labels = tmp.path().join("does_not_exist.labels.txt");
        let result = recover_active_head(
            tmp.path(),
            Some(default_source(&missing_head, &missing_labels)),
            &*synth_loader_ok(),
        )
        .unwrap();
        assert!(matches!(result, RecoveryActiveResult::Unhealthy { .. }));
    }

    /// `default_head: None` skips the bundled-default activation
    /// step and surfaces `Unhealthy` directly, but root staging /
    /// per-workspace recovery still ran via `recover_all`.
    #[test]
    fn recover_active_no_default_head_returns_unhealthy() {
        let tmp = tempfile::tempdir().unwrap();
        let result = recover_active_head(tmp.path(), None, &*synth_loader_ok()).unwrap();
        match result {
            RecoveryActiveResult::Unhealthy { reason } => {
                assert!(
                    reason.contains("head.default not configured"),
                    "diagnostic should surface the root cause: {reason}",
                );
            }
            other => panic!("expected Unhealthy with head.default reason, got {other:?}"),
        }
    }

    // MARK: 1B per-workspace tests

    #[test]
    fn recover_workspace_orphan_head_files_swept() {
        let tmp = tempfile::tempdir().unwrap();
        // Create the workspaces root explicitly; the recovery
        // walker filters by `<root>/workspaces/`.
        std::fs::create_dir_all(workspaces_dir(tmp.path())).unwrap();
        let ws = ws_id(0xAA);
        let head = head_id(0xBB);
        let ws_dir = fresh_workspace_with_head(tmp.path(), ws, head);
        // Drop two unreferenced files into heads/.
        let orphan_id = head_id(0xCC);
        let orphan_mpk = head_artifact_path(&ws_dir, orphan_id);
        let orphan_json = head_manifest_path(&ws_dir, orphan_id);
        std::fs::write(&orphan_mpk, b"ORPHAN").unwrap();
        std::fs::write(&orphan_json, b"{}").unwrap();
        let report = recover_workspaces(tmp.path()).unwrap();
        assert_eq!(report.workspaces_scanned, 1);
        // Two files removed (mpk + json).
        assert_eq!(report.head_orphans_swept, 2);
        // The legitimate head's files survive.
        assert!(head_artifact_path(&ws_dir, head).is_file());
        assert!(head_manifest_path(&ws_dir, head).is_file());
        assert!(!orphan_mpk.exists());
        assert!(!orphan_json.exists());
    }

    #[test]
    fn recover_workspace_head_count_repaired() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(workspaces_dir(tmp.path())).unwrap();
        let ws = ws_id(0xAA);
        let head = head_id(0xBB);
        let ws_dir = fresh_workspace_with_head(tmp.path(), ws, head);
        // Tamper workspace.json to claim head_count=0 even
        // though heads.json has 1 entry.
        let mut core = read_workspace_core(&ws_dir).unwrap();
        core.head_count = 0;
        write_workspace_core(&ws_dir, &core).unwrap();
        let report = recover_workspaces(tmp.path()).unwrap();
        assert_eq!(report.head_count_repaired, 1);
        let core = read_workspace_core(&ws_dir).unwrap();
        assert_eq!(core.head_count, 1);
    }

    #[test]
    fn recover_workspace_dataset_tombstone_completed() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(workspaces_dir(tmp.path())).unwrap();
        let ws = ws_id(0xAA);
        let head = head_id(0xBB);
        let ws_dir = fresh_workspace_with_head(tmp.path(), ws, head);
        // Stage a dataset-delete tombstone + payload that
        // recovery should drain.
        let staging = ws_dir.join(".tmp");
        let tombstone = DeleteTombstone::Dataset {
            job_id: job_id(),
            workspace_id: ws,
            path: Some(AssetPath::parse("audio/cat").unwrap()),
            workspace_revision_id: 6,
            created_at: now_rfc3339(),
        };
        let staged = write_tombstone(&staging, &tombstone).unwrap();
        // Pre-stage some payload bytes.
        let target = tmp.path().join("dataset.wav");
        std::fs::write(&target, b"data").unwrap();
        stage_payload(&target, &staged).unwrap();
        let report = recover_workspaces(tmp.path()).unwrap();
        assert_eq!(report.dataset_tombstones_completed, 1);
        assert!(!staged.tombstone.exists());
        assert!(!staged.stage_dir.exists());
    }

    #[test]
    fn recover_workspace_dataset_stage_orphan_without_tombstone_swept() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(workspaces_dir(tmp.path())).unwrap();
        let ws = ws_id(0xAA);
        let head = head_id(0xBB);
        let ws_dir = fresh_workspace_with_head(tmp.path(), ws, head);
        // Synthesize an orphan stage directory with no tombstone.
        let staging = ws_dir.join(".tmp");
        std::fs::create_dir_all(&staging).unwrap();
        let orphan_dir = staging.join("delete-assets-aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee");
        std::fs::create_dir_all(orphan_dir.join("payload")).unwrap();
        std::fs::write(orphan_dir.join("payload/leftover"), b"x").unwrap();
        let report = recover_workspaces(tmp.path()).unwrap();
        assert_eq!(report.dataset_stage_orphans_swept, 1);
        assert!(!orphan_dir.exists());
    }

    #[test]
    fn recover_workspace_converter_tombstone_completed() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(workspaces_dir(tmp.path())).unwrap();
        let ws = ws_id(0xAA);
        let head = head_id(0xBB);
        let ws_dir = fresh_workspace_with_head(tmp.path(), ws, head);
        // Stage a converter-delete tombstone + payload that
        // recovery should drain.  Same shape as the dataset
        // tombstone with a distinct filename prefix.
        let staging = ws_dir.join(".tmp");
        let tombstone = DeleteTombstone::Converter {
            job_id: job_id(),
            workspace_id: ws,
            path: Some(AssetPath::parse("tfjs/model.json").unwrap()),
            workspace_revision_id: 6,
            created_at: now_rfc3339(),
        };
        let staged = write_tombstone(&staging, &tombstone).unwrap();
        let target = tmp.path().join("model.json");
        std::fs::write(&target, b"manifest-bytes").unwrap();
        stage_payload(&target, &staged).unwrap();
        let report = recover_workspaces(tmp.path()).unwrap();
        assert_eq!(report.converter_tombstones_completed, 1);
        // Dataset counter unchanged: the prefix-dispatch keeps
        // the two trees independent.
        assert_eq!(report.dataset_tombstones_completed, 0);
        assert!(!staged.tombstone.exists());
        assert!(!staged.stage_dir.exists());
    }

    #[test]
    fn recover_workspace_converter_stage_orphan_without_tombstone_swept() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(workspaces_dir(tmp.path())).unwrap();
        let ws = ws_id(0xAA);
        let head = head_id(0xBB);
        let ws_dir = fresh_workspace_with_head(tmp.path(), ws, head);
        let staging = ws_dir.join(".tmp");
        std::fs::create_dir_all(&staging).unwrap();
        let orphan_dir = staging.join("delete-converters-cccccccc-dddd-4eee-8fff-aaaaaaaaaaaa");
        std::fs::create_dir_all(orphan_dir.join("payload")).unwrap();
        std::fs::write(orphan_dir.join("payload/leftover"), b"x").unwrap();
        let report = recover_workspaces(tmp.path()).unwrap();
        assert_eq!(report.converter_stage_orphans_swept, 1);
        assert!(!orphan_dir.exists());
    }

    #[test]
    fn recover_workspace_training_logs_tombstone_completed() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(workspaces_dir(tmp.path())).unwrap();
        let ws = ws_id(0xAA);
        let head = head_id(0xBB);
        let ws_dir = fresh_workspace_with_head(tmp.path(), ws, head);
        // Stage a training-logs-delete tombstone + payload that
        // recovery should drain.  Logs aren't workspace state, so
        // the tombstone variant carries no `workspace_revision_id`
        // (unlike Dataset/Converter); recovery just drains.
        let staging = ws_dir.join(".tmp");
        let tombstone = DeleteTombstone::TrainingLogs {
            job_id: job_id(),
            workspace_id: ws,
            path: Some(AssetPath::parse("aaaaaaaa-1111-4222-8333-444444444444.jsonl").unwrap()),
            created_at: now_rfc3339(),
        };
        let staged = write_tombstone(&staging, &tombstone).unwrap();
        let target = tmp.path().join("training_log.jsonl");
        std::fs::write(&target, b"{\"seq\":1}\n").unwrap();
        stage_payload(&target, &staged).unwrap();
        let report = recover_workspaces(tmp.path()).unwrap();
        assert_eq!(report.training_logs_tombstones_completed, 1);
        // Other tree counters unchanged: prefix-dispatch keeps
        // each tree's recovery independent.
        assert_eq!(report.dataset_tombstones_completed, 0);
        assert_eq!(report.converter_tombstones_completed, 0);
        assert_eq!(report.converter_logs_tombstones_completed, 0);
        assert!(!staged.tombstone.exists());
        assert!(!staged.stage_dir.exists());
    }

    #[test]
    fn recover_workspace_training_logs_stage_orphan_without_tombstone_swept() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(workspaces_dir(tmp.path())).unwrap();
        let ws = ws_id(0xAA);
        let head = head_id(0xBB);
        let ws_dir = fresh_workspace_with_head(tmp.path(), ws, head);
        let staging = ws_dir.join(".tmp");
        std::fs::create_dir_all(&staging).unwrap();
        let orphan_dir = staging.join("delete-training-logs-11111111-2222-4333-8444-555555555555");
        std::fs::create_dir_all(orphan_dir.join("payload")).unwrap();
        std::fs::write(orphan_dir.join("payload/leftover.jsonl"), b"{}").unwrap();
        let report = recover_workspaces(tmp.path()).unwrap();
        assert_eq!(report.training_logs_stage_orphans_swept, 1);
        assert!(!orphan_dir.exists());
    }

    #[test]
    fn recover_workspace_converter_logs_tombstone_completed() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(workspaces_dir(tmp.path())).unwrap();
        let ws = ws_id(0xAA);
        let head = head_id(0xBB);
        let ws_dir = fresh_workspace_with_head(tmp.path(), ws, head);
        // Whole-tree wipe shape: `path = None` records the
        // bare-tree wipe; recovery drains and finalizes the same
        // way.
        let staging = ws_dir.join(".tmp");
        let tombstone = DeleteTombstone::ConverterLogs {
            job_id: job_id(),
            workspace_id: ws,
            path: None,
            created_at: now_rfc3339(),
        };
        let staged = write_tombstone(&staging, &tombstone).unwrap();
        let target = tmp.path().join("converter_logs_tree");
        std::fs::create_dir_all(&target).unwrap();
        std::fs::write(target.join("a.jsonl"), b"{}").unwrap();
        std::fs::write(target.join("b.jsonl"), b"{}").unwrap();
        stage_payload(&target, &staged).unwrap();
        let report = recover_workspaces(tmp.path()).unwrap();
        assert_eq!(report.converter_logs_tombstones_completed, 1);
        // Sibling tree counters unchanged.
        assert_eq!(report.training_logs_tombstones_completed, 0);
        assert_eq!(report.dataset_tombstones_completed, 0);
        assert_eq!(report.converter_tombstones_completed, 0);
        assert!(!staged.tombstone.exists());
        assert!(!staged.stage_dir.exists());
    }

    #[test]
    fn recover_workspace_converter_logs_stage_orphan_without_tombstone_swept() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(workspaces_dir(tmp.path())).unwrap();
        let ws = ws_id(0xAA);
        let head = head_id(0xBB);
        let ws_dir = fresh_workspace_with_head(tmp.path(), ws, head);
        let staging = ws_dir.join(".tmp");
        std::fs::create_dir_all(&staging).unwrap();
        let orphan_dir = staging.join("delete-converter-logs-22222222-3333-4444-8555-666666666666");
        std::fs::create_dir_all(orphan_dir.join("payload")).unwrap();
        std::fs::write(orphan_dir.join("payload/x.jsonl"), b"{}").unwrap();
        let report = recover_workspaces(tmp.path()).unwrap();
        assert_eq!(report.converter_logs_stage_orphans_swept, 1);
        assert!(!orphan_dir.exists());
    }

    #[test]
    fn recover_incomplete_workspace_create_directory_removed() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(workspaces_dir(tmp.path())).unwrap();
        // A directory under workspaces/ without `workspace.json`
        // is an incomplete create; recovery removes it.
        let ws = ws_id(0xAA);
        let ws_dir = crate::file_mgr::schema::workspace_dir_for(tmp.path(), &ws);
        std::fs::create_dir_all(ws_dir.join("heads")).unwrap();
        std::fs::create_dir_all(ws_dir.join(".tmp")).unwrap();
        // No workspace.json.
        let report = recover_workspaces(tmp.path()).unwrap();
        assert_eq!(report.incomplete_creates_removed, 1);
        assert!(!ws_dir.exists());
        // The incomplete dir is NOT counted as scanned.
        assert_eq!(report.workspaces_scanned, 0);
    }

    // MARK: 1C root-staging tests

    #[test]
    fn recover_root_workspace_tombstone_completed() {
        let tmp = tempfile::tempdir().unwrap();
        let staging = tmp.path().join(".tmp");
        std::fs::create_dir_all(&staging).unwrap();
        let ws = ws_id(0xAA);
        let tombstone = DeleteTombstone::Workspace {
            job_id: job_id(),
            workspace_id: ws,
            created_at: now_rfc3339(),
        };
        let staged = write_tombstone(&staging, &tombstone).unwrap();
        // Pre-stage a workspace-shaped payload.
        let target = tmp.path().join("victim_ws");
        std::fs::create_dir_all(target.join("heads")).unwrap();
        std::fs::write(target.join("workspace.json"), b"{}").unwrap();
        stage_payload(&target, &staged).unwrap();
        // Cache cell must be evicted on completion.
        let caches: dashmap::DashMap<WorkspaceId, Arc<WorkspaceCacheCell>> =
            dashmap::DashMap::new();
        caches.insert(
            ws,
            Arc::new(WorkspaceCacheCell::new(
                synth_workspace_core(ws, 0),
                HeadIndex::default(),
            )),
        );
        let report = recover_root_staging(tmp.path(), &caches).unwrap();
        assert_eq!(report.workspace_tombstones_completed, 1);
        assert!(!staged.tombstone.exists());
        assert!(caches.get(&ws).is_none(), "cache cell evicted");
    }

    #[test]
    fn recover_root_workspace_stage_orphan_swept() {
        let tmp = tempfile::tempdir().unwrap();
        let staging = tmp.path().join(".tmp");
        std::fs::create_dir_all(&staging).unwrap();
        // Orphan workspace stage dir with no tombstone.
        let orphan = staging.join("delete-workspace-aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee");
        std::fs::create_dir_all(orphan.join("payload/heads")).unwrap();
        std::fs::write(orphan.join("payload/workspace.json"), b"{}").unwrap();
        let caches: dashmap::DashMap<WorkspaceId, Arc<WorkspaceCacheCell>> =
            dashmap::DashMap::new();
        let report = recover_root_staging(tmp.path(), &caches).unwrap();
        assert_eq!(report.workspace_stage_orphans_swept, 1);
        assert!(!orphan.exists());
    }
}
