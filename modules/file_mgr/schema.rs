//! On-disk schema persistence: byte-level read / write helpers
//! for `workspace.json`, `heads.json`, and per-head
//! `<head_id>.json`.  The shapes themselves live in
//! `common::workspace`; this module owns the filesystem-side
//! discipline (path layout, size cap, atomic rewrite).
//!
//! No locking happens here -- callers hold the per-workspace
//! mutation mutex across read-modify-write cycles.  All writes go
//! through [`crate::file_mgr::fs_atomic::put_atomic`] so a partial
//! write never appears under the published name.
//!
//! # Path layout
//!
//! ```text
//! <workspace_dir>/
//!     workspace.json                        -- WorkspaceCore
//!     heads.json                            -- HeadIndex (<= 2 entries)
//!     heads/
//!         <head_id>.mpk                     -- raw weights
//!         <head_id>.json                    -- HeadManifest
//! ```
//!
//! The `<workspace_dir>` argument is caller-supplied; helpers only
//! join the canonical filenames and do not assume a particular
//! prefix shape.

use crate::common::ids::HeadId;
use crate::common::workspace::{HeadIndex, HeadManifest, MAX_HEADS_PER_WORKSPACE, WorkspaceCore};
use crate::file_mgr::error::{FileError, io_err, metadata_parse_err};
use crate::file_mgr::fs_atomic::put_atomic;
use std::path::{Path, PathBuf};

/// Filename of the per-workspace hot core under `<workspace_dir>/`.
pub const WORKSPACE_CORE_FILENAME: &str = "workspace.json";
/// Filename of the per-workspace head index.
pub const HEAD_INDEX_FILENAME: &str = "heads.json";
/// Subdirectory holding `<head_id>.{mpk,json}` per published head.
pub const HEADS_DIR_NAME: &str = "heads";
/// File extension for raw head weights (Burn `.mpk` artefact).
pub const HEAD_ARTIFACT_EXTENSION: &str = "mpk";
/// File extension for per-head JSON manifest (`HeadManifest`).
pub const HEAD_MANIFEST_EXTENSION: &str = "json";
/// Hard cap on `workspace.json` byte size.  Defends against
/// operator tampering: a 64-MiB-name attack would otherwise blow
/// up the eager-cache resident set; today's nominal core is < 1
/// KiB.
pub const MAX_WORKSPACE_CORE_BYTES: u64 = 64 * 1024;

/// Top-level subdirectory holding every workspace under the
/// daemon's `WORKSPACE_ROOT` (`<root>/workspaces/<workspace_id>/`).
pub const WORKSPACES_DIR_NAME: &str = "workspaces";
/// Top-level subdirectory holding root-scoped staged delete payloads
/// (workspace deletes drop their tombstone + payload here so boot
/// recovery can resume the drain).
pub const ROOT_TMP_DIR_NAME: &str = ".tmp";
/// Top-level subdirectory for the active-head generation tree.
pub const ACTIVE_DIR_NAME: &str = "active";
/// Top-level subdirectory for the deployment-bundled backbone.
pub const BACKBONE_DIR_NAME: &str = "backbone";
/// Subdirectory under `<root>/active/` holding retained generations.
pub const ACTIVE_GENERATIONS_DIR_NAME: &str = "generations";
/// Subdirectory under `<root>/active/` for activation staging.
pub const ACTIVE_TMP_DIR_NAME: &str = ".tmp";
/// Filename of the active pointer JSON (`{ "activation_id": "..." }`).
pub const ACTIVE_CURRENT_FILENAME: &str = "current.json";
/// Filename of the per-generation active-head manifest
/// (`ActiveHeadManifest` body).
pub const ACTIVE_MANIFEST_FILENAME: &str = "manifest.json";
/// Filename of the materialized active-head weights inside a
/// generation directory.
pub const ACTIVE_HEAD_FILENAME: &str = "head.mpk";
/// Filename of the materialized active-head label list inside a
/// generation directory.
pub const ACTIVE_LABELS_FILENAME: &str = "labels.txt";

// MARK: Path helpers

/// `<workspace_dir>/workspace.json`.
#[inline]
pub fn workspace_core_path(workspace_dir: &Path) -> PathBuf {
    workspace_dir.join(WORKSPACE_CORE_FILENAME)
}

/// `<workspace_dir>/heads.json`.
#[inline]
pub fn head_index_path(workspace_dir: &Path) -> PathBuf {
    workspace_dir.join(HEAD_INDEX_FILENAME)
}

/// `<workspace_dir>/heads/`.
#[inline]
pub fn heads_dir(workspace_dir: &Path) -> PathBuf {
    workspace_dir.join(HEADS_DIR_NAME)
}

/// `<workspace_dir>/heads/<head_id>.json`.
#[inline]
pub fn head_manifest_path(workspace_dir: &Path, head_id: HeadId) -> PathBuf {
    heads_dir(workspace_dir).join(format!("{head_id}.{HEAD_MANIFEST_EXTENSION}"))
}

/// `<workspace_dir>/heads/<head_id>.mpk`.
#[inline]
pub fn head_artifact_path(workspace_dir: &Path, head_id: HeadId) -> PathBuf {
    heads_dir(workspace_dir).join(format!("{head_id}.{HEAD_ARTIFACT_EXTENSION}"))
}

// MARK: Root-scoped path helpers

/// `<root>/workspaces/`.  Top-level holder for every workspace dir.
#[inline]
pub fn workspaces_dir(root: &Path) -> PathBuf {
    root.join(WORKSPACES_DIR_NAME)
}

/// `<root>/workspaces/<workspace_id>/`.  Per-workspace directory
/// under the new layout; replaces the legacy `<root>/<id>/` shape.
#[inline]
pub fn workspace_dir_for(root: &Path, id: &crate::common::ids::WorkspaceId) -> PathBuf {
    workspaces_dir(root).join(id.to_string())
}

/// `<root>/.tmp/`.  Root staging area for asynchronous workspace
/// deletes.
#[inline]
pub fn root_tmp_dir(root: &Path) -> PathBuf {
    root.join(ROOT_TMP_DIR_NAME)
}

/// `<root>/active/`.  Holds the active-head generation tree.
#[inline]
pub fn active_dir(root: &Path) -> PathBuf {
    root.join(ACTIVE_DIR_NAME)
}

/// `<root>/active/current.json`.  Atomic pointer at the live
/// generation; absent on a daemon that has never activated.
#[inline]
pub fn active_current_path(root: &Path) -> PathBuf {
    active_dir(root).join(ACTIVE_CURRENT_FILENAME)
}

/// `<root>/active/generations/`.  Retains current + previous
/// generation; older generations are pruned by the active-head
/// writer once `current.json` is durable.
#[inline]
pub fn active_generations_dir(root: &Path) -> PathBuf {
    active_dir(root).join(ACTIVE_GENERATIONS_DIR_NAME)
}

/// `<root>/active/generations/<activation_id>/`.  One generation
/// directory: holds `head.mpk`, `labels.txt`, `manifest.json`.
#[inline]
pub fn active_generation_dir(root: &Path, activation_id: &str) -> PathBuf {
    active_generations_dir(root).join(activation_id)
}

/// `<root>/active/.tmp/`.  Staging directory for pre-publish
/// activation generations.
#[inline]
pub fn active_staging_dir(root: &Path) -> PathBuf {
    active_dir(root).join(ACTIVE_TMP_DIR_NAME)
}

/// `<root>/backbone/`.  Deployment-managed backbone artefacts.
#[inline]
pub fn backbone_dir(root: &Path) -> PathBuf {
    root.join(BACKBONE_DIR_NAME)
}

// MARK: ActiveCurrentPointer (active/current.json)

/// On-disk shape of `<root>/active/current.json`: one field
/// pointing at the live generation directory under
/// `active/generations/<activation_id>/`.  `deny_unknown_fields`
/// fails closed on a hand-edited or future-shape file.
///
/// `activation_id` is the directory name only; not parsed as a
/// `HeadId` because the operator-facing fixture path
/// `misc/heads/00000000-default/` is a directory name, never a
/// `HeadId`.  The string passes
/// [`validate_activation_id`] on deserialize so a hand-edited
/// `current.json` carrying `..` traversal cannot escape the
/// active root when the consumer joins the value into a path.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActiveCurrentPointer {
    /// Generation directory name under
    /// `<root>/active/generations/`.
    #[serde(deserialize_with = "deserialize_activation_id")]
    pub activation_id: String,
}

fn deserialize_activation_id<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    let s = String::deserialize(deserializer)?;
    validate_activation_id(&s).map_err(serde::de::Error::custom)?;
    Ok(s)
}

/// Reasons a string cannot be used as an `activation_id`.
/// Structured per failure mode so the deserialize / FS-path call
/// sites do not have to round-trip through a free-form `String`,
/// and so future consumers (operator-facing error catalogs, golden
/// snapshot tests) can match on the variant rather than the
/// English message.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ActivationIdError {
    #[error("activation_id is empty")]
    Empty,
    #[error("activation_id length {len} > 64")]
    TooLong { len: usize },
    #[error("activation_id must not start with '.'")]
    LeadingDot,
    #[error(
        "activation_id byte 0x{byte:02x} at index {index} is forbidden \
         (allowed: [A-Za-z0-9._-])"
    )]
    BadByte { index: usize, byte: u8 },
}

/// Reject hand-edited `current.json` values that would escape
/// the active root when joined into a filesystem path.  Allows
/// the same byte set as a single `AssetPath` component
/// (`[A-Za-z0-9._-]`, no leading `.`, non-empty, length <= 64);
/// rejects path separators, backslashes, NUL, control bytes,
/// non-ASCII, and any leading-dot variant.
pub fn validate_activation_id(s: &str) -> Result<(), ActivationIdError> {
    if s.is_empty() {
        return Err(ActivationIdError::Empty);
    }
    if s.len() > 64 {
        return Err(ActivationIdError::TooLong { len: s.len() });
    }
    if s.as_bytes()[0] == b'.' {
        return Err(ActivationIdError::LeadingDot);
    }
    for (i, &b) in s.as_bytes().iter().enumerate() {
        let ok = b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b'_');
        if !ok {
            return Err(ActivationIdError::BadByte { index: i, byte: b });
        }
    }
    Ok(())
}

/// Read and parse `<root>/active/current.json`.  Surfaces a
/// missing pointer as `FileError::Io` (kind `NotFound`); callers
/// (boot recovery) handle that distinct from a corrupt JSON file.
pub fn read_active_current(root: &Path) -> Result<ActiveCurrentPointer, FileError> {
    let path = active_current_path(root);
    let bytes = std::fs::read(&path).map_err(|source| io_err(path.display(), source))?;
    serde_json::from_slice(&bytes).map_err(|source| metadata_parse_err(path.display(), source))
}

/// Atomically rewrite `<root>/active/current.json` so the active
/// pointer either reflects the prior state (rename failed before
/// completing) or the new generation (rename succeeded), never an
/// in-between.
pub fn write_active_current(root: &Path, pointer: &ActiveCurrentPointer) -> Result<(), FileError> {
    let path = active_current_path(root);
    let bytes = serde_json::to_vec(pointer)?;
    put_atomic(&path, &bytes)
}

/// Read and parse a generation's `manifest.json`.  Caller is
/// responsible for invoking
/// [`crate::common::workspace::ActiveHeadManifest::validate`] on
/// the result before treating it as trustworthy -- the structural
/// invariants serde cannot catch are owned by that method.
///
/// Validates `activation_id` against
/// [`validate_activation_id`] so a hostile or hand-edited caller
/// cannot pass a path-traversal token that escapes the active
/// root when joined into the final manifest path.
pub fn read_active_manifest(
    root: &Path,
    activation_id: &str,
) -> Result<crate::common::workspace::ActiveHeadManifest, FileError> {
    validate_activation_id(activation_id).map_err(|e| FileError::InvalidName(e.to_string()))?;
    let path = active_generation_dir(root, activation_id).join(ACTIVE_MANIFEST_FILENAME);
    let bytes = std::fs::read(&path).map_err(|source| io_err(path.display(), source))?;
    serde_json::from_slice(&bytes).map_err(|source| metadata_parse_err(path.display(), source))
}

/// Atomically rewrite a generation's `manifest.json`.  Validates
/// `activation_id` against [`validate_activation_id`].
pub fn write_active_manifest(
    root: &Path,
    activation_id: &str,
    manifest: &crate::common::workspace::ActiveHeadManifest,
) -> Result<(), FileError> {
    validate_activation_id(activation_id).map_err(|e| FileError::InvalidName(e.to_string()))?;
    let path = active_generation_dir(root, activation_id).join(ACTIVE_MANIFEST_FILENAME);
    let bytes = serde_json::to_vec(manifest)?;
    put_atomic(&path, &bytes)
}

// MARK: workspace.json

/// Read and parse `<workspace_dir>/workspace.json`.  Returns
/// `FileError::NotFound` shape via the underlying `Io` if the
/// file is absent (callers needing "absent vs corrupt"
/// distinction inspect the inner `io::ErrorKind`).  Caps the
/// in-memory read at [`MAX_WORKSPACE_CORE_BYTES`]; an oversize
/// file fails closed with [`FileError::MetadataTooLarge`].
pub fn read_workspace_core(workspace_dir: &Path) -> Result<WorkspaceCore, FileError> {
    let path = workspace_core_path(workspace_dir);
    let bytes = read_capped(&path, MAX_WORKSPACE_CORE_BYTES)?;
    serde_json::from_slice(&bytes).map_err(|source| metadata_parse_err(path.display(), source))
}

/// Atomically rewrite `<workspace_dir>/workspace.json`.  Refuses
/// to publish an oversize body (defends the hot-path cache
/// budget); on success the file's data + the parent directory
/// entry have both reached stable storage.
///
/// On the Ok arm, records one
/// [`crate::status::WorkspaceMetrics::record_workspace_core_write`]
/// sample with the wall-clock duration of the atomic rewrite
/// (size cap check + serialize + tempfile write + fsync +
/// rename + parent fsync).  Cap-failure / serialize-failure
/// paths do not record the sample because they short-circuit
/// before any disk activity.
pub fn write_workspace_core(workspace_dir: &Path, core: &WorkspaceCore) -> Result<(), FileError> {
    let path = workspace_core_path(workspace_dir);
    let bytes = serde_json::to_vec(core)?;
    if (bytes.len() as u64) > MAX_WORKSPACE_CORE_BYTES {
        return Err(FileError::MetadataTooLarge {
            path: path.display().to_string(),
            observed: bytes.len() as u64,
            max: MAX_WORKSPACE_CORE_BYTES,
        });
    }
    let start = std::time::Instant::now();
    put_atomic(&path, &bytes)?;
    crate::file_mgr::metrics_hooks::emit_workspace_core_write(start.elapsed());
    Ok(())
}

// MARK: heads.json

/// Read and parse `<workspace_dir>/heads.json`.  Enforces the
/// 2-entry sliding-window cap (`MAX_HEADS_PER_WORKSPACE`) on
/// read so a hand-edited or corrupt file with extra entries
/// fails closed rather than feeding stale heads into the
/// activation path.  A `deny_unknown_fields` parse failure
/// surfaces as `FileError::MetadataParse`; an over-cap entry
/// count surfaces as `FileError::MetadataParse` with a
/// synthesized parse error so callers can treat the two
/// shapes uniformly.
pub fn read_head_index(workspace_dir: &Path) -> Result<HeadIndex, FileError> {
    let path = head_index_path(workspace_dir);
    let bytes = std::fs::read(&path).map_err(|source| io_err(path.display(), source))?;
    let index: HeadIndex = serde_json::from_slice(&bytes)
        .map_err(|source| metadata_parse_err(path.display(), source))?;
    if index.heads.len() > MAX_HEADS_PER_WORKSPACE {
        return Err(metadata_parse_err(
            path.display(),
            serde_json::Error::io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "heads.json has {} entries; max is {}",
                    index.heads.len(),
                    MAX_HEADS_PER_WORKSPACE
                ),
            )),
        ));
    }
    Ok(index)
}

/// Atomically rewrite `<workspace_dir>/heads.json`.  Refuses to
/// publish more than `MAX_HEADS_PER_WORKSPACE` entries so the
/// writer-side cap is symmetric with the reader-side cap; the
/// sliding-window logic lives in `head_rotation`.
///
/// On the Ok arm, records one
/// [`crate::status::WorkspaceMetrics::record_head_index_write`]
/// sample with the wall-clock duration of the atomic rewrite.
pub fn write_head_index(workspace_dir: &Path, index: &HeadIndex) -> Result<(), FileError> {
    let path = head_index_path(workspace_dir);
    if index.heads.len() > MAX_HEADS_PER_WORKSPACE {
        return Err(metadata_parse_err(
            path.display(),
            serde_json::Error::io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "refusing to publish {} entries; max is {}",
                    index.heads.len(),
                    MAX_HEADS_PER_WORKSPACE
                ),
            )),
        ));
    }
    let bytes = serde_json::to_vec(index)?;
    let start = std::time::Instant::now();
    put_atomic(&path, &bytes)?;
    crate::file_mgr::metrics_hooks::emit_head_index_write(start.elapsed());
    Ok(())
}

// MARK: per-head <head_id>.json

/// Read and parse `<workspace_dir>/heads/<head_id>.json`.  Runs
/// [`HeadManifest::validate`] after parse so a hand-edited
/// manifest with mismatched `n_classes` / `labels.len()` fails
/// closed instead of flowing into activation.
pub fn read_head_manifest(
    workspace_dir: &Path,
    head_id: HeadId,
) -> Result<HeadManifest, FileError> {
    let path = head_manifest_path(workspace_dir, head_id);
    let bytes = std::fs::read(&path).map_err(|source| io_err(path.display(), source))?;
    let manifest: HeadManifest = serde_json::from_slice(&bytes)
        .map_err(|source| metadata_parse_err(path.display(), source))?;
    manifest.validate().map_err(|e| {
        metadata_parse_err(
            path.display(),
            serde_json::Error::io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("HeadManifest validation: {e}"),
            )),
        )
    })?;
    Ok(manifest)
}

/// Atomically rewrite a per-head manifest file.  Caller is
/// responsible for ensuring `manifest.head_id` matches the
/// filename derived here -- the manifest serializes its own
/// `head_id` field, but this helper trusts the argument and
/// does not cross-check.
pub fn write_head_manifest(workspace_dir: &Path, manifest: &HeadManifest) -> Result<(), FileError> {
    let path = head_manifest_path(workspace_dir, manifest.head_id);
    let bytes = serde_json::to_vec(manifest)?;
    put_atomic(&path, &bytes)
}

// MARK: Internal helpers

/// Read `path` into memory rejecting any file larger than
/// `cap` bytes.  Uses the file's `metadata().len()` for an
/// O(1) precheck so an attacker cannot trick us into
/// allocating a giant buffer before the cap fires.
fn read_capped(path: &Path, cap: u64) -> Result<Vec<u8>, FileError> {
    use std::io::Read;
    let f = std::fs::File::open(path).map_err(|source| io_err(path.display(), source))?;
    let metadata = f
        .metadata()
        .map_err(|source| io_err(path.display(), source))?;
    if metadata.len() > cap {
        return Err(FileError::MetadataTooLarge {
            path: path.display().to_string(),
            observed: metadata.len(),
            max: cap,
        });
    }
    // `take(cap)` is belt-and-suspenders -- if a torn write
    // expanded the file between the metadata stat and the
    // read, we still cap the buffer.
    let mut buf = Vec::with_capacity(metadata.len() as usize);
    let mut limited = f.take(cap);
    limited
        .read_to_end(&mut buf)
        .map_err(|source| io_err(path.display(), source))?;
    Ok(buf)
}

// MARK: Tests

#[cfg(test)]
mod tests {
    #![allow(clippy::disallowed_methods)]
    // Fixture corruption tests intentionally use raw writes.

    use super::*;
    use crate::common::ids::WorkspaceId;
    use crate::common::workspace::{
        HeadIndex, HeadManifest, HeadRecord, MAX_HEADS_PER_WORKSPACE, WorkspaceCore,
        WorkspaceRevision,
    };

    fn ws_id() -> WorkspaceId {
        WorkspaceId::parse("11111111-2222-4333-8444-555555555555").unwrap()
    }

    fn head_id() -> HeadId {
        HeadId::parse("11111111-2222-4333-8444-555555555556").unwrap()
    }

    fn rev(id: u64) -> WorkspaceRevision {
        WorkspaceRevision {
            id,
            at: "2026-05-07T12:00:00Z".to_string(),
        }
    }

    fn sample_core() -> WorkspaceCore {
        WorkspaceCore {
            id: ws_id(),
            name: "main".to_string(),
            tags: Vec::new(),
            created_at: "2026-05-07T12:34:56Z".to_string(),
            workspace_revision: rev(5),
            head_count: 1,
        }
    }

    fn sample_head_record() -> HeadRecord {
        HeadRecord {
            head_id: head_id(),
            workspace_revision: rev(5),
            sha256: "def".to_string(),
            n_classes: 12,
            size_bytes: 4096,
            created_at: "2026-05-07T12:34:56Z".to_string(),
        }
    }

    fn sample_manifest() -> HeadManifest {
        HeadManifest {
            head_id: head_id(),
            workspace_id: ws_id(),
            workspace_revision: rev(5),
            sha256: "def".to_string(),
            n_classes: 2,
            size_bytes: 4096,
            created_at: "2026-05-07T12:34:56Z".to_string(),
            labels: vec!["cat".to_string(), "dog".to_string()],
        }
    }

    // MARK: Path helpers

    #[test]
    fn path_helpers_join_redesign_filenames() {
        let ws = Path::new("/tmp/ws");
        assert_eq!(workspace_core_path(ws), Path::new("/tmp/ws/workspace.json"));
        assert_eq!(head_index_path(ws), Path::new("/tmp/ws/heads.json"));
        assert_eq!(heads_dir(ws), Path::new("/tmp/ws/heads"));
        let id = head_id();
        assert_eq!(
            head_manifest_path(ws, id),
            Path::new("/tmp/ws/heads/11111111-2222-4333-8444-555555555556.json")
        );
        assert_eq!(
            head_artifact_path(ws, id),
            Path::new("/tmp/ws/heads/11111111-2222-4333-8444-555555555556.mpk")
        );
    }

    // MARK: WorkspaceCore round-trip

    #[test]
    fn workspace_core_write_then_read_round_trips() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let core = sample_core();
        write_workspace_core(ws, &core).unwrap();
        // The file is parented under the supplied workspace_dir
        // (no `workspaces/` prefix; the registry layer adds that)
        // -- pinned because path-helper changes ripple into every
        // downstream caller.
        assert!(ws.join("workspace.json").is_file());
        let read = read_workspace_core(ws).unwrap();
        assert_eq!(read, core);
    }

    #[test]
    fn workspace_core_read_missing_file_surfaces_io_not_found() {
        let tmp = tempfile::tempdir().unwrap();
        let res = read_workspace_core(tmp.path());
        match res {
            Err(FileError::Io { source, .. }) => {
                assert_eq!(source.kind(), std::io::ErrorKind::NotFound);
            }
            other => panic!("expected FileError::Io NotFound, got {other:?}"),
        }
    }

    /// Pinned: corrupt JSON returns `MetadataParse` (not `Io`)
    /// so callers can distinguish missing from malformed.
    #[test]
    fn workspace_core_read_rejects_corrupt_json() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join(WORKSPACE_CORE_FILENAME), b"{ not json").unwrap();
        let res = read_workspace_core(tmp.path());
        assert!(matches!(res, Err(FileError::MetadataParse { .. })));
    }

    /// Pinned: the 64 KiB cap is enforced on read.  Mostly
    /// defensive (the writer enforces it too); catches operator
    /// tampering.
    #[test]
    fn workspace_core_read_rejects_oversize_file() {
        let tmp = tempfile::tempdir().unwrap();
        let big = vec![b'a'; (MAX_WORKSPACE_CORE_BYTES + 1) as usize];
        std::fs::write(tmp.path().join(WORKSPACE_CORE_FILENAME), &big).unwrap();
        let res = read_workspace_core(tmp.path());
        assert!(matches!(
            res,
            Err(FileError::MetadataTooLarge { observed, max, .. })
                if observed == MAX_WORKSPACE_CORE_BYTES + 1 && max == MAX_WORKSPACE_CORE_BYTES
        ));
    }

    /// Writer-side cap: refuses to publish a body that would
    /// blow up downstream readers.  Today's `WorkspaceCore`
    /// fits in < 1 KiB; this test injects a pathological name
    /// to verify the cap fires before disk write.
    #[test]
    fn workspace_core_write_rejects_oversize_body() {
        let tmp = tempfile::tempdir().unwrap();
        let mut core = sample_core();
        core.name = "x".repeat(MAX_WORKSPACE_CORE_BYTES as usize);
        let res = write_workspace_core(tmp.path(), &core);
        assert!(matches!(res, Err(FileError::MetadataTooLarge { .. })));
        // No file produced.
        assert!(!tmp.path().join(WORKSPACE_CORE_FILENAME).exists());
    }

    /// `put_atomic` semantics flow through: the temp + rename
    /// guarantees the file appears wholly-new, never partial.
    #[test]
    fn workspace_core_write_is_atomic_via_put_atomic() {
        let tmp = tempfile::tempdir().unwrap();
        // First write.
        let core = sample_core();
        write_workspace_core(tmp.path(), &core).unwrap();
        // Second write replaces atomically.
        let mut core2 = core.clone();
        core2.workspace_revision = rev(6);
        write_workspace_core(tmp.path(), &core2).unwrap();
        let read = read_workspace_core(tmp.path()).unwrap();
        assert_eq!(read.workspace_revision, rev(6));
    }

    // MARK: HeadIndex round-trip

    #[test]
    fn head_index_round_trip_with_capped_two_entries() {
        let tmp = tempfile::tempdir().unwrap();
        let mut index = HeadIndex::default();
        index.heads.push(sample_head_record());
        let mut second = sample_head_record();
        second.head_id = HeadId::parse("11111111-2222-4333-8444-555555555557").unwrap();
        second.workspace_revision = rev(4);
        index.heads.push(second);
        // Two-entry sliding window matches MAX_HEADS_PER_WORKSPACE.
        assert_eq!(index.heads.len(), MAX_HEADS_PER_WORKSPACE);
        write_head_index(tmp.path(), &index).unwrap();
        let read = read_head_index(tmp.path()).unwrap();
        assert_eq!(read, index);
    }

    #[test]
    fn head_index_read_missing_file_surfaces_io_not_found() {
        let tmp = tempfile::tempdir().unwrap();
        match read_head_index(tmp.path()) {
            Err(FileError::Io { source, .. }) => {
                assert_eq!(source.kind(), std::io::ErrorKind::NotFound);
            }
            other => panic!("expected FileError::Io NotFound, got {other:?}"),
        }
    }

    #[test]
    fn head_index_default_is_empty_and_writeable() {
        let tmp = tempfile::tempdir().unwrap();
        let empty = HeadIndex::default();
        write_head_index(tmp.path(), &empty).unwrap();
        let read = read_head_index(tmp.path()).unwrap();
        assert_eq!(read, empty);
        assert!(read.heads.is_empty());
    }

    /// Reader rejects > 2 entries even though `Vec` itself has no
    /// schema-level cap.  Defends against hand-edited
    /// `heads.json`; production never produces this shape because
    /// the rotation primitive enforces the cap on writes.
    #[test]
    fn head_index_read_rejects_over_cap_entries() {
        let tmp = tempfile::tempdir().unwrap();
        // Hand-craft a 3-entry index and write it directly,
        // bypassing `write_head_index`'s symmetric cap.
        let mut over = HeadIndex::default();
        for i in 0..(MAX_HEADS_PER_WORKSPACE + 1) {
            let mut rec = sample_head_record();
            rec.head_id =
                HeadId::parse(&format!("11111111-2222-4333-8444-55555555555{:x}", i + 1)).unwrap();
            over.heads.push(rec);
        }
        let bytes = serde_json::to_vec(&over).unwrap();
        std::fs::write(tmp.path().join(HEAD_INDEX_FILENAME), &bytes).unwrap();
        let res = read_head_index(tmp.path());
        assert!(matches!(res, Err(FileError::MetadataParse { .. })));
    }

    /// Writer-side symmetric cap: refuses to publish > 2
    /// entries even if a buggy caller hands in too many.
    #[test]
    fn head_index_write_rejects_over_cap_entries() {
        let tmp = tempfile::tempdir().unwrap();
        let mut over = HeadIndex::default();
        for i in 0..(MAX_HEADS_PER_WORKSPACE + 1) {
            let mut rec = sample_head_record();
            rec.head_id =
                HeadId::parse(&format!("11111111-2222-4333-8444-55555555555{:x}", i + 1)).unwrap();
            over.heads.push(rec);
        }
        let res = write_head_index(tmp.path(), &over);
        assert!(matches!(res, Err(FileError::MetadataParse { .. })));
        assert!(!tmp.path().join(HEAD_INDEX_FILENAME).exists());
    }

    // MARK: HeadManifest round-trip

    #[test]
    fn head_manifest_round_trips_under_heads_dir() {
        let tmp = tempfile::tempdir().unwrap();
        // The heads/ subdir must exist before we can write a
        // manifest into it; downstream lifecycle code creates
        // it at workspace-create time.  Mirror that here.
        std::fs::create_dir_all(heads_dir(tmp.path())).unwrap();
        let manifest = sample_manifest();
        write_head_manifest(tmp.path(), &manifest).unwrap();
        let expected =
            heads_dir(tmp.path()).join(format!("{}.{}", manifest.head_id, HEAD_MANIFEST_EXTENSION));
        assert!(expected.is_file());
        let read = read_head_manifest(tmp.path(), manifest.head_id).unwrap();
        assert_eq!(read, manifest);
    }

    #[test]
    fn head_manifest_read_rejects_corrupt_json() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(heads_dir(tmp.path())).unwrap();
        let path = head_manifest_path(tmp.path(), head_id());
        std::fs::write(&path, b"{ not json").unwrap();
        let res = read_head_manifest(tmp.path(), head_id());
        assert!(matches!(res, Err(FileError::MetadataParse { .. })));
    }

    /// `read_head_manifest` runs `HeadManifest::validate` after
    /// deserialization so a hand-edited body with mismatched
    /// `n_classes` vs `labels.len()` fails closed.
    #[test]
    fn head_manifest_read_rejects_n_classes_labels_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(heads_dir(tmp.path())).unwrap();
        let mut bad = sample_manifest();
        bad.n_classes = 5; // labels.len() == 2
        let path = head_manifest_path(tmp.path(), bad.head_id);
        std::fs::write(&path, serde_json::to_vec(&bad).unwrap()).unwrap();
        let res = read_head_manifest(tmp.path(), bad.head_id);
        match res {
            Err(FileError::MetadataParse { source, .. }) => {
                assert!(
                    source.to_string().contains("n_classes"),
                    "diagnostic must mention n_classes; got {source}",
                );
            }
            other => panic!("expected MetadataParse on n_classes mismatch; got {other:?}"),
        }
    }

    /// Same shape for the `n_classes = 0` corner case.
    #[test]
    fn head_manifest_read_rejects_zero_classes() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(heads_dir(tmp.path())).unwrap();
        let mut bad = sample_manifest();
        bad.n_classes = 0;
        bad.labels.clear();
        let path = head_manifest_path(tmp.path(), bad.head_id);
        std::fs::write(&path, serde_json::to_vec(&bad).unwrap()).unwrap();
        let res = read_head_manifest(tmp.path(), bad.head_id);
        assert!(
            matches!(res, Err(FileError::MetadataParse { .. })),
            "expected MetadataParse on zero classes; got {res:?}",
        );
    }
}
