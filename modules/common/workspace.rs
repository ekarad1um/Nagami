//! Workspace, head, active-head, and job foundation types.
//!
//! Pure data-shape contracts consumed by `file_mgr` (`workspace.json`,
//! `heads.json`, per-head manifests, active-head manifests), the
//! in-memory `JobRegistry`, and the API response layer.  Schema
//! source of truth; no I/O lives here.
//!
//! # Naming collision
//!
//! [`HeadRecord`] (the on-disk index entry) is intentionally distinct
//! from the Burn-derived `model::HeadRecord` (`Linear` parameter
//! container materialized by `#[derive(Module)]`).  Refer to them by
//! qualified path (`common::workspace::HeadRecord` vs
//! `model::HeadRecord`).

use crate::common::error::{Categorized, ErrorKind};
use crate::common::ids::{HeadId, WorkspaceId, default_runtime_head_id};
use thiserror::Error;

// MARK: WorkspaceRevision

/// Monotonic mutation record covering both daemon-owned trees in a
/// workspace (`datasets/` and `converters/`).  `id` increments by
/// one on every accepted user upload/delete under either tree;
/// `at` records the RFC3339 wall-clock at the mutation moment.
/// Heads snapshot the workspace's revision when their producer
/// job starts so stale-vs-current detection is a single integer
/// compare.
///
/// Boot recovery may advance the revision conservatively: a
/// crash can stale heads without changing user file bytes, but
/// must never leave a head current after a workspace file
/// mutation.  Name/tag edits do not advance the revision.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkspaceRevision {
    /// Monotonic counter; strictly increases on accepted
    /// user file mutations.  Starts at 0 on workspace create.
    pub id: u64,
    /// RFC3339 wall-clock timestamp of the revision bump.
    pub at: String,
}

// MARK: WorkspaceCore (workspace.json)

/// Hot-path workspace metadata persisted at
/// `workspaces/<workspace_id>/workspace.json` and held in the
/// `ArcSwap`-backed core cache.  Workspace listing / summary
/// endpoints read this file (and `heads.json`) only; they never
/// walk `datasets/` or `converters/`.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkspaceCore {
    /// Workspace identifier; matches the directory name.
    pub id: WorkspaceId,
    /// Operator-supplied display name (UTF-8).  Validation (length,
    /// charset, Unicode case-insensitive uniqueness via
    /// `str::to_lowercase`) is owned by the lifecycle handler; this
    /// struct is the on-disk shape, not the validator.
    pub name: String,
    /// Operator-supplied tags (UTF-8).  Validation (trim, length,
    /// charset, Unicode case-insensitive uniqueness, max 32 entries)
    /// is owned by the lifecycle handler.  Tag edits do not advance
    /// `workspace_revision` or affect head status.
    pub tags: Vec<String>,
    /// RFC3339 wall-clock timestamp of workspace creation.
    pub created_at: String,
    /// Current workspace revision.  Bumped before bytes mutate
    /// under `datasets/` or `converters/` so a crash can never
    /// leave a head current after a file change.
    pub workspace_revision: WorkspaceRevision,
    /// Number of heads currently published in `heads.json`.
    /// Derived; boot recovery repairs from `heads.json.heads.len()`.
    pub head_count: u8,
}

// MARK: HeadIndex (heads.json) + HeadRecord

/// On-disk head index persisted at
/// `workspaces/<workspace_id>/heads.json`.  Hard cap at 2 entries
/// (sliding window by completion time, most-recent-first).
/// Failed jobs never appear here -- only successful publishes.
#[derive(Clone, Debug, Default, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeadIndex {
    /// Up to [`MAX_HEADS_PER_WORKSPACE`] entries, most-recent-first.
    pub heads: Vec<HeadRecord>,
}

/// Hard cap on the number of heads retained per workspace.
/// Sliding window -- the third successful train / convert
/// displaces the oldest.
pub const MAX_HEADS_PER_WORKSPACE: usize = 2;

/// Compact head index entry.  Trainer/converter input metadata
/// (dataset path, training-cfg payload, etc.) and labels are
/// intentionally absent so workspace summaries stay small; the
/// full picture lives in the per-head manifest ([`HeadManifest`]).
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeadRecord {
    /// Stable identifier for this head.
    pub head_id: HeadId,
    /// Workspace revision at the moment the producer
    /// snapshotted its job reference.  Stale-vs-current is a
    /// single integer compare against the workspace's current
    /// revision id.
    pub workspace_revision: WorkspaceRevision,
    /// Hex SHA-256 of the published `<head_id>.mpk` bytes; used
    /// by activation pre-load and boot verification.
    pub sha256: String,
    /// Output classes baked into the head.  Persisted so the
    /// API summary surfaces `n_classes` without opening the
    /// `.mpk`.
    pub n_classes: u32,
    /// Size of the published `<head_id>.mpk` in bytes.
    pub size_bytes: u64,
    /// RFC3339 wall-clock at successful publish.
    pub created_at: String,
}

// MARK: HeadManifest (per-head <head_id>.json)

/// Per-head manifest persisted alongside the `.mpk` weights at
/// `workspaces/<workspace_id>/heads/<head_id>.json`.  Index-atomic
/// publish: staged `.mpk` + `.json` are renamed before
/// `heads.json` references them; a crash before the index commit
/// leaves only unreferenced files.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeadManifest {
    /// Stable identifier; matches the file basename.
    pub head_id: HeadId,
    /// Workspace this head belongs to.  Cross-references the
    /// `workspaces/<workspace_id>/` directory.
    pub workspace_id: WorkspaceId,
    /// Workspace revision at producer-snapshot time.
    pub workspace_revision: WorkspaceRevision,
    /// Hex SHA-256 of the published `<head_id>.mpk` bytes.
    pub sha256: String,
    /// Output classes baked into the head.  Must equal
    /// `labels.len()`.
    pub n_classes: u32,
    /// Size of the published `<head_id>.mpk` in bytes.
    pub size_bytes: u64,
    /// RFC3339 wall-clock at successful publish.
    pub created_at: String,
    /// Class labels in inference order.  Materialized inline so
    /// the head publish is index-atomic; activation derives
    /// `active/labels.txt` from this list.
    pub labels: Vec<String>,
}

// MARK: HeadStatus (derived)

/// Derived freshness of a [`HeadRecord`] relative to its owning
/// workspace's current revision.  Computed on demand from the
/// workspace's `workspace_revision` and the head's
/// `workspace_revision` (snapshotted at producer start); never
/// persisted.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HeadStatus {
    /// Head was produced at the workspace's current revision id.
    Current,
    /// Head was produced at an older revision id.  May still
    /// serve correctly; the operator chooses whether to retrain.
    Stale,
}

impl HeadStatus {
    /// Compute freshness from the head's snapshotted revision
    /// and the workspace's current revision.  Comparison is on
    /// `id` only -- the timestamp is operator debugging context.
    /// A head whose recorded id is greater than the workspace's
    /// is corruption (and reads as `Stale` here so the route
    /// layer can fail closed).
    #[inline]
    pub fn from_revisions(head: &WorkspaceRevision, workspace: &WorkspaceRevision) -> Self {
        if head.id == workspace.id {
            Self::Current
        } else {
            Self::Stale
        }
    }
}

/// Structural-invariant failure raised by
/// [`HeadManifest::validate`].
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum HeadValidationError {
    /// `n_classes` was zero.
    #[error("head manifest has n_classes = 0")]
    ZeroClasses,
    /// `n_classes` and `labels.len()` disagreed.
    #[error("head manifest n_classes ({n_classes}) != labels.len() ({labels_len})")]
    ClassCountMismatch {
        /// Recorded `n_classes`.
        n_classes: u32,
        /// Observed `labels.len()`.
        labels_len: usize,
    },
}

impl Categorized for HeadValidationError {
    fn kind(&self) -> ErrorKind {
        // Daemon-internal: a producer bug or a hand-tampered
        // manifest, never operator request input.
        ErrorKind::Internal
    }
}

impl HeadManifest {
    /// Structural invariants beyond what serde catches.
    /// Callers reading a manifest from disk must invoke this
    /// before trusting the values.
    pub fn validate(&self) -> Result<(), HeadValidationError> {
        if self.n_classes == 0 {
            return Err(HeadValidationError::ZeroClasses);
        }
        let labels_len = self.labels.len();
        if (self.n_classes as usize) != labels_len {
            return Err(HeadValidationError::ClassCountMismatch {
                n_classes: self.n_classes,
                labels_len,
            });
        }
        Ok(())
    }
}

// MARK: ActiveOrigin / ActiveHeadManifest (active manifest.json)

/// Provenance of an active head generation.  `Default` means the
/// daemon-bundled fallback; `Head { ... }` carries the source
/// workspace + head id + workspace revision at activation time.
/// Variants flatten into [`ActiveHeadManifest`] under the
/// discriminator `"origin"`.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "origin", rename_all = "snake_case")]
pub enum ActiveOrigin {
    /// Bundled deployment-default head.  No source workspace
    /// fields.
    Default,
    /// Activation sourced from a per-workspace head.
    /// Provenance fields are recorded so a deleted source
    /// workspace surfaces `source_workspace_alive: false` in
    /// `GET /active` without breaking inference.
    Head {
        /// Workspace the head was produced in.
        source_workspace_id: WorkspaceId,
        /// Head id within that workspace.
        source_head_id: HeadId,
        /// Workspace revision snapshot of the source head.
        workspace_revision: WorkspaceRevision,
    },
}

/// Active-head manifest persisted at
/// `active/generations/<activation_id>/manifest.json`.  The
/// generation directory owns independent bytes (`head.mpk`,
/// `labels.txt`); deleting the source workspace does not break
/// inference because the active generation is self-contained.
///
/// Boot recovery streams-hashes `head.mpk` + `labels.txt` against
/// `sha256` / `labels_sha256`; on `labels.txt` mismatch only it
/// regenerates from the in-manifest `labels[]` list.
///
/// `deny_unknown_fields` is omitted because [`ActiveOrigin`]
/// flattens into the parent struct under the `"origin"`
/// discriminator; serde rejects this combination.  The
/// trade-off is intentional: forward-compatible manifest fields
/// are silently ignored, while the `ActiveOrigin` discriminant
/// is validated by the inner enum's tagged shape.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ActiveHeadManifest {
    /// Provenance of this active generation.  Internally tagged
    /// by `"origin"` and flattened into the parent.
    #[serde(flatten)]
    pub origin: ActiveOrigin,
    /// Stable runtime identifier stamped on every emitted
    /// `InferenceFrame.head_id`.  For `Default` origin this is
    /// the bundled-default UUID
    /// ([`crate::common::ids::default_runtime_head_id`]); for
    /// `Head` origin it equals the trained `source_head_id`.
    pub runtime_head_id: HeadId,
    /// Hex SHA-256 of the generation's `head.mpk` bytes.
    pub sha256: String,
    /// Hex SHA-256 of the generation's materialized
    /// `labels.txt` bytes.
    pub labels_sha256: String,
    /// Output classes baked into the active head.
    pub n_classes: u32,
    /// Class labels in inference order.  Canonical recovery
    /// source for `labels.txt` if the file goes missing /
    /// stale.
    pub labels: Vec<String>,
    /// RFC3339 wall-clock at activation.
    pub activated_at: String,
}

/// Structural validation failure for [`ActiveHeadManifest`].
/// Catches invariants serde does not enforce because
/// internally-tagged enums and `#[serde(flatten)]` cannot
/// compose with `deny_unknown_fields`; the active-head writer
/// and boot recovery call [`ActiveHeadManifest::validate`]
/// explicitly to fail closed on a malformed generation.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ActiveHeadValidationError {
    /// `origin: default` was published with a `runtime_head_id`
    /// that doesn't match the bundled-default UUID.  The active
    /// generation cannot be served because the discriminator
    /// would lie about its identity.
    #[error(
        "active head manifest has origin=default but runtime_head_id ({got}) \
         differs from the bundled default ({expected})"
    )]
    DefaultRuntimeIdMismatch {
        /// Recorded id.
        got: HeadId,
        /// Expected bundled-default id.
        expected: HeadId,
    },
    /// `origin: head` was published with a `runtime_head_id`
    /// that doesn't match `source_head_id`.  Inference frames
    /// would carry a head id different from the source head's
    /// recorded id; runtime consumers cannot disambiguate.
    #[error(
        "active head manifest has origin=head with mismatched runtime_head_id ({got}) \
         vs source_head_id ({expected})"
    )]
    HeadRuntimeIdMismatch {
        /// Recorded `runtime_head_id`.
        got: HeadId,
        /// Recorded `source_head_id`.
        expected: HeadId,
    },
}

impl Categorized for ActiveHeadValidationError {
    fn kind(&self) -> ErrorKind {
        ErrorKind::Internal
    }
}

impl ActiveHeadManifest {
    /// Structural invariants beyond what serde catches.  The
    /// flatten + internally-tagged enum shape is necessary for
    /// the on-disk JSON layout but defeats serde's
    /// `deny_unknown_fields`; this method is the explicit
    /// catcher for inconsistencies between the discriminator,
    /// the source provenance fields, and `runtime_head_id`.
    /// Callers (active-head writer, boot recovery) must invoke
    /// this before treating a deserialized manifest as
    /// trustworthy.
    pub fn validate(&self) -> Result<(), ActiveHeadValidationError> {
        match &self.origin {
            ActiveOrigin::Default => {
                let expected = default_runtime_head_id();
                if self.runtime_head_id != expected {
                    return Err(ActiveHeadValidationError::DefaultRuntimeIdMismatch {
                        got: self.runtime_head_id,
                        expected,
                    });
                }
            }
            ActiveOrigin::Head { source_head_id, .. } => {
                if self.runtime_head_id != *source_head_id {
                    return Err(ActiveHeadValidationError::HeadRuntimeIdMismatch {
                        got: self.runtime_head_id,
                        expected: *source_head_id,
                    });
                }
            }
        }
        Ok(())
    }
}

// MARK: ConverterType

/// Selector for the convert pipeline; wire shape is snake_case.  New
/// variants gate on a reviewed converter-specific request payload.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConverterType {
    /// TFJS-bundled-graph -> head conversion (the only
    /// converter the daemon ships today).
    Tfjs,
}

// MARK: JobType / JobReference

/// Discriminator for typed job snapshots.  Matches the long-running
/// operations the daemon bounds via the `JobRegistry`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobType {
    /// Train a new head from a workspace's `datasets/` tree.
    /// Bounded to one unfinished job daemon-wide.
    Train,
    /// Convert workspace converter inputs into a head.
    /// Bounded to one unfinished job daemon-wide.
    Convert,
    /// Async dataset file / tree delete under `datasets/`.
    /// Tombstoned + staged; boot resumes interrupted payload
    /// removal.
    DatasetDelete,
    /// Async converter file / tree delete under `converters/`.
    /// Tombstoned + staged; boot resumes interrupted payload
    /// removal.
    ConverterDelete,
    /// Async training-log file / tree delete under
    /// `training_logs/`.  Tombstoned + staged like
    /// [`Self::DatasetDelete`] but does NOT bump
    /// `workspace_revision` (logs aren't workspace state).
    TrainingLogsDelete,
    /// Async converter-log file / tree delete under
    /// `converter_logs/`.  Mirror of
    /// [`Self::TrainingLogsDelete`] for the converter producer.
    ConverterLogsDelete,
    /// Async workspace delete.  Stages the entire workspace
    /// directory under root `.tmp/` then drains in batches.
    WorkspaceDelete,
}

/// State a running job touches.  Producers and delete jobs register
/// exactly one whole-workspace reference for `WorkspaceDelete`
/// exclusion; uploads and file deletes overlap train/convert without
/// conflict.  Conflict detection reduces to "same workspace_id +
/// at least one side is a `WorkspaceDelete`."
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum JobReference {
    Workspace { workspace_id: WorkspaceId },
}

impl JobReference {
    pub fn workspace_id(&self) -> WorkspaceId {
        match self {
            JobReference::Workspace { workspace_id } => *workspace_id,
        }
    }
}

// MARK: Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::error::{Categorized, ErrorKind};
    use crate::common::ids::{HeadId, WorkspaceId, default_runtime_head_id};

    fn rev(id: u64) -> WorkspaceRevision {
        WorkspaceRevision {
            id,
            at: "2026-05-07T12:00:00Z".to_string(),
        }
    }

    fn ws_id() -> WorkspaceId {
        WorkspaceId::parse("11111111-2222-4333-8444-555555555555").unwrap()
    }

    fn head_id() -> HeadId {
        HeadId::parse("11111111-2222-4333-8444-555555555556").unwrap()
    }

    // MARK: HeadStatus

    #[test]
    fn head_status_current_when_revisions_match() {
        assert_eq!(
            HeadStatus::from_revisions(&rev(5), &rev(5)),
            HeadStatus::Current
        );
    }

    #[test]
    fn head_status_stale_when_revision_id_differs() {
        // Even when the timestamps match, only `id` matters.
        assert_eq!(
            HeadStatus::from_revisions(&rev(4), &rev(5)),
            HeadStatus::Stale
        );
        // Even when the head is "ahead" of the workspace -- the
        // boot-recovery conservative bump may produce this.
        assert_eq!(
            HeadStatus::from_revisions(&rev(6), &rev(5)),
            HeadStatus::Stale
        );
    }

    // MARK: WorkspaceCore + WorkspaceRevision serde

    #[test]
    fn workspace_core_round_trips() {
        let core = WorkspaceCore {
            id: ws_id(),
            name: "main".to_string(),
            tags: vec!["pet-noises".to_string(), "field".to_string()],
            created_at: "2026-05-07T12:34:56Z".to_string(),
            workspace_revision: rev(5),
            head_count: 2,
        };
        let json = serde_json::to_string(&core).unwrap();
        let parsed: WorkspaceCore = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, core);
    }

    /// Default-empty `tags` round-trips through the canonical
    /// shape without surprise.  The lifecycle handler
    /// substitutes `Vec::new()` when `POST /workspace` omits the
    /// optional field.
    #[test]
    fn workspace_core_round_trips_with_empty_tags() {
        let core = WorkspaceCore {
            id: ws_id(),
            name: "main".to_string(),
            tags: Vec::new(),
            created_at: "2026-05-07T12:34:56Z".to_string(),
            workspace_revision: rev(0),
            head_count: 0,
        };
        let json = serde_json::to_string(&core).unwrap();
        assert!(json.contains("\"tags\":[]"));
        let parsed: WorkspaceCore = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, core);
    }

    #[test]
    fn workspace_core_rejects_unknown_fields() {
        let bad = r#"{
            "id": "11111111-2222-4333-8444-555555555555",
            "name": "main",
            "tags": [],
            "created_at": "2026-05-07T12:34:56Z",
            "workspace_revision": { "id": 5, "at": "2026-05-07T13:00:00Z" },
            "head_count": 2,
            "schema_version": 1
        }"#;
        let res: Result<WorkspaceCore, _> = serde_json::from_str(bad);
        assert!(res.is_err(), "deny_unknown_fields must reject extra keys");
    }

    #[test]
    fn workspace_revision_rejects_unknown_fields() {
        let bad = r#"{ "id": 5, "at": "2026-05-07T13:00:00Z", "extra": true }"#;
        assert!(serde_json::from_str::<WorkspaceRevision>(bad).is_err());
    }

    // MARK: HeadIndex / HeadRecord

    #[test]
    fn head_index_default_empty() {
        let idx = HeadIndex::default();
        assert!(idx.heads.is_empty());
        let json = serde_json::to_string(&idx).unwrap();
        let parsed: HeadIndex = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, idx);
    }

    #[test]
    fn head_record_round_trips() {
        let rec = HeadRecord {
            head_id: head_id(),
            workspace_revision: rev(5),
            sha256: "def".to_string(),
            n_classes: 12,
            size_bytes: 2048,
            created_at: "2026-05-07T12:34:56Z".to_string(),
        };
        let json = serde_json::to_string(&rec).unwrap();
        let parsed: HeadRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, rec);
    }

    /// Pinned: dropped fields must not participate in the on-disk
    /// schema.  A round-trip must NOT produce keys for the legacy
    /// names.
    #[test]
    fn head_record_drops_round_1_provenance_fields() {
        let rec = HeadRecord {
            head_id: head_id(),
            workspace_revision: rev(5),
            sha256: "def".to_string(),
            n_classes: 12,
            size_bytes: 2048,
            created_at: "2026-05-07T12:34:56Z".to_string(),
        };
        let json = serde_json::to_string(&rec).unwrap();
        for stale in [
            "dataset_path",
            "dataset_revision_at_train",
            "training_cfg_sha256",
            "training_cfg",
        ] {
            assert!(
                !json.contains(stale),
                "legacy field `{stale}` must not appear in serialized HeadRecord"
            );
        }
    }

    // MARK: HeadManifest

    #[test]
    fn head_manifest_round_trips() {
        let manifest = HeadManifest {
            head_id: head_id(),
            workspace_id: ws_id(),
            workspace_revision: rev(5),
            sha256: "def".to_string(),
            n_classes: 2,
            size_bytes: 2048,
            created_at: "2026-05-07T12:34:56Z".to_string(),
            labels: vec!["cat".to_string(), "dog".to_string()],
        };
        let json = serde_json::to_string(&manifest).unwrap();
        let parsed: HeadManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, manifest);
    }

    #[test]
    fn head_manifest_rejects_unknown_fields() {
        // The legacy shape (which carried `dataset_path`,
        // `dataset_revision_at_train`, `training_cfg_sha256`,
        // `training_cfg`) must fail to parse so a fresh-boot daemon
        // refuses old data.
        let bad = r#"{
            "head_id": "11111111-2222-4333-8444-555555555556",
            "workspace_id": "11111111-2222-4333-8444-555555555555",
            "dataset_path": "audio_dataset",
            "dataset_revision_at_train": { "id": 5, "at": "2026-05-07T13:00:00Z" },
            "training_cfg_sha256": "abc",
            "training_cfg": {},
            "sha256": "def",
            "n_classes": 12,
            "size_bytes": 2048,
            "created_at": "2026-05-07T12:34:56Z",
            "labels": [],
            "schema_version": 2
        }"#;
        assert!(serde_json::from_str::<HeadManifest>(bad).is_err());
    }

    /// `HeadManifest::validate` enforces `n_classes ==
    /// labels.len()`.  A hand-edited manifest with mismatched
    /// fields fails closed at the read boundary.
    #[test]
    fn head_manifest_validate_class_count_consistent() {
        let mut manifest = HeadManifest {
            head_id: head_id(),
            workspace_id: ws_id(),
            workspace_revision: rev(5),
            sha256: "def".to_string(),
            n_classes: 2,
            size_bytes: 2048,
            created_at: "2026-05-07T12:34:56Z".to_string(),
            labels: vec!["cat".to_string(), "dog".to_string()],
        };
        // Consistent shape passes.
        assert!(manifest.validate().is_ok());

        // Bump n_classes without touching labels: rejected.
        manifest.n_classes = 3;
        let err = manifest.validate().unwrap_err();
        assert!(matches!(
            err,
            HeadValidationError::ClassCountMismatch {
                n_classes: 3,
                labels_len: 2
            }
        ));

        // Add an extra label to make labels.len() != n_classes
        // again from the other direction.
        manifest.n_classes = 2;
        manifest.labels.push("bird".to_string());
        let err = manifest.validate().unwrap_err();
        assert!(matches!(
            err,
            HeadValidationError::ClassCountMismatch {
                n_classes: 2,
                labels_len: 3
            }
        ));
    }

    #[test]
    fn head_validation_error_classifies_internal() {
        let err = HeadValidationError::ClassCountMismatch {
            n_classes: 3,
            labels_len: 2,
        };
        assert_eq!(err.kind(), ErrorKind::Internal);
    }

    // MARK: ActiveHeadManifest

    #[test]
    fn active_head_manifest_default_origin_round_trips() {
        let manifest = ActiveHeadManifest {
            origin: ActiveOrigin::Default,
            runtime_head_id: default_runtime_head_id(),
            sha256: "aa".to_string(),
            labels_sha256: "bb".to_string(),
            n_classes: 1,
            labels: vec!["unknown".to_string()],
            activated_at: "2026-05-07T12:34:56Z".to_string(),
        };
        let json = serde_json::to_string(&manifest).unwrap();
        // Discriminator field is `origin: "default"`; no
        // source_* fields appear.
        assert!(json.contains("\"origin\":\"default\""));
        assert!(!json.contains("source_workspace_id"));
        let parsed: ActiveHeadManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, manifest);
    }

    #[test]
    fn active_head_manifest_head_origin_round_trips() {
        let manifest = ActiveHeadManifest {
            origin: ActiveOrigin::Head {
                source_workspace_id: ws_id(),
                source_head_id: head_id(),
                workspace_revision: rev(5),
            },
            runtime_head_id: head_id(),
            sha256: "aa".to_string(),
            labels_sha256: "bb".to_string(),
            n_classes: 2,
            labels: vec!["cat".to_string(), "dog".to_string()],
            activated_at: "2026-05-07T12:34:56Z".to_string(),
        };
        let json = serde_json::to_string(&manifest).unwrap();
        assert!(json.contains("\"origin\":\"head\""));
        assert!(json.contains("source_workspace_id"));
        assert!(json.contains("\"workspace_revision\""));
        assert!(!json.contains("source_dataset_revision"));
        let parsed: ActiveHeadManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, manifest);
    }

    /// The `Head` origin requires its source_* sibling fields.
    /// A manifest tagged `origin: head` without them must fail
    /// to parse so a corrupt active generation cannot be served
    /// as a valid head.
    #[test]
    fn active_head_manifest_head_origin_requires_source_fields() {
        let bad = r#"{
            "origin": "head",
            "runtime_head_id": "11111111-2222-4333-8444-555555555556",
            "sha256": "aa",
            "labels_sha256": "bb",
            "n_classes": 12,
            "labels": [],
            "activated_at": "2026-05-07T12:34:56Z"
        }"#;
        assert!(serde_json::from_str::<ActiveHeadManifest>(bad).is_err());
    }

    /// Serde + flatten + internally-tagged-enum cannot reject
    /// stray sibling fields at parse time -- but `validate()`
    /// catches the meaningful invariants downstream code
    /// depends on.  For `origin: default`, the runtime id MUST
    /// be the bundled-default UUID; a manifest with a different
    /// runtime id is structurally invalid even if it parses.
    #[test]
    fn active_head_validate_default_requires_bundled_runtime_id() {
        let bad = ActiveHeadManifest {
            origin: ActiveOrigin::Default,
            // Wrong: a Head's UUID under origin: default.
            runtime_head_id: head_id(),
            sha256: "aa".to_string(),
            labels_sha256: "bb".to_string(),
            n_classes: 1,
            labels: vec!["unknown".to_string()],
            activated_at: "2026-05-07T12:34:56Z".to_string(),
        };
        assert!(matches!(
            bad.validate(),
            Err(ActiveHeadValidationError::DefaultRuntimeIdMismatch { .. })
        ));

        // Correct shape passes.
        let good = ActiveHeadManifest {
            runtime_head_id: default_runtime_head_id(),
            ..bad
        };
        assert!(good.validate().is_ok());
    }

    /// For `origin: head`, the runtime id MUST equal
    /// `source_head_id` so emitted inference frames carry the
    /// source head's recorded identity.
    #[test]
    fn active_head_validate_head_requires_runtime_eq_source() {
        let other = HeadId::parse("11111111-2222-4333-8444-666666666666").unwrap();
        let bad = ActiveHeadManifest {
            origin: ActiveOrigin::Head {
                source_workspace_id: ws_id(),
                source_head_id: head_id(),
                workspace_revision: rev(5),
            },
            // Wrong: runtime id differs from source_head_id.
            runtime_head_id: other,
            sha256: "aa".to_string(),
            labels_sha256: "bb".to_string(),
            n_classes: 1,
            labels: vec!["cat".to_string()],
            activated_at: "2026-05-07T12:34:56Z".to_string(),
        };
        assert!(matches!(
            bad.validate(),
            Err(ActiveHeadValidationError::HeadRuntimeIdMismatch { .. })
        ));

        // Correct shape passes.
        let good = ActiveHeadManifest {
            runtime_head_id: head_id(),
            ..bad
        };
        assert!(good.validate().is_ok());
    }

    #[test]
    fn active_head_validation_error_classifies_internal() {
        let err = ActiveHeadValidationError::DefaultRuntimeIdMismatch {
            got: head_id(),
            expected: default_runtime_head_id(),
        };
        assert_eq!(err.kind(), ErrorKind::Internal);
    }

    // MARK: ConverterType

    #[test]
    fn converter_type_round_trips_in_snake_case() {
        assert_eq!(
            serde_json::to_string(&ConverterType::Tfjs).unwrap(),
            "\"tfjs\""
        );
        let parsed: ConverterType = serde_json::from_str("\"tfjs\"").unwrap();
        assert_eq!(parsed, ConverterType::Tfjs);
    }

    #[test]
    fn converter_type_rejects_unknown_variant() {
        // Wire shape is closed; new converter variants gate on
        // a reviewed payload contract.
        assert!(serde_json::from_str::<ConverterType>("\"onnx\"").is_err());
    }

    // MARK: JobReference

    #[test]
    fn job_reference_workspace_id_returns_owner() {
        let r = JobReference::Workspace {
            workspace_id: ws_id(),
        };
        assert_eq!(r.workspace_id(), ws_id());
    }

    #[test]
    fn job_reference_round_trips() {
        let r = JobReference::Workspace {
            workspace_id: ws_id(),
        };
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains("\"kind\":\"workspace\""));
        let parsed: JobReference = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, r);
    }

    /// Pinned: legacy dataset-tree / dataset-file variants must not
    /// parse so a stale operator tool cannot smuggle a path reference
    /// past the registry.
    #[test]
    fn job_reference_rejects_round_1_variants() {
        for stale in [
            r#"{"kind":"dataset_tree","workspace_id":"11111111-2222-4333-8444-555555555555","path":"audio"}"#,
            r#"{"kind":"dataset_file","workspace_id":"11111111-2222-4333-8444-555555555555","path":"audio/cat"}"#,
        ] {
            assert!(
                serde_json::from_str::<JobReference>(stale).is_err(),
                "legacy variant `{stale}` must not parse"
            );
        }
    }

    #[test]
    fn job_type_round_trips_in_snake_case() {
        for (jt, expected) in [
            (JobType::Train, "\"train\""),
            (JobType::Convert, "\"convert\""),
            (JobType::DatasetDelete, "\"dataset_delete\""),
            (JobType::ConverterDelete, "\"converter_delete\""),
            (JobType::TrainingLogsDelete, "\"training_logs_delete\""),
            (JobType::ConverterLogsDelete, "\"converter_logs_delete\""),
            (JobType::WorkspaceDelete, "\"workspace_delete\""),
        ] {
            assert_eq!(serde_json::to_string(&jt).unwrap(), expected);
            let parsed: JobType = serde_json::from_str(expected).unwrap();
            assert_eq!(parsed, jt);
        }
    }

    #[test]
    fn max_heads_per_workspace_is_two() {
        // Pinned: the storage contract caps heads at 2 (sliding
        // window).  Bumping this value requires touching the
        // 2-slot rotation contract in `head_rotation`.
        assert_eq!(MAX_HEADS_PER_WORKSPACE, 2);
    }
}
