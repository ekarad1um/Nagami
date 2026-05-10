//! `metadata.json` schema + the `MetadataStore` aggregate of
//! [`WorkspaceMgr`].

use std::sync::Arc;

use crate::common::ids::{AssetId, WorkspaceId};
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

use crate::file_mgr::WorkspaceMgr;
use crate::file_mgr::error::{FileError, io_err, metadata_parse_err};

/// Asset kind taxonomy.  Each variant maps to a
/// `<workspace>/<subdir>/` path prefix and a set of allowed
/// file extensions; both are crate-private since they're
/// enforced through [`WorkspaceMgr`]'s API rather than
/// exposed to consumers.
///
/// `#[serde(rename_all = "snake_case")]` is load-bearing: the
/// `/api/v1/workspaces/{id}/assets` JSON shape pins
/// `kind: "head_mpk"`, not `"HeadMpk"`.
///
/// `#[non_exhaustive]`: the asset taxonomy expects further
/// additions; in-crate impl-block matches are unaffected.
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum AssetKind {
    Dataset,
    BackboneMpk,
    BackboneRknn,
    HeadMpk,
    HeadLabels,
    Metadata,
}

impl AssetKind {
    pub(crate) fn subdir(&self) -> &'static str {
        match self {
            AssetKind::Dataset => "datasets",
            AssetKind::BackboneMpk | AssetKind::BackboneRknn => "weights",
            AssetKind::HeadMpk => "weights",
            AssetKind::HeadLabels => "labels",
            AssetKind::Metadata => ".",
        }
    }

    pub(crate) fn allowed_ext(&self) -> &[&'static str] {
        match self {
            // Dataset accepts both raw archives (tar/zip) AND
            // operator-uploaded TFJS source models (`.json`
            // for `model.json` + `.bin` for the weight
            // shards).
            AssetKind::Dataset => &["tar.gz", "tgz", "zip", "json", "bin"],
            AssetKind::BackboneMpk | AssetKind::HeadMpk => &["mpk"],
            AssetKind::BackboneRknn => &["rknn"],
            AssetKind::HeadLabels => &["txt"],
            AssetKind::Metadata => &["json"],
        }
    }
}

/// Per-asset record stored in `metadata.json.assets`.  The
/// `kind` disambiguates same-name files in different subdirs
/// (a head named `default.mpk` and a backbone named
/// `default.mpk` co-exist).
///
/// The [`AssetId`] newtype's serde shim (`try_from =
/// "String"`) routes wire-side deserialization through
/// [`AssetId::parse`], so a workspace `metadata.json`
/// containing `name: "../etc/passwd"` (or any other
/// path-traversal / non-allowlisted shape) fails to load
/// with a structured error rather than being silently
/// wrapped.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AssetRecord {
    pub kind: AssetKind,
    pub name: AssetId,
    /// SHA-256 of the on-disk file, lowercase hex.
    pub sha256: String,
    pub size_bytes: u64,
}

/// `metadata.json` schema.  Versioned via `schema_version`;
/// bump on breaking change.  v1 is the only version today.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkspaceMetadata {
    pub schema_version: u32,
    pub id: WorkspaceId,
    pub name: String,
    /// RFC3339 string.  Stored as `String` (not
    /// [`OffsetDateTime`]) because TOML serde for time types
    /// differs across versions, while strings round-trip
    /// identically.
    pub created_at: String,
    pub assets: Vec<AssetRecord>,
}

impl WorkspaceMetadata {
    /// Latest schema version this build writes.  Bump
    /// alongside any shape change to [`WorkspaceMetadata`] or
    /// [`AssetRecord`] that can't be deserialized by an older
    /// daemon, so an older daemon reading a workspace
    /// produced by a newer one fails loudly with
    /// [`FileError::SchemaTooNew`] rather than parsing
    /// partial / missing fields and silently corrupting state
    /// on the next write.
    pub const CURRENT: u32 = 1;

    /// Oldest schema version this build still understands.
    /// With no migration code yet, only the latest schema is
    /// accepted.  When v2 lands and v1 metadata can be
    /// upgraded in place, raise `CURRENT` to 2 and keep
    /// `MIN_COMPATIBLE = 1`.  Once v1 support is dropped,
    /// raise `MIN_COMPATIBLE` to match the new floor.
    pub const MIN_COMPATIBLE: u32 = 1;

    pub fn new(id: WorkspaceId, name: String) -> Self {
        Self {
            schema_version: Self::CURRENT,
            id,
            name,
            created_at: OffsetDateTime::now_utc()
                .format(&time::format_description::well_known::Rfc3339)
                .expect("RFC3339 format"),
            assets: Vec::new(),
        }
    }

    /// Find the index of an asset by `(kind, name)`;
    /// convenience for upsert.  `name` is `&str` (not
    /// `&AssetId`) so callers that already have a validated
    /// `&str` (e.g. from
    /// [`crate::file_mgr::validate_asset_name`]) don't need
    /// to allocate an [`AssetId`] just for lookup.
    pub fn find_index(&self, kind: AssetKind, name: &str) -> Option<usize> {
        self.assets
            .iter()
            .position(|a| a.kind == kind && a.name == name)
    }

    /// Case-insensitive collision lookup.  Returns the first
    /// asset of the same `kind` whose name matches `name`
    /// ignoring ASCII case.  Used by upload paths to defend
    /// against case-insensitive filesystems (macOS HFS+,
    /// Windows NTFS) where uploading `Foo.mpk` then `foo.mpk`
    /// would silently overwrite the first file's bytes while
    /// leaving distinct case-sensitive metadata records.
    ///
    /// `eq_ignore_ascii_case` covers the common case but
    /// does not implement full Unicode case folding.  Asset
    /// names that pass [`crate::file_mgr::validate_asset_name`]
    /// may contain arbitrary multi-byte UTF-8, so a non-ASCII
    /// collision (e.g. `Cafe.mpk` vs `cafe.mpk` under HFS+'s
    /// Unicode-normalized case folding) is not caught here.
    pub fn find_case_insensitive(&self, kind: AssetKind, name: &str) -> Option<&AssetRecord> {
        self.assets
            .iter()
            .find(|a| a.kind == kind && a.name.as_str().eq_ignore_ascii_case(name))
    }
}

// MARK: MetadataStore

impl WorkspaceMgr {
    /// Acquire (or lazily allocate) the per-workspace
    /// metadata lock.  Held only by [`Self::with_metadata`]
    /// and the upload paths; callers must NOT hold it across
    /// `.await` boundaries (`parking_lot::Mutex` doesn't
    /// yield).
    pub(crate) fn metadata_lock(&self, id: &WorkspaceId) -> Arc<parking_lot::Mutex<()>> {
        self.metadata_locks
            .entry(*id)
            .or_insert_with(|| Arc::new(parking_lot::Mutex::new(())))
            .clone()
    }

    /// Read the metadata, run `f` on it, write it back, all
    /// under the per-workspace lock so concurrent uploads
    /// can't lose updates.  The closure may fail; the lock
    /// is released either way.
    pub fn with_metadata<F, R>(&self, id: &WorkspaceId, f: F) -> Result<R, FileError>
    where
        F: FnOnce(&mut WorkspaceMetadata) -> R,
    {
        let lock = self.metadata_lock(id);
        let _guard = lock.lock();
        let mut meta = self.read_metadata(id)?;
        let result = f(&mut meta);
        self.write_metadata(id, &meta)?;
        Ok(result)
    }

    /// Load + parse a workspace's `metadata.json`, gating
    /// against forward / backward incompatible schema
    /// versions.
    ///
    /// [`FileError::SchemaTooNew`] is the load-bearing check:
    /// a daemon downgraded after a future schema bump would
    /// otherwise silently parse a newer-shape file (missing
    /// fields default-on), then on the next write serialize
    /// the older shape over the newer one, losing every
    /// field the new schema added.  [`FileError::SchemaTooOld`]
    /// is dormant today (`MIN_COMPATIBLE = 1`); it activates
    /// when a future build raises the floor.
    pub fn read_metadata(&self, id: &WorkspaceId) -> Result<WorkspaceMetadata, FileError> {
        let path = self.workspace_dir(id).join("metadata.json");
        let bytes = std::fs::read(&path).map_err(|e| io_err(path.display(), e))?;
        let meta: WorkspaceMetadata =
            serde_json::from_slice(&bytes).map_err(|e| metadata_parse_err(path.display(), e))?;
        if meta.schema_version > WorkspaceMetadata::CURRENT {
            return Err(FileError::SchemaTooNew {
                path: path.display().to_string(),
                found: meta.schema_version,
                max: WorkspaceMetadata::CURRENT,
            });
        }
        if meta.schema_version < WorkspaceMetadata::MIN_COMPATIBLE {
            return Err(FileError::SchemaTooOld {
                path: path.display().to_string(),
                found: meta.schema_version,
                min: WorkspaceMetadata::MIN_COMPATIBLE,
            });
        }
        Ok(meta)
    }

    /// Atomically rewrite a workspace's `metadata.json`.
    /// Defers to [`crate::file_mgr::put_atomic`] (tempfile,
    /// write, fdatasync, rename, parent fsync), so the
    /// converter / training paths that route through
    /// [`crate::file_mgr::FsService::put_atomic`] see
    /// identical durability semantics.
    pub fn write_metadata(
        &self,
        id: &WorkspaceId,
        meta: &WorkspaceMetadata,
    ) -> Result<(), FileError> {
        let ws = self.workspace_dir(id);
        std::fs::create_dir_all(&ws).map_err(|e| io_err(ws.display(), e))?;
        let bytes = serde_json::to_vec_pretty(meta)?;
        crate::file_mgr::fs_atomic::put_atomic(&ws.join("metadata.json"), &bytes)
    }
}
