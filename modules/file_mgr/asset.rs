//! `AssetStore` aggregate of [`WorkspaceMgr`] -- kind-aware
//! asset reads + the full workspace-walk validation.

use std::path::PathBuf;

use crate::common::ids::{AssetId, WorkspaceId};

use crate::file_mgr::WorkspaceMgr;
use crate::file_mgr::WorkspaceReport;
use crate::file_mgr::error::{FileError, io_err};
use crate::file_mgr::metadata::{AssetKind, AssetRecord};
use crate::file_mgr::validate::{sha256_file_streaming, validate_asset_name};

impl WorkspaceMgr {
    /// Resolve the on-disk path for an asset.  Does NOT verify
    /// the file exists -- used by readers that want to feed the
    /// path to a downstream loader (e.g.,
    /// [`crate::inference::HotHead::load`]).
    ///
    /// Validates `name` in EVERY build (not just debug): an
    /// accidentally-unvalidated `..` escape used to silently
    /// produce a path outside the workspace in release.  The
    /// check now panics on contract violation, matching the
    /// `expect` invariant the upload commit path already
    /// relied on; api / training call sites already run
    /// [`validate_asset_name`] before reaching here so the
    /// panic cannot fire on operator-validated input.  New
    /// call sites that hold a typed [`AssetId`] should prefer
    /// [`Self::asset_path_typed`] which skips the re-check.
    pub fn asset_path(&self, id: &WorkspaceId, kind: AssetKind, name: &str) -> PathBuf {
        // Release-build guard.  The validator is identical to
        // [`AssetId::parse`]; an invalid name reaching here is
        // a missing upstream `validate_asset_name` call, not
        // operator input.
        validate_asset_name(name).unwrap_or_else(|e| {
            panic!(
                "asset_path called with unvalidated name {name:?} ({e}); \
                 callers must run validate_asset_name (or hold an AssetId) first"
            )
        });
        self.workspace_dir(id).join(kind.subdir()).join(name)
    }

    /// Typed sibling of [`Self::asset_path`].  Skips the
    /// release-build re-check because [`AssetId`] already
    /// guarantees the same allowlist on construction.
    /// Preferred at call sites that already hold a typed id
    /// (the upload commit path stores `AssetId` in the
    /// metadata record, for example).
    pub fn asset_path_typed(&self, id: &WorkspaceId, kind: AssetKind, name: &AssetId) -> PathBuf {
        self.workspace_dir(id)
            .join(kind.subdir())
            .join(name.as_str())
    }

    /// List assets of a given kind.  Reads the cached
    /// `metadata.json`; does not walk the filesystem (use
    /// [`Self::validate`] for that).
    pub fn list_assets(
        &self,
        id: &WorkspaceId,
        kind: AssetKind,
    ) -> Result<Vec<AssetRecord>, FileError> {
        let meta = self.read_metadata(id)?;
        Ok(meta.assets.into_iter().filter(|a| a.kind == kind).collect())
    }

    /// Walk the workspace, recompute sha256 for every file in
    /// metadata and check against the recorded value.  Returns
    /// lists of:
    ///
    /// - `missing` -- declared in metadata but not present on
    ///   disk
    /// - `corrupt` -- present but sha256 differs
    /// - `extra`   -- on disk but absent from metadata
    ///
    /// Hashes are computed in a streaming 64 KiB read loop so
    /// a multi-GB dataset asset doesn't drag the whole file
    /// into RAM.
    pub fn validate(&self, id: &WorkspaceId) -> Result<WorkspaceReport, FileError> {
        let meta = self.read_metadata(id)?;
        let ws = self.workspace_dir(id);
        let mut missing = Vec::new();
        let mut corrupt = Vec::new();

        for a in &meta.assets {
            let p = self.asset_path(id, a.kind, a.name.as_str());
            if !p.exists() {
                missing.push((a.kind, a.name.as_str().to_string()));
                continue;
            }
            let digest = sha256_file_streaming(&p)?;
            if digest != a.sha256 {
                corrupt.push((a.kind, a.name.as_str().to_string()));
            }
        }

        // Extras: walk known asset subdirs and compare against
        // metadata.
        let mut extra: Vec<(PathBuf, String)> = Vec::new();
        for subdir in ["datasets", "weights", "labels"] {
            let dir = ws.join(subdir);
            if !dir.exists() {
                continue;
            }
            for e in std::fs::read_dir(&dir).map_err(|err| io_err(dir.display(), err))? {
                let e = e.map_err(|err| io_err(dir.display(), err))?;
                if !e.file_type().is_ok_and(|t| t.is_file()) {
                    continue;
                }
                let name = e.file_name().to_string_lossy().into_owned();
                let known = meta
                    .assets
                    .iter()
                    .any(|a| a.kind.subdir() == subdir && a.name.as_str() == name);
                if !known {
                    extra.push((e.path(), name));
                }
            }
        }

        Ok(WorkspaceReport {
            ok: missing.is_empty() && corrupt.is_empty(),
            missing,
            corrupt,
            extra,
        })
    }
}
