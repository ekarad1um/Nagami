//! Workspace lifecycle (`/workspace`, `/workspace/{id}`); asset and
//! upload routes live in [`crate::api::routes::dataset`].  Boot
//! recovery owns reconciliation, so there is no validate endpoint.
//!
//! ## Path-extractor convention
//!
//! Handlers extract path segments as `Path<String>` and call
//! `WorkspaceId::parse(&id)?` (or `JobId::parse(&job)?`) on the first
//! line. **Do not switch to `Path<WorkspaceId>` /
//! `Path<JobId>` directly:** `serde(transparent)` accepts the same
//! string set our explicit `parse()` does (no validation gain) and
//! axum's `PathRejection` returns plain-text 400 bodies that bypass
//! the `{error, code}` envelope (consistency loss).

use std::sync::Arc;

use crate::common::ids::WorkspaceId;
use crate::common::workspace::{HeadStatus, WorkspaceCore, WorkspaceRevision};
use crate::file_mgr::FsService;
use axum::Router;
use axum::extract::{Path, State};
use axum::response::Json;
use axum::routing::get;
use serde::{Deserialize, Serialize};
use tokio::task;

use crate::api::AppState;
use crate::api::error::ApiError;
use crate::api::extract::ApiJson;

#[derive(Serialize)]
struct WorkspaceListResp {
    workspaces: Vec<WorkspaceListEntry>,
}

#[derive(Serialize)]
struct WorkspaceListEntry {
    id: String,
    name: String,
    created_at: String,
}

async fn list_workspaces(
    State(files): State<Arc<dyn FsService>>,
) -> Result<Json<WorkspaceListResp>, ApiError> {
    // Hot path: reads only the cached `workspace.json` core per
    // workspace, never walks `datasets/`.
    let resp = task::spawn_blocking(move || -> Result<_, ApiError> {
        let ids = files.list_workspaces()?;
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            let summary = files.summary(&id)?;
            out.push(WorkspaceListEntry {
                id: summary.core.id.to_string(),
                name: summary.core.name.clone(),
                created_at: summary.core.created_at.clone(),
            });
        }
        Ok(WorkspaceListResp { workspaces: out })
    })
    .await??;
    Ok(Json(resp))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CreateWorkspaceReq {
    name: String,
    /// Optional operator tags; defaults to empty.  Per-tag validation
    /// runs in `WorkspaceMgr::create_with_tags`.
    #[serde(default)]
    tags: Vec<String>,
}

#[derive(Serialize)]
struct CreateWorkspaceResp {
    id: String,
    name: String,
    tags: Vec<String>,
    created_at: String,
    workspace_revision: WorkspaceRevision,
}

fn workspace_lifecycle_resp(core: &WorkspaceCore) -> CreateWorkspaceResp {
    CreateWorkspaceResp {
        id: core.id.to_string(),
        name: core.name.clone(),
        tags: core.tags.clone(),
        created_at: core.created_at.clone(),
        workspace_revision: core.workspace_revision.clone(),
    }
}

async fn create_workspace(
    State(files): State<Arc<dyn FsService>>,
    ApiJson(req): ApiJson<CreateWorkspaceReq>,
) -> Result<Json<CreateWorkspaceResp>, ApiError> {
    if req.name.is_empty() {
        return Err(ApiError::Bad("name must be non-empty".into()));
    }
    let name = req.name.clone();
    let tags = req.tags.clone();
    let files_for_summary = files.clone();
    let id = task::spawn_blocking(move || files.create_with_tags(&name, &tags)).await??;
    let summary = task::spawn_blocking(move || files_for_summary.summary(&id)).await??;
    Ok(Json(workspace_lifecycle_resp(&summary.core)))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PatchWorkspaceReq {
    /// New display name.  Length / charset / Unicode case-insensitive
    /// uniqueness (excluding the target) enforced daemon-side.
    #[serde(default)]
    name: Option<String>,
    /// New tag set; replaces (does not merge with) the existing tags.
    /// Max 32 entries; trimmed + de-duplicated under Unicode
    /// case-insensitive comparison.
    #[serde(default)]
    tags: Option<Vec<String>>,
}

/// Atomic update of one or both workspace metadata fields.  Name and
/// tag edits do NOT advance `workspace_revision`, `head_count`, or
/// head freshness -- they're operator metadata, not workspace
/// mutations.
async fn patch_workspace(
    State(files): State<Arc<dyn FsService>>,
    Path(id): Path<String>,
    ApiJson(req): ApiJson<PatchWorkspaceReq>,
) -> Result<Json<CreateWorkspaceResp>, ApiError> {
    let id = WorkspaceId::parse(&id)?;
    if req.name.is_none() && req.tags.is_none() {
        return Err(ApiError::Bad(
            "PATCH /workspace requires at least one of `name` or `tags`".into(),
        ));
    }
    let name = req.name.clone();
    let tags = req.tags.clone();
    let core =
        task::spawn_blocking(move || files.patch_workspace(&id, name.as_deref(), tags.as_deref()))
            .await?
            .map_err(|e| classify_workspace_existence_error(&id, e))?;
    Ok(Json(workspace_lifecycle_resp(&core)))
}

#[derive(Serialize)]
struct WorkspaceHeadEntry {
    head_id: String,
    workspace_revision: WorkspaceRevision,
    sha256: String,
    n_classes: u32,
    size_bytes: u64,
    created_at: String,
    status: HeadStatus,
}

#[derive(Serialize)]
struct WorkspaceSummaryResp {
    id: String,
    name: String,
    created_at: String,
    workspace_revision: WorkspaceRevision,
    heads: Vec<WorkspaceHeadEntry>,
}

/// Hot summary read of `{id, name, created_at, workspace_revision,
/// heads: [...]}`.  Does NOT walk or return workspace files;
/// `GET /workspace/{id}/assets` owns that.
async fn get_workspace(
    State(files): State<Arc<dyn FsService>>,
    Path(id): Path<String>,
) -> Result<Json<WorkspaceSummaryResp>, ApiError> {
    let id = WorkspaceId::parse(&id)?;
    let summary = task::spawn_blocking(move || files.summary(&id))
        .await?
        .map_err(|e| classify_workspace_existence_error(&id, e))?;
    let heads = summary
        .heads
        .heads
        .iter()
        .zip(summary.head_statuses.iter())
        .map(|(rec, status)| WorkspaceHeadEntry {
            head_id: rec.head_id.to_string(),
            workspace_revision: rec.workspace_revision.clone(),
            sha256: rec.sha256.clone(),
            n_classes: rec.n_classes,
            size_bytes: rec.size_bytes,
            created_at: rec.created_at.clone(),
            status: *status,
        })
        .collect();
    Ok(Json(WorkspaceSummaryResp {
        id: summary.core.id.to_string(),
        name: summary.core.name.clone(),
        created_at: summary.core.created_at.clone(),
        workspace_revision: summary.core.workspace_revision.clone(),
        heads,
    }))
}

#[derive(Serialize)]
struct DeleteWorkspaceResp {
    job_id: String,
}

/// Async workspace delete: stage the tree under
/// `.tmp/delete-workspace-<job_id>/payload`, return the `JobId`, and
/// drain the payload off the request hot path on the blocking pool.
async fn delete_workspace(
    State(files): State<Arc<dyn FsService>>,
    Path(id): Path<String>,
) -> Result<Json<DeleteWorkspaceResp>, ApiError> {
    let id = WorkspaceId::parse(&id)?;
    let job_id = task::spawn_blocking(move || files.start_delete_workspace(&id))
        .await?
        .map_err(|e| classify_workspace_existence_error(&id, e))?;
    Ok(Json(DeleteWorkspaceResp {
        job_id: job_id.to_string(),
    }))
}

/// Re-classify a `summary` / `read_metadata` failure at the API
/// boundary.  `file_mgr` wraps every IO failure (including a missing
/// `workspace.json`) as `Internal`; the route has the operator's
/// intent (a valid UUID) so it can confidently walk the source chain
/// and promote a genuine ENOENT to `NotFound`.  Other IO shapes
/// (EACCES, ENOSPC, EIO, ELOOP, ...) stay `Internal` so the operator
/// gets a real 500 + diagnostic instead of a misleading 404.
pub(crate) fn classify_workspace_existence_error(
    id: &WorkspaceId,
    err: crate::file_mgr::FsError,
) -> ApiError {
    use crate::common::error::{Categorized, ErrorKind};
    let kind = err.kind();
    if matches!(kind, ErrorKind::NotFound) {
        return ApiError::NotFound(format!("workspace {id}: {err}"));
    }
    if matches!(kind, ErrorKind::Internal) {
        let mut src: Option<&(dyn std::error::Error + 'static)> = std::error::Error::source(&err);
        while let Some(s) = src {
            if let Some(io_err) = s.downcast_ref::<std::io::Error>() {
                if io_err.kind() == std::io::ErrorKind::NotFound {
                    return ApiError::NotFound(format!("workspace {id}: {err}"));
                }
                break;
            }
            src = s.source();
        }
    }
    err.into()
}

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/workspace", get(list_workspaces).post(create_workspace))
        .route(
            "/workspace/{id}",
            get(get_workspace)
                .patch(patch_workspace)
                .delete(delete_workspace),
        )
}
