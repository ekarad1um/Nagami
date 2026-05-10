//! Trained-head routes under `/workspace/{id}/heads/...`:
//!
//! - `GET    /workspace/{id}/heads` -- list with derived
//!   [`HeadStatus`]; cache-only, never walks disk.
//! - `GET    /workspace/{id}/heads/{head_id}` -- per-head manifest
//!   body.  Surfaces 404 when the id is not in the cached index even
//!   if `heads/<head_id>.json` exists as orphan residue.
//! - `DELETE /workspace/{id}/heads/{head_id}` -- sync remove: drops
//!   the index entry, atomic-rewrites `workspace.json` with the
//!   refreshed `head_count`, and unlinks `heads/<head_id>.{mpk,json}`.
//!   Returns 409 `JobConflict` on overlapping running jobs.

use std::sync::Arc;

use crate::common::ids::{HeadId, WorkspaceId};
use crate::common::workspace::{HeadManifest, HeadStatus, WorkspaceRevision};
use crate::file_mgr::FsService;
use axum::Router;
use axum::extract::{Path, State};
use axum::response::Json;
use axum::routing::get;
use serde::Serialize;
use tokio::task;

use crate::api::AppState;
use crate::api::error::ApiError;
use crate::api::routes::workspace::classify_workspace_existence_error;

#[derive(Serialize)]
struct HeadEntry {
    head_id: String,
    workspace_revision: WorkspaceRevision,
    sha256: String,
    n_classes: u32,
    size_bytes: u64,
    created_at: String,
    status: HeadStatus,
}

#[derive(Serialize)]
struct ListHeadsResp {
    heads: Vec<HeadEntry>,
}

/// Reads the cached `heads.json` snapshot + derives [`HeadStatus`]
/// from the current `workspace_revision`.  Wait-free w.r.t.
/// concurrent mutations.
async fn list_heads(
    State(files): State<Arc<dyn FsService>>,
    Path(id): Path<String>,
) -> Result<Json<ListHeadsResp>, ApiError> {
    let id = WorkspaceId::parse(&id)?;
    let summary = task::spawn_blocking(move || files.summary(&id))
        .await?
        .map_err(|e| classify_workspace_existence_error(&id, e))?;
    let heads = summary
        .heads
        .heads
        .iter()
        .zip(summary.head_statuses.iter())
        .map(|(rec, status)| HeadEntry {
            head_id: rec.head_id.to_string(),
            workspace_revision: rec.workspace_revision.clone(),
            sha256: rec.sha256.clone(),
            n_classes: rec.n_classes,
            size_bytes: rec.size_bytes,
            created_at: rec.created_at.clone(),
            status: *status,
        })
        .collect();
    Ok(Json(ListHeadsResp { heads }))
}

/// Validate the head id against the cached index BEFORE reading the
/// manifest so orphan `.json` files (boot-recovery residue) surface
/// as 404 instead of leaking their contents.
async fn get_head_manifest(
    State(files): State<Arc<dyn FsService>>,
    Path((id, head_id)): Path<(String, String)>,
) -> Result<Json<HeadManifest>, ApiError> {
    let id = WorkspaceId::parse(&id)?;
    let head_id = HeadId::parse(&head_id)?;
    let summary = task::spawn_blocking({
        let files = files.clone();
        move || files.summary(&id)
    })
    .await?
    .map_err(|e| classify_workspace_existence_error(&id, e))?;
    let in_index = summary.heads.heads.iter().any(|rec| rec.head_id == head_id);
    if !in_index {
        return Err(ApiError::NotFound(format!(
            "head {head_id} not in workspace {id} (heads.json index)"
        )));
    }
    let workspace_dir = crate::file_mgr::schema::workspace_dir_for(files.root(), &id);
    let manifest = task::spawn_blocking(move || {
        crate::file_mgr::schema::read_head_manifest(&workspace_dir, head_id)
    })
    .await?
    .map_err(|e| classify_workspace_existence_error(&id, crate::file_mgr::FsError::new(e)))?;
    Ok(Json(manifest))
}

#[derive(Serialize)]
struct DeleteHeadResp {
    deleted_head_id: String,
}

/// Synchronous single-head delete; the `WorkspaceMgr` impl gates
/// with the per-workspace mutation mutex and the job-reference
/// conflict check.  Failures classify as 404 (head/workspace missing)
/// or 409 (running job overlap).
async fn delete_head(
    State(files): State<Arc<dyn FsService>>,
    Path((id, head_id)): Path<(String, String)>,
) -> Result<Json<DeleteHeadResp>, ApiError> {
    let id = WorkspaceId::parse(&id)?;
    let head_id = HeadId::parse(&head_id)?;
    task::spawn_blocking(move || files.delete_head(&id, head_id))
        .await?
        .map_err(|e| classify_workspace_existence_error(&id, e))?;
    Ok(Json(DeleteHeadResp {
        deleted_head_id: head_id.to_string(),
    }))
}

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/workspace/{id}/heads", get(list_heads))
        .route(
            "/workspace/{id}/heads/{head_id}",
            get(get_head_manifest).delete(delete_head),
        )
}
