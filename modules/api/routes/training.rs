//! Training endpoints: `POST /workspace/{id}/train` is the producer
//! (body is the flattened [`TrainingCfg`]; no wrapper, no dataset
//! path -- the trainer always walks `<workspace>/datasets/`), with
//! list / status / cancel surfaces under
//! `/workspace/{id}/training[/{job}]`.  Backbone selection is
//! daemon-side: the trainer reads the path from
//! [`crate::api::AppState::training_backbone_path`], which the
//! daemon resolves at boot from the first `kind = "burn"` entry
//! in the launch TOML's `[[backbone.candidates]]`.  No upload API.

use std::sync::Arc;

use crate::common::ids::{HeadId, JobId, WorkspaceId};
use crate::file_mgr::{FsService, TrainingCfg, validate_training_cfg};
use crate::training::{TrainingJob, TrainingRegistry};
use axum::Router;
use axum::extract::{Path, State};
use axum::response::Json;
use axum::routing::{get, post};
use serde::Serialize;

use crate::api::AppState;
use crate::api::error::ApiError;
use crate::api::extract::ApiJson;

/// Producer response: daemon-allocated identifiers caller uses to
/// observe progress through `GET /api/v1/training/{id}/{job}` (or
/// the unified `/jobs/{job_id}` SSE stream).
#[derive(Debug, Serialize)]
struct TrainStartResp {
    /// Pre-allocated head id stable across the job lifetime; the
    /// index entry is committed only after `publish_trained_head`
    /// returns successfully.
    head_id: String,
    job_id: String,
}

#[derive(Serialize)]
struct TrainingListResp {
    jobs: Vec<crate::training::JobView>,
}

async fn start_training(
    State(state): State<AppState>,
    Path(id): Path<String>,
    ApiJson(cfg): ApiJson<TrainingCfg>,
) -> Result<Json<TrainStartResp>, ApiError> {
    let workspace_id = WorkspaceId::parse(&id)?;
    // Wire shape (deny_unknown_fields) already ran inside `ApiJson`;
    // this enforces only numeric range gates (epochs / batch / lr).
    validate_training_cfg(&cfg).map_err(|e| ApiError::Bad(e.to_string()))?;

    // Resolve the backbone path BEFORE the workspace check: a
    // launch-config without a Burn candidate is a deployment
    // misconfiguration the operator should see immediately, not
    // after a successful workspace lookup that suggests the
    // request is otherwise valid.  `state` is owned (axum's
    // `State<AppState>` extractor clones the AppState once at
    // the request boundary), so move the field out directly
    // rather than re-cloning the `PathBuf` inside.
    let backbone_path = state.training_backbone_path.ok_or_else(|| {
        ApiError::File(crate::file_mgr::io_err(
            "<launch.backbone.candidates[kind=\"burn\"]>",
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "no Burn backbone configured in launch TOML",
            ),
        ))
    })?;
    if !backbone_path.is_file() {
        return Err(ApiError::File(crate::file_mgr::io_err(
            backbone_path.display(),
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "deployment backbone not found",
            ),
        )));
    }

    // Workspace existence + revision snapshot on the blocking pool
    // so the runtime stays free under eMMC pressure.  The cached
    // `summary` read also seeds the per-workspace cell on first hit.
    let files = state.files.clone();
    let training = state.training.clone();
    let workspace_id_for_check = workspace_id;
    let files_for_check = files.clone();
    let workspace_revision: crate::common::workspace::WorkspaceRevision =
        tokio::task::spawn_blocking(move || {
            files_for_check
                .summary(&workspace_id_for_check)
                .map_err(ApiError::from)
                .map(|s| s.core.workspace_revision.clone())
        })
        .await??;

    // Allocate head id before spawn so the response can return it; the
    // publish at job end reuses the same id verbatim.
    let head_id = HeadId::new();
    let job = TrainingJob {
        workspace_id,
        head_id,
        workspace_revision,
        training_cfg: cfg,
        backbone_path,
    };

    let training_for_spawn = training.clone();
    let files_for_spawn = files.clone();
    let job_id =
        tokio::task::spawn_blocking(move || training_for_spawn.spawn(files_for_spawn, job))
            .await??;
    Ok(Json(TrainStartResp {
        head_id: head_id.to_string(),
        job_id: job_id.to_string(),
    }))
}

async fn list_training(
    State(training): State<Arc<dyn TrainingRegistry>>,
    State(files): State<Arc<dyn FsService>>,
    Path(id): Path<String>,
) -> Result<Json<TrainingListResp>, ApiError> {
    let workspace_id = WorkspaceId::parse(&id)?;
    // Existence proof so a missing workspace returns 404 rather than
    // `200 { jobs: [] }`.  Cached `summary()` read -- never walks
    // `datasets/`.
    let workspace_id_for_existence = workspace_id;
    let files_for_existence = files.clone();
    tokio::task::spawn_blocking(move || files_for_existence.summary(&workspace_id_for_existence))
        .await?
        .map_err(|e| {
            crate::api::routes::workspace::classify_workspace_existence_error(&workspace_id, e)
        })?;
    Ok(Json(TrainingListResp {
        jobs: training.list_for_workspace(&workspace_id),
    }))
}

async fn get_training(
    State(training): State<Arc<dyn TrainingRegistry>>,
    Path((id, job)): Path<(String, String)>,
) -> Result<Json<crate::training::JobView>, ApiError> {
    let workspace_id = WorkspaceId::parse(&id)?;
    let job_id = JobId::parse(&job)?;
    Ok(Json(training.status(&workspace_id, job_id)?))
}

async fn cancel_training(
    State(training): State<Arc<dyn TrainingRegistry>>,
    Path((id, job)): Path<(String, String)>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let workspace_id = WorkspaceId::parse(&id)?;
    let job_id = JobId::parse(&job)?;
    training.cancel(&workspace_id, job_id)?;
    Ok(Json(serde_json::json!({"ok": true})))
}

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/workspace/{id}/train", post(start_training))
        .route("/workspace/{id}/training", get(list_training))
        .route(
            "/workspace/{id}/training/{job}",
            get(get_training).delete(cancel_training),
        )
}
