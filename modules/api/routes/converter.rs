//! `POST /workspace/{id}/convert` -- producer for the head-extraction
//! convert pipeline.  The route validates the request, snapshots
//! workspace revision, resolves each converter-rooted input path on
//! the blocking pool, takes a `Workspace` job-reference (so a
//! concurrent `WorkspaceDelete` is excluded but uploads and
//! file-deletes overlap freely), allocates the destination
//! `head_id`, then spawns the conversion and returns
//! `{head_id, job_id}` immediately.  Convert concurrency is bounded
//! by the `max_convert_jobs` semaphore acquired before admission.

use std::path::PathBuf;
use std::sync::Arc;

use crate::common::asset_path::AssetPath;
use crate::common::ids::{HeadId, WorkspaceId};
use crate::common::workspace::{JobReference, JobType, WorkspaceRevision};
use crate::file_mgr::{ConvertRequest, FsService, JobRegistry, validate_convert_request};
use axum::Router;
use axum::extract::{Path, State};
use axum::response::Json;
use axum::routing::post;
use serde::Serialize;
use tokio::task;

use crate::api::AppState;
use crate::api::error::ApiError;
use crate::api::extract::ApiJson;

/// SHA-256 hex helper kept `pub(crate)` so the dataset-upload
/// streaming tests can verify the production digest against an
/// independent hash.  Test-only today; the `allow(dead_code)`
/// keeps non-test builds clippy-clean.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::Digest;
    let d = sha2::Sha256::digest(bytes);
    d.iter().map(|b| format!("{b:02x}")).collect()
}

/// Producer response: the daemon-allocated identifiers the caller
/// uses to follow progress through `converter_logs/<job_id>.jsonl`
/// and the `/jobs/{job_id}` SSE stream.
#[derive(Debug, Serialize)]
struct ConvertStartResp {
    /// Pre-allocated head id stable across the job lifetime; the
    /// index entry is committed only after `publish_trained_head`
    /// returns successfully.
    head_id: String,
    job_id: String,
}

/// Resolve a converter-rooted input to its on-disk path, promoting
/// ENOENT to 404 and rejecting non-regular files (dir/symlink) as 400.
fn resolve_converter_input(
    files: &Arc<dyn FsService>,
    workspace_id: &WorkspaceId,
    path: &AssetPath,
) -> Result<PathBuf, ApiError> {
    let (resolved, md) = files.open_workspace_file(workspace_id, path).map_err(|e| {
        use std::error::Error as _;
        let is_not_found = e
            .source()
            .and_then(|s| s.downcast_ref::<crate::file_mgr::FileError>())
            .map(|fe| {
                matches!(
                    fe,
                    crate::file_mgr::FileError::Io { source, .. }
                        if source.kind() == std::io::ErrorKind::NotFound
                )
            })
            .unwrap_or(false);
        if is_not_found {
            ApiError::NotFound(format!(
                "convert input not found: /{}",
                strip_converter_prefix(path)
            ))
        } else {
            ApiError::from(e)
        }
    })?;
    if !md.is_file() {
        return Err(ApiError::Bad(format!(
            "convert input /{} is not a regular file",
            strip_converter_prefix(path),
        )));
    }
    Ok(resolved)
}

/// Render a `converters/<sub>` workspace-rooted path back to the
/// wire-form `<sub>` for operator-facing diagnostics only.
fn strip_converter_prefix(path: &AssetPath) -> &str {
    path.as_str()
        .strip_prefix("converters/")
        .expect("converter input path starts with converters/")
}

async fn start_convert(
    State(files): State<Arc<dyn FsService>>,
    State(jobs): State<Arc<JobRegistry>>,
    Path(id): Path<String>,
    ApiJson(req): ApiJson<ConvertRequest>,
) -> Result<Json<ConvertStartResp>, ApiError> {
    let workspace_id = WorkspaceId::parse(&id)?;
    // Path traversal was rejected at deserialize via ConverterPath; this
    // pass enforces only cardinality (shards >= 1, <= MAX_CONVERT_SHARDS).
    validate_convert_request(&req).map_err(|e| ApiError::Bad(e.to_string()))?;

    // Workspace existence + revision snapshot + per-file regular-file
    // resolution all happen on the blocking pool so the runtime stays
    // free under eMMC pressure.
    let files_for_resolve = files.clone();
    let req_for_resolve = req.clone();
    let (workspace_revision, model_json_resolved, shards_resolved, labels_resolved, labels_format): (
        WorkspaceRevision,
        PathBuf,
        Vec<PathBuf>,
        PathBuf,
        crate::file_mgr::LabelsFormat,
    ) = task::spawn_blocking(move || {
        let summary = files_for_resolve
            .summary(&workspace_id)
            .map_err(ApiError::from)?;
        // New converters land behind the discriminator without
        // touching the surrounding admission shape.
        let ConvertRequest::Tfjs(params) = &req_for_resolve;
        let model_json =
            resolve_converter_input(&files_for_resolve, &workspace_id, params.model_json_path.workspace_path())?;
        let mut shards: Vec<PathBuf> = Vec::with_capacity(params.shards.len());
        for s in &params.shards {
            shards.push(resolve_converter_input(
                &files_for_resolve,
                &workspace_id,
                s.workspace_path(),
            )?);
        }
        let labels = resolve_converter_input(
            &files_for_resolve,
            &workspace_id,
            params.labels_path.workspace_path(),
        )?;
        Ok::<_, ApiError>((
            summary.core.workspace_revision.clone(),
            model_json,
            shards,
            labels,
            params.labels_format,
        ))
    })
    .await??;

    // Single-tenant by design (`max_convert_jobs = 1`); a second
    // concurrent request gets `ConvertError::Busy` -> 409 here, before
    // any job-reference lease is taken.
    let convert_permit = crate::converter::acquire_convert_permit()?;

    // The registry-allocated job id is reused for the JSONL log
    // filename so operators can correlate `GET /jobs` and the on-disk
    // log by id.  The single Workspace reference excludes only
    // `WorkspaceDelete`; uploads and file-deletes overlap freely.
    let job_handle = jobs
        .try_acquire(
            JobType::Convert,
            vec![JobReference::Workspace { workspace_id }],
            None,
        )
        .map_err(|c| ApiError::File(crate::file_mgr::FileError::from(c)))?;
    let job_id = job_handle.job_id();

    // Allocate head id before spawn so the response can return it; the
    // publish at job end reuses the same id verbatim.
    let head_id = HeadId::new();

    let files_for_worker = files.clone();
    let job = crate::converter::ConvertJob {
        job_id,
        workspace_id,
        head_id,
        workspace_revision,
        model_json_path: model_json_resolved,
        shard_paths: shards_resolved,
        labels_path: labels_resolved,
        labels_format,
    };
    tokio::task::spawn_blocking(move || {
        // Permit moves into the worker so the slot stays held until
        // the job terminates.
        let _convert_permit = convert_permit;
        match crate::converter::run_convert_job(files_for_worker, job) {
            Ok(out) => {
                job_handle.succeed(Some(crate::file_mgr::RegistryJobResult::Convert {
                    head_id: out.head_id,
                    sha256: out.sha256,
                    n_classes: out.n_classes,
                }));
            }
            Err(e) => {
                tracing::warn!(
                    target: "converter",
                    job_id = %job_id,
                    workspace_id = %workspace_id,
                    err = %e,
                    "convert job failed",
                );
                job_handle.fail(format!("{e}"));
            }
        }
    });

    Ok(Json(ConvertStartResp {
        head_id: head_id.to_string(),
        job_id: job_id.to_string(),
    }))
}

pub fn router() -> Router<AppState> {
    Router::new().route("/workspace/{id}/convert", post(start_convert))
}
