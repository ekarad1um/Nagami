//! Workspace asset routes (upload, list, get, delete) split out of
//! `routes/workspace.rs` so the lifecycle file stays focused on
//! `workspace.json` shape:
//!
//! - `POST /workspace/{id}/upload` -- multipart `{path, file}`;
//!   returns `{path, sha256, size_bytes, workspace_revision_id}`.
//! - `GET  /workspace/{id}/assets[?path=&offset=&limit=]` -- paginated
//!   direct-child listing.
//! - `GET  /workspace/{id}/assets/{*path}` -- file stream or directory
//!   listing depending on the resolved kind.
//! - `DELETE /workspace/{id}/assets/{*path}` -- async delete; returns
//!   `{job_id}`.

use std::path::PathBuf;
use std::sync::Arc;

use crate::common::asset_path::AssetPath;
use crate::common::ids::WorkspaceId;
use crate::file_mgr::log_page::{DEFAULT_LOG_PAGE_LIMIT, read_jsonl_page};
use crate::file_mgr::{
    DEFAULT_DATASET_LIST_LIMIT, DatasetListing, DatasetUploadReceipt, FsService,
    MAX_DATASET_LIST_LIMIT, content_type_from_path, hex_lowercase,
};
use axum::Router;
use axum::body::Body;
use axum::extract::{DefaultBodyLimit, Path, State};
use axum::http::header;
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::task;

use crate::api::AppState;
use crate::api::error::ApiError;
use crate::api::extract::{ApiMultipart, ApiQuery};
use crate::api::routes::workspace::classify_workspace_existence_error;

/// Parse a `{*path}` capture into an [`AssetPath`].  axum URL-decodes
/// wildcard captures, so a literal `%2E%2E%2F` arrives as `../` and
/// is rejected by the parser via `LeadingDot`.
fn parse_asset_path(raw: &str) -> Result<AssetPath, ApiError> {
    AssetPath::parse(raw).map_err(|e| ApiError::Bad(format!("invalid asset path: {e}")))
}

// MARK: GET /workspace/{id}/assets (paginated direct-child listing)

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ListAssetsQuery {
    /// Optional subdirectory under `datasets/`.  Defaults to
    /// the workspace's `datasets/` root.
    #[serde(default)]
    path: Option<String>,
    /// Default 0.
    #[serde(default)]
    offset: Option<usize>,
    /// Default [`DEFAULT_DATASET_LIST_LIMIT`]; clamped to
    /// [`MAX_DATASET_LIST_LIMIT`].
    #[serde(default)]
    limit: Option<usize>,
}

async fn list_assets(
    State(files): State<Arc<dyn FsService>>,
    Path(id): Path<String>,
    ApiQuery(q): ApiQuery<ListAssetsQuery>,
) -> Result<Json<DatasetListing>, ApiError> {
    let id = WorkspaceId::parse(&id)?;
    let dir = q
        .path
        .as_deref()
        .map(AssetPath::parse)
        .transpose()
        .map_err(|e| ApiError::Bad(e.to_string()))?;
    let offset = q.offset.unwrap_or(0);
    let limit = q
        .limit
        .unwrap_or(DEFAULT_DATASET_LIST_LIMIT)
        .min(MAX_DATASET_LIST_LIMIT);
    // `dir = None` lists the workspace root; `Some(p)` lists
    // `<workspace_dir>/<p>`.  `.tmp/` is excluded so internal staging
    // stays unaddressable.
    let listing = task::spawn_blocking(move || {
        files.list_workspace_children(&id, dir.as_ref(), offset, limit)
    })
    .await?
    .map_err(|e| classify_workspace_existence_error(&id, e))?;
    Ok(Json(listing))
}

// MARK: GET /workspace/{id}/assets/{*path} (file stream OR dir list OR JSONL page)

/// Per-call query for `GET /assets/{*path}`.  `?after_seq=` and
/// `?limit=` switch the response to a JSONL page (see
/// [`read_jsonl_page`]); both are accepted only on `.jsonl`
/// files so the byte-stream surface stays unambiguous for any
/// future RFC 7233 byte-range implementation.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GetAssetQuery {
    /// Cursor: yield only events with `seq > after_seq`.
    /// Optional; absent means "from the start".
    #[serde(default)]
    after_seq: Option<u64>,
    /// Per-call ceiling on lines returned.  Server clamps to
    /// [`crate::file_mgr::log_page::MAX_LOG_PAGE_LIMIT`].  Optional;
    /// absent means [`DEFAULT_LOG_PAGE_LIMIT`].
    #[serde(default)]
    limit: Option<usize>,
}

impl GetAssetQuery {
    /// Whether the caller is asking for JSONL paging rather than
    /// the byte-stream / directory-listing default.
    fn is_paging_request(&self) -> bool {
        self.after_seq.is_some() || self.limit.is_some()
    }
}

async fn get_asset(
    State(files): State<Arc<dyn FsService>>,
    Path((id, raw_path)): Path<(String, String)>,
    ApiQuery(q): ApiQuery<GetAssetQuery>,
) -> Result<Response, ApiError> {
    let id = WorkspaceId::parse(&id)?;
    let asset_path = parse_asset_path(&raw_path)?;

    // Resolve + stat off-mutex (read-only filesystem op) so we can
    // distinguish file vs directory before reading.
    let files_for_resolve = files.clone();
    let asset_path_for_resolve = asset_path.clone();
    let resolved: PathBuf = task::spawn_blocking(move || {
        files_for_resolve.workspace_asset_path(&id, &asset_path_for_resolve)
    })
    .await?
    .map_err(|e| classify_workspace_existence_error(&id, e))?;

    let stat_path = resolved.clone();
    let md = match task::spawn_blocking(move || std::fs::symlink_metadata(&stat_path)).await? {
        Ok(md) => md,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(ApiError::NotFound(format!(
                "asset {} in workspace {id}",
                asset_path.as_str(),
            )));
        }
        Err(source) => {
            return Err(ApiError::File(crate::file_mgr::io_err(
                resolved.display(),
                source,
            )));
        }
    };

    if md.is_dir() {
        // Paging applies to JSONL files only; surfacing it on a
        // directory would silently mask a client typo (e.g. a
        // missing trailing `<id>.jsonl`).  Reject up front.
        if q.is_paging_request() {
            return Err(ApiError::Bad(format!(
                "?after_seq= / ?limit= apply only to .jsonl files; \
                 path {:?} resolved to a directory",
                asset_path.as_str(),
            )));
        }
        let files_for_dir = files.clone();
        let asset_for_dir = asset_path.clone();
        let listing = task::spawn_blocking(move || {
            files_for_dir.list_workspace_children(
                &id,
                Some(&asset_for_dir),
                0,
                DEFAULT_DATASET_LIST_LIMIT,
            )
        })
        .await?
        .map_err(|e| classify_workspace_existence_error(&id, e))?;
        return Ok(Json(listing).into_response());
    }
    if !md.is_file() {
        return Err(ApiError::Bad(format!(
            "asset {} is not a regular file or directory",
            asset_path.as_str(),
        )));
    }

    // JSONL paging branch: caller asked for `?after_seq=` /
    // `?limit=`, the resolved path is a file, and the file
    // extension says `.jsonl`.  Reads through
    // `crate::file_mgr::log_page::read_jsonl_page`, the canonical
    // forward-only paginator for the daemon's per-job JSONL
    // backstop (`<workspace>/{training,converter}_logs/<id>.jsonl`).
    if q.is_paging_request() {
        let is_jsonl = resolved
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("jsonl"));
        if !is_jsonl {
            return Err(ApiError::Bad(format!(
                "?after_seq= / ?limit= apply only to .jsonl files; \
                 path {:?} has a different extension",
                asset_path.as_str(),
            )));
        }
        let after_seq = q.after_seq.unwrap_or(0);
        let limit = q.limit.unwrap_or(DEFAULT_LOG_PAGE_LIMIT);
        let path_for_page = resolved.clone();
        let page = task::spawn_blocking(move || read_jsonl_page(&path_for_page, after_seq, limit))
            .await?
            .map_err(|source| {
                ApiError::File(crate::file_mgr::io_err(resolved.display(), source))
            })?;
        return Ok(Json(page).into_response());
    }

    // The workspace mutex is NOT held during streaming -- the resolve
    // already validated the path; the body flows through
    // `tokio::fs::File` so concurrent mutations are not blocked.
    let f = tokio::fs::File::open(&resolved)
        .await
        .map_err(|source| ApiError::File(crate::file_mgr::io_err(resolved.display(), source)))?;
    let stream = tokio_util::io::ReaderStream::new(f);
    let body = Body::from_stream(stream);
    let content_type = content_type_from_path(&resolved);
    let resp = Response::builder()
        .header(header::CONTENT_TYPE, content_type)
        .header(header::CONTENT_LENGTH, md.len())
        .body(body)
        .map_err(|e| {
            ApiError::File(crate::file_mgr::io_err(
                resolved.display(),
                std::io::Error::other(e),
            ))
        })?;
    Ok(resp)
}

// MARK: DELETE /workspace/{id}/assets/{*path}

/// Async-delete response: returned for `datasets/...` /
/// `converters/...` paths and for the bare `datasets` /
/// `converters` whole-tree wipes.  Wire status is `202 Accepted`
/// — the rename + tombstone landed durably under the
/// per-workspace mutex but the staged drain runs in the
/// background, so the request is queued, not done.
#[derive(Serialize)]
struct AsyncDeleteResp {
    job_id: String,
}

/// Sync-delete response: returned for `training_logs[/...]` and
/// `converter_logs[/...]` paths.  Wire status is `200 OK` — the
/// unlinks have already happened.  `removed` is the count of
/// `.jsonl` files unlinked (0 if the dir was already empty or
/// the requested file did not exist).
#[derive(Serialize)]
struct SyncRemovedResp {
    removed: usize,
}

async fn delete_asset(
    State(files): State<Arc<dyn FsService>>,
    Path((id, raw_path)): Path<(String, String)>,
) -> Result<Response, ApiError> {
    let id = WorkspaceId::parse(&id)?;
    let asset_path = parse_asset_path(&raw_path)?;
    let outcome =
        task::spawn_blocking(move || files.start_workspace_asset_delete(&id, &asset_path))
            .await?
            .map_err(|e| classify_workspace_existence_error(&id, e))?;
    Ok(match outcome {
        crate::file_mgr::AssetDeleteOutcome::Async { job_id } => (
            axum::http::StatusCode::ACCEPTED,
            Json(AsyncDeleteResp {
                job_id: job_id.to_string(),
            }),
        )
            .into_response(),
        crate::file_mgr::AssetDeleteOutcome::SyncRemoved { removed } => {
            // 200 OK: the wipe is done before the response
            // returns.
            Json(SyncRemovedResp { removed }).into_response()
        }
    })
}

// MARK: POST /workspace/{id}/upload (multipart {path, file})

/// Multipart contract: `path` MUST arrive before `file`.  The
/// streaming write needs the validated path before allocating the
/// tempfile, so a `file` field seen first is rejected with 400.
async fn upload_dataset(
    State(files): State<Arc<dyn FsService>>,
    Path(id): Path<String>,
    ApiMultipart(mut multipart): ApiMultipart,
) -> Result<Json<DatasetUploadReceipt>, ApiError> {
    let id = WorkspaceId::parse(&id)?;

    // Existence check before streaming so an upload to a phantom
    // workspace never creates an orphan tempfile.
    let id_for_existence = id;
    let files_for_existence = files.clone();
    task::spawn_blocking(move || files_for_existence.summary(&id_for_existence))
        .await?
        .map_err(|e| classify_workspace_existence_error(&id, e))?;

    // Permit before any body bytes are consumed; drops at fn exit.
    let _permit = files.acquire_upload_permit()?;

    let mut path: Option<AssetPath> = None;
    let mut receipt: Option<DatasetUploadReceipt> = None;
    while let Some(field) = multipart.next_field().await? {
        let fname = field.name().map(|s| s.to_string()).unwrap_or_default();
        match fname.as_str() {
            "kind" => {
                // Reject explicitly so old clients get a clear diagnostic
                // rather than a silent misclassification.
                return Err(ApiError::Bad(
                    "multipart `kind` is no longer accepted; the upload route \
                     takes `{path, file}`.  Use `path` to give the \
                     workspace-rooted destination, e.g. \
                     `datasets/<class>/sample.wav` or \
                     `converters/tfjs/model.json` (a leading slash is also \
                     accepted)."
                        .into(),
                ));
            }
            "path" => {
                if receipt.is_some() {
                    return Err(ApiError::Bad(
                        "duplicate `path` field after upload commit".into(),
                    ));
                }
                if path.is_some() {
                    return Err(ApiError::Bad("duplicate `path` field".into()));
                }
                let text = read_field_text_capped(field, MAX_TEXT_FIELD_BYTES).await?;
                // Accept `/datasets/...` or `datasets/...` (and same
                // for `converters/`); canonical form drops the leading
                // slash before `AssetPath::parse` runs syntactic
                // validation and the file_mgr enforces the top-level
                // + non-empty-child constraints.
                let canonical = text.trim().trim_start_matches('/');
                let parsed = AssetPath::parse(canonical)
                    .map_err(|e| ApiError::Bad(format!("invalid `path`: {e}")))?;
                path = Some(parsed);
            }
            "file" => {
                if receipt.is_some() {
                    return Err(ApiError::Bad("duplicate `file` field".into()));
                }
                let asset_path = path.clone().ok_or_else(|| {
                    ApiError::Bad("multipart `path` field must precede `file`".into())
                })?;

                let tmp_dir = files.workspace_tmpdir(&id);
                tokio::fs::create_dir_all(&tmp_dir)
                    .await
                    .map_err(|e| ApiError::File(crate::file_mgr::io_err(tmp_dir.display(), e)))?;
                let tmp_dir_for_spawn = tmp_dir.clone();
                let tmp_file = tokio::task::spawn_blocking(move || {
                    tempfile::NamedTempFile::new_in(&tmp_dir_for_spawn)
                })
                .await?
                .map_err(|e| ApiError::File(crate::file_mgr::io_err(tmp_dir.display(), e)))?;

                // Reject mid-stream the moment cumulative bytes cross
                // the per-request cap; the tempfile drops on Err
                // without partial commit.
                let max_upload_bytes = files.max_upload_bytes();
                let tmp_reopened = tmp_file.reopen().map_err(|e| {
                    ApiError::File(crate::file_mgr::io_err(tmp_file.path().display(), e))
                })?;
                let mut writer = tokio::fs::File::from_std(tmp_reopened);
                use futures_util::TryStreamExt as _;
                use tokio::io::{AsyncReadExt, AsyncWriteExt};
                let stream = field.map_err(std::io::Error::other);
                let mut reader = tokio_util::io::StreamReader::new(stream);
                let mut buf = vec![0u8; 64 * 1024];
                let mut total: u64 = 0;
                let mut hasher = Sha256::new();
                loop {
                    let n = reader.read(&mut buf).await.map_err(|e| {
                        ApiError::File(crate::file_mgr::io_err("<upload-stream>", e))
                    })?;
                    if n == 0 {
                        break;
                    }
                    total = total.saturating_add(n as u64);
                    if total > max_upload_bytes {
                        return Err(ApiError::Fs(crate::file_mgr::FsError::new(
                            crate::file_mgr::FileError::PayloadTooLarge {
                                observed: total,
                                max: max_upload_bytes,
                            },
                        )));
                    }
                    hasher.update(&buf[..n]);
                    writer.write_all(&buf[..n]).await.map_err(|e| {
                        ApiError::File(crate::file_mgr::io_err(tmp_file.path().display(), e))
                    })?;
                }
                writer.flush().await.map_err(|e| {
                    ApiError::File(crate::file_mgr::io_err(tmp_file.path().display(), e))
                })?;
                writer.sync_all().await.map_err(|e| {
                    ApiError::File(crate::file_mgr::io_err(tmp_file.path().display(), e))
                })?;
                drop(writer);

                let digest = hex_lowercase(&hasher.finalize());

                // Atomic commit on the blocking pool.
                let files_for_install = files.clone();
                let id_for_install = id;
                let asset_path_for_install = asset_path.clone();
                let r = tokio::task::spawn_blocking(move || {
                    files_for_install.upload_workspace_file(
                        &id_for_install,
                        &asset_path_for_install,
                        tmp_file.path(),
                        &digest,
                        total,
                    )
                })
                .await??;
                receipt = Some(r);
            }
            other => {
                tracing::debug!(target: "api", field = other, "draining unknown multipart field");
                drain_field(field).await?;
            }
        }
    }
    receipt
        .ok_or_else(|| ApiError::Bad("multipart upload missing `path` and / or `file`".into()))
        .map(Json)
}

/// Cap for small text multipart fields (`path`).  256 bytes is the
/// `AssetPath` total cap; the extra room is whitespace tolerance.
pub(crate) const MAX_TEXT_FIELD_BYTES: usize = 320;

async fn read_field_text_capped(
    mut field: axum::extract::multipart::Field<'_>,
    cap: usize,
) -> Result<String, ApiError> {
    let mut buf = Vec::new();
    while let Some(chunk) = field.chunk().await? {
        if buf.len().saturating_add(chunk.len()) > cap {
            return Err(ApiError::Bad(format!(
                "multipart text field exceeds {cap} bytes"
            )));
        }
        buf.extend_from_slice(&chunk);
    }
    String::from_utf8(buf)
        .map_err(|_| ApiError::Bad("multipart text field is not valid UTF-8".into()))
}

async fn drain_field(mut field: axum::extract::multipart::Field<'_>) -> Result<(), ApiError> {
    while let Some(_chunk) = field.chunk().await? {}
    Ok(())
}

pub fn router() -> Router<AppState> {
    Router::new()
        .route(
            "/workspace/{id}/upload",
            post(upload_dataset).layer(DefaultBodyLimit::max(256 * 1024 * 1024)),
        )
        .route("/workspace/{id}/assets", get(list_assets))
        .route(
            "/workspace/{id}/assets/{*path}",
            get(get_asset).delete(delete_asset),
        )
}
