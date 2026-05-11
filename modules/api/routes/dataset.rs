//! Workspace asset routes (list, get, put/upload, delete) split out
//! of `routes/workspace.rs` so the lifecycle file stays focused on
//! `workspace.json` shape.  All addressed-asset operations share
//! the same URI family `/workspace/{id}/assets/{*path}`; the HTTP
//! method picks the operation:
//!
//! - `GET    /workspace/{id}/assets[?offset=&limit=]` --
//!   paginated direct-child listing rooted at the workspace dir.
//! - `GET    /workspace/{id}/assets/{*path}` -- file stream OR
//!   directory listing OR JSONL page OR byte-range slice,
//!   dispatched by the resolved kind + query.  Byte-range slices
//!   (`?byte_offset=` / `?byte_limit=`) work on any regular file
//!   (including `.jsonl`); they are mutually exclusive with the
//!   JSONL-paging namespace (`?after_seq=` / `?limit=`).
//! - `PUT    /workspace/{id}/assets/{*path}` -- upload raw file
//!   bytes in the request body; returns
//!   `{path, sha256, size_bytes, workspace_revision_id}`.
//!   Idempotent at the bytes-on-disk level: the same body to the
//!   same path produces the same final state (atomic
//!   rename-into-tree); each call still bumps
//!   `workspace.json.workspace_revision` per the daemon's revision
//!   discipline.
//! - `DELETE /workspace/{id}/assets/{*path}` -- async delete;
//!   returns `{job_id}`.

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
use axum::routing::get;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::task;

use crate::api::AppState;
use crate::api::error::ApiError;
use crate::api::extract::ApiQuery;
use crate::api::routes::workspace::classify_workspace_existence_error;

/// Parse a `{*path}` capture into an [`AssetPath`].  axum URL-decodes
/// wildcard captures, so a literal `%2E%2E%2F` arrives as `../` and
/// is rejected by the parser via `LeadingDot`.
fn parse_asset_path(raw: &str) -> Result<AssetPath, ApiError> {
    AssetPath::parse(raw).map_err(|e| ApiError::Bad(format!("invalid asset path: {e}")))
}

// MARK: GET /workspace/{id}/assets (paginated workspace-root listing)

/// Per-call query for the root listing.  Subdirectory listings
/// live on the wildcard form (`GET /assets/{*path}`) which carries
/// the same `?offset=&limit=` fields, so a `?path=` here would be
/// redundant with the wildcard URL.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ListAssetsQuery {
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
    let offset = q.offset.unwrap_or(0);
    let limit = q
        .limit
        .unwrap_or(DEFAULT_DATASET_LIST_LIMIT)
        .min(MAX_DATASET_LIST_LIMIT);
    // `None` here lists the workspace root; subdirectory listings
    // route through `get_asset` via `/assets/{*path}` which shares
    // the same `?offset=&limit=` fields.  `.tmp/` is excluded so
    // internal staging stays unaddressable.
    let listing =
        task::spawn_blocking(move || files.list_workspace_children(&id, None, offset, limit))
            .await?
            .map_err(|e| classify_workspace_existence_error(&id, e))?;
    Ok(Json(listing))
}

// MARK: GET /workspace/{id}/assets/{*path} (file stream OR dir list OR JSONL page)

/// Per-call query for `GET /assets/{*path}`.  Two resolved kinds
/// (directory or regular file) accept three query namespaces;
/// the per-kind dispatch keeps the wire surface unambiguous:
///
/// * Directory listing -- `?offset=` + `?limit=` (same shape as
///   the root-listing endpoint).  Server clamps `limit` to
///   [`MAX_DATASET_LIST_LIMIT`] and defaults to
///   [`DEFAULT_DATASET_LIST_LIMIT`].
/// * `.jsonl` page (file only) -- `?after_seq=` cursor +
///   `?limit=` line ceiling (see [`read_jsonl_page`]).  Either
///   field alone triggers JSONL-page mode; server clamps `limit`
///   to [`crate::file_mgr::log_page::MAX_LOG_PAGE_LIMIT`].
/// * Byte-range slice (file only, any extension) --
///   `?byte_offset=` + `?byte_limit=`.  Either field alone
///   triggers byte-slice mode; the response is the requested
///   byte range with `Content-Length` reflecting the slice size.
///   Mutually exclusive with the JSONL-page namespace -- mixing
///   `byte_offset`/`byte_limit` with `after_seq`/`limit` returns
///   400.
///
/// A regular file with no query falls through all three branches
/// and streams the full bytes.
///
/// `after_seq` is JSONL-only (rejected on dirs and non-`.jsonl`
/// files); `offset` is dir-only (rejected on any file);
/// `byte_offset` / `byte_limit` are file-only (rejected on dirs).
/// `limit` is shared between dir-listing and JSONL paging because
/// the per-kind clamp keeps the contract honest.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GetAssetQuery {
    /// JSONL paging cursor: yield only events with
    /// `seq > after_seq`.  Optional.  Rejected with 400 when the
    /// resolved path is not a `.jsonl` file (including
    /// directories).
    #[serde(default)]
    after_seq: Option<u64>,
    /// Pagination offset for directory listings.  Optional;
    /// absent means 0.  Rejected with 400 when the resolved path
    /// is a file (regular or `.jsonl`).
    #[serde(default)]
    offset: Option<usize>,
    /// Per-call ceiling.  Interpretation depends on the resolved
    /// kind: for dir listings, max entries returned (clamped to
    /// [`MAX_DATASET_LIST_LIMIT`]); for `.jsonl` pages, max lines
    /// returned (clamped to
    /// [`crate::file_mgr::log_page::MAX_LOG_PAGE_LIMIT`]);
    /// rejected with 400 on non-`.jsonl` files.
    #[serde(default)]
    limit: Option<usize>,
    /// Byte-slice start (inclusive).  Optional; absent means 0.
    /// Rejected with 400 when the resolved path is a directory,
    /// or when combined with the JSONL-paging namespace
    /// (`after_seq` / `limit`).  `byte_offset > file_size`
    /// returns 400 (out of range).
    #[serde(default)]
    byte_offset: Option<u64>,
    /// Byte-slice ceiling (max bytes returned).  Optional;
    /// absent means "from `byte_offset` to EOF".  Rejected with
    /// 400 in the same situations as `byte_offset`; silently
    /// clamps to the remainder if `byte_offset + byte_limit`
    /// exceeds the file size.
    #[serde(default)]
    byte_limit: Option<u64>,
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
        // Per-kind query scoping: `after_seq` is the JSONL paging
        // cursor and is meaningless on a directory; reject up
        // front so a client typo (e.g. a missing trailing
        // `<id>.jsonl`) doesn't get silently swallowed.
        if q.after_seq.is_some() {
            return Err(ApiError::Bad(format!(
                "?after_seq= applies only to .jsonl files; \
                 path {:?} resolved to a directory",
                asset_path.as_str(),
            )));
        }
        if q.byte_offset.is_some() || q.byte_limit.is_some() {
            return Err(ApiError::Bad(format!(
                "?byte_offset= / ?byte_limit= apply only to regular files; \
                 path {:?} resolved to a directory",
                asset_path.as_str(),
            )));
        }
        let offset = q.offset.unwrap_or(0);
        let limit = q
            .limit
            .unwrap_or(DEFAULT_DATASET_LIST_LIMIT)
            .min(MAX_DATASET_LIST_LIMIT);
        let files_for_dir = files.clone();
        let asset_for_dir = asset_path.clone();
        let listing = task::spawn_blocking(move || {
            files_for_dir.list_workspace_children(&id, Some(&asset_for_dir), offset, limit)
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

    // File branch.  `?offset=` is dir-only; reject early so the
    // operator sees a clear "wrong shape for this resource"
    // rather than a silently-ignored parameter.
    if q.offset.is_some() {
        return Err(ApiError::Bad(format!(
            "?offset= applies only to directory listings; \
             path {:?} resolved to a file",
            asset_path.as_str(),
        )));
    }

    let wants_byte_range = q.byte_offset.is_some() || q.byte_limit.is_some();
    let wants_jsonl_page = q.after_seq.is_some() || q.limit.is_some();

    // Byte-range slice and JSONL paging are mutually exclusive
    // namespaces: the slice surface returns raw bytes, JSONL
    // paging returns a parsed event page.  Mixing the two is a
    // client confusion -- surface 400 rather than silently
    // picking one.
    if wants_byte_range && wants_jsonl_page {
        return Err(ApiError::Bad(format!(
            "?byte_offset= / ?byte_limit= cannot combine with \
             ?after_seq= / ?limit= (byte-range slice and JSONL \
             paging are mutually exclusive); path {:?}",
            asset_path.as_str(),
        )));
    }

    // JSONL paging branch.  Either `?after_seq=` or `?limit=`
    // signals JSONL-page intent on a file (the cursor names the
    // first event to skip; `limit` caps the line count).  Both
    // are accepted only on `.jsonl` files so the byte-stream
    // surface stays unambiguous; clients fetching raw `.jsonl`
    // bytes use `?byte_offset=` / `?byte_limit=` instead.  Reads
    // through [`read_jsonl_page`], the canonical forward-only
    // paginator for the daemon's per-job JSONL backstop
    // (`<workspace>/{training,converter}_logs/<id>.jsonl`).
    if wants_jsonl_page {
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

    // Byte-stream branch -- full file or `?byte_offset=` /
    // `?byte_limit=` slice.  The `Content-Length` always
    // reflects the slice (or full size) the response actually
    // carries; out-of-range `byte_offset` is 400, an oversized
    // `byte_limit` clamps silently to the remainder.
    let total = md.len();
    let (byte_offset, take_limit) = if wants_byte_range {
        let off = q.byte_offset.unwrap_or(0);
        if off > total {
            return Err(ApiError::Bad(format!(
                "?byte_offset={off} exceeds file size {total} for {:?}",
                asset_path.as_str(),
            )));
        }
        let remaining = total - off;
        let take = q.byte_limit.unwrap_or(remaining).min(remaining);
        (off, take)
    } else {
        (0, total)
    };

    // The workspace mutex is NOT held during streaming -- the resolve
    // already validated the path; the body flows through
    // `tokio::fs::File` so concurrent mutations are not blocked.
    use tokio::io::{AsyncReadExt as _, AsyncSeekExt as _};
    let mut f = tokio::fs::File::open(&resolved)
        .await
        .map_err(|source| ApiError::File(crate::file_mgr::io_err(resolved.display(), source)))?;
    if byte_offset > 0 {
        f.seek(std::io::SeekFrom::Start(byte_offset))
            .await
            .map_err(|source| {
                ApiError::File(crate::file_mgr::io_err(resolved.display(), source))
            })?;
    }
    let stream = tokio_util::io::ReaderStream::new(f.take(take_limit));
    let body = Body::from_stream(stream);
    let content_type = content_type_from_path(&resolved);
    let resp = Response::builder()
        .header(header::CONTENT_TYPE, content_type)
        .header(header::CONTENT_LENGTH, take_limit)
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

/// Async-delete response.  Wire status is always `202 Accepted`
/// — the rename + tombstone landed durably under the
/// per-workspace mutex but the staged drain runs in the
/// background, so the request is queued, not done.  Clients poll
/// `GET /jobs/{job_id}` or stream `GET /jobs/{job_id}/events` for
/// terminal state.
#[derive(Serialize)]
struct AsyncDeleteResp {
    job_id: String,
}

async fn delete_asset(
    State(files): State<Arc<dyn FsService>>,
    Path((id, raw_path)): Path<(String, String)>,
) -> Result<Response, ApiError> {
    let id = WorkspaceId::parse(&id)?;
    let asset_path = parse_asset_path(&raw_path)?;
    let job_id = task::spawn_blocking(move || files.start_workspace_asset_delete(&id, &asset_path))
        .await?
        .map_err(|e| classify_workspace_existence_error(&id, e))?;
    Ok((
        axum::http::StatusCode::ACCEPTED,
        Json(AsyncDeleteResp {
            job_id: job_id.to_string(),
        }),
    )
        .into_response())
}

// MARK: PUT /workspace/{id}/assets/{*path}

/// Upload a single file into `<workspace>/{datasets,converters}/<...>`.
///
/// The destination is the URL-path wildcard segment after `/assets/`;
/// the request body is the raw file bytes (no multipart envelope).
/// Same wildcard shape as the sibling `GET` and `DELETE` on
/// `/assets/{*path}`, so the asset surface addresses paths
/// consistently across read / write / delete via the HTTP method.
///
/// PUT (rather than POST) reflects the operator-named-URI semantics:
/// the caller fully specifies the destination, and a successful
/// upload places the named bytes at that URI.  Replays with the
/// same body produce the same final state on disk (the atomic
/// rename-into-tree replaces in place); the daemon's per-write
/// `workspace_revision` bump is bookkeeping, not part of the
/// resource state PUT addresses.
///
/// Body size is bounded two ways:
///
/// * A `DefaultBodyLimit::max(256 MiB)` middleware on the route.
///   tower-http's `RequestBodyLimit` short-circuits with HTTP 413
///   when an incoming `Content-Length` exceeds the cap; for chunked
///   transfers without `Content-Length`, the cap is enforced as
///   bytes flow and surfaces as a stream error mid-handler (mapped
///   to HTTP 500 via `ApiError::File` -- a degenerate-input
///   backstop, not the operator-facing path).
/// * The handler's mid-stream check against
///   `FsService::max_upload_bytes()` (operator-tunable, expected
///   to be <= the hard ceiling).  Returns the typed
///   `PayloadTooLarge` error which the operator-facing surface
///   maps to HTTP 400 `bad_request`.
///
/// Pipeline: validate the `{*path}` wildcard + workspace exists,
/// acquire the global upload permit, stream body bytes into a
/// workspace-local tempfile while computing sha256, fsync the
/// tempfile, hand off to `FsService::upload_workspace_file` for the
/// atomic rename-into-tree + `workspace.json` revision bump.
async fn upload_asset(
    State(files): State<Arc<dyn FsService>>,
    Path((id, raw_path)): Path<(String, String)>,
    body: Body,
) -> Result<Json<DatasetUploadReceipt>, ApiError> {
    let id = WorkspaceId::parse(&id)?;
    // `parse_asset_path` enforces the 256-byte total cap, the
    // `datasets/<class>/<file>` depth gate, and traversal rejection.
    // axum URL-decodes the wildcard before the handler sees it, so
    // a literal `%2E%2E%2F` arrives as `../` and the parser surfaces
    // `LeadingDot`.
    let asset_path = parse_asset_path(&raw_path)?;

    // Existence check before allocating the tempfile so an upload to
    // a phantom workspace never creates an orphan directory under
    // `<root>/workspaces/<phantom>/.tmp/`.
    let id_for_existence = id;
    let files_for_existence = files.clone();
    task::spawn_blocking(move || files_for_existence.summary(&id_for_existence))
        .await?
        .map_err(|e| classify_workspace_existence_error(&id, e))?;

    // Permit before any body bytes are consumed; drops at fn exit.
    let _permit = files.acquire_upload_permit()?;

    let tmp_dir = files.workspace_tmpdir(&id);
    tokio::fs::create_dir_all(&tmp_dir)
        .await
        .map_err(|e| ApiError::File(crate::file_mgr::io_err(tmp_dir.display(), e)))?;
    let tmp_dir_for_spawn = tmp_dir.clone();
    let tmp_file =
        task::spawn_blocking(move || tempfile::NamedTempFile::new_in(&tmp_dir_for_spawn))
            .await?
            .map_err(|e| ApiError::File(crate::file_mgr::io_err(tmp_dir.display(), e)))?;

    // Reject mid-stream the moment cumulative bytes cross the per-
    // request cap; the tempfile drops on Err without partial commit.
    let max_upload_bytes = files.max_upload_bytes();
    let tmp_reopened = tmp_file
        .reopen()
        .map_err(|e| ApiError::File(crate::file_mgr::io_err(tmp_file.path().display(), e)))?;
    let mut writer = tokio::fs::File::from_std(tmp_reopened);
    use futures_util::TryStreamExt as _;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let stream = body.into_data_stream().map_err(std::io::Error::other);
    let mut reader = tokio_util::io::StreamReader::new(stream);
    let mut buf = vec![0u8; 64 * 1024];
    let mut total: u64 = 0;
    let mut hasher = Sha256::new();
    loop {
        let n = reader
            .read(&mut buf)
            .await
            .map_err(|e| ApiError::File(crate::file_mgr::io_err("<upload-stream>", e)))?;
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
        writer
            .write_all(&buf[..n])
            .await
            .map_err(|e| ApiError::File(crate::file_mgr::io_err(tmp_file.path().display(), e)))?;
    }
    writer
        .flush()
        .await
        .map_err(|e| ApiError::File(crate::file_mgr::io_err(tmp_file.path().display(), e)))?;
    writer
        .sync_all()
        .await
        .map_err(|e| ApiError::File(crate::file_mgr::io_err(tmp_file.path().display(), e)))?;
    drop(writer);

    let digest = hex_lowercase(&hasher.finalize());

    // Atomic commit on the blocking pool.
    let files_for_install = files.clone();
    let id_for_install = id;
    let r = task::spawn_blocking(move || {
        files_for_install.upload_workspace_file(
            &id_for_install,
            &asset_path,
            tmp_file.path(),
            &digest,
            total,
        )
    })
    .await??;
    Ok(Json(r))
}

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/workspace/{id}/assets", get(list_assets))
        .route(
            "/workspace/{id}/assets/{*path}",
            get(get_asset)
                .put(upload_asset)
                .delete(delete_asset)
                // 256 MiB hard ceiling enforced by tower-http's
                // `RequestBodyLimit`.  Layered on the
                // `MethodRouter` so it applies to the PUT (upload)
                // path; GET / DELETE don't consume a request body
                // so the limit is a no-op there.
                .layer(DefaultBodyLimit::max(256 * 1024 * 1024)),
        )
}
