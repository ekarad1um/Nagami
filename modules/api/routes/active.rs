//! `POST /active` + `GET /active` -- active-head activation and read.
//!
//! `POST /active` stages a fresh generation under
//! `<root>/active/.tmp/<activation_id>/`, pre-loads + validates it,
//! atomically publishes `current.json`, then installs the
//! prevalidated runtime candidate into `HotHead`.  The daemon's
//! first-boot path uses the same primitive so on-disk + runtime
//! states never drift.  The route is idempotent: re-activating the
//! same head produces a fresh `activation_id` so operators can
//! confirm the write landed via `GET /active`.
//!
//! Activations serialize through the global `active_mutex`.  Head
//! origin staging currently fails closed if a source workspace/head
//! disappears mid-copy; the intended active-then-workspace lock order
//! is documented at the staging call site until `FsService` grows a
//! lock-only workspace guard.

use std::path::Path;

use axum::Router;
use axum::extract::State;
use axum::response::Json;
use axum::routing::post;
use serde::{Deserialize, Serialize};
use tokio::task;

use crate::api::AppState;
use crate::api::error::ApiError;
use crate::api::extract::ApiJson;
use crate::common::ids::{HeadId, WorkspaceId};
use crate::common::workspace::{ActiveHeadManifest, ActiveOrigin, WorkspaceRevision};
use crate::file_mgr::active_head_writer::{
    ActivationError, ActivationOriginInput, ActivationResult, HeadInnerLoader, PendingActivation,
    prune_old_generations, publish_active_generation, stage_and_validate_activation,
    staging_path_for,
};

/// `POST /active` body: either `{workspace_id, head_id}` to activate
/// a workspace head or `{default: true}` to activate the bundled
/// default.  `untagged` is required because the two shapes share no
/// discriminator; `deny_unknown_fields` lives on each variant
/// (untagged + parent-level `deny_unknown_fields` do not compose) so
/// stray keys surface as 422.
#[derive(Deserialize, Debug)]
#[serde(untagged)]
enum ActivateRequest {
    Head(HeadActivate),
    /// `default` MUST be `true`; the route rejects `default: false`
    /// explicitly so we can disambiguate from a future no-op shape.
    Default(DefaultActivate),
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct HeadActivate {
    workspace_id: WorkspaceId,
    head_id: HeadId,
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct DefaultActivate {
    default: bool,
}

/// Wire shape for both `POST /active` and (with one extra field)
/// `GET /active`.  Mirrors [`ActiveHeadManifest`] with the `origin`
/// discriminator flattened so operator tooling that consumes the
/// per-generation `manifest.json` works unmodified.
#[derive(Serialize, Debug)]
struct ActiveResp {
    sha256: String,
    labels_sha256: String,
    n_classes: u32,
    labels: Vec<String>,
    /// Stable runtime identifier stamped on every emitted
    /// `InferenceFrame.head_id`.
    runtime_head_id: String,
    activated_at: String,
    /// `"head"` | `"default"`.
    origin: &'static str,
    /// Head-origin only.
    #[serde(skip_serializing_if = "Option::is_none")]
    source_workspace_id: Option<String>,
    /// Head-origin only.
    #[serde(skip_serializing_if = "Option::is_none")]
    source_head_id: Option<String>,
    /// Head-origin only.
    #[serde(skip_serializing_if = "Option::is_none")]
    workspace_revision: Option<WorkspaceRevision>,
    /// Activation directory name; correlates with on-disk
    /// `<root>/active/generations/<activation_id>/`.
    activation_id: String,
    /// GET-only, Head-origin only: whether the source workspace's
    /// dir is still on disk.  Inference keeps serving even when
    /// `false` because the active generation owns independent bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    source_workspace_alive: Option<bool>,
}

impl ActiveResp {
    fn from_manifest(
        manifest: &ActiveHeadManifest,
        activation_id: &str,
        source_workspace_alive: Option<bool>,
    ) -> Self {
        let (origin, src_ws, src_head, src_rev) = match &manifest.origin {
            ActiveOrigin::Default => ("default", None, None, None),
            ActiveOrigin::Head {
                source_workspace_id,
                source_head_id,
                workspace_revision,
            } => (
                "head",
                Some(source_workspace_id.to_string()),
                Some(source_head_id.to_string()),
                Some(workspace_revision.clone()),
            ),
        };
        Self {
            sha256: manifest.sha256.clone(),
            labels_sha256: manifest.labels_sha256.clone(),
            n_classes: manifest.n_classes,
            labels: manifest.labels.clone(),
            runtime_head_id: manifest.runtime_head_id.to_string(),
            activated_at: manifest.activated_at.clone(),
            origin,
            source_workspace_id: src_ws,
            source_head_id: src_head,
            workspace_revision: src_rev,
            activation_id: activation_id.to_string(),
            source_workspace_alive,
        }
    }
}

/// `POST /active`: lock active mutex, snapshot prior `current.json`,
/// stage + validate + pre-load, atomically publish, install the
/// prevalidated candidate, then best-effort prune old generations.
/// The runtime install is reachable only via daemon-bug paths
/// (mismatched `HeadInner` types) because the candidate already
/// passed `HeadInner::validate`.
async fn post_active(
    State(state): State<AppState>,
    ApiJson(req): ApiJson<ActivateRequest>,
) -> Result<Json<ActiveResp>, ApiError> {
    // `untagged` accepts `default: false` as the Default arm; reject
    // it here so we can disambiguate from a future no-op shape.
    if let ActivateRequest::Default(DefaultActivate { default }) = &req
        && !*default
    {
        return Err(ApiError::Bad(
            "`default` must be true; use `{workspace_id, head_id}` to activate a workspace head"
                .into(),
        ));
    }

    // Head-origin fast-fail: heads.json existence check OUTSIDE
    // the active mutex so a typo'd id costs nothing under
    // contention.  Race-tolerant: the staging primitive re-checks
    // and fails closed if the source disappears between here and
    // the publish.
    if let ActivateRequest::Head(HeadActivate {
        workspace_id,
        head_id,
    }) = &req
    {
        let workspace_id = *workspace_id;
        let head_id = *head_id;
        let files = state.files.clone();
        let summary = task::spawn_blocking(move || files.summary(&workspace_id))
            .await?
            .map_err(|e| {
                crate::api::routes::workspace::classify_workspace_existence_error(
                    &workspace_id,
                    e,
                )
            })?;
        if !summary.heads.heads.iter().any(|r| r.head_id == head_id) {
            return Err(ApiError::NotFound(format!(
                "head {head_id} not in workspace {workspace_id} (heads.json index)"
            )));
        }
    }

    // Run read+stage+publish+install+prune on a single
    // `spawn_blocking` worker, holding `active_mutex` end-to-end.
    // Sync mutex (`parking_lot::Mutex`) is correct because the
    // guard never crosses `.await`.
    //
    // The lock is load-bearing across the WHOLE chain because
    // two invariants span it:
    //   * publish + install must not reorder across requests --
    //     otherwise `current.json` and the runtime `HotHead`
    //     diverge;
    //   * prune must not run while a peer can publish a fresh
    //     generation -- otherwise the peer's directory is not in
    //     this request's `keep` and gets deleted, leaving
    //     `current.json` pointing at a missing dir.
    let active_mutex = state.active_mutex.clone();
    let files = state.files.clone();
    let bundled = state.bundled_default_dir.clone();
    let head = state.head.clone();
    let (activation_id, manifest) =
        task::spawn_blocking(move || -> Result<(String, ActiveHeadManifest), ApiError> {
            let _guard = active_mutex.lock();

            // Snapshot the prior activation id for the prune
            // keep-list.  Absent pointer = first-boot / wiped
            // state; a parse error here means boot recovery
            // missed a corrupt pointer and the operator should
            // know.
            let previous_activation_id =
                match crate::file_mgr::schema::read_active_current(files.root()) {
                    Ok(p) => Some(p.activation_id),
                    Err(crate::file_mgr::FileError::Io { source, .. })
                        if source.kind() == std::io::ErrorKind::NotFound =>
                    {
                        None
                    }
                    Err(e) => return Err(ApiError::File(e)),
                };

            // Stage + validate + publish (writes current.json).
            // The per-workspace mutation mutex would ideally guard
            // the staging step (lock order: active first, then
            // workspace) so a concurrent workspace delete cannot
            // race the source copy.  The current `FsService`
            // couples mutex acquisition with a metadata read that
            // no longer matches the workspace shape; until a
            // lock-only handle lands, the staging primitive itself
            // fails closed (`ActivationError::NotFound` /
            // `HashMismatch`) if the source disappears mid-flight,
            // and a successful publish owns independent bytes so
            // subsequent deletes do not affect it.
            let result = match req {
                ActivateRequest::Head(HeadActivate {
                    workspace_id,
                    head_id,
                }) => {
                    let workspace_dir =
                        crate::file_mgr::schema::workspace_dir_for(files.root(), &workspace_id);
                    stage_and_publish_activation(
                        files.root(),
                        ActivationOriginInput::Head {
                            workspace_dir: &workspace_dir,
                            workspace_id,
                            head_id,
                        },
                        &bundled,
                    )?
                }
                ActivateRequest::Default(_) => stage_and_publish_activation(
                    files.root(),
                    ActivationOriginInput::Default,
                    &bundled,
                )?,
            };

            // Install AFTER `current.json` is durable so on-disk
            // and runtime cannot diverge.  Receipt is intentionally
            // dropped: read-your-write goes through `current.json`
            // (next `GET /active`), not the `HotHead` version.
            let _swap_receipt = head.install_prevalidated(result.candidate)?;

            // Best-effort prune.  Held under the lock so a peer
            // activation cannot publish a generation between our
            // `read_dir` and our remove pass; that generation
            // would land outside our `keep` and be deleted.
            // Failure logs and continues -- boot recovery sweeps
            // any residue on next start.
            let keep: Vec<&str> = std::iter::once(result.activation_id.as_str())
                .chain(previous_activation_id.as_deref())
                .collect();
            if let Err(e) = prune_old_generations(files.root(), &keep) {
                tracing::warn!(
                    target: "acoustics",
                    err = %e,
                    activation_id = %result.activation_id,
                    "active generation prune failed; residue will be swept on next boot",
                );
            }

            Ok((result.activation_id, result.manifest))
        })
        .await??;

    Ok(Json(ActiveResp::from_manifest(
        &manifest,
        &activation_id,
        None,
    )))
}

fn stage_and_publish_activation(
    root: &Path,
    origin_input: ActivationOriginInput<'_>,
    bundled_default_dir: &Path,
) -> Result<ActivationResult, ActivationError> {
    let staged = stage_and_validate_activation(
        PendingActivation {
            root,
            origin_input,
            bundled_default_dir,
            now_rfc3339: crate::file_mgr::now_rfc3339(),
        },
        &runtime_head_loader(),
    )?;
    publish_active_generation(
        root,
        &staging_path_for(root, &staged.activation_id),
        &staged.manifest,
        &staged.activation_id,
    )?;
    Ok(staged)
}

/// Pre-load closure for the activation primitive.  Boxed as
/// `dyn Any + Send` so the file_mgr-side primitive stays decoupled
/// from the inference crate.
fn runtime_head_loader() -> Box<HeadInnerLoader> {
    Box::new(|head_mpk, labels, head_id| {
        let head = crate::inference::HotHead::load(head_mpk, labels, head_id)
            .map_err(|e| format!("{e}"))?;
        // `HotHead::load` wrapped the inner in a `VersionedSwap`;
        // clone it once for the install-step downcast.
        let inner = (*head.snapshot()).clone();
        Ok(Box::new(inner) as Box<dyn std::any::Any + Send>)
    })
}

/// `GET /active`: read `<root>/active/current.json`, load + validate
/// the pointed manifest, then add the GET-only
/// `source_workspace_alive` field.  Wait-free; never takes the
/// active mutex.
async fn get_active(State(state): State<AppState>) -> Result<Json<ActiveResp>, ApiError> {
    let root = state.files.root().to_path_buf();
    let (pointer, manifest) = task::spawn_blocking(
        move || -> Result<(crate::file_mgr::ActiveCurrentPointer, ActiveHeadManifest), ApiError> {
            let pointer =
                crate::file_mgr::schema::read_active_current(&root).map_err(|e| match e {
                    crate::file_mgr::FileError::Io { source, .. }
                        if source.kind() == std::io::ErrorKind::NotFound =>
                    {
                        ApiError::NotFound("no active head: current.json absent".into())
                    }
                    other => ApiError::File(other),
                })?;
            let manifest =
                crate::file_mgr::schema::read_active_manifest(&root, &pointer.activation_id)?;
            // Validate before trusting deserialized fields.
            manifest
                .validate()
                .map_err(|e| ApiError::Bad(format!("active manifest validation: {e}")))?;
            Ok((pointer, manifest))
        },
    )
    .await??;

    let source_workspace_alive = match &manifest.origin {
        ActiveOrigin::Default => None,
        ActiveOrigin::Head {
            source_workspace_id,
            ..
        } => {
            let ws_dir =
                crate::file_mgr::schema::workspace_dir_for(state.files.root(), source_workspace_id);
            Some(ws_dir.is_dir())
        }
    };

    Ok(Json(ActiveResp::from_manifest(
        &manifest,
        &pointer.activation_id,
        source_workspace_alive,
    )))
}

pub fn router() -> Router<AppState> {
    Router::new().route("/active", post(post_active).get(get_active))
}
