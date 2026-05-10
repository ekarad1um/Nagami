//! `GET /mic` + `POST /mic/policy` (and the legacy `POST /mic`
//! alias).  Mic catalogue is launch-immutable; only the policy is
//! mutable via this surface.

use std::sync::Arc;

use crate::audio_io::mic_arbitrator::{MicCatalogue, MicPolicy};
use crate::config::MicSettingsHandle;
use axum::Router;
use axum::extract::State;
use axum::response::Json;
use axum::routing::{get, post};
use serde::{Deserialize, Serialize};
use tokio::task;

use crate::api::error::ApiError;
use crate::api::extract::{ApiJson, ApiQuery};
use crate::api::{AppState, VersionQuery, check_min_version};

/// `GET /mic` returns both layers separately:
///
/// * `catalogue` -- read-only deployment manifest (which mics +
///   channels exist).  Operators change this by editing the launch
///   TOML and restarting the daemon.  The API has no endpoint to
///   mutate it.
/// * `policy` -- current user preference.  Operators change this
///   via `POST /mic/policy` or by hot-reloading the user TOML.
///
/// Returning both in one round-trip lets a UI show "what mics
/// exist" alongside "which one is active" without a second call.
#[derive(Serialize)]
struct MicResp {
    catalogue: MicCatalogue,
    policy: MicPolicy,
    /// Version stamp for read-your-writes.  Increments
    /// on every successful `try_set_policy` /
    /// `try_set_policy_no_persist` mutation.
    version: u64,
}

async fn get_mic(
    State(mic_settings): State<Arc<dyn MicSettingsHandle>>,
    ApiQuery(q): ApiQuery<VersionQuery>,
) -> Result<Json<MicResp>, ApiError> {
    // `snapshot_with_version` atomically pairs value + version
    // from a single `VersionedSwap` guard; separate calls would
    // tear under a concurrent `try_set_policy`.
    // `?min_version=N` returns 425 until the live cell reaches
    // `N` (read-your-writes after a prior POST).
    let (live, cur) = mic_settings.snapshot_with_version();
    check_min_version(cur, q.min_version)?;
    Ok(Json(MicResp {
        catalogue: (*live.catalogue).clone(),
        policy: live.policy.clone(),
        version: cur.get(),
    }))
}

/// `POST /mic` (and `POST /mic/policy`) accepts the user-pref
/// layer ([`MicPolicy`]) only.  The launch [`MicCatalogue`] is
/// immutable post-boot -- the API has no endpoint to mutate it.
/// Operators change available mics + channels by editing the
/// launch TOML and restarting the daemon.
///
/// The submitted policy is cross-validated against the live
/// catalogue (see `validate_against`) before commit; an unknown
/// `Fixed { id }` or out-of-whitelist `Fixed { channel }` returns
/// 400 with a diagnostic.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SetMicPolicyReq {
    policy: MicPolicy,
}

async fn post_mic_policy(
    State(mic_settings): State<Arc<dyn MicSettingsHandle>>,
    ApiJson(req): ApiJson<SetMicPolicyReq>,
) -> Result<Json<MicResp>, ApiError> {
    let new_policy = req.policy;
    // Echo the just-installed policy paired with `receipt.version`;
    // a post-swap `snapshot()` could pick up a later writer's policy.
    let echoed_policy = new_policy.clone();

    // `try_set_policy` runs validation + atomic in-memory swap +
    // TOML persist in one shot.  The persist hits disk, so wrap
    // in `spawn_blocking` to keep the async worker free.
    let mic_settings_for_spawn = mic_settings.clone();
    let receipt = task::spawn_blocking(
        move || -> Result<crate::common::version::SwapReceipt, ApiError> {
            mic_settings_for_spawn
                .try_set_policy(new_policy)
                .map_err(Into::into)
        },
    )
    .await??;

    // Catalogue is launch-immutable so any snapshot's
    // `catalogue` Arc is fine; we just need a cheap clone.
    let catalogue = (*mic_settings.snapshot().catalogue).clone();
    Ok(Json(MicResp {
        catalogue,
        policy: echoed_policy,
        // Surface the post-swap version.  Callers
        // round-trip via `GET /mic?min_version=N` to confirm the
        // write has settled (read-your-writes).
        version: receipt.version.get(),
    }))
}

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/mic", get(get_mic).post(post_mic_policy))
        .route("/mic/policy", post(post_mic_policy))
}
