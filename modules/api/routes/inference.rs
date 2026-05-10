//! Inference cadence endpoints.
//!
//! `GET/POST /inference` exposes the live `InferenceCfg`
//! (hop_samples + top_k).  The active classifier head is read
//! through `GET /api/v1/active` (which returns the same
//! head_id + n_classes plus the full active manifest); the
//! retired `GET/POST /inference/head` routes were dropped along
//! with the workspace redesign cleanup.

use std::sync::Arc;

use crate::config::ConfigHandle;
use crate::inference::InferenceCfg;
use arc_swap::ArcSwap;
use axum::Router;
use axum::extract::State;
use axum::response::{IntoResponse, Json};
use axum::routing::get;
use serde::{Deserialize, Serialize};

use crate::api::AppState;
use crate::api::error::ApiError;
use crate::api::extract::ApiJson;

#[derive(Serialize)]
struct InferenceResp {
    cfg: InferenceCfg,
}

async fn get_inference(
    State(inference_cfg): State<Arc<ArcSwap<InferenceCfg>>>,
) -> impl IntoResponse {
    let cfg = **inference_cfg.load();
    Json(InferenceResp { cfg })
}

/// API contract: every input DTO denies unknown fields so a client
/// typo (e.g. `top-k` vs `top_k`, `hop-samples` vs `hop_samples`)
/// surfaces as an explicit 422 instead of silently using the prior
/// value via `Option::None.unwrap_or(...)`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SetInferenceReq {
    /// Stride in samples between successive windows.  Must be
    /// in `1..=InferenceCfg::MAX_HOP_SAMPLES` (= `WaveformLen *
    /// 3 / 4`) so successive windows overlap by at least 25%.
    pub hop_samples: Option<usize>,
    /// Number of TopK entries per emitted frame.  1..=64.
    pub top_k: Option<usize>,
}

async fn post_inference(
    State(inference_cfg): State<Arc<ArcSwap<InferenceCfg>>>,
    State(config): State<Arc<dyn ConfigHandle>>,
    ApiJson(req): ApiJson<SetInferenceReq>,
) -> Result<Json<InferenceResp>, ApiError> {
    // Validate request fields BEFORE entering the lock so we don't
    // hold the config lock while returning a 400.  The bounds are
    // owned by `InferenceCfg` (single source of truth shared with
    // the config loader); we form the would-be merged value and
    // delegate to its validator so any future tightening of the
    // contract takes effect everywhere.
    let current = **inference_cfg.load();
    let candidate = InferenceCfg {
        hop_samples: req.hop_samples.unwrap_or(current.hop_samples),
        top_k: req.top_k.unwrap_or(current.top_k),
    };
    candidate.validate().map_err(ApiError::Bad)?;

    // Compute the merged config INSIDE the lock so concurrent
    // partial updates (e.g. caller A sets hop_samples, caller B sets
    // top_k) are NOT computed from the same stale snapshot.  Without
    // this, A's mutate_then sees the original top_k; A's write
    // overwrites B's top_k.  With the guard pattern, A's mutation
    // runs against the latest config (post-B if B ran first under
    // the lock), and the after-hook stores the new InferenceCfg
    // into the runtime ArcSwap WHILE STILL HOLDING the mutate lock
    // -- so the on-disk + runtime states stay in sync against
    // concurrent mutators.  ( the
    // `mutate_then<F, G, R>` was generic on R; the object-safe
    // guard captures `next` directly via the caller's stack frame
    // instead.)
    let inference_cfg_for_after = inference_cfg.clone();
    let mut guard = config.open_mutation()?;
    let next = {
        let c = guard.config();
        let mut next = c.inference;
        if let Some(h) = req.hop_samples {
            next.hop_samples = h;
        }
        if let Some(k) = req.top_k {
            next.top_k = k;
        }
        c.inference = next;
        next
    };
    guard.commit(Box::new(move |c| {
        inference_cfg_for_after.store(Arc::new(c.inference))
    }))?;
    Ok(Json(InferenceResp { cfg: next }))
}

pub fn router() -> Router<AppState> {
    Router::new().route("/inference", get(get_inference).post(post_inference))
}
