//! `GET /status` -- daemon-wide status snapshot.

use std::sync::Arc;

use crate::common::traits::lag_source::LagSource;
use crate::status::{StatusReporter, StatusSnapshot};
use axum::Router;
use axum::extract::State;
use axum::response::Json;
use axum::routing::get;

use crate::api::AppState;
use crate::api::error::ApiError;

async fn get_status(
    State(monitor): State<Arc<dyn StatusReporter>>,
    State(broadcast_lag_reader): State<Arc<dyn LagSource>>,
) -> Result<Json<StatusSnapshot>, ApiError> {
    // Wait-free.  Process-wide metrics come from
    // the StatusMonitor's background ArcSwap (the daemon called
    // `start_sampler` at boot pointing at the workspace root); the
    // request path no longer touches sysinfo.  Both `monitor.snapshot`
    // and `broadcast_lag_reader.snapshot` are atomic loads + a
    // DashMap walk.  No `spawn_blocking` slot consumed; no mutex.
    let lags = broadcast_lag_reader.snapshot();
    let snap = monitor.snapshot(lags);
    Ok(Json(snap))
}

pub fn router() -> Router<AppState> {
    Router::new().route("/status", get(get_status))
}
