//! Job inspection + logs routes.
//!
//! Memory-only `GET /jobs` + `GET /jobs/{job_id}` snapshots and
//! the `GET /jobs/{job_id}/events` SSE stream.  Reads go
//! through the in-process [`crate::file_mgr::JobRegistry`];
//! no log files are opened by these handlers.  The durable JSONL
//! surface lives on
//! `GET /workspace/{id}/assets/{training,converter}_logs/{job_id}.jsonl?after_seq=&limit=`.
//!
//! ## Wire shape
//!
//! - `GET /jobs` -> `Vec<JobSnapshot>` newest-first; bounded by
//!   `cfg.max_recent_jobs`.  `?limit=` clamps to that ceiling.
//! - `GET /jobs/{job_id}` -> `JobSnapshot`; 404 when the job
//!   id is not retained.
//! - `GET /jobs/{job_id}/events?after_seq=N&logs=true|false`
//!   -> `text/event-stream`.  Replays in-memory ring events
//!   strictly after `N`, then follows the broadcast channel
//!   until terminal state OR client disconnect.  When `N` is
//!   older than the ring's oldest seq, returns 409 with a
//!   `{error, code: "event_gap", oldest_seq, latest_seq}`
//!   envelope.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::extract::{Path, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::get;
use futures_util::stream::{self, Stream};
use serde::{Deserialize, Serialize};

use crate::api::AppState;
use crate::api::error::ApiError;
use crate::api::extract::ApiQuery;
use crate::common::ids::JobId;
use crate::file_mgr::{EventGap, JobEvent, JobRegistry, JobSnapshot};

#[derive(Debug, Deserialize)]
struct JobsListQuery {
    /// Optional client-side limit; clamped to
    /// `cfg.max_recent_jobs` server-side.
    #[serde(default)]
    limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct JobEventsQuery {
    /// Cursor: replay strictly after this seq.  `0` (or absent)
    /// means "fresh subscribe" -- replay whatever the ring holds.
    #[serde(default)]
    after_seq: Option<u64>,
    /// Whether to include log-line events.  Defaults to `true`
    /// so SSE clients see progress + log without per-call opt-in.
    #[serde(default)]
    logs: Option<bool>,
}

/// Render an [`EventGap`] error as the redesign §5 409 body
/// (`{error, code: "event_gap", oldest_seq, latest_seq}`).
#[derive(Serialize)]
struct EventGapBody {
    error: &'static str,
    code: &'static str,
    oldest_seq: u64,
    latest_seq: u64,
}

fn job_event_to_sse(event: JobEvent, include_logs: bool) -> (Event, bool) {
    let terminal = event.state.is_some_and(|state| !state.is_active());
    if !include_logs && event.message.is_some() {
        return (Event::default().comment("log filtered"), terminal);
    }
    let json = serde_json::to_string(&event).unwrap_or_default();
    (Event::default().event("job").data(json), terminal)
}

async fn list_jobs(
    State(jobs): State<Arc<JobRegistry>>,
    ApiQuery(q): ApiQuery<JobsListQuery>,
) -> Json<Vec<JobSnapshot>> {
    let cap = jobs.cfg().max_recent_jobs;
    let limit = q.limit.unwrap_or(cap).min(cap);
    Json(jobs.recent(limit))
}

async fn get_job(
    State(jobs): State<Arc<JobRegistry>>,
    Path(job_id): Path<String>,
) -> Result<Json<JobSnapshot>, ApiError> {
    let job_id = JobId::parse(&job_id)?;
    jobs.snapshot(job_id)
        .map(Json)
        .ok_or_else(|| ApiError::NotFound(format!("job not found: {job_id}")))
}

async fn job_events(
    State(jobs): State<Arc<JobRegistry>>,
    Path(job_id): Path<String>,
    ApiQuery(q): ApiQuery<JobEventsQuery>,
) -> Result<Response, ApiError> {
    let job_id = JobId::parse(&job_id)?;
    // Existence gate: 404 if the job id is not retained at all.
    if jobs.snapshot(job_id).is_none() {
        return Err(ApiError::NotFound(format!("job not found: {job_id}")));
    }
    let after_seq = q.after_seq.unwrap_or(0);
    let include_logs = q.logs.unwrap_or(true);
    let stream = match jobs.subscribe_events(job_id, after_seq) {
        Ok(s) => s,
        Err(EventGap {
            oldest_seq,
            latest_seq,
        }) => {
            // 409 with the same `{error, code}` envelope shape
            // every other api error carries.  Clients page the
            // JSONL log to backfill before reconnecting with a
            // fresher cursor.
            let body = EventGapBody {
                error: "event ring overflow; backfill via /{training,converter}_logs",
                code: "event_gap",
                oldest_seq,
                latest_seq,
            };
            return Ok((StatusCode::CONFLICT, Json(body)).into_response());
        }
    };
    // Count this connection on `sse_clients_current` for the
    // lifetime of the SSE stream.  The RAII guard is owned by
    // the unfold state and dropped with it; client disconnect,
    // terminal-state close, and abrupt drop all decrement.
    // `Option` because the metrics global may not be installed
    // in tests / boot-without-counters.
    let sse_guard: Option<crate::status::SseClientGuard> =
        crate::status::workspace_metrics::global().map(|m| m.sse_client_guard());

    // Build the SSE stream by chaining replay + live recv.
    // Each item is a `Result<Event, Infallible>`; axum's `Sse`
    // closes the connection cleanly on terminal state or
    // client disconnect.  KeepAlive at 15 s prevents idle
    // timeouts on long-running jobs.
    let event_stream = stream::unfold(
        (stream, false, include_logs, sse_guard),
        move |(mut s, terminal_emitted, include_logs, sse_guard)| async move {
            // Drain replay first.  Replay events are filtered
            // by the registry; we only filter logs here.
            if let Some(e) = s.next_replay() {
                let (event, terminal) = job_event_to_sse(e, include_logs);
                let terminal_emitted = terminal_emitted || terminal;
                return Some((Ok(event), (s, terminal_emitted, include_logs, sse_guard)));
            }
            // Replay drained.  If a terminal event has already
            // been observed (either pre-subscribe or via replay),
            // close the stream; otherwise wait for live events.
            if terminal_emitted || s.terminal_seen() {
                drop(sse_guard);
                return None;
            }
            match s.recv().await {
                Ok(e) => {
                    let (event, terminal) = job_event_to_sse(e, include_logs);
                    Some((
                        Ok::<_, Infallible>(event),
                        (s, terminal, include_logs, sse_guard),
                    ))
                }
                Err(_) => {
                    drop(sse_guard);
                    None
                } // Lagged or Closed: end stream.
            }
        },
    );
    let pinned: std::pin::Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send>> =
        Box::pin(event_stream);
    let sse = Sse::new(pinned).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)));
    let mut response = sse.into_response();
    // Disable buffering on intermediate proxies (nginx is the
    // documented production topology in `docs/BUILD.md`).
    // `Cache-Control: no-cache` is the SSE spec recommendation;
    // `X-Accel-Buffering: no` is the nginx-specific opt-out so
    // event delivery stays sub-second under reverse-proxy.
    let headers = response.headers_mut();
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    headers.insert("X-Accel-Buffering", HeaderValue::from_static("no"));
    Ok(response)
}

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/jobs", get(list_jobs))
        .route("/jobs/{job_id}", get(get_job))
        .route("/jobs/{job_id}/events", get(job_events))
}
