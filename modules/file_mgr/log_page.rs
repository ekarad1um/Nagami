//! Forward-only JSONL log paging.
//!
//! Reads a `<workspace>/{training,converter}_logs/<job_id>.jsonl`
//! file (or any other JSONL file under the workspace) one
//! [`LogEvent`] per line and returns a bounded page filtered by
//! the caller's `after_seq` cursor.
//!
//! Canonical reader for the daemon's per-job JSONL backstop;
//! `GET /api/v1/workspace/{id}/assets/{*path}` routes through
//! [`read_jsonl_page`] when `?after_seq=` or `?limit=` is set on
//! a `.jsonl` file.
//!
//! ## Design
//!
//! - **Forward-only.**  We scan the file from the start; the
//!   typical caller asks for `?after_seq=N` where `N` is close
//!   to the latest written seq, but the JSONL is small enough
//!   (capped by `max_log_line_bytes` per line and rotated on
//!   delete) that a re-scan is fine.  No mtime or seek index.
//! - **Malformed lines silently skipped.**  The registry's
//!   broadcast channel is the authoritative event source; the
//!   JSONL is a backstop.  A truncated mid-write line should not
//!   500 the page.  The skipped lines do **not** advance the
//!   cursor — the caller's next `?after_seq=` is the highest
//!   `seq` actually returned.
//! - **Missing file is empty page.**  A page on a file that has
//!   not been created yet (the producer hasn't emitted an event)
//!   returns `events: []` with `next_after_seq` echoing the
//!   caller's input — no 404 surface for "no events yet".

use crate::file_mgr::JobProgress;
use serde::{Deserialize, Serialize};
use std::io;
use std::path::Path;

/// Per-call default page size when the caller omits `?limit=`.
pub const DEFAULT_LOG_PAGE_LIMIT: usize = 200;

/// Hard ceiling on `?limit=` regardless of the caller's request.
/// The ceiling protects against unbounded buffering in the
/// blocking scan; the SSE stream is the right shape for a
/// follow-along consumer that expects no upper bound.
pub const MAX_LOG_PAGE_LIMIT: usize = 1000;

/// One line of a JSONL backstop, deserialised forgivingly.  The
/// shape mirrors [`crate::file_mgr::JobEvent`] but drops the
/// `state: Option<JobState>` typing in favour of `Option<String>`:
/// producers (`ConvertJobLog`, `TrainJobLog`) emit free-form
/// lifecycle phases (`"started"`, `"read_model_json"`,
/// `"completed"`, `"failed"`, ...) that are richer than the
/// registry's terminal-state enum and would otherwise drop on
/// the strict deserialize.  `progress` and `message` are typed
/// the same way as `JobEvent` so the wire surfaces are
/// byte-compatible for non-state fields.
#[derive(Debug, Serialize, Deserialize)]
pub struct LogEvent {
    pub seq: u64,
    pub at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub progress: Option<JobProgress>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// Page response shape echoed to the wire by the route layer.
#[derive(Debug, Serialize)]
pub struct LogPageResp {
    /// Bounded slice of events; up to `limit` entries with
    /// `seq > after_seq`.  Empty when the file is missing or
    /// when no event satisfies the cursor.
    pub events: Vec<LogEvent>,
    /// Next cursor to pass back as `?after_seq=` to continue
    /// paging.  Equals the last event's `seq` when the page is
    /// non-empty; echoes the caller's `after_seq` otherwise so a
    /// poll that catches up reads `next_after_seq == after_seq`
    /// and knows no progress has been made.
    pub next_after_seq: u64,
}

/// Read one bounded page from `path`.  `limit` is clamped to
/// `[1, MAX_LOG_PAGE_LIMIT]`; values outside that range are
/// rounded silently because the route's query DTO already
/// validates against operator-supplied bogus values.
pub fn read_jsonl_page(path: &Path, after_seq: u64, limit: usize) -> io::Result<LogPageResp> {
    use std::io::{BufRead, BufReader};
    let limit = limit.clamp(1, MAX_LOG_PAGE_LIMIT);
    let file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            return Ok(LogPageResp {
                events: Vec::new(),
                next_after_seq: after_seq,
            });
        }
        Err(e) => return Err(e),
    };
    let mut events = Vec::with_capacity(limit.min(64));
    let reader = BufReader::new(file);
    let mut next_after_seq = after_seq;
    for line in reader.lines() {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        let evt: LogEvent = match serde_json::from_str(&line) {
            Ok(e) => e,
            Err(_) => continue,
        };
        if evt.seq <= after_seq {
            continue;
        }
        next_after_seq = evt.seq;
        events.push(evt);
        if events.len() >= limit {
            break;
        }
    }
    Ok(LogPageResp {
        events,
        next_after_seq,
    })
}
