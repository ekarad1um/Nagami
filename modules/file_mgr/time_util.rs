//! Tiny RFC3339 wall-clock helper used by the workspace
//! lifecycle (`workspace.json::created_at`,
//! `workspace_revision.at`, etc.).
//!
//! Centralized so every workspace write stamps the same format.
//! `time::OffsetDateTime::now_utc().format(Rfc3339)` never fails
//! for the canonical formatter; the `unwrap_or_else` fallback to
//! a sentinel is defensive only -- a callsite that observes the
//! sentinel has bigger problems than a stale timestamp.

use std::time::SystemTime;
use time::{OffsetDateTime, format_description::well_known::Rfc3339};

/// Current UTC wall-clock formatted as RFC3339.  Never panics.
pub fn now_rfc3339() -> String {
    OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .unwrap_or_else(|_| String::from("1970-01-01T00:00:00Z"))
}

/// Format a [`SystemTime`] as an RFC3339 UTC string.  Used by the
/// asset-listing surface to render filesystem `mtime` values in
/// the same canonical format the workspace lifecycle emits via
/// [`now_rfc3339`].  Pre-epoch and out-of-range timestamps fall
/// back to the same sentinel `now_rfc3339` uses on its own
/// formatter failure -- a returned sentinel is a "this filesystem
/// reported nonsense" signal, not a daemon bug.
pub fn rfc3339_from(t: SystemTime) -> String {
    OffsetDateTime::from(t)
        .format(&Rfc3339)
        .unwrap_or_else(|_| String::from("1970-01-01T00:00:00Z"))
}
