//! Bounded UTF-8 truncation for operator-supplied log lines.
//!
//! Both the converter and training JSONL writers (`<workspace>/
//! converter_logs/<job_id>.jsonl` and `<workspace>/training_logs/
//! <job_id>.jsonl`) accept an optional operator-supplied
//! `message` payload.  Letting that payload through unbounded
//! would let a malicious or runaway producer flood the on-disk
//! log; the per-line cap below ([`MAX_LOG_LINE_BYTES`]) keeps a
//! single event small enough to scan with `head`/`jq` even when
//! the producer misbehaves.
//!
//! Slicing a `&str` at a hard byte offset would panic if the cap
//! landed inside a multi-byte UTF-8 codepoint; this helper snaps
//! down to the nearest char boundary so a multi-byte glyph
//! straddling the cap is dropped rather than corrupted.  The
//! suffix `"...[truncated]"` is appended verbatim on truncation
//! so operators can grep for it.
//!
//! `file_mgr::job_registry`'s per-event ring uses a separate
//! truncation routine with a configurable cap and a different
//! suffix marker (`" ... [truncated]"`).  Don't conflate -- the
//! two are observably distinct on disk.

/// Per-event log-line cap, in bytes.  8 KiB is the threshold
/// above which a single JSONL line stops being scannable with
/// `head -n` / `jq -c` by hand; producers writing past this cap
/// signal a malformed message and are truncated.
pub const MAX_LOG_LINE_BYTES: usize = 8 * 1024;

/// Truncate `m` to at most [`MAX_LOG_LINE_BYTES`], snapping to a
/// UTF-8 char boundary so a multi-byte codepoint straddling the
/// cap does not panic the slice.  Appends `"...[truncated]"`
/// verbatim on truncation; messages within the cap are returned
/// unchanged.
pub fn truncate_log_message(m: &str) -> String {
    if m.len() <= MAX_LOG_LINE_BYTES {
        return m.to_string();
    }
    // UTF-8 codepoints are at most 4 bytes, so the snap-down
    // loop runs at most 3 iterations.
    let mut idx = MAX_LOG_LINE_BYTES;
    while idx > 0 && !m.is_char_boundary(idx) {
        idx -= 1;
    }
    let mut s = m[..idx].to_string();
    s.push_str("...[truncated]");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Short messages pass through unchanged.
    #[test]
    fn short_message_is_unchanged() {
        let m = "hello world";
        assert_eq!(truncate_log_message(m), "hello world");
    }

    /// A message of exactly the cap length passes through
    /// untruncated -- the cap is inclusive.
    #[test]
    fn exact_cap_is_unchanged() {
        let m: String = std::iter::repeat_n('a', MAX_LOG_LINE_BYTES).collect();
        let out = truncate_log_message(&m);
        assert_eq!(out, m);
        assert!(!out.contains("...[truncated]"));
    }

    /// Naive `&str[..cap]` would panic when the cap lands inside
    /// a 4-byte UTF-8 codepoint; pin the boundary-snapping
    /// behavior so a future revert surfaces as a test failure.
    #[test]
    fn four_byte_codepoint_at_boundary_snaps_down() {
        // 4-byte codepoint (U+1F600 grinning face).  Build a
        // message of exactly cap - 1 ASCII bytes + one 4-byte
        // codepoint -> total cap + 3 bytes, with the codepoint
        // straddling byte index (cap - 1)..=(cap + 2).  Slice at
        // MAX_LOG_LINE_BYTES is mid-codepoint.
        let mut m = String::with_capacity(MAX_LOG_LINE_BYTES + 4);
        for _ in 0..(MAX_LOG_LINE_BYTES - 1) {
            m.push('a');
        }
        m.push('\u{1F600}');
        let out = truncate_log_message(&m);
        assert!(
            out.ends_with("...[truncated]"),
            "expected truncation marker; got len={}",
            out.len(),
        );
        // The truncated body itself is valid UTF-8 (snap-down
        // landed on a char boundary).  Length <= cap because the
        // 4-byte codepoint at the boundary is dropped.
        let body = out.trim_end_matches("...[truncated]");
        assert!(
            body.is_char_boundary(body.len()),
            "body must end on a char boundary",
        );
        assert!(
            body.len() <= MAX_LOG_LINE_BYTES,
            "body must not exceed cap; got {}",
            body.len(),
        );
    }

    /// 2-byte codepoint variant: cap lands inside a `é` (U+00E9,
    /// 2 bytes in UTF-8).  Total output stays within cap plus
    /// suffix length, ends with the truncation marker, and the
    /// snap-down loop backs off exactly one byte to the
    /// codepoint's start.
    #[test]
    fn two_byte_codepoint_at_boundary_snaps_down() {
        // Prepend one ASCII byte so subsequent `é` codepoints
        // start at odd byte offsets; the cap (MAX_LOG_LINE_BYTES,
        // even) then lands inside an `é` and forces the snap-down
        // loop to back off one byte to the codepoint's start.
        // Without the leading ASCII byte, every even byte offset
        // (including the cap) is a codepoint boundary in a string
        // of `é`s and the snap-down would never execute.
        let mut s = String::from("a");
        while s.len() < MAX_LOG_LINE_BYTES + 1024 {
            s.push('é');
        }
        let truncated = truncate_log_message(&s);
        // Snap-down dropped the straddling `é`: body is one ASCII
        // byte + N whole `é`s, so body.len() = MAX_LOG_LINE_BYTES - 1.
        let body = truncated.trim_end_matches("...[truncated]");
        assert_eq!(body.len(), MAX_LOG_LINE_BYTES - 1);
        assert!(
            body.is_char_boundary(body.len()),
            "body must end on a char boundary",
        );
        // Capped under the cap + suffix budget.
        assert!(
            truncated.len() <= MAX_LOG_LINE_BYTES + b"...[truncated]".len(),
            "truncated len {} > cap + suffix",
            truncated.len(),
        );
        assert!(truncated.ends_with("...[truncated]"));
        // Round-trip: the truncated body is still valid UTF-8
        // (snap-down to char boundary held).  String already
        // implies UTF-8, so a successful chars().count() is the
        // assertion.
        let _ = truncated.chars().count();
    }
}
