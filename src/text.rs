//! Lexical WGSL text helpers: comment stripping, directive splitting,
//! whole-token scans, and the parser-free compaction the bailout paths ship.
//! Byte-level and grammar-agnostic; nothing here touches naga.

use std::borrow::Cow;
use std::ops::Range;

/// Rewrite lone `\r` to `\n`; leave `\r\n` intact (`str::lines`
/// handles it).  Borrows the input when no lone `\r` exists.
//
// UTF-8 safety: `0x0D` cannot appear inside a multi-byte sequence
// (continuation bytes are `0x80..=0xBF`, valid lead bytes `0xC2..=0xF4`),
// so byte-level `\r` matches always land on character boundaries -
// the slicing below is guaranteed valid UTF-8.
pub(crate) fn normalize_line_endings(source: &str) -> Cow<'_, str> {
    let bytes = source.as_bytes();
    let lone_cr_at = |i: usize| bytes[i] == b'\r' && bytes.get(i + 1) != Some(&b'\n');
    if !source.contains('\r') || !(0..bytes.len()).any(lone_cr_at) {
        return Cow::Borrowed(source);
    }
    let mut out = String::with_capacity(source.len());
    let mut i = 0;
    while i < bytes.len() {
        if lone_cr_at(i) {
            out.push('\n');
            i += 1;
        } else {
            let start = i;
            while i < bytes.len() && !lone_cr_at(i) {
                i += 1;
            }
            out.push_str(&source[start..i]);
        }
    }
    Cow::Owned(out)
}

/// `true` when `b` can appear inside a WGSL identifier (ASCII alphanumeric
/// or underscore).
fn is_ident_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// `true` when a WGSL line break starts at byte `i`: LF / VT / FF / CR plus
/// NEL (U+0085), LS (U+2028), and PS (U+2029) in their UTF-8 forms.
fn wgsl_line_break_at(bytes: &[u8], i: usize) -> bool {
    match bytes[i] {
        0x0A..=0x0D => true,
        0xC2 => bytes.get(i + 1) == Some(&0x85),
        0xE2 => {
            bytes.get(i + 1) == Some(&0x80) && matches!(bytes.get(i + 2), Some(&0xA8) | Some(&0xA9))
        }
        _ => false,
    }
}

/// Replace WGSL line and block comments with spaces while preserving
/// byte offsets and line breaks, so subsequent lexical scans see only
/// real code but any positional diagnostics stay accurate.  Borrows
/// comment-free input.
pub(crate) fn strip_wgsl_comments(source: &str) -> Cow<'_, str> {
    if !source.contains("//") && !source.contains("/*") {
        return Cow::Borrowed(source);
    }
    let bytes = source.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        // Code up to the next `/` is copied whole.
        let slash = bytes[i..]
            .iter()
            .position(|&b| b == b'/')
            .map_or(bytes.len(), |p| i + p);
        out.extend_from_slice(&bytes[i..slash]);
        i = slash;
        if i >= bytes.len() {
            break;
        }
        if bytes.get(i + 1) == Some(&b'/') {
            // Line comment.  WGSL ends it at ANY line-break code point
            // (https://www.w3.org/TR/WGSL/#line-break), not just `\n`: stopping
            // early would blank a live statement that follows e.g. a lone `\r`
            // on the same `str::lines` line - the bailout paths ship this text.
            while i < bytes.len() && !wgsl_line_break_at(bytes, i) {
                out.push(b' ');
                i += 1;
            }
        } else if bytes.get(i + 1) == Some(&b'*') {
            // Block comment.  WGSL (https://www.w3.org/TR/WGSL/#comments)
            // permits nesting; a non-nesting scrub would close at the
            // inner `*/` and expose outer-comment `f16`/`enable` content
            // to token scans.  Track depth.
            out.extend_from_slice(b"  ");
            i += 2;
            let mut depth: u32 = 1;
            while i + 1 < bytes.len() && depth > 0 {
                if bytes[i] == b'/' && bytes[i + 1] == b'*' {
                    depth += 1;
                    out.extend_from_slice(b"  ");
                    i += 2;
                } else if bytes[i] == b'*' && bytes[i + 1] == b'/' {
                    depth -= 1;
                    out.extend_from_slice(b"  ");
                    i += 2;
                } else {
                    out.push(if bytes[i] == b'\n' { b'\n' } else { b' ' });
                    i += 1;
                }
            }
            if depth > 0 {
                // Unterminated block comment; consume the remainder so
                // the scrubbed output still matches the input byte count.
                while i < bytes.len() {
                    out.push(if bytes[i] == b'\n' { b'\n' } else { b' ' });
                    i += 1;
                }
            }
        } else {
            out.push(b'/');
            i += 1;
        }
    }
    // Every replacement emits ASCII space or newline, so the resulting
    // bytes stay valid UTF-8.
    Cow::Owned(String::from_utf8(out).expect("comment stripping preserves UTF-8"))
}

/// `true` when an identifier token names a 16-bit-float type: the scalar
/// `f16`, or a predeclared half-precision vector / matrix alias
/// (`vec2h`..`vec4h`, `mat2x2h`..`mat4x4h`).  Every one of these requires
/// `enable f16;` yet only `f16` itself contains the substring "f16".
fn is_f16_type_token(tok: &[u8]) -> bool {
    matches!(
        tok,
        b"f16"
            | b"vec2h"
            | b"vec3h"
            | b"vec4h"
            | b"mat2x2h"
            | b"mat2x3h"
            | b"mat2x4h"
            | b"mat3x2h"
            | b"mat3x3h"
            | b"mat3x4h"
            | b"mat4x2h"
            | b"mat4x3h"
            | b"mat4x4h"
    )
}

/// `true` when `source` uses any construct that requires `enable f16;`:
/// the `f16` keyword, a predeclared half-precision type alias
/// (`vec2h`/.../`mat4x4h`), or a numeric literal carrying the `h`
/// f16 suffix (`1.0h`, `0h`, `1.5e2h`, `0x1p2h`).  Scans the
/// comment-stripped text token by token so a longer identifier
/// (`myf16var`, `mesh`) and a comment never trigger a false match.
///
/// A spurious positive is harmless: naga tolerates a redundant
/// `enable f16;`, and the emitter drops the directive from the output
/// whenever the final module uses no f16 - so detection errs broad.
pub(crate) fn references_f16_token(source: &str) -> bool {
    cleaned_references_f16_token(&strip_wgsl_comments(source))
}

/// [`references_f16_token`] over already comment-stripped text.
pub(crate) fn cleaned_references_f16_token(cleaned: &str) -> bool {
    let bytes = cleaned.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    while i < len {
        let b = bytes[i];
        if is_ident_char(b) && !b.is_ascii_digit() {
            // Identifier / keyword token: letters, digits, `_`, not
            // leading with a digit.  Match the whole token so a longer
            // identifier that merely contains `f16`/`...h` is excluded.
            let start = i;
            while i < len && is_ident_char(bytes[i]) {
                i += 1;
            }
            if is_f16_type_token(&bytes[start..i]) {
                return true;
            }
        } else if b.is_ascii_digit() || (b == b'.' && i + 1 < len && bytes[i + 1].is_ascii_digit())
        {
            // Numeric literal: consume mantissa, hex digits, the
            // `e`/`E`/`p`/`P` exponent (with its optional sign), and the
            // trailing type-suffix letters.  A literal whose suffix is
            // `h` is an f16 value; any letters inside belong to the
            // literal, so only the final byte can be that suffix.
            let start = i;
            i += 1;
            while i < len {
                let c = bytes[i];
                // A `+`/`-` continues the literal only as an exponent sign
                // (right after `e`/`E`/`p`/`P`); otherwise it ends the token.
                let part_of_literal = c.is_ascii_alphanumeric()
                    || c == b'.'
                    || ((c == b'+' || c == b'-') && matches!(bytes[i - 1] | 0x20, b'e' | b'p'));
                if !part_of_literal {
                    break;
                }
                i += 1;
            }
            if bytes[i - 1] == b'h' && i - start >= 2 {
                return true;
            }
        } else {
            i += 1;
        }
    }
    false
}

/// [`has_enable_directive`] for `f16`.
pub(crate) fn has_enable_f16_directive(source: &str) -> bool {
    has_enable_directive(source, "f16")
}

/// `true` when `source` declares `enable <ext>;`, including as one entry of a
/// comma-separated list (`enable f16, clip_distances;`) and regardless of how
/// the directives are split across lines.  A false negative is not harmless:
/// the preamble guard in [`crate::run`] turns it into a hard error on valid input, so
/// EVERY directive is scanned, not just the first on a line.
fn has_enable_directive(source: &str, ext: &str) -> bool {
    cleaned_has_enable_directive(&strip_wgsl_comments(source), ext)
}

/// [`has_enable_directive`] over already comment-stripped text.
pub(crate) fn cleaned_has_enable_directive(cleaned: &str, ext: &str) -> bool {
    // Each `;`-terminated segment is one directive; a directive lists one or
    // more comma-separated extensions.  Scanning all segments handles several
    // directives on one line (`enable a; enable f16;`) - the callers include
    // arbitrary user-authored preamble text.
    for segment in cleaned.split(';') {
        let Some(list) = segment.trim_start().strip_prefix("enable") else {
            continue;
        };
        // `enable` must be followed by whitespace to be the keyword, not an
        // identifier prefix like `enablef16` / `enable_x`.  Line breaks count
        // (`enable\nf16;` is valid WGSL), matching `split_directives`; omitting
        // them makes the f16 preamble guard reject a preamble that DOES enable f16.
        if !list.starts_with([' ', '\t', '\n', '\r']) {
            continue;
        }
        if list.split(',').any(|e| e.trim() == ext) {
            return true;
        }
    }
    false
}

/// Byte spans to blank so `ext` disappears from every `requires` directive in
/// comment-stripped `cleaned`: the whole directive when `ext` is all it lists,
/// otherwise each `ext` entry through to the next entry (a last one from the
/// previous entry to the `;`).  Only the leading directive block is scanned -
/// a `requires` after a declaration is already invalid and stays naga's to
/// report.
pub(crate) fn requires_entry_spans(cleaned: &str, ext: &str) -> Vec<Range<usize>> {
    let directives = split_directives(cleaned).0;
    let mut spans = Vec::new();
    let mut start = 0;
    for (semi, _) in directives.match_indices(';') {
        let segment = &directives[start..semi];
        let keyword = start + (segment.len() - segment.trim_start().len());
        start = semi + 1;
        let Some(list) = segment.trim_start().strip_prefix("requires") else {
            continue;
        };
        if !list.starts_with([' ', '\t', '\n', '\r']) {
            continue;
        }
        // Trimmed bounds of each entry; an empty one is a trailing comma.
        let mut entries: Vec<(usize, usize)> = Vec::new();
        let mut pos = semi - list.len();
        for raw in list.split(',') {
            let s = pos + (raw.len() - raw.trim_start().len());
            let e = s + raw.trim().len();
            if s < e {
                entries.push((s, e));
            }
            pos += raw.len() + 1;
        }
        let is_ext = |&(s, e): &(usize, usize)| &directives[s..e] == ext;
        if !entries.is_empty() && entries.iter().all(is_ext) {
            spans.push(keyword..semi + 1);
            continue;
        }
        // Some entry is not `ext` (else the directive went above), so a last
        // `ext` entry always has a predecessor.
        for (k, entry) in entries.iter().enumerate().filter(|(_, e)| is_ext(e)) {
            spans.push(match entries.get(k + 1) {
                Some(&(next, _)) => entry.0..next,
                None => entries[k - 1].1..semi,
            });
        }
    }
    spans
}

/// Lexically compact WGSL text that never goes through the generator: strip
/// comments, then collapse every whitespace run, keeping a single space only
/// where joining would merge tokens.  Used on the bailout paths (input naga
/// cannot parse or validate) and on the naga-emitter fallback, which
/// otherwise ship fully un-minified text.
///
/// Grammar-agnostic and token-safe by construction, so it needs no parser:
/// * a space survives between two identifier-ish chars (WGSL identifiers are
///   XID; approximated by ASCII alphanumeric, `_`, and EVERY non-ASCII char -
///   XID's exotic members like U+2118 fail `is_alphanumeric`, and the
///   over-approximation only ever keeps a redundant space), covering
///   `enable f16`, `else if`, `let x`;
/// * a space survives where maximal munch would fuse two tokens of valid
///   WGSL into one: `- -x` (`--` is reserved), `+ +` (likewise), `& &x` /
///   `| |` (would form `&&`/`||`), and `x / *p` (would open a `/*` comment);
///   `> >` joins deliberately - WGSL's template-list disambiguation reads
///   nested `>>` correctly;
/// * everything else joins.
///
/// Idempotent: re-running splits at exactly the kept spaces and re-keeps
/// them.  Whole non-whitespace chunks are copied verbatim, so multi-byte
/// characters pass through untouched (ASCII whitespace never splits a
/// UTF-8 sequence).
pub(crate) fn compact_wgsl_text(source: &str) -> String {
    let stripped = strip_wgsl_comments(source);
    let ident_ish = |c: char| c.is_ascii_alphanumeric() || c == '_' || !c.is_ascii();
    let mut out = String::with_capacity(stripped.len());
    let mut prev_char: Option<char> = None;
    for chunk in stripped.split_ascii_whitespace() {
        if let (Some(prev), Some(next)) = (prev_char, chunk.chars().next()) {
            let keep = (ident_ish(prev) && ident_ish(next))
                || (prev == next && matches!(prev, '-' | '+' | '&' | '|'))
                || (prev == '/' && matches!(next, '*' | '/'));
            if keep {
                out.push(' ');
            }
        }
        out.push_str(chunk);
        prev_char = chunk.chars().next_back();
    }
    out
}

/// `true` when comment-stripped `cleaned` uses `token` as a whole identifier
/// token (so a longer identifier like `my_binding_array` never triggers for
/// `binding_array`).
pub(crate) fn cleaned_references_whole_token(cleaned: &str, token: &str) -> bool {
    let bytes = cleaned.as_bytes();
    let mut i = 0;
    while let Some(off) = cleaned[i..].find(token) {
        let start = i + off;
        let end = start + token.len();
        let before_ok = start == 0 || !is_ident_char(bytes[start - 1]);
        let after_ok = end >= bytes.len() || !is_ident_char(bytes[end]);
        if before_ok && after_ok {
            return true;
        }
        i = start + 1;
    }
    false
}

/// If `bytes[i..]` begins a `//` line comment or a (nesting-aware) `/* */`
/// block comment, return the byte index just past it; otherwise `None`.  An
/// unterminated block comment returns `len`.  Shared by [`split_directives`]'
/// leading-trivia skip and its `;`-terminator scan so both treat comments
/// identically - a `;` inside a comment must never terminate a directive.
fn skip_comment(bytes: &[u8], i: usize, len: usize) -> Option<usize> {
    if i + 1 < len && bytes[i] == b'/' && bytes[i + 1] == b'/' {
        // WGSL ends a line comment at ANY line break, not just `\n`; stopping at
        // `\n` alone would swallow a directive that follows a `\r`/VT-terminated
        // comment into the "comment", so `split_directives` would misplace it
        // (matches `strip_wgsl_comments`; a directive lost here ships past a
        // preamble's declarations - invalid, exit 0).
        let mut j = i + 2;
        while j < len && !wgsl_line_break_at(bytes, j) {
            j += 1;
        }
        return Some(j);
    }
    if i + 1 < len && bytes[i] == b'/' && bytes[i + 1] == b'*' {
        let mut j = i + 2;
        let mut depth = 1usize;
        while j + 1 < len && depth > 0 {
            if bytes[j] == b'/' && bytes[j + 1] == b'*' {
                depth += 1;
                j += 2;
            } else if bytes[j] == b'*' && bytes[j + 1] == b'/' {
                depth -= 1;
                j += 2;
            } else {
                j += 1;
            }
        }
        return Some(if depth > 0 { len } else { j });
    }
    None
}

/// Split `source` into its leading directive block and the remaining body.
/// WGSL requires every `enable` / `requires` / `diagnostic` directive
/// before any declaration, so a caller prepending a preamble splices the
/// source's directives ahead of the preamble's declarations.
pub(crate) fn split_directives(source: &str) -> (&str, &str) {
    let bytes = source.as_bytes();
    let len = bytes.len();
    // `boundary` is the committed end of the leading directive region; it
    // advances only past a fully `;`-terminated directive (plus any trailing
    // blank lines).  Scanning by `;` rather than by line is what makes this
    // correct on *compact* generator output, where the whole module is one
    // physical line (`enable f16;@fragment ...`) - a line-based scan would
    // misclassify the entire module as one directive and drop the body,
    // mis-ordering a prepended preamble's directives after declarations.
    let mut boundary = 0usize;
    let mut pos = 0usize;
    loop {
        // Skip whitespace and `//` / `/* */` comments WITHOUT committing the
        // boundary, so leading trivia before a NON-directive is not hoisted.
        // Only ASCII blankspace is skipped (directives are ASCII); a UTF-8
        // lead byte (>= 0xC2) is never ASCII blankspace, so the byte cursor
        // can never land inside a multi-byte sequence - `&source[scan..]`
        // below is always on a char boundary.  VT (0x0B) is WGSL blankspace
        // but `is_ascii_whitespace` omits it, so a `//` comment ended by a VT
        // (see `skip_comment`) would otherwise leave the VT unskipped and the
        // following directive unrecognised.
        let mut scan = pos;
        loop {
            while scan < len && (bytes[scan].is_ascii_whitespace() || bytes[scan] == 0x0B) {
                scan += 1;
            }
            if let Some(next) = skip_comment(bytes, scan, len) {
                scan = next;
                continue;
            }
            break;
        }
        if scan >= len {
            // Only trivia remains - preserve the old contract of treating a
            // trivia-only prefix as "all directives" (harmless: no decls).
            boundary = len;
            break;
        }
        // A directive keyword must end on a word boundary so user identifiers
        // like `requires_foo` / `diagnostic_counter` / `enablef16` are not
        // hoisted.  `diagnostic` may also be followed immediately by `(`
        // (the canonical `diagnostic(severity, rule);` form).
        let rest = &source[scan..];
        let is_directive = if let Some(a) = rest.strip_prefix("enable") {
            a.starts_with([' ', '\t', '\n', '\r'])
        } else if let Some(a) = rest.strip_prefix("requires") {
            a.starts_with([' ', '\t', '\n', '\r'])
        } else if let Some(a) = rest.strip_prefix("diagnostic") {
            a.starts_with(['(', ' ', '\t', '\n', '\r'])
        } else {
            false
        };
        if !is_directive {
            break;
        }
        // Consume through the terminating `;`, skipping comments so a `;`
        // inside a `//` or `/* */` comment between the directive keyword and
        // its real terminator does not split the directive mid-comment.
        let mut j = scan;
        while j < len && bytes[j] != b';' {
            if let Some(next) = skip_comment(bytes, j, len) {
                j = next;
                continue;
            }
            j += 1;
        }
        if j >= len {
            // Unterminated directive (already-invalid WGSL): commit the rest.
            boundary = len;
            break;
        }
        // Swallow one trailing line break plus any following blank lines so
        // the directive block ends cleanly (mirrors the old line-based form).
        let mut k = j + 1;
        while k < len && (bytes[k] == b' ' || bytes[k] == b'\t') {
            k += 1;
        }
        let mut m = k;
        if m < len && bytes[m] == b'\r' {
            m += 1;
        }
        if m < len && bytes[m] == b'\n' {
            k = m + 1;
            loop {
                let mut x = k;
                while x < len && (bytes[x] == b' ' || bytes[x] == b'\t') {
                    x += 1;
                }
                let mut y = x;
                if y < len && bytes[y] == b'\r' {
                    y += 1;
                }
                if y < len && bytes[y] == b'\n' {
                    k = y + 1;
                } else {
                    break;
                }
            }
        }
        boundary = k;
        pos = k;
    }
    (&source[..boundary], &source[boundary..])
}

/// Concatenate `fragments`, each non-empty one `\n`-terminated: directive
/// blocks and preamble bodies may or may not carry a trailing newline.
pub(crate) fn join_with_newline(fragments: &[&str]) -> String {
    let cap: usize = fragments.iter().map(|f| f.len()).sum::<usize>() + fragments.len();
    let mut out = String::with_capacity(cap);
    for fragment in fragments {
        if fragment.is_empty() {
            continue;
        }
        out.push_str(fragment);
        if !out.ends_with('\n') {
            out.push('\n');
        }
    }
    out
}

#[cfg(test)]
#[path = "text_tests.rs"]
mod tests;
