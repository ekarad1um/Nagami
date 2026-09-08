//! Lexical WGSL text helpers: comment stripping, directive splitting,
//! whole-token scans, and the parser-free compaction the bailout paths ship.
//! Byte-level and grammar-agnostic; nothing here touches naga.

use std::borrow::Cow;
use std::ops::Range;

/// Rewrite lone `\r` to `\n`; leave `\r\n` intact (`str::lines`
/// handles it).  Borrows the input when no lone `\r` exists.
//
// UTF-8 safety: `0x0D` never occurs inside a multi-byte sequence, so
// byte-level `\r` matches are char boundaries and the slicing is valid.
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

/// Blanks comments to spaces, preserving byte offsets and line breaks so
/// positional diagnostics stay accurate; borrows comment-free input.
pub(crate) fn strip_wgsl_comments(source: &str) -> Cow<'_, str> {
    if !source.contains("//") && !source.contains("/*") {
        return Cow::Borrowed(source);
    }
    let bytes = source.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
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
            // WGSL ends a line comment at any line-break code point, not just
            // `\n`; ending it at `\n` alone would blank a live statement after
            // a lone `\r`, and the bailout paths ship this text.
            while i < bytes.len() && !wgsl_line_break_at(bytes, i) {
                out.push(b' ');
                i += 1;
            }
        } else if bytes.get(i + 1) == Some(&b'*') {
            // WGSL block comments nest; a non-nesting scrub would close at the
            // inner `*/` and expose outer-comment `f16`/`enable` text to token
            // scans.
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

/// Every one of these requires `enable f16;` yet only `f16` itself contains
/// the substring `f16`.
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

/// [`cleaned_references_f16_token`] with the comment stripping production
/// does once up front, kept here so that handling stays under test.
#[cfg(test)]
pub(crate) fn references_f16_token(source: &str) -> bool {
    cleaned_references_f16_token(&strip_wgsl_comments(source))
}

/// `true` when comment-stripped `cleaned` needs `enable f16;`: the `f16`
/// keyword, a half-precision alias (`vec2h`..`mat4x4h`) or an `h`-suffixed
/// literal (`1.0h`, `0x1p2h`), matched token-wise so `myf16var` and `mesh`
/// never trigger.  A spurious positive is harmless - naga tolerates a
/// redundant `enable f16;` and the emitter drops it when the final module uses
/// no f16 - so detection errs broad.
pub(crate) fn cleaned_references_f16_token(cleaned: &str) -> bool {
    let bytes = cleaned.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    while i < len {
        let b = bytes[i];
        if is_ident_char(b) && !b.is_ascii_digit() {
            let start = i;
            while i < len && is_ident_char(bytes[i]) {
                i += 1;
            }
            if is_f16_type_token(&bytes[start..i]) {
                return true;
            }
        } else if b.is_ascii_digit() || (b == b'.' && i + 1 < len && bytes[i + 1].is_ascii_digit())
        {
            // Whole numeric literal (mantissa, hex digits, signed exponent,
            // type suffix): letters inside belong to the literal, so only the
            // final byte can be the `h` suffix.
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

/// [`cleaned_has_enable_directive`] for `f16`, comments included.
#[cfg(test)]
pub(crate) fn has_enable_f16_directive(source: &str) -> bool {
    has_enable_directive(source, "f16")
}

/// [`cleaned_has_enable_directive`] with comment stripping.
#[cfg(test)]
fn has_enable_directive(source: &str, ext: &str) -> bool {
    cleaned_has_enable_directive(&strip_wgsl_comments(source), ext)
}

/// WGSL blankspace: the ASCII set plus NEL / LS / PS / LRM / RLM.  Rust's
/// `is_ascii_whitespace` omits VT, which WGSL accepts wherever a space goes,
/// so a keyword boundary tested with it rejects `enable<VT>f16;` that tint
/// takes.
pub(crate) fn is_wgsl_blankspace(c: char) -> bool {
    matches!(c, '\u{20}' | '\u{09}'..='\u{0D}')
        || matches!(
            c,
            '\u{85}' | '\u{200E}' | '\u{200F}' | '\u{2028}' | '\u{2029}'
        )
}

/// [`is_wgsl_blankspace`] restricted to one byte, for scans that must stay on
/// a UTF-8 char boundary.
fn is_ascii_blankspace(b: u8) -> bool {
    b == b' ' || (0x09..=0x0D).contains(&b)
}

/// `true` when comment-stripped `cleaned` declares `enable <ext>;`, also as
/// one entry of a comma-separated list and however the directives are split
/// across lines.  A false negative becomes a hard error on valid input at the
/// preamble guard, so every directive is scanned, not just the first on a
/// line.
pub(crate) fn cleaned_has_enable_directive(cleaned: &str, ext: &str) -> bool {
    // One `;`-terminated segment per directive, however many share a line.
    for segment in cleaned.split(';') {
        let Some(list) = segment.trim_start().strip_prefix("enable") else {
            continue;
        };
        // Whitespace after `enable` separates the keyword from identifiers
        // like `enablef16`; line breaks count (`enable\nf16;` is valid WGSL),
        // or the f16 preamble guard rejects a preamble that does enable f16.
        if !list.starts_with(is_wgsl_blankspace) {
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
        if !list.starts_with(is_wgsl_blankspace) {
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
        // Some entry is not `ext` (else the whole directive was blanked), so
        // a last `ext` entry always has a predecessor.
        for (k, entry) in entries.iter().enumerate().filter(|(_, e)| is_ext(e)) {
            spans.push(match entries.get(k + 1) {
                Some(&(next, _)) => entry.0..next,
                None => entries[k - 1].1..semi,
            });
        }
    }
    spans
}

/// Lexically compact WGSL text that never goes through the generator (the
/// bailout paths and the naga-emitter fallback): strip comments, then
/// collapse every whitespace run, keeping a single space only where joining
/// would merge tokens.  Grammar-agnostic and token-safe without a parser:
/// * a space survives between two identifier-ish chars (XID approximated by
///   ASCII alphanumeric, `_` and EVERY non-ASCII char - exotic XID members
///   like U+2118 fail `is_alphanumeric`, and the over-approximation only
///   keeps a redundant space), covering `enable f16`, `else if`, `let x`;
/// * a space survives where maximal munch would fuse two tokens of valid
///   WGSL into one: `- -x` and `+ +` (`--`/`++` are reserved), `& &x` /
///   `| |` (`&&`/`||`), and `x / *p` (a `/*` comment); `> >` joins
///   deliberately, WGSL's template-list disambiguation reads nested `>>`;
/// * everything else joins.
///
/// Idempotent: re-running splits at exactly the kept spaces and re-keeps
/// them.  Whole non-whitespace chunks are copied verbatim, so multi-byte
/// characters pass through untouched.
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

/// Whole-token match on comment-stripped text, so `my_binding_array` never
/// triggers for `binding_array`.
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

/// Index just past the `//` or nesting `/* */` comment starting at `i`
/// (`len` when unterminated), else `None`.  Shared by the leading-trivia
/// skip and the `;`-terminator scan so a `;` inside a comment never
/// terminates a directive.
fn skip_comment(bytes: &[u8], i: usize, len: usize) -> Option<usize> {
    if i + 1 < len && bytes[i] == b'/' && bytes[i + 1] == b'/' {
        // Any WGSL line break ends the comment, not just `\n`: otherwise a
        // directive after a `\r`/VT-terminated comment is swallowed and ships
        // past a preamble's declarations (invalid output, exit 0).
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
    // `boundary` advances only past a fully `;`-terminated directive (plus
    // trailing blank lines).  Scanning by `;` rather than by line keeps this
    // correct on compact generator output, where the whole module is one
    // physical line (`enable f16;@fragment ...`) that a line scan would
    // classify as one directive, dropping the body.
    let mut boundary = 0usize;
    let mut pos = 0usize;
    loop {
        // Trivia is skipped without committing the boundary, so trivia before
        // a non-directive is not hoisted.  Only ASCII blankspace is skipped
        // and a UTF-8 lead byte never is, so `scan` stays on a char boundary.
        let mut scan = pos;
        loop {
            while scan < len && is_ascii_blankspace(bytes[scan]) {
                scan += 1;
            }
            if let Some(next) = skip_comment(bytes, scan, len) {
                scan = next;
                continue;
            }
            break;
        }
        if scan >= len {
            // Trivia-only source counts as all directives (harmless: no decls).
            boundary = len;
            break;
        }
        // A keyword must end on a word boundary so `requires_foo` / `enablef16`
        // are not hoisted; `diagnostic(` is the canonical form.
        let rest = &source[scan..];
        let is_directive = if let Some(a) = rest.strip_prefix("enable") {
            a.starts_with(is_wgsl_blankspace)
        } else if let Some(a) = rest.strip_prefix("requires") {
            a.starts_with(is_wgsl_blankspace)
        } else if let Some(a) = rest.strip_prefix("diagnostic") {
            a.starts_with(|c| c == '(' || is_wgsl_blankspace(c))
        } else {
            false
        };
        if !is_directive {
            break;
        }
        // Through the terminating `;`; a `;` inside a comment does not count.
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
        // One trailing line break plus following blank lines belong to the
        // directive block.
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
