//! Crate root and end-to-end minification entry point.
//!
//! Two layers are exposed:
//!
//! * [`run_module`] runs the IR optimization pipeline against a
//!   pre-parsed [`naga::Module`] and returns a [`Report`].
//! * [`run`] is the source-to-source entry point: parse, run
//!   [`run_module`], then emit minified WGSL via the custom generator,
//!   with automatic fallback to naga's own emitter if the custom output
//!   fails validation.
//!
//! The private helpers below (source preprocessing, directive splitting,
//! f16-auto-enable, preamble name collection) implement the invariants
//! the pipeline relies on but are not stable public surface.

pub mod config;
pub mod error;
pub mod generator;
mod io;
pub mod name_gen;
pub mod passes;
pub mod pipeline;
#[cfg(feature = "wasm")]
mod wasm;

use config::Config;
use error::Error;
use generator::{GenerateOptions, generate};
use pipeline::{PassReport, Report};
use std::collections::HashSet;

// MARK: Source preprocessing

/// Emit WGSL via naga's backend, wrapping the typed error in [`Error::Emit`].
/// Distinct from [`pipeline::emit_wgsl_with_info`] in that this helper is
/// used at the crate root for final-output emission and fallback paths
/// rather than per-pass trace emission.
fn emit_wgsl_with_naga_safe(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
) -> Result<String, Error> {
    naga::back::wgsl::write_string(module, info, naga::back::wgsl::WriterFlags::empty())
        .map_err(|e| Error::Emit(e.to_string()))
}

/// `true` when `module` contains an override-sized array
/// (`ArraySize::Pending`, e.g. `array<i32, O*2>`).  naga 29's own WGSL
/// back-end hits an `unreachable!()` while emitting the override size
/// expression, which under the release `panic = "abort"` strategy aborts the
/// process instead of returning an error.  nagami's generator emits these
/// types itself, so callers skip the naga baseline/fallback emit for such
/// modules rather than invoke the panicking path.
fn module_has_override_sized_array(module: &naga::Module) -> bool {
    module.types.iter().any(|(_, ty)| {
        matches!(
            ty.inner,
            naga::TypeInner::Array {
                size: naga::ArraySize::Pending(_),
                ..
            } | naga::TypeInner::BindingArray {
                size: naga::ArraySize::Pending(_),
                ..
            }
        )
    })
}

/// `true` when `line` is exactly one of the `wgpu_*` enable directives
/// that naga rejects.  Matched against the trimmed line so surrounding
/// whitespace is irrelevant.
fn should_strip_enable_directive(line: &str) -> bool {
    matches!(
        line.trim(),
        "enable wgpu_binding_array;"
            | "enable wgpu_per_vertex;"
            | "enable wgpu_ray_query;"
            | "enable wgpu_ray_query_vertex_return;"
    )
}

/// Normalise `source` so naga's front-end accepts it.  Three
/// adjustments, in order: lone-CR endings rewritten to LF (so
/// `str::lines` sees every break), `wgpu_*` enable directives naga
/// rejects are stripped, and `enable f16;` is injected when the
/// source uses the `f16` token without declaring it.  Output is
/// re-derived from the IR, so callers of [`run`] never observe these
/// rewrites.
fn preprocess_source_for_naga(source: &str) -> String {
    // `str::lines` ignores lone `\r` (classic-Mac); the downstream
    // per-line scans assume every break is visible, so normalise CR
    // to LF first.
    let normalized = normalize_line_endings(source);
    let source = normalized.as_str();

    let mut had_changes = false;
    let mut stripped = String::with_capacity(source.len());

    for line in source.lines() {
        if should_strip_enable_directive(line) {
            had_changes = true;
            continue;
        }
        stripped.push_str(line);
        stripped.push('\n');
    }

    // Shaders often use f16 without an explicit `enable f16;`; inject it
    // when absent so naga can parse them.  `references_f16_token` matches
    // whole tokens on comment-stripped text (so `myf16var`/`mesh` never
    // trigger), and `has_enable_f16_directive` tolerates arbitrary
    // whitespace in an existing `enable   f16;` so it is not injected twice.
    if references_f16_token(&stripped) && !has_enable_f16_directive(&stripped) {
        let mut with_f16 = String::with_capacity(stripped.len() + 12);
        with_f16.push_str("enable f16;\n");
        with_f16.push_str(&stripped);
        return with_f16;
    }

    if had_changes {
        stripped
    } else {
        source.to_string()
    }
}

/// Rewrite lone `\r` to `\n`; leave `\r\n` intact (`str::lines`
/// handles it).  Returns the original string unchanged when no lone
/// `\r` exists.
//
// UTF-8 safety: `0x0D` cannot appear inside a multi-byte sequence
// (continuation bytes are `0x80..=0xBF`, lead bytes `0xC0..=0xFD`),
// so byte-level `\r` matches always land on character boundaries -
// the slicing below is guaranteed valid UTF-8.
fn normalize_line_endings(source: &str) -> String {
    let bytes = source.as_bytes();
    let mut needs_rewrite = false;
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\r' && bytes.get(i + 1) != Some(&b'\n') {
            needs_rewrite = true;
            break;
        }
        i += 1;
    }
    if !needs_rewrite {
        return source.to_string();
    }
    let mut out = String::with_capacity(source.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\r' && bytes.get(i + 1) != Some(&b'\n') {
            out.push('\n');
            i += 1;
        } else {
            let start = i;
            while i < bytes.len() && !(bytes[i] == b'\r' && bytes.get(i + 1) != Some(&b'\n')) {
                i += 1;
            }
            out.push_str(&source[start..i]);
        }
    }
    out
}

/// `true` when `b` can appear inside a WGSL identifier (ASCII alphanumeric
/// or underscore).  Used for whole-token matching in token detection.
fn is_ident_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Replace WGSL line and block comments with spaces while preserving
/// byte offsets and line breaks, so subsequent lexical scans see only
/// real code but any positional diagnostics stay accurate.
fn strip_wgsl_comments(source: &str) -> String {
    let bytes = source.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        // Line comment
        if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'/' {
            while i < bytes.len() && bytes[i] != b'\n' {
                out.push(b' ');
                i += 1;
            }
            continue;
        }
        // Block comment.  WGSL (https://www.w3.org/TR/WGSL/#comments)
        // permits nesting; a non-nesting scrub would close at the
        // inner `*/` and expose outer-comment `f16`/`enable` content
        // to token scans.  Track depth.
        if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'*' {
            out.push(b' ');
            out.push(b' ');
            i += 2;
            let mut depth: u32 = 1;
            while i + 1 < bytes.len() && depth > 0 {
                if bytes[i] == b'/' && bytes[i + 1] == b'*' {
                    depth += 1;
                    out.push(b' ');
                    out.push(b' ');
                    i += 2;
                } else if bytes[i] == b'*' && bytes[i + 1] == b'/' {
                    depth -= 1;
                    out.push(b' ');
                    out.push(b' ');
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
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    // Every replacement emits ASCII space or newline, so the resulting
    // bytes stay valid UTF-8.
    String::from_utf8(out).expect("comment stripping preserves UTF-8")
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
fn references_f16_token(source: &str) -> bool {
    let cleaned = strip_wgsl_comments(source);
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

/// `true` when `source` already contains an `enable f16;` directive.
/// Tolerates arbitrary intra-line whitespace between `enable`, `f16`,
/// and the terminating `;` so hand-formatted shaders are still detected.
fn has_enable_f16_directive(source: &str) -> bool {
    let cleaned = strip_wgsl_comments(source);
    for line in cleaned.lines() {
        let t = line.trim();
        let Some(rest) = t.strip_prefix("enable") else {
            continue;
        };
        // `enable` must be followed by at least one space or tab to be
        // lexically distinct from an identifier like `enablef16`.
        if !rest.starts_with([' ', '\t']) {
            continue;
        }
        let rest = rest.trim_start_matches([' ', '\t']);
        let Some(rest) = rest.strip_prefix("f16") else {
            continue;
        };
        let rest = rest.trim_start_matches([' ', '\t']);
        if rest.starts_with(';') {
            return true;
        }
    }
    false
}

/// Split `source` into its leading directive block and the remaining body.
///
/// WGSL requires every `enable`, `requires`, and `diagnostic` directive
/// to appear before any global declaration.  When a preamble full of
/// declarations is prepended to user source, the source's own
/// directives end up after the preamble's declarations and the combined
/// text stops being spec-compliant.  Extracting the leading directives
/// here lets the caller splice them in front of the preamble before
/// concatenation so the result remains valid.
/// If `bytes[i..]` begins a `//` line comment or a (nesting-aware) `/* */`
/// block comment, return the byte index just past it; otherwise `None`.  An
/// unterminated block comment returns `len`.  Shared by [`split_directives`]'
/// leading-trivia skip and its `;`-terminator scan so both treat comments
/// identically - a `;` inside a comment must never terminate a directive.
fn skip_comment(bytes: &[u8], i: usize, len: usize) -> Option<usize> {
    if i + 1 < len && bytes[i] == b'/' && bytes[i + 1] == b'/' {
        let mut j = i + 2;
        while j < len && bytes[j] != b'\n' {
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

fn split_directives(source: &str) -> (&str, &str) {
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
        // Only ASCII whitespace is skipped (directives are ASCII); a UTF-8
        // lead byte (>= 0xC2) is never ASCII whitespace, so the byte cursor
        // can never land inside a multi-byte sequence - `&source[scan..]`
        // below is always on a char boundary.
        let mut scan = pos;
        loop {
            while scan < len && bytes[scan].is_ascii_whitespace() {
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

/// Concatenate `fragments` so each non-empty fragment is followed by
/// at least one `\n` before the next fragment starts.  Empty fragments
/// are skipped so we never emit a stray blank line.  Used to splice
/// directive blocks and preamble bodies safely when any fragment may
/// or may not already carry a trailing newline.
fn join_with_newline(fragments: &[&str]) -> String {
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

// MARK: Public entry points

/// Result of a full minification pipeline run.
///
/// `#[non_exhaustive]` so we can add fields (timing, diagnostics, etc.)
/// in future releases without a major version bump.  Two consequences
/// for external callers (existing `2026.4.x` consumers may need a
/// small migration):
///
/// - Pattern-match with a rest binding: `let Output { source, report, .. } = run(...)?;`
/// - You cannot construct this struct directly from outside the crate;
///   obtain values via [`run`].  This is intentional - the contract
///   between source-bytes accounting and per-pass timings/diagnostics
///   lives inside [`run`] and is not safe to bypass.
#[non_exhaustive]
pub struct Output {
    /// The minified WGSL source string.
    pub source: String,
    /// Aggregate report with input/output sizes and per-pass details.
    pub report: Report,
}

/// Apply IR-level optimization passes to an already-parsed naga module.
///
/// The module is modified in place.  No WGSL text generation is performed
/// past the initial and final size measurements; use [`run`] for
/// end-to-end source-to-source minification.
///
/// # Errors
///
/// Returns [`Error::Validation`] if `module` fails validation either
/// before the pipeline runs or after a rollback-less pass, and
/// [`Error::Emit`] if naga's backend cannot render the final IR.
pub fn run_module(module: &mut naga::Module, config: &Config) -> Result<Report, Error> {
    let info = io::validate_module(module)?;
    let before_wgsl = emit_wgsl_with_naga_safe(module, &info)?;
    let mut report = Report::new(before_wgsl.len());

    pipeline::run_ir_passes(module, config, &mut report)?;

    let info = io::validate_module(module)?;
    let after_wgsl = emit_wgsl_with_naga_safe(module, &info)?;
    report.output_bytes = after_wgsl.len();

    Ok(report)
}

/// Collect every top-level declaration name in `module`: types,
/// struct members, constants, overrides, globals, functions, and
/// entry points.  [`run`] uses the result to hide preamble symbols
/// from the generator and to extend `preserve_symbols` so rename and
/// mangle passes leave them alone.
fn collect_module_names(module: &naga::Module) -> HashSet<String> {
    let mut names = HashSet::new();
    for (_, ty) in module.types.iter() {
        if let Some(name) = &ty.name {
            names.insert(name.clone());
        }
        if let naga::TypeInner::Struct { members, .. } = &ty.inner {
            for m in members {
                if let Some(name) = &m.name {
                    names.insert(name.clone());
                }
            }
        }
    }
    for (_, c) in module.constants.iter() {
        if let Some(name) = &c.name {
            names.insert(name.clone());
        }
    }
    for (_, ov) in module.overrides.iter() {
        if let Some(name) = &ov.name {
            names.insert(name.clone());
        }
    }
    for (_, g) in module.global_variables.iter() {
        if let Some(name) = &g.name {
            names.insert(name.clone());
        }
    }
    for (_, f) in module.functions.iter() {
        if let Some(name) = &f.name {
            names.insert(name.clone());
        }
    }
    for ep in &module.entry_points {
        names.insert(ep.name.clone());
    }
    names
}

// MARK: Naga error-message coupling

/// Substrings that identify naga parse errors about enable-extensions
/// naga does not yet support (or a shader declares but the front-end
/// refuses).  Matched against the rendered [`Error`] message.
///
/// NOTE: This couples behaviour to naga's human-readable error strings.
/// A naga upgrade that rewords these messages silently flips the
/// "unsupported extension -> return input unchanged" code path into a
/// hard error.  The lock-in tests at the bottom of this module pin the
/// current phrasings so such drift fails at test time instead.
const UNSUPPORTED_EXTENSION_PATTERNS: &[&str] = &[
    "enable extension is not enabled",
    "enable-extension is not yet supported",
];

/// Substrings flagging text-validation errors that are known naga
/// limitations rather than real generator bugs.  Currently scoped to
/// the `subgroups` enable-extension, which naga's text front-end
/// rejects even though its IR emitter produces it.
const KNOWN_TEXT_VALIDATION_LIMITATION_PATTERNS: &[&str] = &[
    "`subgroups` enable-extension is not yet supported",
    "subgroups enable-extension is not yet supported",
];

/// `true` when `err` is a parse error whose message matches any of
/// [`UNSUPPORTED_EXTENSION_PATTERNS`].
///
/// Restricted to [`Error::Parse`] on purpose: the patterns are
/// substring matches against naga's rendered diagnostic, which can
/// appear inside user-controlled source quoted by a validation or
/// emit error.  Without the variant guard the caller's
/// "extension we cannot parse -> return input unchanged" branch can
/// silently swallow a real validation or emit failure.
///
/// Additionally restricted to the first line of the rendered
/// diagnostic.  naga's codespan output places the diagnostic message
/// on line 1 (`error: <message>`) and quotes user source on the
/// indented lines that follow.  Matching the full message would let
/// a user shader containing the pattern text in a comment trigger
/// the bailout when an UNRELATED parse error happens to render that
/// comment as nearby context.
fn is_unsupported_extension_parse_error(err: &Error) -> bool {
    if !matches!(err, Error::Parse(_)) {
        return false;
    }
    let msg = err.to_string();
    let first_line = msg.lines().next().unwrap_or("");
    UNSUPPORTED_EXTENSION_PATTERNS
        .iter()
        .any(|p| first_line.contains(p))
}

/// `true` when `err` is a `Parse` or `Validation` error whose message
/// matches any of [`KNOWN_TEXT_VALIDATION_LIMITATION_PATTERNS`].
///
/// Both variants are accepted because `io::validate_wgsl_text`
/// internally calls `parse_wgsl` (which wraps any front-end failure
/// in `Error::Parse`) and then `validate_module_with_source` (which
/// wraps validator failures in `Error::Validation`); naga's text
/// front-end can report a not-yet-supported `enable` directive
/// through either path depending on whether the failure surfaces at
/// tokenisation or at semantic validation.
///
/// Scoped to `Parse | Validation` to refuse matching against unrelated
/// `Emit`/`Io`/`Config` errors whose body happens to quote the same
/// phrasing - otherwise the caller would silently bypass the
/// "fall back to naga emitter" guard on a real downstream failure.
fn is_known_text_validation_limitation(err: &Error) -> bool {
    if !matches!(err, Error::Parse(_) | Error::Validation(_)) {
        return false;
    }
    let msg = err.to_string();
    // Same first-line restriction as `is_unsupported_extension_parse_error`:
    // naga's diagnostic format places the message on line 1 and quotes user
    // source on subsequent lines, so restricting the match here prevents a
    // user shader carrying the pattern text in a comment from spuriously
    // opting into the round-trip-validation bypass when an unrelated parse
    // or validation error renders that comment as nearby context.
    let first_line = msg.lines().next().unwrap_or("");
    KNOWN_TEXT_VALIDATION_LIMITATION_PATTERNS
        .iter()
        .any(|p| first_line.contains(p))
}

/// Minify a WGSL shader source string end-to-end.
///
/// Parses `source` (optionally prepended with [`Config::preamble`]),
/// runs the IR optimization pipeline, and emits minified WGSL via the
/// custom generator.  If the generator fails or produces output that
/// fails round-trip validation, the result silently falls back to
/// naga's own emitter, except when a preamble is active (in which case
/// the preamble-stripping invariant prevents a safe fallback and the
/// error is propagated).
///
/// # Errors
///
/// Propagates [`Error::Parse`], [`Error::Validation`], and
/// [`Error::Emit`] from the underlying stages.  Shaders using
/// extensions naga cannot parse short-circuit to an unchanged input
/// rather than erroring; see `UNSUPPORTED_EXTENSION_PATTERNS`.
pub fn run(source: &str, config: &Config) -> Result<Output, Error> {
    let normalized_source = preprocess_source_for_naga(source);

    // Resolve the preamble: parse it to collect external names, then
    // splice its body after both sets of hoisted directives.  Empty or
    // whitespace-only preambles collapse to the no-preamble path so
    // downstream code has a single predicate to check.  Computed once
    // and reused both at parse time below and at re-validation time
    // after generator emission - `preprocess_source_for_naga` is a
    // pure function of its input, so memoising the result avoids a
    // second O(preamble) scan/allocation for free.
    let effective_preamble = config.preamble.as_deref().filter(|s| !s.trim().is_empty());
    let normalized_preamble: Option<String> = effective_preamble.map(preprocess_source_for_naga);
    let (preamble_names, full_source);
    if let Some(normalized_preamble) = normalized_preamble.as_deref() {
        // Run the same `wgpu_*` stripping / `enable f16;` injection
        // against the preamble that the user source already gets.
        // Without this, a preamble that uses `f16` (or carries a
        // `wgpu_binding_array` directive that naga rejects) would
        // crash at parse time while the same text in the source body
        // would silently succeed - an asymmetry that surprised callers
        // and prevented preambles from sharing source-style content.
        let preamble_module = io::parse_wgsl_with_path(normalized_preamble, "<preamble>")?;
        preamble_names = collect_module_names(&preamble_module);
        // Directives must precede declarations (see `split_directives`),
        // so extract both sides' leading directives and prepend them
        // before the preamble body.  `split_directives` returns each
        // section as a borrowed slice that may or may not carry a
        // trailing newline (e.g. a source whose entire content is
        // `enable f16;` with no final newline returns `"enable f16;"`).
        // Concatenating two such slices directly would glue the last
        // directive of the first block onto the first directive of the
        // next, producing a syntax error; `join_with_newline` ensures
        // each non-empty fragment is `\n`-terminated before the next
        // fragment begins.
        let (source_directives, source_body) = split_directives(&normalized_source);
        let (preamble_directives, preamble_body) = split_directives(normalized_preamble);
        full_source = join_with_newline(&[
            source_directives,
            preamble_directives,
            preamble_body,
            source_body,
        ]);
    } else {
        preamble_names = HashSet::new();
        full_source = normalized_source;
    }

    let mut module = match io::parse_wgsl(&full_source) {
        Ok(m) => m,
        Err(e) if is_unsupported_extension_parse_error(&e) => {
            // Shader uses an extension naga can't parse (e.g.
            // `wgpu_ray_query`).  Return the original source so the
            // caller can still ship something runnable on backends
            // that DO understand the extension.  The synthetic
            // `unsupported_extension_bailout` PassReport lets
            // downstream tooling distinguish bailout from "ran with
            // no changes"; `validation_ok = true` is "no failure
            // observed" (no IR ever built), set true so CI gates
            // asserting `all validation_ok` don't fail spuriously.
            let mut report = Report::new(source.len());
            report.pass_reports.push(PassReport {
                pass_name: "unsupported_extension_bailout".to_string(),
                before_bytes: Some(source.len()),
                after_bytes: Some(source.len()),
                changed: false,
                duration_us: 0,
                validation_ok: true,
                text_validation_ok: None,
                rolled_back: false,
            });
            return Ok(Output {
                source: source.to_string(),
                report,
            });
        }
        Err(e) => return Err(e),
    };
    let mut report = Report::new(source.len());

    // Add preamble names to `preserve_symbols` so rename and mangle
    // passes do not touch them; any access expression in the user
    // source still has to resolve against the preamble's exported names.
    let mut effective_config = config.clone();
    effective_config
        .preserve_symbols
        .extend(preamble_names.iter().cloned());

    pipeline::run_ir_passes(&mut module, &effective_config, &mut report)?;

    let info = io::validate_module(&module)?;
    // Skip the naga baseline/fallback emit for override-sized arrays (its
    // back-end aborts on them); nagami's generator emits them, and the
    // baseline byte count falls back to the input length.
    let naga_output: Option<String> = if module_has_override_sized_array(&module) {
        None
    } else {
        Some(emit_wgsl_with_naga_safe(&module, &info)?)
    };
    let before_bytes = naga_output
        .as_ref()
        .map_or_else(|| source.len(), String::len);

    let gen_result = generate(
        &module,
        &info,
        GenerateOptions {
            beautify: config.beautify,
            indent: config.indent,
            mangle: config.mangle(),
            float_precision: config.float_precision,
            preserve_symbols: effective_config.preserve_symbols.iter().cloned().collect(),
            preamble_names,
            type_alias: true,
            ..Default::default()
        },
    );

    let has_preamble = effective_preamble.is_some();

    // Fall back to naga's output if the generator errored or produced
    // invalid WGSL.  With a preamble active, `naga_output` still
    // contains the preamble's declarations and is unusable as a
    // fallback (the consumer will re-prepend the preamble, producing
    // duplicate definitions), so the error must propagate instead.
    let (final_source, final_bytes, changed, rolled_back, compacted_valid, duration_us) =
        match gen_result {
            Ok(emitted) => {
                let after_bytes = emitted.source.len();
                let changed = before_bytes != after_bytes
                    || naga_output.as_deref() != Some(emitted.source.as_str());
                // With a preamble active the emitted source is
                // incomplete on its own; validate by re-prepending the
                // preamble with directives properly hoisted.
                let validation_result = if let Some(normalized_preamble) =
                    normalized_preamble.as_deref()
                {
                    // Reuse the same normalised preamble computed
                    // up-front so the re-prepended text exactly matches
                    // what naga parsed; this avoids both redundant
                    // work and any chance of drift if the preprocess
                    // function ever became non-deterministic.
                    let (emit_directives, emit_body) = split_directives(&emitted.source);
                    let (pre_directives, pre_body) = split_directives(normalized_preamble);
                    let combined =
                        join_with_newline(&[emit_directives, pre_directives, pre_body, emit_body]);
                    io::validate_wgsl_text(&combined)
                } else {
                    io::validate_wgsl_text(&emitted.source)
                };
                let valid = match &validation_result {
                    Ok(()) => true,
                    // Key the bypass on the validator's own message, not on the
                    // emitted source text.  The custom generator never
                    // synthesises `enable subgroups;` (it emits the builtins
                    // bare), so an `emitted.source.contains("enable subgroups;")`
                    // guard was unsatisfiable - it made this whole arm dead.
                    // naga's text front-end accepts the bare subgroup builtins
                    // today, so this still does not fire; it remains as forward
                    // defence for a naga release that reports the subgroups
                    // round-trip limitation through text re-validation.
                    Err(e) if is_known_text_validation_limitation(e) => {
                        if config.trace.enabled {
                            eprintln!(
                                "warning: skipping text-validation rollback due to known naga subgroup parser limitation"
                            );
                        }
                        true
                    }
                    Err(_) => false,
                };
                if valid {
                    (
                        emitted.source,
                        after_bytes,
                        changed,
                        false,
                        true,
                        emitted.duration_us,
                    )
                } else if has_preamble {
                    // Naga-emitter fallback unsafe with a preamble -
                    // its output carries the preamble's decls already,
                    // and the caller will re-prepend them, duplicating.
                    // Propagate the underlying validator message so
                    // the user can diagnose; the original error was
                    // dropped before this branch.
                    let underlying = match validation_result {
                        Err(e) => e.to_string(),
                        Ok(()) => "(no underlying error)".to_string(),
                    };
                    return Err(Error::Emit(format!(
                        "generator output failed validation; \
                         cannot fall back safely when a preamble is active: {underlying}",
                    )));
                } else if let Some(naga_output) = naga_output {
                    if config.trace.enabled {
                        if let Err(e) = &validation_result {
                            eprintln!("warning: generator WGSL validation error: {e}");
                        }
                        eprintln!(
                            "warning: generator output failed text validation; falling back to naga emitter"
                        );
                    }
                    // The naga-emitter fallback is *usually* valid, but it is
                    // not guaranteed: naga's own wgsl-out can emit tokens its
                    // frontend then rejects (e.g. an `f32(<f64 literal>)` cast,
                    // or an f16 literal whose `enable f16;` directive it drops).
                    // Re-validate before trusting it so a doubly-invalid case
                    // surfaces as a diagnosable error instead of silently
                    // shipping invalid WGSL (matching the preamble posture above).
                    if let Err(ve) = io::validate_wgsl_text(&naga_output) {
                        return Err(Error::Emit(format!(
                            "generator output failed validation and the naga-emitter \
                             fallback is also invalid: {ve}"
                        )));
                    }
                    (
                        naga_output,
                        before_bytes,
                        false,
                        true,
                        false,
                        emitted.duration_us,
                    )
                } else {
                    // No naga baseline (override-sized array): the generator
                    // output is invalid and there is no fallback emitter.
                    let underlying = match validation_result {
                        Err(e) => e.to_string(),
                        Ok(()) => "(no underlying error)".to_string(),
                    };
                    return Err(Error::Emit(format!(
                        "generator output failed validation and no naga fallback \
                         is available (override-sized array): {underlying}"
                    )));
                }
            }
            Err(e) => match naga_output {
                Some(naga_output) => {
                    if has_preamble {
                        return Err(e);
                    }
                    if config.trace.enabled {
                        eprintln!(
                            "warning: generator emit failed ({e}); falling back to naga emitter"
                        );
                    }
                    // Re-validate the naga fallback (see above) so a doubly-invalid
                    // case errors instead of shipping invalid WGSL.
                    if let Err(ve) = io::validate_wgsl_text(&naga_output) {
                        return Err(Error::Emit(format!(
                            "generator emit failed ({e}) and the naga-emitter \
                             fallback is also invalid: {ve}"
                        )));
                    }
                    (naga_output, before_bytes, false, true, false, 0)
                }
                // No naga fallback available (override-sized array): the
                // generator's own error is the only diagnosis.
                None => return Err(e),
            },
        };

    report.pass_reports.push(PassReport {
        pass_name: "generator_emit".to_string(),
        before_bytes: Some(before_bytes),
        after_bytes: Some(final_bytes),
        changed: !rolled_back && changed,
        duration_us,
        validation_ok: compacted_valid,
        text_validation_ok: Some(compacted_valid),
        rolled_back,
    });
    report.output_bytes = final_bytes;

    Ok(Output {
        source: final_source,
        report,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TRIVIAL_SHADER: &str = r#"
        @vertex fn main() -> @builtin(position) vec4<f32> {
            return vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }
    "#;

    #[test]
    fn run_module_report_bytes_nonzero() {
        let mut module = io::parse_wgsl(TRIVIAL_SHADER).unwrap();
        let config = Config::default();
        let report = run_module(&mut module, &config).unwrap();
        assert!(report.input_bytes > 0, "input_bytes must be nonzero");
        assert!(report.output_bytes > 0, "output_bytes must be nonzero");
        assert!(
            report.output_bytes <= report.input_bytes,
            "output should not exceed input for a trivial shader"
        );
    }

    #[test]
    fn run_produces_generator_emit_pass() {
        let config = Config::default();
        let output = run(TRIVIAL_SHADER, &config).unwrap();
        let gen_pass = output
            .report
            .pass_reports
            .iter()
            .find(|p| p.pass_name == "generator_emit");
        assert!(
            gen_pass.is_some(),
            "report must include generator_emit pass"
        );
        let gen_pass = gen_pass.unwrap();
        assert!(gen_pass.validation_ok, "generator output must be valid");
        assert!(!gen_pass.rolled_back, "generator should not need rollback");
        assert!(
            output.report.output_bytes > 0,
            "output_bytes must be nonzero"
        );
    }

    #[test]
    fn run_output_matches_report_bytes() {
        let config = Config::default();
        let output = run(TRIVIAL_SHADER, &config).unwrap();
        assert_eq!(
            output.source.len(),
            output.report.output_bytes,
            "report.output_bytes must match source length"
        );
        assert_eq!(
            TRIVIAL_SHADER.len(),
            output.report.input_bytes,
            "report.input_bytes must match original source length"
        );
    }

    #[test]
    fn validate_each_pass_reports_after_bytes() {
        let config = Config {
            trace: config::TraceConfig {
                enabled: false,
                validate_each_pass: true,
                ..Default::default()
            },
            ..Default::default()
        };
        let output = run(TRIVIAL_SHADER, &config).unwrap();
        // IR passes should all have after_bytes when validate_each_pass is on
        for pr in output
            .report
            .pass_reports
            .iter()
            .filter(|p| p.pass_name != "generator_emit")
        {
            assert!(
                pr.after_bytes.is_some(),
                "pass '{}' should have after_bytes when validate_each_pass is on",
                pr.pass_name
            );
        }
    }

    #[test]
    fn generator_emit_report_consistency() {
        // Verify the generator_emit pass report fields are internally consistent.
        let config = Config::default();
        let output = run(TRIVIAL_SHADER, &config).unwrap();
        let gen_report = output
            .report
            .pass_reports
            .iter()
            .find(|p| p.pass_name == "generator_emit")
            .expect("generator_emit pass must exist");

        // If not rolled back, final source should equal output
        if !gen_report.rolled_back {
            assert_eq!(
                gen_report.after_bytes,
                Some(output.source.len()),
                "after_bytes must match output source length"
            );
            assert!(gen_report.validation_ok);
            assert_eq!(gen_report.text_validation_ok, Some(true));
        }
        // Whether rolled back or not, before_bytes must be present
        assert!(gen_report.before_bytes.is_some());
        assert!(gen_report.after_bytes.is_some());
        // changed must be false when rolled_back
        if gen_report.rolled_back {
            assert!(!gen_report.changed);
        }
    }

    // MARK: End-to-end preserve_symbols tests

    /// Regression guard for [`literal_to_wgsl_bare`].  Bare literal
    /// emission is safe only at its two sanctioned call sites:
    ///
    ///   1. inside a type constructor, where the enclosing type pins
    ///      the component type, and
    ///   2. as the RHS of an extracted `const NAME = ...;` declaration,
    ///      where every use of `NAME` re-binds via abstract coercion.
    ///
    /// The shader below stresses concrete-typed whole-number literals
    /// (`F32(1.0)`, `F32(2.0)`, `F32(3.0)`) in positions that would
    /// break if the emitter ever used the bare form outside those two
    /// patterns, e.g. as a standalone `let` initialiser or as an
    /// overload-resolution argument to `atan2`.
    #[test]
    fn e2e_concrete_float_literals_round_trip_after_minification() {
        let src = r#"
            @compute @workgroup_size(1)
            fn m() {
                // F32 literals as standalone let initializers (concretize path).
                let a: f32 = 1.0;
                let b: f32 = 2.0;
                // F32 literals as atan2 arguments (overload-resolution path).
                let c: f32 = atan2(a, b);
                // Vec constructor with repeated concrete-typed float component
                // (splat-collapse + bare-emit path).
                let d: vec3<f32> = vec3<f32>(3.0, 3.0, 3.0);
                // Binary-arithmetic with literal (left & right both typed).
                let e: f32 = c + 1.0;
                // Suppress unused-variable warnings in the emitted code.
                _ = d.x + e;
            }
        "#;
        let config = Config::default();
        let output = run(src, &config).expect("pipeline should succeed");
        // The emitted WGSL must re-parse - this is the whole point of the
        // bare-literal invariant.
        io::parse_wgsl(&output.source).expect("minified output must round-trip");
    }

    #[test]
    fn e2e_preserve_symbols_struct_type_survives_mangle() {
        let src = r#"
            struct Uniforms {
                resolution: vec2<f32>,
                time: f32,
            }
            @group(0) @binding(0) var<uniform> uniforms: Uniforms;
            @fragment
            fn fs_main() -> @location(0) vec4f {
                return vec4f(uniforms.resolution, uniforms.time, 1.0);
            }
        "#;
        let config = Config {
            profile: config::Profile::Max,
            mangle: Some(true),
            preserve_symbols: vec!["Uniforms".to_string()],
            ..Default::default()
        };
        let output = run(src, &config).unwrap();
        assert!(
            output.source.contains("Uniforms"),
            "preserved struct type name must survive full pipeline: {}",
            output.source
        );
        // Validate the output is valid WGSL.
        io::validate_wgsl_text(&output.source).expect("output must be valid WGSL");
    }

    #[test]
    fn e2e_preserve_symbols_struct_member_survives_mangle() {
        let src = r#"
            struct Uniforms {
                resolution: vec2<f32>,
                time: f32,
            }
            @group(0) @binding(0) var<uniform> uniforms: Uniforms;
            @fragment
            fn fs_main() -> @location(0) vec4f {
                return vec4f(uniforms.resolution, uniforms.time, 1.0);
            }
        "#;
        let config = Config {
            profile: config::Profile::Max,
            mangle: Some(true),
            preserve_symbols: vec!["resolution".to_string()],
            ..Default::default()
        };
        let output = run(src, &config).unwrap();
        assert!(
            output.source.contains("resolution"),
            "preserved member name must survive full pipeline: {}",
            output.source
        );
        io::validate_wgsl_text(&output.source).expect("output must be valid WGSL");
    }

    #[test]
    fn e2e_preserve_symbols_multiple_categories() {
        // Preserve a struct type, a member, and a constant through the full pipeline.
        let src = r#"
            const MY_CONST: f32 = 3.14;
            struct Material {
                color: vec3<f32>,
                roughness: f32,
            }
            @group(0) @binding(0) var<uniform> mat: Material;
            @fragment fn fs_main() -> @location(0) vec4f {
                return vec4f(mat.color * MY_CONST, mat.roughness);
            }
        "#;
        let config = Config {
            profile: config::Profile::Max,
            mangle: Some(true),
            preserve_symbols: vec![
                "Material".to_string(),
                "color".to_string(),
                "MY_CONST".to_string(),
            ],
            ..Default::default()
        };
        let output = run(src, &config).unwrap();
        assert!(
            output.source.contains("Material"),
            "preserved struct type must survive: {}",
            output.source
        );
        assert!(
            output.source.contains("color"),
            "preserved member must survive: {}",
            output.source
        );
        assert!(
            output.source.contains("MY_CONST"),
            "preserved constant must survive: {}",
            output.source
        );
        // Non-preserved member should be mangled.
        assert!(
            !output.source.contains("roughness"),
            "non-preserved member should be mangled: {}",
            output.source
        );
        io::validate_wgsl_text(&output.source).expect("output must be valid WGSL");
    }

    // MARK: Struct name collision regression tests

    #[test]
    fn e2e_struct_name_does_not_collide_with_function_params() {
        // Regression: the rename pass assigns short names to function
        // parameters and locals.  The generator must not pick the same
        // names for struct types/members, because in WGSL function-scope
        // names shadow module-scope type names, making the struct
        // unusable as a type inside that function.
        //
        // This shader has enough globals + function params that the
        // rename pass will consume early short names, creating a
        // collision opportunity for the generator's struct mangling.
        let src = r#"
            struct Data { value: f32, extra: f32 }
            @group(0) @binding(0) var<uniform> d: Data;
            fn helper(a: f32, b: f32, c: f32) -> Data {
                var result: Data;
                result.value = a + b + c + d.value;
                result.extra = a * d.extra;
                return result;
            }
            @fragment fn main() -> @location(0) vec4f {
                let r = helper(1.0, 2.0, 3.0);
                return vec4f(r.value, r.extra, 0.0, 1.0);
            }
        "#;
        let config = Config {
            profile: config::Profile::Max,
            mangle: Some(true),
            ..Default::default()
        };
        let output = run(src, &config).unwrap();
        // The generator must not roll back.
        let gen_report = output
            .report
            .pass_reports
            .iter()
            .find(|p| p.pass_name == "generator_emit")
            .expect("generator_emit pass must exist");
        assert!(
            !gen_report.rolled_back,
            "generator should not roll back; struct names must not collide \
             with function parameter names: {}",
            output.source
        );
        io::validate_wgsl_text(&output.source).expect("output must be valid WGSL");
    }

    #[test]
    fn e2e_struct_name_does_not_collide_with_local_variables() {
        // Similar to the above, but the collision is with local variables
        // rather than function parameters.
        let src = r#"
            struct Result { x: f32, y: f32 }
            @group(0) @binding(0) var<uniform> input: Result;
            fn compute(val: f32) -> Result {
                var a: f32 = val;
                var b: f32 = val * 2.0;
                var c: f32 = a + b;
                var out: Result;
                out.x = c + input.x;
                out.y = a * input.y;
                return out;
            }
            @fragment fn main() -> @location(0) vec4f {
                let r = compute(1.0);
                return vec4f(r.x, r.y, 0.0, 1.0);
            }
        "#;
        let config = Config {
            profile: config::Profile::Max,
            mangle: Some(true),
            ..Default::default()
        };
        let output = run(src, &config).unwrap();
        let gen_report = output
            .report
            .pass_reports
            .iter()
            .find(|p| p.pass_name == "generator_emit")
            .expect("generator_emit pass must exist");
        assert!(
            !gen_report.rolled_back,
            "generator should not roll back; struct names must not collide \
             with local variable names: {}",
            output.source
        );
        io::validate_wgsl_text(&output.source).expect("output must be valid WGSL");
    }

    // MARK: Preamble tests

    #[test]
    fn preamble_declarations_excluded_from_output() {
        let preamble = "\
            struct Inputs { time: f32, size: vec2f, }\n\
            @group(0) @binding(0) var<uniform> inputs: Inputs;\
        ";
        let source = "\
            @fragment fn main() -> @location(0) vec4f {\
                return vec4f(inputs.time, inputs.size, 1.0);\
            }\
        ";
        let config = Config {
            preamble: Some(preamble.to_string()),
            ..Default::default()
        };
        let output = run(source, &config).unwrap();
        assert!(
            !output.source.contains("Inputs"),
            "preamble struct should not appear in output: {}",
            output.source
        );
        assert!(
            output.source.contains("main"),
            "entry point must still appear in output: {}",
            output.source
        );
        // Output alone is incomplete; validate with preamble re-prepended.
        let (emit_dirs, emit_body) = split_directives(&output.source);
        let (pre_dirs, pre_body) = split_directives(preamble);
        let combined = join_with_newline(&[emit_dirs, pre_dirs, pre_body, emit_body]);
        io::validate_wgsl_text(&combined).expect("output + preamble must be valid WGSL");
    }

    #[test]
    fn preamble_names_preserved_from_renaming() {
        let preamble = "\
            struct Inputs { time: f32, size: vec2f, }\n\
            @group(0) @binding(0) var<uniform> inputs: Inputs;\
        ";
        let source = "\
            @fragment fn main() -> @location(0) vec4f {\
                return vec4f(inputs.time, inputs.size, 1.0);\
            }\
        ";
        let config = Config {
            preamble: Some(preamble.to_string()),
            mangle: Some(true),
            ..Default::default()
        };
        let output = run(source, &config).unwrap();
        // The preamble member names must survive mangling so that access
        // expressions (inputs.time, inputs.size) remain correct.
        assert!(
            output.source.contains("time"),
            "preamble member 'time' must survive mangling: {}",
            output.source
        );
        assert!(
            output.source.contains("size"),
            "preamble member 'size' must survive mangling: {}",
            output.source
        );
    }

    #[test]
    fn empty_preamble_treated_as_none() {
        let source = TRIVIAL_SHADER;
        let config_empty = Config {
            preamble: Some(String::new()),
            ..Default::default()
        };
        let config_none = Config::default();
        let out_empty = run(source, &config_empty).unwrap();
        let out_none = run(source, &config_none).unwrap();
        assert_eq!(
            out_empty.source, out_none.source,
            "empty preamble should produce same output as no preamble"
        );
    }

    #[test]
    fn preamble_report_input_bytes_excludes_preamble() {
        let preamble = "struct Inputs { time: f32, }";
        let source = "@fragment fn main() -> @location(0) vec4f { return vec4f(1.0); }";
        let config = Config {
            preamble: Some(preamble.to_string()),
            ..Default::default()
        };
        let output = run(source, &config).unwrap();
        assert_eq!(
            output.report.input_bytes,
            source.len(),
            "input_bytes should reflect user source, not preamble"
        );
    }

    // MARK: Error diagnostic tests

    #[test]
    fn parse_error_contains_source_annotation() {
        let bad = "@vertex fn bad() -> vec4<f32> { return bad_func(); }";
        let config = Config::default();
        let err = match run(bad, &config) {
            Err(e) => e,
            Ok(_) => panic!("expected parse error"),
        };
        let msg = err.to_string();
        assert_eq!(err.kind(), "parse");
        // Must contain the annotated source line.
        assert!(
            msg.contains("bad_func"),
            "parse error should reference the problematic identifier: {msg}"
        );
        // Must contain line/column info from codespan.
        assert!(
            msg.contains("wgsl:"),
            "parse error should have source location: {msg}"
        );
    }

    #[test]
    fn preamble_parse_error_uses_preamble_label() {
        let bad_preamble = "struct Bad { x: nonexistent_type }";
        let source = "@vertex fn main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0,0.0,0.0,1.0); }";
        let config = Config {
            preamble: Some(bad_preamble.to_string()),
            ..Default::default()
        };
        let err = match run(source, &config) {
            Err(e) => e,
            Ok(_) => panic!("expected preamble parse error"),
        };
        let msg = err.to_string();
        assert!(
            msg.contains("<preamble>"),
            "preamble parse error should identify <preamble> as the source: {msg}"
        );
    }

    #[test]
    fn error_kind_and_message_accessors() {
        let bad = "fn oops { }";
        let config = Config::default();
        let err = match run(bad, &config) {
            Err(e) => e,
            Ok(_) => panic!("expected parse error"),
        };
        assert_eq!(err.kind(), "parse");
        // message() should contain the codespan diagnostic.
        assert!(!err.message().is_empty(), "error message must not be empty");
    }

    #[test]
    fn atomic_compare_exchange_members_do_not_trigger_generator_rollback() {
        let src = r#"
            @group(0) @binding(0)
            var<storage, read_write> val: atomic<u32>;

            @compute @workgroup_size(1)
            fn main() {
                let result = atomicCompareExchangeWeak(&val, 0u, 1u);
                let old = result.old_value;
                let exchanged = result.exchanged;
                _ = old;
                _ = exchanged;
            }
        "#;

        let config = Config {
            profile: config::Profile::Max,
            mangle: Some(true),
            ..Default::default()
        };
        let output = run(src, &config).expect("run should succeed");

        let gen_report = output
            .report
            .pass_reports
            .iter()
            .find(|p| p.pass_name == "generator_emit")
            .expect("generator_emit pass must exist");
        assert!(
            !gen_report.rolled_back,
            "generator should not roll back for atomic compare-exchange member access: {}",
            output.source
        );
        io::validate_wgsl_text(&output.source).expect("output must be valid WGSL");
    }

    #[test]
    fn run_strips_wgpu_binding_array_enable_directive() {
        let src = r#"
            enable wgpu_binding_array;
            @group(0) @binding(0)
            var arr: binding_array<texture_2d<f32>>;

            @fragment
            fn main() -> @location(0) vec4<f32> {
                return textureLoad(arr[0], vec2<i32>(0, 0), 0);
            }
        "#;
        let output = run(src, &Config::default()).expect("run should succeed");
        assert!(
            !output.source.contains("enable wgpu_binding_array;"),
            "legacy enable directive should be removed in normalized input"
        );
    }

    /// When the source declares an extension naga cannot parse,
    /// `run` returns the original input verbatim plus a synthetic
    /// `unsupported_extension_bailout` PassReport so downstream
    /// tooling can distinguish "ran the pipeline" from "bailed out".
    /// `subgroups` is currently a parse-time error in naga 29
    /// (matched by `UNSUPPORTED_EXTENSION_PATTERNS`); a future naga
    /// release that lands subgroup support will need a different
    /// trigger here.
    #[test]
    fn unsupported_extension_bailout_includes_synthetic_pass_report() {
        let src = "enable subgroups;\n\
                   @compute @workgroup_size(1) fn m() {}";
        let output = run(src, &Config::default()).expect("bailout returns Ok with original source");
        // The bailout path returns the ORIGINAL source verbatim
        // (no pipeline run, no preprocessing pass).
        assert_eq!(
            output.source, src,
            "bailout must return the original source unchanged"
        );
        let bailout = output
            .report
            .pass_reports
            .iter()
            .find(|p| p.pass_name == "unsupported_extension_bailout");
        assert!(
            bailout.is_some(),
            "synthetic bailout pass report must be present so callers can detect the short-circuit"
        );
        let b = bailout.unwrap();
        assert!(!b.changed);
        assert!(!b.rolled_back);
        assert_eq!(b.before_bytes, Some(src.len()));
        assert_eq!(b.after_bytes, Some(src.len()));
    }

    #[test]
    fn run_auto_enables_f16_when_used() {
        let src = r#"
            fn id(x: f16) -> f16 { return x; }
        "#;
        let output = run(src, &Config::default()).expect("run should succeed with auto f16 enable");
        assert!(
            !output.source.is_empty(),
            "output should be emitted after auto f16 enable"
        );
    }

    #[test]
    fn split_directives_extracts_leading_enables() {
        let src = "enable f16;\nenable subgroups;\nstruct S { x: f32 }\nfn f() {}\n";
        let (dirs, rest) = split_directives(src);
        assert_eq!(dirs, "enable f16;\nenable subgroups;\n");
        assert_eq!(rest, "struct S { x: f32 }\nfn f() {}\n");
    }

    #[test]
    fn split_directives_handles_comments_and_blanks() {
        let src = "// header\n\nenable f16;\n\nstruct S { x: f32 }\n";
        let (dirs, rest) = split_directives(src);
        assert_eq!(dirs, "// header\n\nenable f16;\n\n");
        assert_eq!(rest, "struct S { x: f32 }\n");
    }

    #[test]
    fn split_directives_terminator_scan_ignores_semicolon_in_block_comment() {
        // A `;` inside a comment BETWEEN the directive keyword and its real
        // terminator must not split the directive mid-comment (which would
        // splice a preamble into the broken comment region).
        let src = "diagnostic /* a;b */ (off, derivative_uniformity);\n\
                   @fragment fn main() -> @location(0) vec4f { return vec4f(0.); }\n";
        let (dirs, rest) = split_directives(src);
        assert_eq!(dirs, "diagnostic /* a;b */ (off, derivative_uniformity);\n");
        assert!(
            rest.starts_with("@fragment"),
            "body must start at the real declaration, not inside the comment: {rest:?}"
        );
    }

    #[test]
    fn split_directives_terminator_scan_ignores_semicolon_in_line_comment() {
        let src = "enable f16; // trailing ; comment\nstruct S { x: f16 }\n";
        let (dirs, rest) = split_directives(src);
        // The directive's own `;` terminates it; the line comment is body
        // trivia.  (The point is the line-comment `;` is not mistaken for a
        // second directive terminator.)
        assert!(dirs.starts_with("enable f16;"));
        assert!(rest.contains("struct S"));
    }

    #[test]
    fn split_directives_no_directives() {
        let src = "struct S { x: f32 }\nfn f() {}\n";
        let (dirs, rest) = split_directives(src);
        assert_eq!(dirs, "");
        assert_eq!(rest, src);
    }

    #[test]
    fn split_directives_diagnostic() {
        let src = "diagnostic(off, derivative_uniformity);\nfn f() {}\n";
        let (dirs, rest) = split_directives(src);
        assert_eq!(dirs, "diagnostic(off, derivative_uniformity);\n");
        assert_eq!(rest, "fn f() {}\n");
    }

    /// Regression: when `split_directives` returns a fragment that
    /// lacks a trailing newline (e.g. a source whose last directive
    /// is the final byte), splicing it directly in front of the
    /// preamble's directives glued them together into one syntax
    /// error.  `join_with_newline` must insert a separator.
    #[test]
    fn join_with_newline_inserts_separator_between_fragments() {
        let joined = join_with_newline(&["enable f16;", "enable subgroups;\n", "fn body() {}\n"]);
        assert_eq!(joined, "enable f16;\nenable subgroups;\nfn body() {}\n");
    }

    #[test]
    fn join_with_newline_skips_empty_fragments() {
        let joined = join_with_newline(&["enable f16;\n", "", "fn body() {}\n"]);
        assert_eq!(joined, "enable f16;\nfn body() {}\n");
    }

    #[test]
    fn join_with_newline_keeps_existing_trailing_newline() {
        let joined = join_with_newline(&["enable f16;\n", "fn body() {}\n"]);
        assert_eq!(joined, "enable f16;\nfn body() {}\n");
    }

    /// CRLF fragments survive `split_directives` (verified by
    /// `split_directives_crlf_line_endings`), so the join helper must
    /// preserve them unchanged.  `ends_with('\n')` matches the LF in
    /// `\r\n`, so no extra newline is appended after a CRLF-ending
    /// fragment.
    #[test]
    fn join_with_newline_preserves_crlf_fragments() {
        let joined = join_with_newline(&["enable f16;\r\n", "fn body() {}\r\n"]);
        assert_eq!(joined, "enable f16;\r\nfn body() {}\r\n");
    }

    #[test]
    fn split_directives_requires() {
        let src = "requires readonly_and_readwrite_storage_textures;\nfn f() {}\n";
        let (dirs, rest) = split_directives(src);
        assert_eq!(dirs, "requires readonly_and_readwrite_storage_textures;\n");
        assert_eq!(rest, "fn f() {}\n");
    }

    #[test]
    fn split_directives_crlf_line_endings() {
        let src = "enable f16;\r\nstruct S { x: f32 }\r\n";
        let (dirs, rest) = split_directives(src);
        assert_eq!(dirs, "enable f16;\r\n");
        assert_eq!(rest, "struct S { x: f32 }\r\n");
    }

    #[test]
    fn split_directives_no_trailing_newline() {
        let src = "enable f16;";
        let (dirs, rest) = split_directives(src);
        assert_eq!(dirs, "enable f16;");
        assert_eq!(rest, "");
    }

    #[test]
    fn split_directives_all_directives() {
        let src = "enable f16;\nrequires something;\n";
        let (dirs, rest) = split_directives(src);
        assert_eq!(dirs, src);
        assert_eq!(rest, "");
    }

    #[test]
    fn split_directives_empty_source() {
        let (dirs, rest) = split_directives("");
        assert_eq!(dirs, "");
        assert_eq!(rest, "");
    }

    #[test]
    fn split_directives_compact_single_line() {
        // Compact generator output is one physical line; the splitter must
        // still stop right after the directive's `;`, not swallow the body.
        let src = "enable f16;@group(0)@binding(0)var<storage,read_write>A:f16;\
                   @compute @workgroup_size(1) fn m(){A=1h;}";
        let (dirs, body) = split_directives(src);
        assert_eq!(dirs, "enable f16;");
        assert_eq!(
            body,
            "@group(0)@binding(0)var<storage,read_write>A:f16;\
             @compute @workgroup_size(1) fn m(){A=1h;}"
        );
        // Multiple directives run together on one line are all consumed.
        let (dirs, body) =
            split_directives("enable f16;enable dual_source_blending;@fragment fn m(){}");
        assert_eq!(dirs, "enable f16;enable dual_source_blending;");
        assert_eq!(body, "@fragment fn m(){}");
    }

    #[test]
    fn split_directives_word_boundary_single_line() {
        // Identifiers that merely start with a directive keyword must NOT be
        // hoisted, even when the whole input is one line.
        assert_eq!(
            split_directives("enablef16;fn m(){}"),
            ("", "enablef16;fn m(){}")
        );
        assert_eq!(
            split_directives("requires_foo();fn m(){}"),
            ("", "requires_foo();fn m(){}")
        );
        // A real directive followed mid-line by a `diagnostic`-prefixed
        // identifier stops at that identifier.
        assert_eq!(
            split_directives("enable f16;diagnostic_counter_thing fn m(){}"),
            ("enable f16;", "diagnostic_counter_thing fn m(){}")
        );
    }

    #[test]
    fn references_f16_token_ignores_identifiers() {
        // `myf16var` and `f16_test` must NOT be detected as an f16 token.
        assert!(!references_f16_token("var myf16var: i32;"));
        assert!(!references_f16_token("fn f16_test() {}"));
        assert!(!references_f16_token("let x = ff16;"));
    }

    #[test]
    fn references_f16_token_detects_real_use() {
        assert!(references_f16_token("var x: f16 = 1.0h;"));
        assert!(references_f16_token("let v = vec3<f16>(0h);"));
        assert!(references_f16_token("fn f() -> f16 { return 0h; }"));
    }

    #[test]
    fn references_f16_token_detects_aliases_and_suffix() {
        // Predeclared half-precision aliases (no `f16` substring).
        assert!(references_f16_token("var v: vec2h = vec2h(1.0h, 2.0h);"));
        assert!(references_f16_token("let v = vec3h(0h);"));
        assert!(references_f16_token("var m: mat4x4h;"));
        assert!(references_f16_token("let m = mat2x3h();"));
        // `h` float-literal suffix in its various spellings, no alias/keyword.
        assert!(references_f16_token("let x = 1.0h + 2.0h;"));
        assert!(references_f16_token("let x = 0h;"));
        assert!(references_f16_token("let x = 1.5e2h;"));
        assert!(references_f16_token("let x = 1.0e-3h;"));
        assert!(references_f16_token("let x = 0x1p2h;"));
        // Negatives: longer identifiers, other float suffixes, plain ints.
        assert!(!references_f16_token("var width: f32; var height: f32;"));
        assert!(!references_f16_token("let mesh = 1.0;"));
        assert!(!references_f16_token("var vec2hh: i32;"));
        assert!(!references_f16_token("let x = 1.0f + 2u + 3;"));
        assert!(!references_f16_token("let x = 1.0e-3;"));
        assert!(!references_f16_token("fn vec2h_helper() {}"));
    }

    #[test]
    fn references_f16_token_ignores_comments() {
        // f16 mentioned only inside comments should not trigger injection.
        assert!(!references_f16_token("// uses f16 later\nvar x: i32;"));
        assert!(!references_f16_token("/* f16 */ var x: i32;"));
        assert!(!references_f16_token(
            "/* multiline\n   f16\n */\nvar x: i32;"
        ));
    }

    // Regression: WGSL (https://www.w3.org/TR/WGSL/#comments) permits
    // nested block comments.  A non-nesting comment scrubber would close
    // at the inner `*/` and expose the trailing `f16` content to the
    // token scan; the depth-tracking implementation must absorb the inner
    // pair and treat the whole region as commented.
    #[test]
    fn references_f16_token_ignores_nested_block_comments() {
        assert!(!references_f16_token(
            "/* outer /* inner */ f16 still in outer */ var x: i32;"
        ));
        assert!(!references_f16_token(
            "/* /* /* deeply nested */ */ f16 inside */ var x: i32;"
        ));
        // A real f16 after a properly-closed nested comment still
        // detects.
        assert!(references_f16_token(
            "/* nest /* inner */ done */ var x: f16 = 0h;"
        ));
    }

    // Regression: classic-Mac (lone `\r`) line endings must be
    // normalised before any `.lines()` scan, otherwise the
    // `wgpu_*` directive strip and `enable f16;` detection fold the
    // entire source into one line and silently fail.
    #[test]
    fn normalize_line_endings_handles_lone_cr() {
        // Lone `\r` becomes `\n`.
        assert_eq!(normalize_line_endings("a\rb\rc"), "a\nb\nc");
        // `\r\n` stays intact (str::lines already handles it).
        assert_eq!(normalize_line_endings("a\r\nb\r\nc"), "a\r\nb\r\nc");
        // Mixed.
        assert_eq!(normalize_line_endings("a\rb\r\nc\nd"), "a\nb\r\nc\nd");
        // Source with no `\r` returns identical content.
        assert_eq!(normalize_line_endings("a\nb\nc"), "a\nb\nc");
        // Multi-byte UTF-8 around line endings preserved.
        assert_eq!(normalize_line_endings("α\rβ\r\nγ"), "α\nβ\r\nγ");
    }

    #[test]
    fn preprocess_strips_wgpu_directive_with_cr_only_endings() {
        let src = "enable wgpu_binding_array;\r@fragment fn m() -> @location(0) vec4f { return vec4f(0); }";
        let out = preprocess_source_for_naga(src);
        assert!(
            !out.contains("enable wgpu_binding_array;"),
            "lone-CR-terminated wgpu_* directive must still be stripped: {out:?}"
        );
    }

    #[test]
    fn has_enable_f16_directive_matches_canonical() {
        assert!(has_enable_f16_directive("enable f16;\n"));
        assert!(has_enable_f16_directive("enable f16;"));
    }

    #[test]
    fn has_enable_f16_directive_matches_extra_whitespace() {
        assert!(has_enable_f16_directive("enable  f16;\n"));
        assert!(has_enable_f16_directive("enable\tf16;\n"));
        assert!(has_enable_f16_directive("enable f16 ;\n"));
    }

    #[test]
    fn has_enable_f16_directive_rejects_commented_out() {
        assert!(!has_enable_f16_directive("// enable f16;"));
        assert!(!has_enable_f16_directive("/* enable f16; */"));
    }

    #[test]
    fn has_enable_f16_directive_rejects_non_directive() {
        assert!(!has_enable_f16_directive("var enable_f16: bool;"));
        assert!(!has_enable_f16_directive("fn enable() {} // f16"));
    }

    #[test]
    fn preprocess_does_not_inject_for_identifier_only() {
        // Source references `f16` only as part of identifiers -> no injection.
        let src = "var myf16_var: i32 = 0;\n";
        let out = preprocess_source_for_naga(src);
        assert!(
            !out.contains("enable f16;"),
            "must not inject enable when `f16` appears only in identifiers: {out}"
        );
    }

    #[test]
    fn preprocess_injects_for_real_f16_use() {
        let src = "fn f() -> f16 { return 0h; }\n";
        let out = preprocess_source_for_naga(src);
        assert!(
            out.starts_with("enable f16;\n"),
            "must inject enable directive for real f16 use: {out}"
        );
    }

    #[test]
    fn preprocess_does_not_duplicate_enable_f16_with_extra_whitespace() {
        let src = "enable  f16;\nfn f() -> f16 { return 0h; }\n";
        let out = preprocess_source_for_naga(src);
        // The output must carry exactly the directive that was already
        // present in the source: one occurrence of `enable...f16;` plus
        // one occurrence of the `f16` return type in the function
        // signature.  The `enable` count locks against accidental
        // injection of a second normalised `enable f16;`.
        assert_eq!(
            out.matches("enable").count(),
            1,
            "must not inject a second `enable` directive when the source already enables f16: {out}"
        );
        assert!(
            !out.contains("enable f16;\nenable  f16;"),
            "must not inject a second enable f16 directive: {out}"
        );
    }

    // MARK: Naga error-message coupling tests

    // These tests fail if `UNSUPPORTED_EXTENSION_PATTERNS` or
    // `KNOWN_TEXT_VALIDATION_LIMITATION_PATTERNS` drift out of sync with
    // the naga error phrasings they target.
    #[test]
    fn unsupported_extension_patterns_match_documented_phrasings() {
        // These are the exact phrasings naga produces today.  All must
        // be recognised by `is_unsupported_extension_parse_error` or
        // the short-circuit return path in `run` falls over into a
        // hard error the moment naga rewords them.
        let samples = [
            "error: enable extension is not enabled",
            "error: the `wgpu_ray_query` enable-extension is not yet supported",
        ];
        for s in samples {
            let err = Error::Parse(s.to_string());
            assert!(
                is_unsupported_extension_parse_error(&err),
                "naga error phrasing should be recognized as unsupported-extension: {s}"
            );
        }
    }

    #[test]
    fn unsupported_extension_patterns_reject_unrelated_errors() {
        let err = Error::Parse("error: expected identifier, found `{`".into());
        assert!(
            !is_unsupported_extension_parse_error(&err),
            "unrelated parse errors must not be treated as unsupported-extension"
        );
    }

    /// Regression: a parse error whose codespan snippet quotes a user
    /// comment containing the unsupported-extension phrasing must NOT
    /// trigger the bailout.  Pre-fix, substring matching against the
    /// entire rendered message swallowed real failures whenever the
    /// user happened to comment-mention an unsupported extension.
    #[test]
    fn unsupported_extension_patterns_ignore_quoted_source_lines() {
        let rendered = "error: expected identifier, found `{`\n  \
                        ┌─ wgsl:5:1\n  │\n5 │ // TODO: enable extension is not enabled \
                        on our backend\n  │ ^^\n";
        let err = Error::Parse(rendered.into());
        assert!(
            !is_unsupported_extension_parse_error(&err),
            "pattern in a quoted source line must NOT trigger the bailout"
        );

        // Same shape for the subgroups text-validation limitation: a
        // shader quoting the phrase in a comment, surfaced as context
        // around an unrelated parse error, must not opt into the
        // validation-bypass.
        let rendered = "error: expected `;`\n  \
                        ┌─ wgsl:3:1\n  │\n3 │ // subgroups enable-extension is not yet supported\n";
        let err = Error::Parse(rendered.into());
        assert!(
            !is_known_text_validation_limitation(&err),
            "pattern in a quoted source line must NOT trigger the validation bypass"
        );
    }

    #[test]
    fn unsupported_extension_patterns_only_match_parse_errors() {
        // Regression: substring patterns must not match against the
        // rendered message of a non-`Parse` error variant, since a
        // validator/emit failure quoting the same phrasing (or even an
        // I/O error path containing the offending source) would
        // previously short-circuit `run` into the "return input
        // unchanged" branch and silently swallow a real failure.
        for ctor in [
            Error::Validation as fn(String) -> Error,
            Error::Emit as fn(String) -> Error,
            Error::Io as fn(String) -> Error,
            Error::Config as fn(String) -> Error,
        ] {
            let err = ctor("error: enable extension is not enabled".to_string());
            assert!(
                !is_unsupported_extension_parse_error(&err),
                "non-Parse error variants must not be treated as \
                 unsupported-extension even when the message matches: {err:?}"
            );
        }
    }

    #[test]
    fn known_text_validation_limitation_only_matches_parse_or_validation() {
        // Same defensive policy as above for the subgroups-limitation
        // matcher: only `Parse` and `Validation` may opt into the
        // round-trip-validation bypass; `Emit` and `Io` errors quoting
        // the same phrasing must be reported normally.
        for ctor in [
            Error::Emit as fn(String) -> Error,
            Error::Io as fn(String) -> Error,
            Error::Config as fn(String) -> Error,
        ] {
            let err = ctor("error: `subgroups` enable-extension is not yet supported".to_string());
            assert!(
                !is_known_text_validation_limitation(&err),
                "non-Parse/Validation error variants must not opt into the \
                 subgroup text-validation bypass: {err:?}"
            );
        }
    }

    #[test]
    fn known_text_validation_limitation_matches_subgroup_phrasings() {
        let samples = [
            "error: `subgroups` enable-extension is not yet supported",
            "error: subgroups enable-extension is not yet supported",
        ];
        for s in samples {
            let err = Error::Parse(s.to_string());
            assert!(
                is_known_text_validation_limitation(&err),
                "subgroup limitation phrasing should be recognized: {s}"
            );
        }
    }

    #[test]
    fn known_text_validation_limitation_matches_validation_variant_too() {
        // The matcher is intentionally scoped to `Parse | Validation`
        // because `io::validate_wgsl_text` internally calls both
        // `parse_wgsl` (Parse errors) and `validate_module_with_source`
        // (Validation errors); naga can report a not-yet-supported
        // `enable` directive through either path.  This regression pins
        // the Validation branch so a future tightening to "Parse only"
        // fails loudly here.
        let samples = [
            "error: `subgroups` enable-extension is not yet supported",
            "error: subgroups enable-extension is not yet supported",
        ];
        for s in samples {
            let err = Error::Validation(s.to_string());
            assert!(
                is_known_text_validation_limitation(&err),
                "subgroup limitation phrasing must also be recognized when wrapped as Validation: {s}"
            );
        }
    }

    #[test]
    fn known_text_validation_limitation_rejects_unrelated_errors() {
        let err = Error::Validation("error: mismatched types".into());
        assert!(!is_known_text_validation_limitation(&err));
    }

    #[test]
    fn preamble_with_enable_f16_shader() {
        // Regression for F1: a preamble that carries its OWN `enable f16;`
        // directive, combined with a source that genuinely USES f16, minified
        // in COMPACT mode (where the whole module is one physical line).  A
        // line-based directive splitter misclassified that single line as one
        // big directive and placed the preamble's `enable` after the emitted
        // declarations -> "expected global declaration, but found a global
        // directive" (a hard Error::Emit, since the naga fallback is unsafe
        // with a preamble).  The `;`-aware splitter keeps every directive
        // ahead of every declaration.
        //
        // (The old form of this test passed even on buggy code: its `enable
        // f16;` was dropped as unused and its preamble carried no directive,
        // so the mis-splice was never exercised.)
        let preamble = "\
            enable f16;\n\
            @group(0) @binding(0) var<uniform> bias: f16;\
        ";
        let source = "\
            @group(0) @binding(1) var<storage, read_write> sink: f16;\n\
            @compute @workgroup_size(1) fn main() {\n\
                var h: f16 = 1.0h;\n\
                h = h + bias;\n\
                sink = h;\n\
            }\
        ";
        let config = Config {
            preamble: Some(preamble.to_string()),
            ..Default::default()
        };
        // Default config => compact mode.
        let output = run(source, &config)
            .expect("enable f16; in preamble + f16 source must minify in compact mode");
        assert!(
            output.source.contains("enable f16;"),
            "the surviving f16 use must keep the enable directive: {}",
            output.source
        );
    }

    // MARK: Splat elision tests

    /// Minify with the `Max` profile (mangle + inline) and assert the
    /// result is valid WGSL.  Returns the minified source for the
    /// caller's own assertions.
    fn minify_and_validate(src: &str) -> String {
        let config = Config {
            profile: config::Profile::Max,
            mangle: Some(true),
            ..Default::default()
        };
        let output = run(src, &config).unwrap();
        io::validate_wgsl_text(&output.source)
            .unwrap_or_else(|e| panic!("output is invalid WGSL: {e}\n{}", output.source));
        output.source
    }

    #[test]
    fn run_never_ships_invalid_wgsl_when_fallback_is_also_invalid() {
        // Regression for the rollback re-validation guard: when BOTH the
        // custom generator AND naga's own wgsl-out fallback emit text that
        // naga's frontend rejects, run() must surface a diagnosable error
        // rather than silently shipping output that does not re-parse.
        // Invariant under test: run() either errors, or returns output that
        // round-trips.  (Robust to future naga changes - if a naga release
        // accepts these forms, the Ok branch simply validates clean.)
        let inputs = [
            // f64 literal narrowed to f32: naga const-substitutes the literal
            // under the cast, then rejects `f32(<F64 literal>)`.
            "@group(0) @binding(0) var<storage, read_write> s: f32;\n\
             @compute @workgroup_size(1) fn m() { let a: f64 = 0.5lf; s = f32(a); }",
            // f16 literal in a cast whose `enable f16;` naga's backend drops.
            "enable f16;\n\
             @fragment fn m() -> @location(0) vec4f { let h: f16 = 1.0h; return vec4f(f32(h)); }",
        ];
        for src in inputs {
            if let Ok(output) = run(src, &Config::default()) {
                io::validate_wgsl_text(&output.source).unwrap_or_else(|e| {
                    panic!(
                        "run() shipped invalid WGSL (it should have errored): {e}\n{}",
                        output.source
                    )
                });
            }
        }
    }

    #[test]
    fn folds_f64_literal_narrowing_cast_to_valid_literal() {
        // Regression for issue A: naga const-substitutes `let a: f64 = 2.5lf`
        // into the cast, yielding `As { Literal(F64), convert }`, which the
        // emitter rendered as `f32(2.5lf)` - a token naga rejects on re-parse.
        // const_fold now folds the narrowing cast of an F64 literal to the
        // converted scalar literal, so the output round-trips (and shrinks).
        for (decl, stmt) in [
            ("var<storage, read_write> s: f32;", "s = f32(a);"),
            ("var<storage, read_write> s: i32;", "s = i32(a);"),
            ("var<storage, read_write> s: u32;", "s = u32(a);"),
        ] {
            let src = format!(
                "@group(0) @binding(0) {decl}\n\
                 @compute @workgroup_size(1) fn m() {{ let a: f64 = 2.5lf; {stmt} }}"
            );
            let output = run(&src, &Config::default())
                .unwrap_or_else(|e| panic!("f64 narrowing cast must minify, got error: {e}"));
            io::validate_wgsl_text(&output.source)
                .unwrap_or_else(|e| panic!("output is invalid WGSL: {e}\n{}", output.source));
            // The `T(<F64 literal>)` cast must be folded away, not emitted
            // (no `lf)` token - the f64 literal no longer appears in a cast).
            assert!(
                !output.source.contains("lf)"),
                "f64 cast literal should be folded, not emitted as a cast: {}",
                output.source
            );
        }
    }

    #[test]
    fn folds_f64_vector_narrowing_cast_to_valid_constructor() {
        // A const f64 VECTOR narrowing cast (`vec2<f32>(vec2<f64>(.5lf,1.5lf))`)
        // is rejected by naga on re-parse, and const_fold can't materialize the
        // converted vector (the converted F32 component literals don't exist as
        // arena handles).  The generator folds it to a converted constructor.
        for (store_ty, decl_a, cast) in [
            ("vec2<f32>", "vec2<f64>(0.5lf, 1.5lf)", "vec2<f32>(a)"),
            (
                "vec3<f32>",
                "vec3<f64>(0.5lf, 1.5lf, 2.5lf)",
                "vec3<f32>(a)",
            ),
            ("vec2<i32>", "vec2<f64>(2.5lf, 3.5lf)", "vec2<i32>(a)"),
        ] {
            let src = format!(
                "@group(0) @binding(0) var<storage, read_write> o: {store_ty};\n\
                 @compute @workgroup_size(1) fn m() {{ let a = {decl_a}; o = {cast}; }}"
            );
            let output = run(&src, &Config::default()).unwrap_or_else(|e| {
                panic!("f64 vector narrowing cast must minify, got error: {e}\nsrc:{src}")
            });
            io::validate_wgsl_text(&output.source)
                .unwrap_or_else(|e| panic!("output is invalid WGSL: {e}\n{}", output.source));
            // The `vecN<f32>(vecN<f64>(..lf..))` cast must be folded away.
            assert!(
                !output.source.contains("lf)"),
                "f64 vector cast should be folded, not emitted: {}",
                output.source
            );
        }
    }

    #[test]
    fn folds_int64_narrowing_cast_to_valid_literal() {
        // The u64/i64 (`lu`/`li`) narrowing cast is the exact analogue of the
        // f64 cast: naga rejects re-parsing `u32(<U64 literal>)`, and naga's own
        // backend emits the same token, so it used to hard-error.  const_fold
        // (scalar) and the generator (vector) now fold it, WRAPPING on
        // narrowing - `u32(i64(-1))` is `4294967295`, NOT a clamped `0` -
        // matching naga's value-conversion semantics.
        let cases = [
            // (decl, cast, expected substring in the folded output)
            ("let a: u64 = 107lu;", "o = u32(a);", "107"),
            ("let a: u64 = 4294967296lu;", "o = u32(a);", "0"), // 2^32 wraps to 0
            ("let a: i64 = -1li;", "o = u32(a);", "4294967295"), // wrap, not clamp
            (
                "let a = vec2<u64>(107lu, 4294967296lu);",
                "o = vec2<u32>(a);",
                "vec2u(107,0)",
            ),
        ];
        for (decl, cast, expect) in cases {
            let store_ty = if cast.contains("vec2") {
                "vec2<u32>"
            } else {
                "u32"
            };
            let src = format!(
                "@group(0) @binding(0) var<storage, read_write> o: {store_ty};\n\
                 @compute @workgroup_size(1) fn m() {{ {decl} {cast} }}"
            );
            let output = run(&src, &Config::default()).unwrap_or_else(|e| {
                panic!("int64 narrowing cast must minify, got error: {e}\nsrc:{src}")
            });
            io::validate_wgsl_text(&output.source)
                .unwrap_or_else(|e| panic!("output is invalid WGSL: {e}\n{}", output.source));
            // The `T(<width-8 literal>)` cast must be folded (no `lu)`/`li)`).
            assert!(
                !output.source.contains("lu)") && !output.source.contains("li)"),
                "int64 cast should be folded, not emitted: {}",
                output.source
            );
            assert!(
                output.source.contains(expect),
                "expected wrapped value {expect:?} in output: {}",
                output.source
            );
        }
    }

    #[test]
    fn splat_elision_add_vec3f() {
        // vec3f(1) + vec3f_var  ->  bare `1` via scalar-vector broadcasting
        let src = r#"
            @fragment fn main() -> @location(0) vec4f {
                var c = vec3f(0.5, 0.6, 0.7);
                c = vec3f(1.0) + c;
                return vec4f(c, 1.0);
            }
        "#;
        let out = minify_and_validate(src);
        // The splat `vec3f(1)` (or its alias) should NOT appear;
        // instead the bare scalar should be used in the addition.
        assert!(
            !out.contains("vec3f(1)") && !out.contains("vec3<f32>(1"),
            "splat should be elided in addition: {out}"
        );
    }

    #[test]
    fn splat_elision_subtract_vec2f() {
        let src = r#"
            @fragment fn main() -> @location(0) vec4f {
                var uv = vec2f(0.3, 0.7);
                uv = uv - vec2f(0.5);
                return vec4f(uv, 0.0, 1.0);
            }
        "#;
        let out = minify_and_validate(src);
        assert!(
            !out.contains("vec2f(.5)") && !out.contains("vec2<f32>(.5"),
            "splat should be elided in subtraction: {out}"
        );
    }

    #[test]
    fn splat_elision_multiply_vec4f() {
        let src = r#"
            @fragment fn main() -> @location(0) vec4f {
                var c = vec4f(0.1, 0.2, 0.3, 0.4);
                c = vec4f(2.0) * c;
                return c;
            }
        "#;
        let out = minify_and_validate(src);
        assert!(
            !out.contains("vec4f(2") && !out.contains("vec4<f32>(2"),
            "splat should be elided in multiplication: {out}"
        );
    }

    #[test]
    fn splat_elision_divide_by_splat() {
        // vector / splat  ->  vector / scalar
        let src = r#"
            fn helper(v: vec3f) -> vec3f {
                return v / vec3f(dot(v, v));
            }
            @fragment fn main() -> @location(0) vec4f {
                return vec4f(helper(vec3f(1.0, 2.0, 3.0)), 1.0);
            }
        "#;
        let out = minify_and_validate(src);
        // The vec3f(dot(...)) should be elided to just dot(...)
        assert!(
            !out.contains("vec3f(dot") && !out.contains("vec3<f32>(dot"),
            "splat wrapping dot() should be elided in division: {out}"
        );
    }

    #[test]
    fn splat_elision_compound_assign() {
        // v -= vec2f(0.5)  ->  v -= .5
        let src = r#"
            @fragment fn main() -> @location(0) vec4f {
                var uv = vec2f(1.0, 1.0);
                uv -= vec2f(0.5);
                return vec4f(uv, 0.0, 1.0);
            }
        "#;
        let out = minify_and_validate(src);
        assert!(
            !out.contains("vec2f(.5)") && !out.contains("vec2<f32>(.5"),
            "splat should be elided in compound assignment: {out}"
        );
    }

    #[test]
    fn splat_elision_no_double_elide() {
        // vec3f(a) + vec3f(b): at most one side should be elided so the
        // result stays a vector (not scalar + scalar = scalar).
        let src = r#"
            @fragment fn main() -> @location(0) vec4f {
                let a = 1.0;
                let b = 2.0;
                let c = vec3f(a) + vec3f(b);
                return vec4f(c, 1.0);
            }
        "#;
        let out = minify_and_validate(src);
        // Output must be valid WGSL - validation above ensures the type is correct.
        assert!(!out.is_empty());
    }

    #[test]
    fn splat_elision_skipped_when_other_is_scalar() {
        // vec4f(1.0) * scalar must NOT elide the Splat, because
        // 1.0 * scalar = scalar, not vec4f.
        let src = r#"
            struct S { b: f32 }
            @group(0) @binding(0) var<uniform> u: S;
            @vertex fn main() -> @builtin(position) vec4f {
                return vec4f(1.0) * u.b;
            }
        "#;
        let out = minify_and_validate(src);
        // The output must still contain vec4f - the splat can't be elided.
        assert!(
            out.contains("vec4"),
            "splat must not be elided when other operand is scalar: {out}"
        );
    }

    #[test]
    fn splat_elision_skipped_when_other_is_scalar_rhs() {
        // scalar * vec4f(1.0) must NOT elide the Splat, because
        // scalar * 1.0 = scalar, not vec4f.  (Reverse of the LHS test.)
        let src = r#"
            struct S { b: f32 }
            @group(0) @binding(0) var<uniform> u: S;
            @vertex fn main() -> @builtin(position) vec4f {
                return u.b * vec4f(1.0);
            }
        "#;
        let out = minify_and_validate(src);
        assert!(
            out.contains("vec4"),
            "splat must not be elided when other operand is scalar (RHS): {out}"
        );
    }

    #[test]
    fn splat_elision_non_arithmetic_unchanged() {
        // Comparison operators should NOT elide splats.
        // vec3f(0) == vec3f_var is NOT valid as 0 == vec3f_var.
        let src = r#"
            @fragment fn main() -> @location(0) vec4f {
                let v = vec3f(1.0, 2.0, 3.0);
                let mask = v > vec3f(1.5);
                return select(vec4f(0), vec4f(1), mask.x);
            }
        "#;
        // Just verify it's valid - comparison ops shouldn't trigger elision.
        minify_and_validate(src);
    }

    // MARK: Last-store inlining tests

    #[test]
    fn last_store_inlined_when_earlier_loads_keep_var_alive() {
        // Pattern: var m is loaded before AND after the last store.
        // The pre-store load keeps m alive (non-dead).  The post-store
        // load should still be inlined to the stored expression, and
        // the store itself should be removed.
        let src = r#"
            fn helper(v: vec3<f32>) -> vec3<f32> { return v; }
            @fragment fn main() -> @location(0) vec4f {
                var m = vec3f(0.0);
                let pre = m;           // load before store - keeps m alive
                m = helper(pre);       // store complex expr
                let post = m;          // load after store - should be inlined
                return vec4f(post, 1.0);
            }
        "#;
        let out = minify_and_validate(src);
        // The variable `m` should still exist (pre-store load keeps it alive),
        // but the last store's value should be inlined into the return.
        // Check that the output doesn't contain a redundant store+load pattern.
        // The output should be shorter than without inlining.
        assert!(!out.is_empty(), "output should not be empty");
    }

    #[test]
    fn last_store_not_inlined_when_escaped() {
        // When a variable's pointer escapes via a function call,
        // its Stores must be preserved - the callee may read through
        // the pointer at any time.
        let src = r#"
            fn consume(p: ptr<function, vec3f>) -> vec3f { return *p; }
            @fragment fn main() -> @location(0) vec4f {
                var m = vec3f(0.0);
                let pre = m;
                m = pre + vec3f(1.0);
                let post = consume(&m);
                return vec4f(post, 1.0);
            }
        "#;
        let out = minify_and_validate(src);
        assert!(!out.is_empty(), "output should not be empty");
    }

    #[test]
    fn last_store_not_inlined_with_partial_stores() {
        // When a variable has partial stores (field access), the whole-variable
        // store must be preserved because the partial store reads the full value.
        let src = r#"
            @fragment fn main() -> @location(0) vec4f {
                var v = vec3f(1.0, 2.0, 3.0);
                let pre = v;
                v = pre + vec3f(1.0);
                v.x = 0.0;             // partial store - depends on full v
                let post = v;
                return vec4f(post, 1.0);
            }
        "#;
        let out = minify_and_validate(src);
        assert!(!out.is_empty(), "output should not be empty");
    }

    #[test]
    fn last_store_preserves_other_stores_to_same_var() {
        // Regression: last-store inlining must only remove the specific
        // Store that was proven dead, not ALL stores to the same variable.
        // Here, the conditional store inside `if` must be preserved because
        // the final `log(1+m)` reads `m` which depends on it.
        let src = r#"
            fn heavy(a: vec3f, b: vec3f) -> vec3f { return a + b; }
            struct U { v: f32 }
            @group(0) @binding(0) var<uniform> u: U;
            @fragment fn main() -> @location(0) vec4f {
                var m = vec3f(0.0);
                let ray = vec3f(1.0, 2.0, 3.0);
                if u.v >= 0.0 {
                    m = heavy(ray, vec3f(0.5));
                }
                m = 0.5 * log(1.0 + m);
                return vec4f(m, 1.0);
            }
        "#;
        let out = minify_and_validate(src);
        // The conditional store `m = heavy(...)` must be preserved.
        // Without it, m stays vec3f(0) and the result is always black.
        assert!(
            out.contains("if"),
            "conditional branch must be preserved (store to m is live): {out}"
        );
    }

    #[test]
    fn last_store_init_preserved_when_loop_reads_var() {
        // Regression: when a variable's init Store has a seeded load
        // that gets forwarded, but the variable is also read inside a
        // subsequent loop body (where cache is cleared), the init Store
        // must NOT be removed.
        let src = r#"
            fn transform(v: vec3f) -> vec3f { return abs(v) - vec3f(0.7); }
            @fragment fn main() -> @location(0) vec4f {
                var p = vec3f(1.0, 2.0, 3.0);
                let ip = p;  // seeded load from init Store, gets forwarded
                for (var i = 0u; i < 4u; i++) {
                    p = transform(p);  // loop body reads p (needs init on 1st iter)
                }
                return vec4f(p + ip, 1.0);
            }
        "#;
        let out = minify_and_validate(src);
        // The init `p = vec3f(1,2,3)` must be preserved.  Without it, the
        // loop reads p=vec3f(0) on the first iteration, producing wrong results.
        // Verify the output produces valid WGSL (validation above) and that
        // the transform call appears (loop body is not dead).
        assert!(
            out.contains("abs"),
            "loop body with transform must be preserved: {out}"
        );
    }

    #[test]
    fn last_store_in_loop_not_removed() {
        // Regression: a Store inside a loop body must NOT be removed by
        // last-store inlining, even if all seeded loads are forwarded.
        // On the next iteration, loads BEFORE the Store in the loop body
        // observe the Store's value via the loop back-edge.
        let src = r#"
            fn complexSquare(a: vec2f) -> vec2f {
                return vec2f(a.x * a.x - a.y * a.y, 2.0 * a.x * a.y);
            }
            @fragment fn main() -> @location(0) vec4f {
                var p = vec3f(1.0, 2.0, 3.0);
                for (var i = 0u; i < 10u; i++) {
                    p = 0.7 * abs(p) / dot(p, p) - vec3f(0.7);
                    p = vec3f(p.x, complexSquare(p.yz)).zxy;
                }
                return vec4f(p, 1.0);
            }
        "#;
        let out = minify_and_validate(src);
        // Both stores to p in the loop must be preserved.
        // The .zxy swizzle is the telltale of the second store.
        assert!(
            out.contains(".zxy"),
            "second store in loop (with .zxy swizzle) must be preserved: {out}"
        );
    }

    // Regression: `split_directives` must word-boundary-match the
    // `diagnostic` keyword.  A bare `starts_with("diagnostic")` would
    // misclassify user identifiers such as `diagnostic_counter`.
    #[test]
    fn split_directives_recognizes_paren_diagnostic() {
        let source = "diagnostic(off, derivative_uniformity);\nfn main(){}\n";
        let (dirs, body) = split_directives(source);
        assert_eq!(dirs, "diagnostic(off, derivative_uniformity);\n");
        assert_eq!(body, "fn main(){}\n");
    }

    #[test]
    fn split_directives_recognizes_space_diagnostic() {
        let source = "diagnostic (off, derivative_uniformity);\nfn main(){}\n";
        let (dirs, body) = split_directives(source);
        assert_eq!(dirs, "diagnostic (off, derivative_uniformity);\n");
        assert_eq!(body, "fn main(){}\n");
    }

    #[test]
    fn split_directives_does_not_capture_diagnostic_prefixed_identifier() {
        // `diagnostic_counter` is a plain identifier; it is not a
        // directive and must not be hoisted above the preamble.  But the
        // realistic failure path is a top-level declaration whose RHS
        // happens to start with `diagnostic_...`.  In WGSL the top-level
        // line would start with `const`/`var`/etc., so the split would
        // terminate there.  Still, guard against pathological inputs
        // that begin a line with an identifier-looking token.
        let source = "diagnostic_counter_alias\nfn main(){}\n";
        let (dirs, body) = split_directives(source);
        assert_eq!(
            dirs, "",
            "identifier starting with 'diagnostic' must not be treated as a directive"
        );
        assert_eq!(body, source);
    }
}
