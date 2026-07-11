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
/// (`ArraySize::Pending`, e.g. `array<i32, O*2>`).  naga's WGSL back-end has no
/// arm for the override size expression in `write_possibly_const_expression`
/// (it hits `_ => unreachable!()`), so it ABORTS under the release
/// `panic = "abort"` strategy rather than returning an error.  nagami's generator
/// emits these types itself, so callers skip the naga baseline/fallback emit for
/// such modules rather than invoke the panicking path.
///
/// Crate-visible because the pipeline's trace / validate-each-pass text-emission
/// path needs the same guard.
pub(crate) fn module_has_override_sized_array(module: &naga::Module) -> bool {
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

/// `true` when any `override` or module-scope `var` initializer contains an
/// expression naga's WGSL back-end cannot write.
///
/// naga's `write_possibly_const_expression` (the writer for override and
/// global-variable initializers) handles only `Literal`, `Constant`,
/// `ZeroValue`, `Compose`, `Splat`, and `Override`; any other variant - e.g. the
/// `Binary` in `override height = 2 * depth;` - falls through to
/// `_ => unreachable!()` and ABORTS under `panic = "abort"`.  (Fully-const
/// initializers such as `2.0 * 3.0` are folded to a `Literal` by naga's
/// front-end before the writer, so only override-dependent initializers trip
/// it.)  nagami emits these correctly, so callers skip the naga baseline/fallback
/// for such modules, as for [`module_has_override_sized_array`].
pub(crate) fn module_has_non_const_global_initializer(module: &naga::Module) -> bool {
    use naga::Expression as E;

    // Iterative DFS over every expression reachable from an override or
    // global-var initializer, mirroring naga's `write_possibly_const_expression`
    // recursion exactly.  `visited` dedupes shared sub-trees (init arenas are
    // acyclic, so it is a memo, not a cycle guard).  Return on the first node
    // outside naga's writable set.
    let mut visited = vec![false; module.global_expressions.len()];
    let mut stack: Vec<naga::Handle<naga::Expression>> = module
        .overrides
        .iter()
        .filter_map(|(_, o)| o.init)
        .chain(module.global_variables.iter().filter_map(|(_, g)| g.init))
        .collect();

    while let Some(handle) = stack.pop() {
        if std::mem::replace(&mut visited[handle.index()], true) {
            continue;
        }
        match &module.global_expressions[handle] {
            E::Literal(_) | E::ZeroValue(_) | E::Override(_) => {}
            // naga writes a *named* constant by name (safe leaf), but for an
            // anonymous constant it recurses into the constant's own init - so
            // a `Binary` there would still abort.  Mirror that descent.
            E::Constant(c) => {
                let konst = &module.constants[*c];
                if konst.name.is_none() {
                    stack.push(konst.init);
                }
            }
            E::Compose { components, .. } => stack.extend(components.iter().copied()),
            E::Splat { value, .. } => stack.push(*value),
            // `Binary` / `Unary` / `Math` / `As` / `Access` / ... have no arm
            // in naga's `write_possibly_const_expression` (`_ => unreachable!()`).
            _ => return true,
        }
    }
    false
}

/// `true` when `module` uses ray queries.  naga 30's WGSL back-end has no arm
/// for `Statement::RayQuery` or for the `RayQueryGetIntersection` /
/// `RayQueryVertexPositions` expressions (both `unreachable!()`), so it ABORTS
/// under the release `panic = "abort"` strategy rather than returning an
/// error.  In a validated module every one of those constructs requires a
/// `ray_query`-typed local, so detecting the *type* in the arena covers them
/// all without walking statement trees.  A dead-but-declared `ray_query` type
/// over-triggers, which only costs the naga baseline byte-count - nagami's
/// generator emits ray-query code itself.  (Ray-tracing-*pipeline* modules
/// without ray queries are fine: the writer handles their stages, payloads,
/// and builtins.)
pub(crate) fn module_has_ray_query(module: &naga::Module) -> bool {
    module
        .types
        .iter()
        .any(|(_, ty)| matches!(ty.inner, naga::TypeInner::RayQuery { .. }))
}

/// `true` when callers must skip the naga WGSL baseline/fallback emit because
/// naga's back-end would abort (`panic = "abort"`) rather than error on this
/// module.  Every site that emits via naga's WGSL backend gates on this so the
/// abort-trigger set stays defined in one place.
///
/// Known residual hole (unreachable from [`run`]): exotic subgroup collective
/// combinations hit `unimplemented!()` in naga's writer, but naga's WGSL
/// front-end cannot parse `enable subgroups;` at all, so no text input reaches
/// them; only a hand-built module fed to [`run_module`] could.
pub(crate) fn module_needs_naga_baseline_skip(module: &naga::Module) -> bool {
    module_has_override_sized_array(module)
        || module_has_non_const_global_initializer(module)
        || module_has_ray_query(module)
}

/// Normalise `source` so naga's front-end accepts it: rewrite lone-CR endings
/// to LF (so the per-line scans see every break), then inject the `enable`
/// directives naga 30 requires to parse a feature the text uses but does not
/// declare (`enable f16;`, `enable wgpu_binding_array;`).  naga 30 implements
/// every `wgpu_*` extension, so nothing is stripped.  Output is re-derived from
/// the IR, so callers of [`run`] never observe these rewrites.
fn preprocess_source_for_naga(source: &str) -> String {
    let normalized = normalize_line_endings(source);

    // Older toolchains made these directives optional, or the source targeted a
    // different compiler.  Detection is whole-token on comment-stripped text so
    // a longer identifier never triggers, and the has-directive guards avoid a
    // duplicate.
    let mut prefix = String::new();
    if references_f16_token(&normalized) && !has_enable_f16_directive(&normalized) {
        prefix.push_str("enable f16;\n");
    }
    if references_binding_array_token(&normalized)
        && !has_enable_directive(&normalized, "wgpu_binding_array")
    {
        prefix.push_str("enable wgpu_binding_array;\n");
    }
    if prefix.is_empty() {
        return normalized;
    }
    prefix.push_str(&normalized);
    prefix
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
    has_enable_directive(source, "f16")
}

/// `true` when `source` declares `enable <ext>;`, including as one entry of a
/// comma-separated list (`enable f16, clip_distances;`) and regardless of how
/// the directives are split across lines.  A false negative is not harmless:
/// the preamble guard in [`run`] turns it into a hard error on valid input, so
/// EVERY directive is scanned, not just the first on a line.
fn has_enable_directive(source: &str, ext: &str) -> bool {
    let cleaned = strip_wgsl_comments(source);
    // Each `;`-terminated segment is one directive; a directive lists one or
    // more comma-separated extensions.  Scanning all segments handles several
    // directives on one line (`enable a; enable f16;`) - the callers include
    // arbitrary user-authored preamble text.
    for segment in cleaned.split(';') {
        let Some(list) = segment.trim_start().strip_prefix("enable") else {
            continue;
        };
        // `enable` must be followed by whitespace to be the keyword, not an
        // identifier prefix like `enablef16` / `enable_x`.
        if !list.starts_with([' ', '\t']) {
            continue;
        }
        if list.split(',').any(|e| e.trim() == ext) {
            return true;
        }
    }
    false
}

/// Strip the dead `return;` naga's WGSL front-end appends after a diverging
/// tail construct in a non-void function (and entry point).  naga never proves
/// a `loop` non-falling-through, so it appends `Statement::Return { value:
/// None }`, which naga 30's validator rejects as `InvalidReturnType`.  Removing
/// it only where the preceding construct PROVABLY diverges
/// ([`block_definitely_terminates`]) is semantics-preserving: the appended
/// return is unreachable there.  The generator re-synthesises the return in its
/// output on the same predicate, so the round-trip is unchanged.
fn strip_front_end_appended_returns(module: &mut naga::Module) {
    fn strip(func: &mut naga::Function) {
        if func.result.is_none() {
            return; // void function: a bare `return;` tail is legitimate
        }
        if func.body.len() < 2 {
            return; // need a diverging construct BEFORE the appended return
        }
        if !matches!(
            func.body.last(),
            Some(naga::Statement::Return { value: None })
        ) {
            return;
        }
        let last_span = func
            .body
            .span_iter()
            .last()
            .map_or(naga::Span::UNDEFINED, |(_, s)| *s);
        let len = func.body.len();
        func.body.cull(len - 1..);
        // Keep the strip only if what now sits at the tail provably diverges;
        // otherwise restore the return (the function genuinely falls through,
        // which stays invalid and takes the bailout below).
        if !crate::passes::dead_branch::block_definitely_terminates(&func.body) {
            func.body
                .push(naga::Statement::Return { value: None }, last_span);
        }
    }
    for (_, func) in module.functions.iter_mut() {
        strip(func);
    }
    for ep in module.entry_points.iter_mut() {
        strip(&mut ep.function);
    }
}

/// naga-only `enable wgpu_*;` directives the generator emits so naga can PARSE
/// the feature, but which tint/Dawn reject and the shipped tint-facing output
/// must omit:
///
/// * `wgpu_binding_array` - tint supports binding arrays NATIVELY without an
///   enable, so a stripped output is fully tint-valid.
/// * `wgpu_int16` - tint has no `i16`/`u16`, so stripping is right only in
///   the SPURIOUS case (a dead `frexp(f16)` whose i16 exponent lingers in
///   the type arena, the emitted body free of 16-bit tokens).  Text that
///   genuinely uses them - only the naga FALLBACK can produce it, since the
///   generator has no i16/u16 spelling and always falls back - keeps the
///   enable: such output is wgpu-facing by necessity, and stripping would
///   leave it invalid for every consumer.  `strip_naga_only_enables` gates
///   on that token check.
const NAGA_ONLY_ENABLES: [&str; 2] = ["enable wgpu_binding_array;", "enable wgpu_int16;"];

/// Remove the [`NAGA_ONLY_ENABLES`] directives (each with one immediately
/// following newline, if any) from generator output.  Each is emitted at most
/// once.
fn strip_naga_only_enables(mut source: String) -> String {
    for directive in NAGA_ONLY_ENABLES {
        // `wgpu_int16` is load-bearing when the text uses 16-bit integer
        // tokens; strip only the spurious lingering-type-arena case.
        if directive == "enable wgpu_int16;"
            && (references_whole_token(&source, "i16") || references_whole_token(&source, "u16"))
        {
            continue;
        }
        if let Some(pos) = source.find(directive) {
            let after = &source[pos + directive.len()..];
            let tail = after.strip_prefix('\n').unwrap_or(after).to_owned();
            source.truncate(pos);
            source.push_str(&tail);
        }
    }
    source
}

/// Lexically compact WGSL text that never goes through the generator: strip
/// comments, then collapse every whitespace run, keeping a single space only
/// where joining would merge tokens.  Used on the bailout paths (input naga
/// cannot parse or validate) and on the naga-emitter fallback, which
/// otherwise ship fully un-minified text.
///
/// Grammar-agnostic and token-safe by construction, so it needs no parser:
/// * a space survives between two identifier-ish chars (Unicode-aware -
///   WGSL identifiers are XID, approximated by `char::is_alphanumeric` plus
///   `_`), covering `enable f16`, `else if`, `let x`;
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
fn compact_wgsl_text(source: &str) -> String {
    let stripped = strip_wgsl_comments(source);
    let ident_ish = |c: char| c.is_alphanumeric() || c == '_';
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

/// Finish the naga-emitter fallback text for shipping: lexically compact it
/// (naga's writer pretty-prints), then drop the naga-only enables exactly
/// like the generator path - the shipped, tint-facing output must not carry
/// them.  Compaction is verified by a naga re-parse; on failure (a compactor
/// bug) the pretty-but-valid text ships instead.  The caller has already
/// validated `naga_output` itself.
fn finalize_naga_fallback_text(naga_output: String) -> String {
    let compacted = compact_wgsl_text(&naga_output);
    let text = if io::validate_wgsl_text(&compacted).is_ok() {
        compacted
    } else {
        naga_output
    };
    strip_naga_only_enables(text)
}

/// The [`NAGA_ONLY_ENABLES`] present in generator output `emit_source`, as a
/// directive prefix.  Prepended to the preamble self-check text: naga REQUIRES
/// these to parse the feature, but the shipped body omits them and a user
/// preamble legitimately lacks them, so without injecting them the naga check
/// would reject a body that is nonetheless tint-valid.  Genuine tint-required
/// enables (f16, ...) are deliberately NOT injected.
fn naga_only_enable_prefix(emit_source: &str) -> String {
    let mut prefix = String::new();
    for directive in NAGA_ONLY_ENABLES {
        if emit_source.contains(directive) {
            prefix.push_str(directive);
            prefix.push('\n');
        }
    }
    prefix
}

/// `true` when comment-stripped `source` uses `token` as a whole identifier
/// token (so a longer identifier like `my_binding_array` never triggers for
/// `binding_array`).
fn references_whole_token(source: &str, token: &str) -> bool {
    let cleaned = strip_wgsl_comments(source);
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

/// naga 30 requires `enable wgpu_binding_array;` to PARSE a `binding_array`
/// type, so the preprocessor injects the directive when the type-generator is
/// used without one.
fn references_binding_array_token(source: &str) -> bool {
    references_whole_token(source, "binding_array")
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
    let before_wgsl = emit_module_for_report(module, &info, config)?;
    let mut report = Report::new(before_wgsl.len());

    pipeline::run_ir_passes(module, config, &mut report)?;

    let info = io::validate_module(module)?;
    let after_wgsl = emit_module_for_report(module, &info, config)?;
    report.output_bytes = after_wgsl.len();

    Ok(report)
}

/// Render `module` to WGSL text for [`run_module`]'s byte accounting.
///
/// Prefers naga's WGSL back-end, but that back-end ABORTS (release
/// `panic = "abort"`) rather than errors on modules with an override-sized
/// array or a non-const global initializer - see
/// [`module_needs_naga_baseline_skip`].  For those, render via nagami's own
/// generator (which handles them and is what nagami actually ships), so the
/// public `run_module` entry point never crashes on validator-accepted input.
fn emit_module_for_report(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    config: &Config,
) -> Result<String, Error> {
    if module_needs_naga_baseline_skip(module) {
        let emitted = generate(
            module,
            info,
            GenerateOptions {
                beautify: config.beautify,
                indent: config.indent,
                mangle: config.mangle(),
                float_precision: config.float_precision,
                preserve_symbols: config.preserve_symbols.iter().cloned().collect(),
                type_alias: true,
                ..Default::default()
            },
        )?;
        Ok(emitted.source)
    } else {
        emit_wgsl_with_naga_safe(module, info)
    }
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

/// What the fallback ladder in [`resolve_generator_output`] resolved for
/// one emission attempt.  [`run`] may still veto it (never-grow guard)
/// and ship the input verbatim instead.
struct EmitOutcome {
    /// Final WGSL text of this outcome.
    source: String,
    /// Byte length of `source`.
    bytes: usize,
    /// Success arm: output differs from the naga baseline in text or
    /// byte count (vacuously true when the baseline emit was skipped).
    /// Dead on the fallback arms - the report masks it with
    /// `!rolled_back`.
    changed: bool,
    /// `true` when the generator's output was discarded - or its emit
    /// attempt failed outright - in favor of the naga-emitter fallback.
    rolled_back: bool,
    /// `false` exactly on the fallback paths.  Not "text validation
    /// passed": the known-limitation carve-out ships generator output
    /// as `true` despite a failed validation (see
    /// [`is_known_text_validation_limitation`]).
    validation_ok: bool,
    /// Generator wall-clock cost; zero when the generator never
    /// produced text.
    duration_us: u64,
}

/// Settle the generator's emission attempt against the fallback ladder.
///
/// Falls back to naga's output if the generator errored or produced
/// invalid WGSL.  With a preamble active (`normalized_preamble` /
/// `effective_preamble` are `Some` together), `naga_output` still
/// contains the preamble's declarations and is unusable as a fallback
/// (the consumer will re-prepend the preamble, producing duplicate
/// definitions), so the error must propagate instead.
fn resolve_generator_output(
    gen_result: Result<generator::Emission, Error>,
    naga_output: Option<String>,
    normalized_preamble: Option<&str>,
    effective_preamble: Option<&str>,
    before_bytes: usize,
    trace_enabled: bool,
) -> Result<EmitOutcome, Error> {
    let has_preamble = effective_preamble.is_some();
    match gen_result {
        Ok(emitted) => {
            // WGSL requires every directive (`enable`/`requires`/
            // `diagnostic`) to precede all declarations.  An active preamble
            // is prepended ahead of this output, so the body must carry no
            // leading directives - the preamble owns them.  Strip them and
            // validate the order the consumer ships, [preamble, body], so a
            // directive the preamble omits errors here, not on the GPU.  With
            // no preamble the whole module is the output and its directives
            // lead it.
            let validation_result = if let Some(normalized_preamble) = normalized_preamble {
                // The generator's naga-ONLY enables (wgpu_binding_array /
                // wgpu_int16) must be present for naga to PARSE the feature,
                // but they are dropped from the shipped body and a user
                // preamble legitimately lacks them - inject them for THIS
                // naga check only.  Genuine tint-required enables (f16, ...)
                // are NOT injected, so a body needing one the preamble omits
                // still fails here (or hits the f16 guard below).
                let emit_body = split_directives(&emitted.source).1;
                let (pre_directives, pre_body) = split_directives(normalized_preamble);
                let naga_only = naga_only_enable_prefix(&emitted.source);
                let combined =
                    join_with_newline(&[naga_only.as_str(), pre_directives, pre_body, emit_body]);
                io::validate_wgsl_text(&combined)
            } else {
                io::validate_wgsl_text(&emitted.source)
            };
            let valid = match &validation_result {
                Ok(()) => true,
                // Keyed on the validator's message (not the output text):
                // forward defence for a future naga that rejects its own valid
                // subgroup-builtin round-trip.  Inert on current naga, which
                // accepts the bare builtins.
                Err(e) if is_known_text_validation_limitation(e) => {
                    if trace_enabled {
                        eprintln!(
                            "warning: skipping text-validation rollback due to known naga subgroup parser limitation"
                        );
                    }
                    true
                }
                Err(_) => false,
            };
            if valid {
                // Read the naga-baseline comparison before `emitted.source`
                // may be moved into `final_source`.
                let differs_from_baseline = naga_output.as_deref() != Some(emitted.source.as_str());
                let final_source = if has_preamble {
                    // A preamble owns all directives, so the body's leading
                    // directives (including any `enable wgpu_binding_array;`)
                    // are already dropped here.
                    split_directives(&emitted.source).1.to_owned()
                } else {
                    // Strip the naga-only `enable wgpu_*;` directives the
                    // generator emitted for the self-check: tint rejects
                    // them, so the shipped, tint-facing output must not carry
                    // them.
                    strip_naga_only_enables(emitted.source)
                };
                // With a preamble active the body's directives are stripped
                // (the preamble owns them).  If the stripped body still needs
                // `enable f16;` but the consumer's preamble does not declare
                // it, the shipped `[preamble, body]` document is invalid, and
                // WGSL forbids re-emitting the directive after the preamble's
                // declarations, so nagami cannot repair it.  The self-check
                // above validates against the f16-INJECTED normalised preamble
                // (see `preprocess_source_for_naga`) and masks this, so guard
                // explicitly and surface a diagnosable error.
                if has_preamble
                    && references_f16_token(&final_source)
                    && !has_enable_f16_directive(effective_preamble.unwrap_or(""))
                {
                    return Err(Error::Emit(
                        "shader body requires `enable f16;` but the preamble \
                             does not declare it; add `enable f16;` to the preamble"
                            .to_string(),
                    ));
                }
                let after_bytes = final_source.len();
                let changed = before_bytes != after_bytes || differs_from_baseline;
                Ok(EmitOutcome {
                    source: final_source,
                    bytes: after_bytes,
                    changed,
                    rolled_back: false,
                    validation_ok: true,
                    duration_us: emitted.duration_us,
                })
            } else if has_preamble {
                // The naga fallback already embeds the preamble's declarations,
                // so the caller re-prepending the preamble would duplicate them
                // - unusable.  Propagate the validator message so the user can
                // diagnose.
                let underlying = match validation_result {
                    Err(e) => e.to_string(),
                    Ok(()) => "(no underlying error)".to_string(),
                };
                Err(Error::Emit(format!(
                    "generator output failed validation; \
                         cannot fall back safely when a preamble is active: {underlying}",
                )))
            } else if let Some(naga_output) = naga_output {
                if trace_enabled {
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
                let fallback = finalize_naga_fallback_text(naga_output);
                let final_bytes = fallback.len();
                Ok(EmitOutcome {
                    source: fallback,
                    bytes: final_bytes,
                    changed: final_bytes != before_bytes,
                    rolled_back: true,
                    validation_ok: false,
                    duration_us: emitted.duration_us,
                })
            } else {
                // No naga baseline (the module is in naga's writer-abort
                // set): the generator output is invalid and there is no
                // fallback emitter.
                let underlying = match validation_result {
                    Err(e) => e.to_string(),
                    Ok(()) => "(no underlying error)".to_string(),
                };
                Err(Error::Emit(format!(
                    "generator output failed validation and no naga fallback \
                         is available (naga's writer would abort on this module): \
                         {underlying}"
                )))
            }
        }
        Err(e) => match naga_output {
            Some(naga_output) => {
                if has_preamble {
                    return Err(e);
                }
                if trace_enabled {
                    eprintln!("warning: generator emit failed ({e}); falling back to naga emitter");
                }
                // Re-validate the naga fallback (see above) so a doubly-invalid
                // case errors instead of shipping invalid WGSL.
                if let Err(ve) = io::validate_wgsl_text(&naga_output) {
                    return Err(Error::Emit(format!(
                        "generator emit failed ({e}) and the naga-emitter \
                             fallback is also invalid: {ve}"
                    )));
                }
                let fallback = finalize_naga_fallback_text(naga_output);
                let final_bytes = fallback.len();
                Ok(EmitOutcome {
                    source: fallback,
                    bytes: final_bytes,
                    changed: final_bytes != before_bytes,
                    rolled_back: true,
                    validation_ok: false,
                    duration_us: 0,
                })
            }
            // No naga fallback available (the module is in naga's
            // writer-abort set): the generator's own error is the only
            // diagnosis.
            None => Err(e),
        },
    }
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
            // `subgroups`).  Ship the source lexically compacted -
            // comments and whitespace need no parser to remove, and
            // this path otherwise ships fully un-minified text - so
            // the caller still gets something runnable on backends
            // that DO understand the extension.  The synthetic
            // `unsupported_extension_bailout` PassReport lets
            // downstream tooling distinguish bailout from "ran with
            // no changes"; `validation_ok = true` is "no failure
            // observed" (no IR ever built), set true so CI gates
            // asserting `all validation_ok` don't fail spuriously.
            let compacted = compact_wgsl_text(source);
            let mut report = Report::new(source.len());
            report.output_bytes = compacted.len();
            report.pass_reports.push(PassReport {
                pass_name: "unsupported_extension_bailout".to_string(),
                before_bytes: Some(source.len()),
                after_bytes: Some(compacted.len()),
                changed: compacted != source,
                duration_us: 0,
                validation_ok: true,
                text_validation_ok: None,
                rolled_back: false,
            });
            return Ok(Output {
                source: compacted,
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

    // naga's WGSL front-end appends an implicit `return;` (value `None`) to a
    // function body whose tail is a `loop` - it never proves a loop
    // non-falling-through, even one whose body always returns - and naga 30's
    // validator now rejects that as `InvalidReturnType` in a non-void function.
    // Such input (a function ending in a diverging loop) is valid WGSL that
    // tint/Dawn accept and that naga 29 also validated; strip the dead appended
    // return so nagami can still minimise it.  The strip is gated on the SAME
    // `block_definitely_terminates` predicate the generator uses to RE-synthesise
    // the trailing return in its output, so input and output stay in lockstep.
    strip_front_end_appended_returns(&mut module);

    // Bail out on input naga's validator rejects.  naga's front-end PARSES some
    // shaders it then rejects at validation - notably a const-expression
    // division / modulo by zero (a WGSL shader-creation error naga 30 enforces
    // that tint/Dawn accept leniently).  nagami cannot optimise such input, so
    // return it verbatim (like the unsupported-extension bailout) rather than
    // error and break a batch run.  Doing this BEFORE the passes also means the
    // post-pass `validate_module` failure below provably indicates a pass bug
    // (valid in, invalid out) and rightly stays a hard error.
    if io::validate_module(&module).is_err() {
        // Same lexical compaction as the unsupported-extension bailout:
        // rejected-but-parseable input (e.g. const division by zero) still
        // deserves comment/whitespace removal on its way through.
        let compacted = compact_wgsl_text(source);
        report.pass_reports.push(PassReport {
            pass_name: "validation_bailout".to_string(),
            before_bytes: Some(source.len()),
            after_bytes: Some(compacted.len()),
            changed: compacted != source,
            duration_us: 0,
            validation_ok: false,
            text_validation_ok: None,
            rolled_back: true,
        });
        report.output_bytes = compacted.len();
        return Ok(Output {
            source: compacted,
            report,
        });
    }

    pipeline::run_ir_passes(&mut module, &effective_config, &mut report)?;

    let info = io::validate_module(&module)?;
    // Skip the naga baseline/fallback emit for modules naga's back-end would
    // abort on (the trigger set lives on `module_needs_naga_baseline_skip`);
    // nagami's generator emits these itself, and the baseline byte count falls
    // back to the input length.
    let naga_output: Option<String> = if module_needs_naga_baseline_skip(&module) {
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

    let EmitOutcome {
        source: final_source,
        bytes: final_bytes,
        changed,
        rolled_back,
        validation_ok: compacted_valid,
        duration_us,
    } = resolve_generator_output(
        gen_result,
        naga_output,
        normalized_preamble.as_deref(),
        effective_preamble,
        before_bytes,
        config.trace.enabled,
    )?;

    // Never ship output larger than the input.  Rare shapes can grow (an
    // already-minimal file whose emission spends more on scaffolding than it
    // saves, or a loop-exit-preservation keep).  Two exemptions: beautify
    // mode grows output on purpose, and with a preamble the input body may
    // carry leading directives that the shipped [preamble, body] order
    // forbids, so the original text is not a valid substitute.  The input is
    // shipped VERBATIM, not compacted - it never went through the emit
    // self-checks.
    let (final_source, final_bytes, changed) =
        if !config.beautify && !has_preamble && final_bytes > source.len() {
            (source.to_string(), source.len(), false)
        } else {
            (final_source, final_bytes, changed)
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
#[path = "lib_tests.rs"]
mod tests;
