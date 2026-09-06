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
//! The private helpers below (naga-abort guards, source preprocessing, the
//! emit fallback ladder) implement the invariants the pipeline relies on but
//! are not stable public surface; the byte-level text scans live in `text`.

pub mod config;
pub mod error;
pub mod generator;
mod io;
pub mod json;
pub mod name_gen;
pub mod name_map;
pub mod passes;
pub mod pipeline;
mod text;
#[cfg(feature = "wasm")]
mod wasm;

use config::Config;
use error::Error;
use generator::{GenerateOptions, generate};
use pipeline::{PassReport, Report};
use std::borrow::Cow;
use std::collections::HashSet;
use text::{
    cleaned_has_enable_directive, cleaned_references_f16_token, cleaned_references_whole_token,
    compact_wgsl_text, has_enable_f16_directive, join_with_newline, normalize_line_endings,
    references_f16_token, split_directives, strip_wgsl_comments,
};

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
/// the IR, so callers of [`run`] never observe these rewrites.  Borrows the
/// input when nothing needs rewriting.
fn preprocess_source_for_naga(source: &str) -> Cow<'_, str> {
    let normalized = normalize_line_endings(source);

    // Older toolchains made these directives optional, or the source targeted a
    // different compiler.  Detection is whole-token on comment-stripped text so
    // a longer identifier never triggers, and the has-directive guards avoid a
    // duplicate.
    let cleaned = strip_wgsl_comments(&normalized);
    let mut prefix = String::new();
    if cleaned_references_f16_token(&cleaned) && !cleaned_has_enable_directive(&cleaned, "f16") {
        prefix.push_str("enable f16;\n");
    }
    if cleaned_references_whole_token(&cleaned, "binding_array")
        && !cleaned_has_enable_directive(&cleaned, "wgpu_binding_array")
    {
        prefix.push_str("enable wgpu_binding_array;\n");
    }
    if prefix.is_empty() {
        return normalized;
    }
    prefix.push_str(&normalized);
    Cow::Owned(prefix)
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
    // `wgpu_int16` is load-bearing when the text uses 16-bit integer tokens;
    // strip only the spurious lingering-type-arena case.
    let keep_int16 = source.contains("enable wgpu_int16;") && {
        let cleaned = strip_wgsl_comments(&source);
        cleaned_references_whole_token(&cleaned, "i16")
            || cleaned_references_whole_token(&cleaned, "u16")
    };
    for directive in NAGA_ONLY_ENABLES {
        if directive == "enable wgpu_int16;" && keep_int16 {
            continue;
        }
        if let Some(pos) = source.find(directive) {
            let mut end = pos + directive.len();
            if source[end..].starts_with('\n') {
                end += 1;
            }
            source.replace_range(pos..end, "");
        }
    }
    source
}

/// Bailout paths ship the input body only lexically compacted, so any leading
/// directives stay inside it; with a preamble active the consumer's
/// [preamble, body] concatenation would place them after the preamble's
/// declarations - invalid WGSL for every consumer, shipped with exit 0.  The
/// generator path strips body directives (the preamble owns them); a bailout
/// has no parser to do that, so it hard-errors instead (mirroring the f16
/// preamble guard).  A directive-free body concatenates cleanly and still
/// ships.
fn preamble_bailout_guard(effective_preamble: Option<&str>, source: &str) -> Result<(), Error> {
    if effective_preamble.is_some() && !split_directives(source).0.trim().is_empty() {
        return Err(Error::Emit(
            "input cannot be minified (naga bailout) and carries leading directives; \
             the shipped [preamble, body] order would misplace them - move the \
             directives into the preamble or minify without --preamble"
                .to_string(),
        ));
    }
    Ok(())
}

/// Ship the input compacted; `reason` is the ORIGINAL naga error (a
/// partially repaired module's second error would not describe the input).
/// No pass report: callers key on [`Report::bailout`].
fn bailout_output(
    reason: String,
    source: &str,
    effective_preamble: Option<&str>,
    mut report: Report,
) -> Result<Output, Error> {
    preamble_bailout_guard(effective_preamble, source)?;
    let compacted = compact_wgsl_text(source);
    report.bailout = Some(reason);
    report.output_bytes = compacted.len();
    Ok(Output {
        source: compacted,
        report,
        name_map: None,
    })
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
    /// [`name_map::NameMap`], or `None` whenever the shipped text does not
    /// carry the pipeline's renames (bailouts, the verbatim guard, the
    /// naga-emitter fallback): names unchanged from the visible output.
    pub name_map: Option<name_map::NameMap>,
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

    let (_, info) = pipeline::run_ir_passes(module, info, config, &mut report)?;

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

/// Every declaration name in `module` (types, struct members, and the
/// module-scope declarations).  [`run`] uses the result to hide preamble
/// symbols from the generator and to extend `preserve_symbols` so rename
/// and mangle leave them alone.
fn collect_module_names(module: &naga::Module) -> HashSet<String> {
    name_gen::module_scope_names(module)
        .chain(name_gen::type_names(module))
        .chain(name_gen::struct_member_names(module))
        .map(str::to_owned)
        .collect()
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

/// `true` when the FIRST line of `err`'s rendering contains one of
/// `patterns`.  naga's codespan output puts the message on line 1 and quotes
/// user source below it, so matching the whole rendering would let a shader
/// comment containing the pattern text trigger on an unrelated error.
fn first_line_matches(err: &Error, patterns: &[&str]) -> bool {
    let msg = err.to_string();
    let first_line = msg.lines().next().unwrap_or("");
    patterns.iter().any(|p| first_line.contains(p))
}

/// `true` for a `Parse` error about an enable-extension naga cannot parse
/// ([`UNSUPPORTED_EXTENSION_PATTERNS`]).  Restricted to `Parse`: a
/// validation or emit error quoting the same text must stay a hard error
/// rather than take the "ship the input compacted" bailout.
fn is_unsupported_extension_parse_error(err: &Error) -> bool {
    matches!(err, Error::Parse(_)) && first_line_matches(err, UNSUPPORTED_EXTENSION_PATTERNS)
}

/// `true` for a `Parse` / `Validation` error matching
/// [`KNOWN_TEXT_VALIDATION_LIMITATION_PATTERNS`] (`validate_wgsl_text`
/// reports the not-yet-supported `enable` through either variant depending
/// on whether tokenisation or semantic validation trips).  Other variants
/// never opt into the fallback bypass.
fn is_known_text_validation_limitation(err: &Error) -> bool {
    matches!(err, Error::Parse(_) | Error::Validation(_))
        && first_line_matches(err, KNOWN_TEXT_VALIDATION_LIMITATION_PATTERNS)
}

/// What the fallback ladder in [`resolve_generator_output`] resolved for
/// one emission attempt.  [`run`] may still veto it (never-grow guard)
/// and ship the input verbatim instead.
struct EmitOutcome {
    /// Final WGSL text of this outcome.
    source: String,
    /// Success arm: output differs from the naga baseline in text or
    /// byte count (vacuously true when the baseline emit was skipped).
    /// Dead on the fallback arms - the report masks it with
    /// `!rolled_back`.
    changed: bool,
    /// `true` when the naga-emitter fallback (or the compacted input) shipped
    /// instead of the generator's text.  Its negation is the report's
    /// validation verdict - not "text validation passed": the known-limitation
    /// carve-out ships generator output despite a failed validation (see
    /// [`is_known_text_validation_limitation`]).
    rolled_back: bool,
    /// Generator wall-clock cost; zero when the generator never
    /// produced text.
    duration_us: u64,
    /// Untextable-IR rung only; forwarded into [`Report::bailout`].
    untextable_reason: Option<String>,
}

/// The last rung of the fallback ladder: both the generator output AND the
/// naga-emitter fallback fail text re-validation, i.e. the optimized IR has
/// no valid WGSL spelling at all (a pass-manufactured literal pair whose
/// runtime value is const-eval-rejected, e.g. `5f/0f` -> inf).  Ship the
/// INPUT lexically compacted, mirroring the parse/validation bailouts, so a
/// batch run gets un-optimized-but-correct output instead of a hard error.
fn untextable_ir_bailout(source: &str, before_bytes: usize, reason: String) -> EmitOutcome {
    let compacted = compact_wgsl_text(source);
    EmitOutcome {
        changed: compacted.len() != before_bytes,
        source: compacted,
        rolled_back: true,
        duration_us: 0,
        untextable_reason: Some(reason),
    }
}

/// Settle the generator's emission attempt against the fallback ladder.
///
/// Falls back to naga's output if the generator errored or produced
/// invalid WGSL; if THAT text also fails re-validation, degrades once more
/// to [`untextable_ir_bailout`].  With a preamble active
/// (`normalized_preamble` / `effective_preamble` are `Some` together),
/// `naga_output` still contains the preamble's declarations and is unusable
/// as a fallback (the consumer will re-prepend the preamble, producing
/// duplicate definitions), so the error must propagate instead.
fn resolve_generator_output(
    gen_result: Result<generator::Emission, Error>,
    naga_output: Option<String>,
    normalized_preamble: Option<&str>,
    effective_preamble: Option<&str>,
    source: &str,
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
                let changed = before_bytes != final_source.len() || differs_from_baseline;
                Ok(EmitOutcome {
                    source: final_source,
                    changed,
                    rolled_back: false,
                    duration_us: emitted.duration_us,
                    untextable_reason: None,
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
                // Ungated like the bailout warning; the codespan block stays
                // trace-gated.
                eprintln!(
                    "warning: generator output failed text validation; \
                     shipping naga emitter output (still IR-minified)"
                );
                if trace_enabled && let Err(e) = &validation_result {
                    eprintln!("warning: generator WGSL validation error: {e}");
                }
                // The naga-emitter fallback is *usually* valid, but it is
                // not guaranteed: naga's own wgsl-out can emit tokens its
                // frontend then rejects (e.g. an `f32(<f64 literal>)` cast,
                // or an f16 literal whose `enable f16;` directive it drops).
                // Re-validate before trusting it.  A doubly-invalid case is
                // an IR no emitter can round-trip through WGSL text (e.g. a
                // pass-manufactured `5f/0f` whose runtime value is inf -
                // inexpressible as a const-expression for ANY consumer), not
                // necessarily a pass bug: degrade to the compacted-input
                // bailout, loudly, instead of failing a batch run on valid
                // input.
                if let Err(ve) = io::validate_wgsl_text(&naga_output) {
                    eprintln!(
                        "warning: minified IR cannot round-trip WGSL text ({ve}); \
                         shipping the input lexically compacted"
                    );
                    return Ok(untextable_ir_bailout(source, before_bytes, ve.to_string()));
                }
                let fallback = finalize_naga_fallback_text(naga_output);
                Ok(EmitOutcome {
                    changed: fallback.len() != before_bytes,
                    source: fallback,
                    rolled_back: true,
                    duration_us: emitted.duration_us,
                    untextable_reason: None,
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
                eprintln!(
                    "warning: generator emit failed ({e}); \
                     shipping naga emitter output (still IR-minified)"
                );
                // Re-validate the naga fallback; doubly-invalid degrades to the
                // compacted-input bailout (see the un-textable-IR note above).
                if let Err(ve) = io::validate_wgsl_text(&naga_output) {
                    eprintln!(
                        "warning: generator emit failed ({e}); minified IR cannot \
                         round-trip WGSL text ({ve}); shipping the input lexically compacted"
                    );
                    return Ok(untextable_ir_bailout(source, before_bytes, ve.to_string()));
                }
                let fallback = finalize_naga_fallback_text(naga_output);
                Ok(EmitOutcome {
                    changed: fallback.len() != before_bytes,
                    source: fallback,
                    rolled_back: true,
                    duration_us: 0,
                    untextable_reason: None,
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
    let normalized_preamble: Option<Cow<'_, str>> =
        effective_preamble.map(preprocess_source_for_naga);
    let (preamble_names, full_source): (HashSet<String>, Cow<'_, str>);
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
        full_source = Cow::Owned(join_with_newline(&[
            source_directives,
            preamble_directives,
            preamble_body,
            source_body,
        ]));
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
            // that DO understand the extension.
            return bailout_output(
                format!("naga cannot parse the input: {e}"),
                source,
                effective_preamble,
                Report::new(source.len()),
            );
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
    let info = match io::validate_module(&module) {
        Ok(info) => info,
        Err(validation_err) => {
            // One repair attempt; an incomplete repair reports the ORIGINAL
            // error, which describes the input.
            let recovered =
                passes::specialize_ptr_params::specialize_ptr_params(&mut module, &preamble_names)
                    .then(|| io::validate_module(&module).ok())
                    .flatten();
            let Some(info) = recovered else {
                let mut reason = format!("naga rejects the input: {validation_err}");
                if reason.contains("is a pointer of space") {
                    reason.push_str(
                        "\nnote: pointer-parameter recovery covers whole-variable arguments \
                         only (f(&v), not f(&v.m) or f(&v[i]))",
                    );
                }
                return bailout_output(reason, source, effective_preamble, report);
            };
            report.pass_reports.push(PassReport {
                pass_name: "specialize_ptr_params".to_string(),
                before_bytes: None,
                after_bytes: None,
                changed: true,
                duration_us: 0,
                validation_ok: true,
                text_validation_ok: None,
                rolled_back: false,
            });
            info
        }
    };

    let (name_log, info) =
        pipeline::run_ir_passes(&mut module, info, &effective_config, &mut report)?;
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

    // Taken before `resolve_generator_output` consumes the result.
    let gen_name_tables = gen_result
        .as_ref()
        .ok()
        .map(|emission| (emission.structs.clone(), emission.live_const_names.clone()));

    let EmitOutcome {
        source: final_source,
        changed,
        rolled_back,
        duration_us,
        untextable_reason,
    } = resolve_generator_output(
        gen_result,
        naga_output,
        normalized_preamble.as_deref(),
        effective_preamble,
        source,
        before_bytes,
        config.trace.enabled,
    )?;

    // Same signal as the parse/validation bailouts.
    if let Some(reason) = untextable_reason {
        report.bailout = Some(format!(
            "the optimized IR has no valid WGSL text form, the input shipped compacted: {reason}"
        ));
    }

    // Never ship output larger than the input.  Rare shapes can grow (an
    // already-minimal file whose emission spends more on scaffolding than it
    // saves, or a loop-exit-preservation keep).  Two exemptions: beautify
    // mode grows output on purpose, and with a preamble the input body may
    // carry leading directives that the shipped [preamble, body] order
    // forbids, so the original text is not a valid substitute.  The input is
    // shipped VERBATIM, not compacted - it never went through the emit
    // self-checks.
    let (final_source, changed, shipped_input_verbatim) =
        if !config.beautify && !has_preamble && final_source.len() > source.len() {
            (source.to_string(), false, true)
        } else {
            (final_source, changed, false)
        };
    let final_bytes = final_source.len();

    let name_map = (!rolled_back && !shipped_input_verbatim).then(|| {
        let (structs, live_const_names) = gen_name_tables.unwrap_or_default();
        name_map::NameMap::assemble(&module, &name_log, structs, &live_const_names)
    });

    report.pass_reports.push(PassReport {
        pass_name: "generator_emit".to_string(),
        before_bytes: Some(before_bytes),
        after_bytes: Some(final_bytes),
        changed: !rolled_back && changed,
        duration_us,
        validation_ok: !rolled_back,
        text_validation_ok: Some(!rolled_back),
        rolled_back,
    });
    report.output_bytes = final_bytes;

    Ok(Output {
        source: final_source,
        report,
        name_map,
    })
}

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;
