//! Crate root: [`run_module`] optimises a pre-parsed [`naga::Module`] and
//! returns a [`Report`]; [`run`] is source-to-source: preprocess, parse,
//! optimise, emit via the custom generator with a fallback ladder down to
//! naga's own emitter and the lexically compacted input.

pub mod config;
pub mod error;
pub mod generator;
pub(crate) mod handle_set;
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
    compact_wgsl_text, join_with_newline, normalize_line_endings, requires_entry_spans,
    split_directives, strip_wgsl_comments,
};

// MARK: Source preprocessing

fn emit_wgsl_with_naga_safe(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
) -> Result<String, Error> {
    naga::back::wgsl::write_string(module, info, naga::back::wgsl::WriterFlags::empty())
        .map_err(|e| Error::Emit(e.to_string()))
}

/// naga's WGSL back-end has no arm for an override-sized array
/// (`ArraySize::Pending`, e.g. `array<i32, O*2>`) and hits `unreachable!()`,
/// an abort under `panic = "abort"`; nagami's generator emits these itself,
/// so callers skip the naga baseline/fallback emit for such modules.
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

/// naga's `write_possibly_const_expression` (override and global-variable
/// initializers) handles only `Literal`, `Constant`, `ZeroValue`, `Compose`,
/// `Splat` and `Override`; anything else - the `Binary` in
/// `override height = 2 * depth;` (fully-const initializers are folded to
/// a `Literal` by the front-end) - hits `unreachable!()` and aborts under
/// `panic = "abort"`, so callers skip the naga baseline/fallback emit.
pub(crate) fn module_has_non_const_global_initializer(module: &naga::Module) -> bool {
    use naga::Expression as E;

    // Mirrors naga's writer recursion exactly; `visited` is a memo for shared
    // sub-trees (init arenas are acyclic).
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
            // naga writes a named constant by name but recurses into an
            // anonymous constant's init, where a `Binary` would still abort.
            E::Constant(c) => {
                let konst = &module.constants[*c];
                if konst.name.is_none() {
                    stack.push(konst.init);
                }
            }
            E::Compose { components, .. } => stack.extend(components.iter().copied()),
            E::Splat { value, .. } => stack.push(*value),
            _ => return true,
        }
    }
    false
}

/// naga's WGSL back-end has no arm for `Statement::RayQuery` or the
/// `RayQueryGetIntersection` / `RayQueryVertexPositions` expressions
/// (`unreachable!()`, an abort under `panic = "abort"`).  In a validated
/// module each requires a `ray_query`-typed local, so the type arena covers
/// them all without walking statements; a dead-but-declared type
/// over-triggers, costing only the naga baseline byte count.
/// Ray-tracing-pipeline modules without ray queries are fine.
pub(crate) fn module_has_ray_query(module: &naga::Module) -> bool {
    module
        .types
        .iter()
        .any(|(_, ty)| matches!(ty.inner, naga::TypeInner::RayQuery { .. }))
}

/// The one gate for every naga WGSL back-end emit: modules it would abort
/// on (`panic = "abort"`) rather than reject.  Residual hole, unreachable
/// from [`run`]: exotic subgroup collectives hit `unimplemented!()` in the
/// writer, but naga's front-end cannot parse `enable subgroups;`, so only a
/// hand-built module fed to [`run_module`] could reach them.
pub(crate) fn module_needs_naga_baseline_skip(module: &naga::Module) -> bool {
    module_has_override_sized_array(module)
        || module_has_non_const_global_initializer(module)
        || module_has_ray_query(module)
}

/// Normalise `source` so naga's front-end accepts it: lone-CR endings to
/// LF, the `enable` directives naga requires for a feature the text uses
/// but does not declare (`f16`, `wgpu_binding_array`) injected, and the one
/// `requires` entry naga refuses blanked (naga implements every `wgpu_*`
/// extension, so none is stripped).  Output is re-derived from the IR, so
/// callers of [`run`] never observe these rewrites; borrows the input when
/// nothing needs rewriting.
fn preprocess_source_for_naga(source: &str) -> Cow<'_, str> {
    let normalized = normalize_line_endings(source);

    // Older toolchains made these directives optional, or the source targets
    // another compiler; the has-directive guards avoid a duplicate.
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
    // naga parses pointer parameters in every address space yet rejects the
    // directive announcing them (wgpu #5158), stopping the very shaders
    // pointer specialization exists for at the parser.  The directive is
    // advisory, so blanking changes nothing the IR sees; same length with
    // line breaks kept because naga's diagnostics quote these offsets.
    let blanks = requires_entry_spans(&cleaned, "unrestricted_pointer_parameters");
    if prefix.is_empty() && blanks.is_empty() {
        return normalized;
    }
    let base = prefix.len();
    let mut out = prefix;
    out.push_str(&normalized);
    for span in blanks {
        let span = span.start + base..span.end + base;
        let blanked: String = out.as_bytes()[span.clone()]
            .iter()
            .map(|&b| if b == b'\n' { '\n' } else { ' ' })
            .collect();
        out.replace_range(span, &blanked);
    }
    Cow::Owned(out)
}

/// naga's front-end appends `Return { value: None }` after a `loop` tail
/// (it never proves a loop non-falling-through), which the validator
/// rejects as `InvalidReturnType` in a non-void function.  Removing it only
/// where the tail provably diverges is semantics-preserving, and the
/// generator re-synthesises the return on the same predicate, so the
/// round-trip is unchanged.
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
        // Restore the return when the new tail does not provably diverge: a
        // genuine fall-through stays invalid and takes the bailout.
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

/// naga-only `enable wgpu_*;` directives naga needs to parse a feature and
/// tint/Dawn reject as unknown, paired with the tokens whose presence in the
/// shipped text means the feature survived minification.  Dropping one buys
/// a tint-consumable body; dropping one the body still needs ships text
/// NEITHER compiler accepts, so each is kept when the output uses it and the
/// input asked for it.
///
/// `wgpu_binding_array`: tint's own `binding_array` is narrow (two template
/// arguments, sampled-texture element), so a wgpu bindless shader - samplers,
/// buffers or a runtime size - is tint-invalid whatever we do, and only naga
/// can consume it.  A source that never declared the extension is targeting
/// tint and keeps the stripped form.  `wgpu_int16`: tint has no `i16`/`u16`,
/// so stripping is right only in the spurious case (a dead `frexp(f16)` whose
/// i16 exponent lingers in the type arena, the body free of 16-bit tokens);
/// the generator has no i16/u16 spelling, so genuine use reaches the text
/// only through the naga fallback, which is wgpu-facing by necessity.
/// `(directive, tokens whose presence means the output still needs it, whether
/// keeping it also requires the input to have declared it)`.
const NAGA_ONLY_ENABLES: [(&str, &[&str], bool); 2] = [
    ("enable wgpu_binding_array;", &["binding_array"], true),
    ("enable wgpu_int16;", &["i16", "u16"], false),
];

/// Drop the naga-only directives `text` no longer needs.  `input` is the
/// user's source: an extension it never declared is not reinstated, so a
/// tint-targeted shader keeps the stripped body.  Each directive is emitted
/// at most once; one following newline goes with it.
fn strip_naga_only_enables(mut text: String, input: &str) -> String {
    if !NAGA_ONLY_ENABLES.iter().any(|(d, _, _)| text.contains(d)) {
        return text;
    }
    // Both verdicts before the first edit, so the scan borrows `text` once.
    let keep = {
        let cleaned = strip_wgsl_comments(&text);
        let declared = strip_wgsl_comments(input);
        NAGA_ONLY_ENABLES.map(|(directive, tokens, opt_in)| {
            tokens
                .iter()
                .any(|t| cleaned_references_whole_token(&cleaned, t))
                && (!opt_in
                    || cleaned_has_enable_directive(
                        &declared,
                        directive
                            .trim_start_matches("enable ")
                            .trim_end_matches(';'),
                    ))
        })
    };
    for (i, (directive, _, _)) in NAGA_ONLY_ENABLES.into_iter().enumerate() {
        let Some(pos) = text.find(directive) else {
            continue;
        };
        if keep[i] {
            continue;
        }
        let mut end = pos + directive.len();
        if text[end..].starts_with('\n') {
            end += 1;
        }
        text.replace_range(pos..end, "");
    }
    text
}

/// A bailout ships the input body only lexically compacted, leading
/// directives included; with a preamble active the consumer's
/// [preamble, body] concatenation would place them after declarations -
/// invalid WGSL shipped with exit 0.  The generator path strips body
/// directives (the preamble owns them); a bailout has no parser to, so it
/// hard-errors like the f16 preamble guard.
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

/// Compacts the pretty-printed fallback (a naga re-parse guards against a
/// compactor bug; the pretty-but-valid text ships on failure) and drops the
/// naga-only enables like the generator path.  `naga_output` is already
/// validated by the caller.
fn finalize_naga_fallback_text(naga_output: String, input: &str) -> String {
    let compacted = compact_wgsl_text(&naga_output);
    let text = if io::validate_wgsl_text(&compacted).is_ok() {
        compacted
    } else {
        naga_output
    };
    strip_naga_only_enables(text, input)
}

/// Prefix for the preamble self-check: naga needs these to parse the
/// feature, the shipped body omits them and a user preamble legitimately
/// lacks them, so without injection the check would reject a tint-valid
/// body.  Genuine tint-required enables (f16, ...) are deliberately not
/// injected.
fn naga_only_enable_prefix(emit_source: &str) -> String {
    let mut prefix = String::new();
    for (directive, _, _) in NAGA_ONLY_ENABLES {
        if emit_source.contains(directive) {
            prefix.push_str(directive);
            prefix.push('\n');
        }
    }
    prefix
}

// MARK: Public entry points

/// Result of a full minification run; `#[non_exhaustive]` so fields can be
/// added without a major bump (destructure with `..`, construct via [`run`]).
#[non_exhaustive]
pub struct Output {
    /// Minified WGSL.
    pub source: String,
    /// Aggregate report with input/output sizes and per-pass details.
    pub report: Report,
    /// [`name_map::NameMap`], or `None` whenever the shipped text does not
    /// carry the pipeline's renames (bailouts, the verbatim guard, the
    /// naga-emitter fallback): names unchanged from the visible output.
    pub name_map: Option<name_map::NameMap>,
}

/// Apply the IR passes to a parsed module in place; text is emitted only
/// for the report's before/after sizes.  Fails with [`Error::Validation`]
/// on invalid input or an unrecoverable pass failure, [`Error::Emit`] when
/// naga's backend cannot render the final IR.
pub fn run_module(module: &mut naga::Module, config: &Config) -> Result<Report, Error> {
    let info = io::validate_module(module)?;
    let before_wgsl = emit_module_for_report(module, &info, config)?;
    let mut report = Report::new(before_wgsl.len());

    let (_, info) = pipeline::run_ir_passes(module, info, config, &mut report)?;

    let after_wgsl = emit_module_for_report(module, &info, config)?;
    report.output_bytes = after_wgsl.len();

    Ok(report)
}

/// Byte accounting for [`run_module`]: naga's back-end, or nagami's
/// generator for the modules naga would abort on, so the public entry point
/// never crashes on validator-accepted input.
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

/// Every declaration name, module-scope plus types and struct members; a
/// preamble's are hidden from the generator and preserved from renaming.
fn collect_module_names(module: &naga::Module) -> HashSet<String> {
    name_gen::module_scope_names(module)
        .chain(name_gen::type_names(module))
        .chain(name_gen::struct_member_names(module))
        .map(str::to_owned)
        .collect()
}

// MARK: Naga error-message coupling

/// First-line keys of the parse errors naga raises for a directive it
/// declines: `unknown enable-extension`, `unknown language extension`, and
/// the "the `x` ... extension is not {yet supported, enabled, supported in
/// the current environment}" forms.  Every key spans a space because naga
/// quotes user identifiers on that line ("no definition in scope for
/// identifier: `extension_of_life`"); a bare word would file an invalid
/// shader as a bailout.  Coupled to naga's wording: a test parses real
/// shaders so a rewording fails at test time, not as a hard error in the
/// field.
const UNSUPPORTED_EXTENSION_PATTERNS: &[&str] = &[
    "extension is not",
    "unknown enable-extension",
    "unknown language extension",
];

/// Text-validation errors that are naga limitations, not generator bugs:
/// the `subgroups` enable naga's text front-end rejects even though its IR
/// emitter produces it.
const KNOWN_TEXT_VALIDATION_LIMITATION_PATTERNS: &[&str] =
    &["`subgroups` enable-extension is not yet supported"];

/// First line only: naga's codespan rendering quotes user source below the
/// message, where a shader comment could contain a pattern.
fn first_line_matches(err: &Error, patterns: &[&str]) -> bool {
    let msg = err.to_string();
    let first_line = msg.lines().next().unwrap_or("");
    patterns.iter().any(|p| first_line.contains(p))
}

/// `Parse` only: a validation or emit error quoting the same text must stay
/// a hard error, not take the compacted-input bailout.
fn is_unsupported_extension_parse_error(err: &Error) -> bool {
    matches!(err, Error::Parse(_)) && first_line_matches(err, UNSUPPORTED_EXTENSION_PATTERNS)
}

/// `validate_wgsl_text` reports the unsupported `enable` as `Parse` or
/// `Validation` depending on whether tokenisation or semantics trips; no
/// other variant opts into the fallback bypass.
fn is_known_text_validation_limitation(err: &Error) -> bool {
    matches!(err, Error::Parse(_) | Error::Validation(_))
        && first_line_matches(err, KNOWN_TEXT_VALIDATION_LIMITATION_PATTERNS)
}

/// One emission attempt after the fallback ladder; [`run`] may still veto
/// it with the never-grow guard.
struct EmitOutcome {
    source: String,
    /// Output differs from the naga baseline in text or byte count
    /// (vacuously true without a baseline); dead on the fallback arms, the
    /// report masks it with `!rolled_back`.
    changed: bool,
    /// The naga fallback or the compacted input shipped instead of the
    /// generator's text.  Its negation is the report's validation verdict,
    /// not "text validation passed": the known-limitation carve-out ships
    /// generator output despite a failed validation.
    rolled_back: bool,
    /// Generator wall-clock cost; zero when it produced no text.
    duration_us: u64,
    /// Untextable-IR rung only; forwarded into [`Report::bailout`].
    untextable_reason: Option<String>,
}

/// Last rung: generator output and naga fallback both fail text
/// re-validation, i.e. the optimized IR has no valid WGSL spelling (a
/// pass-manufactured `5f/0f`, whose value inf is const-eval-rejected).
/// Ships the input compacted, like the parse/validation bailouts, so a
/// batch run gets correct output instead of a hard error.
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

/// Ship `naga_output` instead of the generator's text.  naga's wgsl-out can
/// emit tokens its own front-end rejects (an `f32(<f64 literal>)` cast, an
/// f16 literal whose `enable f16;` it drops), so re-validate first; a
/// doubly-invalid IR has no WGSL spelling for any consumer and is not
/// necessarily a pass bug, hence the loud degrade to the compacted input.
/// `context` names the rung that got here, for that warning.
fn ship_naga_fallback(
    naga_output: String,
    source: &str,
    before_bytes: usize,
    duration_us: u64,
    context: &str,
) -> EmitOutcome {
    if let Err(ve) = io::validate_wgsl_text(&naga_output) {
        eprintln!(
            "warning: {context}; minified IR cannot round-trip WGSL text ({ve}); \
             shipping the input lexically compacted"
        );
        return untextable_ir_bailout(source, before_bytes, ve.to_string());
    }
    let fallback = finalize_naga_fallback_text(naga_output, source);
    EmitOutcome {
        changed: fallback.len() != before_bytes,
        source: fallback,
        rolled_back: true,
        duration_us,
        untextable_reason: None,
    }
}

/// The fallback ladder: naga's output when the generator errs or emits
/// invalid WGSL, the compacted input when that text fails re-validation
/// too.  With a preamble (`normalized_preamble` / `effective_preamble` are
/// `Some` together) `naga_output` still embeds the preamble's declarations,
/// which the consumer re-prepends, so the error propagates instead.
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
            // Directives must precede all declarations, so with a preamble the
            // body carries none (the preamble owns them); validate the order
            // the consumer ships, [preamble, body], so a directive the preamble
            // omits errors here, not on the GPU.
            let validation_result = if let Some(normalized_preamble) = normalized_preamble {
                // The naga-only enables are injected for this check only; a
                // genuine enable the preamble omits still fails here (or at
                // the f16 guard).
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
                // Keyed on the validator's message, not the output text:
                // forward defence for a naga that rejects its own valid
                // subgroup-builtin round-trip; inert today.
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
                // Before `emitted.source` may be moved.
                let differs_from_baseline = naga_output.as_deref() != Some(emitted.source.as_str());
                let final_source = if has_preamble {
                    // The preamble owns all directives, naga-only ones included.
                    split_directives(&emitted.source).1.to_owned()
                } else {
                    strip_naga_only_enables(emitted.source, source)
                };
                // The preamble owns every directive, so a body whose feature
                // it never declares ships an invalid [preamble, body] document
                // nagami cannot repair (WGSL forbids directives after
                // declarations).  The self-check normalises the preamble with
                // f16 and the naga-only enables injected, which masks exactly
                // this, so each feature the body still uses is checked here.
                if has_preamble {
                    let body = strip_wgsl_comments(&final_source);
                    let preamble = strip_wgsl_comments(effective_preamble.unwrap_or(""));
                    // A binding array is only naga-facing when the source opted
                    // into the extension; otherwise tint consumes it directly
                    // and no directive is due.
                    let needs = [
                        (cleaned_references_f16_token(&body), "f16"),
                        (
                            cleaned_references_whole_token(&body, "binding_array")
                                && cleaned_has_enable_directive(
                                    &strip_wgsl_comments(source),
                                    "wgpu_binding_array",
                                ),
                            "wgpu_binding_array",
                        ),
                    ];
                    for (needed, ext) in needs {
                        if needed && !cleaned_has_enable_directive(&preamble, ext) {
                            return Err(Error::Emit(format!(
                                "shader body requires `enable {ext};` but the preamble \
                                 does not declare it; add `enable {ext};` to the preamble"
                            )));
                        }
                    }
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
                // Unusable fallback: it embeds the preamble's declarations.
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
                Ok(ship_naga_fallback(
                    naga_output,
                    source,
                    before_bytes,
                    emitted.duration_us,
                    "generator output failed text validation",
                ))
            } else {
                // No baseline (naga's writer-abort set) means no fallback emitter.
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
                Ok(ship_naga_fallback(
                    naga_output,
                    source,
                    before_bytes,
                    /*duration_us=*/ 0,
                    &format!("generator emit failed ({e})"),
                ))
            }
            // No fallback (writer-abort set): the generator's error is the
            // only diagnosis.
            None => Err(e),
        },
    }
}

/// Minify `source` end-to-end: parse (with [`Config::preamble`] prepended),
/// optimise, and emit via the custom generator, falling back to naga's
/// emitter when the generator fails or its output does not round-trip -
/// except with a preamble active, where the fallback is unusable and the
/// error propagates.  A directive naga's front-end declines ships the input
/// lexically compacted with [`Report::bailout`] set instead of an error.
/// Fails with [`Error::Parse`], [`Error::Validation`] or [`Error::Emit`]
/// from the underlying stages.
pub fn run(source: &str, config: &Config) -> Result<Output, Error> {
    let normalized_source = preprocess_source_for_naga(source);

    // Empty or whitespace-only preambles collapse to the no-preamble path,
    // one predicate downstream; the normalised preamble is reused by the
    // post-emission self-check.
    let effective_preamble = config.preamble.as_deref().filter(|s| !s.trim().is_empty());
    let normalized_preamble: Option<Cow<'_, str>> =
        effective_preamble.map(preprocess_source_for_naga);
    let (preamble_names, full_source): (HashSet<String>, Cow<'_, str>);
    if let Some(normalized_preamble) = normalized_preamble.as_deref() {
        // The preamble gets the body's preprocessing, or `f16` in a preamble
        // would fail at parse time while the same text in the body succeeds.
        let preamble_module = match io::parse_wgsl_with_path(normalized_preamble, "<preamble>") {
            Ok(m) => m,
            // The consumer's own preamble carries the directive, so the
            // compacted body still concatenates into a shader their compiler
            // accepts.
            Err(e) if is_unsupported_extension_parse_error(&e) => {
                return bailout_output(
                    format!("naga cannot parse the preamble: {e}"),
                    source,
                    effective_preamble,
                    Report::new(source.len()),
                );
            }
            Err(e) => return Err(e),
        };
        preamble_names = collect_module_names(&preamble_module);
        // Directives must precede declarations, so both sides' leading
        // directives go ahead of the preamble body.
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
            // A directive naga declines (`enable subgroups;`, `requires
            // texel_buffers;`): the compacted source still runs on backends
            // that understand it.
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

    // Preamble names must survive rename and mangle: user-source accesses
    // resolve against the consumer's declarations.
    let mut effective_config = config.clone();
    effective_config
        .preserve_symbols
        .extend(preamble_names.iter().cloned());

    // A function ending in a diverging loop is valid WGSL tint/Dawn accept;
    // naga's appended dead return would fail validation.
    strip_front_end_appended_returns(&mut module);

    // naga rejects pointer parameters outside function / private space that
    // WGSL admits (wgpu #5158): whole-variable call sites are specialized
    // away first (naga-valid, smaller output), the rest validates through
    // the stand-in.
    let specialized = passes::specialize_ptr_params::specialize_ptr_params(
        &mut module,
        &effective_config
            .preserve_symbols
            .iter()
            .cloned()
            .collect::<HashSet<_>>(),
    );

    // naga parses some shaders it then rejects at validation (a
    // const-expression division by zero tint/Dawn accept leniently); such
    // input ships compacted rather than breaking a batch run, and validating
    // before the passes makes a post-pass validation failure provably a pass
    // bug.
    let info = match io::validate_module(&module) {
        Ok(info) => info,
        Err(validation_err) => {
            return bailout_output(
                format!("naga rejects the input: {validation_err}"),
                source,
                effective_preamble,
                report,
            );
        }
    };
    if specialized {
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
    }

    let (name_log, info) =
        pipeline::run_ir_passes(&mut module, info, &effective_config, &mut report)?;
    // Without a baseline the byte count falls back to the input length.
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

    // Never ship output larger than the input (an already-minimal file can
    // grow on scaffolding or a loop-exit keep), except under beautify, which
    // grows on purpose, and with a preamble, where the input's leading
    // directives are not a valid substitute in [preamble, body] order.
    // Shipped verbatim, not compacted, so the input must stand on its own:
    // it never went through the emit self-checks, and a shader whose missing
    // `enable` the parse injected grows by exactly that directive - reverting
    // to the input would re-ship text no compiler accepts.
    let (final_source, changed, shipped_input_verbatim) = if !config.beautify
        && !has_preamble
        && final_source.len() > source.len()
        && io::validate_wgsl_text(source).is_ok()
    {
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
