//! Crate root: [`run_module`] optimises a pre-parsed [`naga::Module`] and
//! returns a [`Report`]; [`run`] is source-to-source: preprocess, parse,
//! optimise, emit via the custom generator with a fallback ladder down to
//! naga's own emitter and the lexically compacted input.

/// The IR crate [`run_module`] takes; re-exported so a consumer builds its
/// module with the version this crate validates against.
pub use naga;

pub(crate) mod analysis;
pub mod config;
pub mod error;
pub mod generator;
pub(crate) mod handle_set;
mod io;
pub(crate) mod ir;
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

/// Map a parse label from the text naga parsed back to `source`.  The parsed
/// text is `normalized_source` (`source` behind the injected directive lines,
/// same length otherwise) with, when a preamble is active, the preamble
/// spliced between the source's directives and its body starting at
/// `body_start` (`0`: no preamble, the texts coincide).  A label inside the
/// injected or preamble text has no position in `source`.
fn relocate_parse_label(
    mut err: Error,
    body_start: usize,
    normalized_source: &str,
    source: &str,
) -> Error {
    let Error::Parse(diag) = &mut err else {
        return err;
    };
    let Some(loc) = diag.location else {
        return err;
    };
    let offset = loc.offset as usize;
    let (source_directives, source_body) = split_directives(normalized_source);
    let in_normalized = if body_start == 0 || offset < source_directives.len() {
        Some(offset)
    } else if offset >= body_start {
        Some(offset - body_start + normalized_source.len() - source_body.len())
    } else {
        None
    };
    let injected = normalized_source.len() - source.len();
    // Resolved on the normalized text (same length as `source`, lone CRs
    // already LF): `Span::location` counts only `\n`.  A label past the end
    // sits on the newline the preamble splice appended after a body without
    // one, so it clamps to the end rather than dropping.
    let normalized = &normalized_source[injected..];
    diag.location = in_normalized
        .and_then(|o| o.checked_sub(injected))
        .map(|start| {
            let start = start.min(normalized.len());
            let end = (start + loc.length as usize).min(normalized.len());
            naga::Span::new(start as u32, end as u32).location(normalized)
        });
    err
}

/// naga's front-end appends `Return { value: None }` after a `loop` tail
/// (it never proves a loop non-falling-through), which the validator
/// rejects as `InvalidReturnType` in a non-void function.  Removing it only
/// where the tail provably diverges is semantics-preserving, and the
/// generator re-synthesises the return on the same predicate, so the
/// round-trip is unchanged.
///
/// `naga::proc::ensure_block_returns` descends into the LAST statement's
/// nested blocks - a `Block`, both arms of an `If`, every non-fall-through
/// `Switch` case - and appends there, so a diverging loop inside a branch
/// carries the dead return inside the arm, never at the function tail.
/// Mirrored exactly: stripping only the tail left those modules failing
/// validation on their own INPUT and bailing out to lexical compaction.
fn strip_front_end_appended_returns(module: &mut naga::Module) {
    /// The blocks `ensure_block_returns` would have appended to, in its own
    /// order: the tail first, then whatever the new tail nests.
    fn strip_block(block: &mut naga::Block) {
        if matches!(block.last(), Some(naga::Statement::Return { value: None })) {
            // Nothing before it means no diverging construct to justify the
            // removal; a bare `return;` is invalid WGSL in a non-void
            // function, so what is left is always naga's.
            if block.len() < 2 {
                return;
            }
            let last_span = block
                .span_iter()
                .last()
                .map_or(naga::Span::UNDEFINED, |(_, s)| *s);
            let len = block.len();
            block.cull(len - 1..);
            strip_nested(block);
            // Restore the return when the new tail does not provably diverge:
            // a genuine fall-through stays invalid and takes the bailout.
            if !crate::passes::dead_branch::block_definitely_terminates(block) {
                block.push(naga::Statement::Return { value: None }, last_span);
            }
            return;
        }
        strip_nested(block);
    }

    fn strip_nested(block: &mut naga::Block) {
        match block.last_mut() {
            Some(naga::Statement::Block(inner)) => strip_block(inner),
            Some(naga::Statement::If { accept, reject, .. }) => {
                strip_block(accept);
                strip_block(reject);
            }
            Some(naga::Statement::Switch { cases, .. }) => {
                for case in cases.iter_mut() {
                    if !case.fall_through {
                        strip_block(&mut case.body);
                    }
                }
            }
            _ => {}
        }
    }

    fn strip(func: &mut naga::Function) {
        if func.result.is_none() {
            return; // void function: a bare `return;` tail is legitimate
        }
        strip_block(&mut func.body);
    }
    ir::visit::for_each_function_mut(&mut module.functions, &mut module.entry_points, &mut strip);
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
    /// [`name_map::NameMap`]; `None` when the shipped text carries no rename
    /// (bailouts, the verbatim guard).  The naga-emitter fallback gets a map
    /// spelled as that emitter prints.
    pub name_map: Option<name_map::NameMap>,
}

/// Apply the IR passes to a parsed module in place; text is emitted only
/// for the report's before/after sizes.  Fails with [`Error::Validation`]
/// on invalid input or an unrecoverable pass failure, [`Error::Emit`] when
/// naga's backend cannot render the final IR.
pub fn run_module(module: &mut naga::Module, config: &Config) -> Result<Report, Error> {
    let info = io::validate_module(module)?;
    let mut effective_config = Cow::Borrowed(config);
    if config.preserve_interface {
        let (symbols, _) = interface_names(module);
        effective_config.to_mut().preserve_symbols.extend(symbols);
    }
    let config = effective_config.as_ref();
    let before_wgsl = emit_module_for_report(module, &info, config)?;
    let mut report = Report::new(before_wgsl.len());

    let info = pipeline::run_ir_passes(module, info, config, &mut report, HashSet::new())?.info;

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
        let emitted = generate(module, info, GenerateOptions::from_config(config))?;
        Ok(emitted.source)
    } else {
        emit_wgsl_with_naga_safe(module, info)
    }
}

/// A preamble's declaration names (module scope plus types), which the
/// generator hides from the output, and its struct member names.  Both are
/// preserved from renaming; only the first can stand for "owned by the
/// preamble", since a member name lives in its struct's own namespace and
/// says nothing about a same-named module-scope declaration.
fn collect_module_names(module: &naga::Module) -> (HashSet<String>, Vec<String>) {
    let declarations = name_gen::module_scope_names(module)
        .chain(name_gen::type_names(module))
        .map(str::to_owned)
        .collect();
    let members = name_gen::struct_member_names(module)
        .map(str::to_owned)
        .collect();
    (declarations, members)
}

/// The names [`Config::preserve_interface`] keeps, split into declarations
/// and struct members as [`collect_module_names`] splits a preamble's.
pub(crate) fn interface_names(module: &naga::Module) -> (Vec<String>, Vec<String>) {
    let mut symbols: Vec<String> = Vec::new();
    let mut members = Vec::new();
    let mut pending: Vec<naga::Handle<naga::Type>> = Vec::new();
    for (_, gv) in module.global_variables.iter() {
        if gv.binding.is_some() {
            symbols.extend(gv.name.clone());
            pending.push(gv.ty);
        }
    }
    symbols.extend(module.overrides.iter().filter_map(|(_, o)| o.name.clone()));
    let mut seen = vec![false; module.types.len()];
    while let Some(ty) = pending.pop() {
        if std::mem::replace(&mut seen[ty.index()], true) {
            continue;
        }
        match &module.types[ty].inner {
            naga::TypeInner::Struct {
                members: fields, ..
            } => {
                symbols.extend(module.types[ty].name.clone());
                for field in fields {
                    members.extend(field.name.clone());
                    pending.push(field.ty);
                }
            }
            naga::TypeInner::Array { base, .. }
            | naga::TypeInner::BindingArray { base, .. }
            | naga::TypeInner::Pointer { base, .. } => pending.push(*base),
            _ => {}
        }
    }
    (symbols, members)
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
    /// naga's emitter printed the (renamed) module; the reason, forwarded
    /// into [`Report::fallback`].  The name map is built from that emitter's
    /// spelling.
    naga_fallback: Option<String>,
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
        naga_fallback: None,
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
        naga_fallback: Some(context.to_string()),
    }
}

/// The fallback ladder: naga's output when the generator errs or emits
/// invalid WGSL, the compacted input when that fails re-validation too.
/// `fallback_blocked` names why naga's text cannot stand in; then the error
/// propagates.
#[allow(clippy::too_many_arguments)]
fn resolve_generator_output(
    gen_result: Result<generator::Emission, Error>,
    naga_output: Option<String>,
    normalized_preamble: Option<&str>,
    effective_preamble: Option<&str>,
    source: &str,
    before_bytes: usize,
    trace_enabled: bool,
    fallback_blocked: &dyn Fn() -> Option<String>,
) -> Result<EmitOutcome, Error> {
    let has_preamble = effective_preamble.is_some();
    match gen_result {
        Ok(emitted) => {
            // The preamble owns every directive, so a body whose feature it
            // never declares ships an invalid [preamble, body] document nagami
            // cannot repair (WGSL forbids directives after declarations).
            // Named before the self-check below, whose failure on the same
            // omission would only quote naga's "extension is not enabled".
            if let Some(preamble) = effective_preamble {
                let body = strip_wgsl_comments(split_directives(&emitted.source).1);
                let preamble = strip_wgsl_comments(preamble);
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
            if validation_result.is_ok() {
                // Before `emitted.source` may be moved.
                let differs_from_baseline = naga_output.as_deref() != Some(emitted.source.as_str());
                let final_source = if has_preamble {
                    // The preamble owns all directives, naga-only ones
                    // included.  A module-level `diagnostic(...)` only the
                    // body declared would vanish with them - unseen by the
                    // self-check, whose validator has no uniformity analysis
                    // - so each emitted one must be among the preamble's own
                    // directives, blankspace aside (a `@diagnostic` attribute
                    // on a preamble function is not one).
                    let (directives, body) = split_directives(&emitted.source);
                    let squeeze = |s: &str| s.split(text::is_wgsl_blankspace).collect::<String>();
                    let preamble = strip_wgsl_comments(effective_preamble.unwrap_or(""));
                    let owned: Vec<String> = split_directives(&preamble)
                        .0
                        .split(';')
                        .map(squeeze)
                        .collect();
                    for directive in directives.split(';').map(squeeze) {
                        if directive.starts_with("diagnostic") && !owned.contains(&directive) {
                            return Err(Error::Emit(format!(
                                "shader body declares `{directive};` but the preamble does \
                                 not; move the directive into the preamble"
                            )));
                        }
                    }
                    body.to_owned()
                } else {
                    strip_naga_only_enables(emitted.source, source)
                };
                let changed = before_bytes != final_source.len() || differs_from_baseline;
                Ok(EmitOutcome {
                    source: final_source,
                    changed,
                    rolled_back: false,
                    duration_us: emitted.duration_us,
                    untextable_reason: None,
                    naga_fallback: None,
                })
            } else if let Some(why) = fallback_blocked() {
                let underlying = match validation_result {
                    Err(e) => e.to_string(),
                    Ok(()) => "(no underlying error)".to_string(),
                };
                Err(Error::Emit(format!(
                    "generator output failed validation; cannot fall back safely: {why}: \
                     {underlying}",
                )))
            } else if let Some(naga_output) = naga_output {
                // The CLI warns from `Report::fallback`; the diagnostic
                // itself stays trace-gated.
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
                if let Some(why) = fallback_blocked() {
                    return Err(Error::Emit(format!("{e}; cannot fall back safely: {why}")));
                }
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

/// The text handed to naga, with the preamble's [`collect_module_names`]
/// lists (empty without one).
struct SplicedSource<'s> {
    preamble_names: HashSet<String>,
    preamble_members: Vec<String>,
    full_source: Cow<'s, str>,
    /// Byte offset of the body inside `full_source`; [`relocate_parse_label`]
    /// maps a parse position back through it.
    body_start: usize,
}

/// Splice `normalized_preamble` ahead of `normalized_source`, both sides'
/// leading directives hoisted ahead of it per [`split_directives`], or pass
/// the source through when there is none.
fn splice_preamble<'s>(
    normalized_source: &Cow<'s, str>,
    normalized_preamble: Option<&str>,
) -> Result<SplicedSource<'s>, Error> {
    let Some(normalized_preamble) = normalized_preamble else {
        return Ok(SplicedSource {
            preamble_names: HashSet::new(),
            preamble_members: Vec::new(),
            full_source: normalized_source.clone(),
            body_start: 0,
        });
    };
    let preamble_module = io::parse_wgsl_with_path(normalized_preamble, "<preamble>")?;
    let (preamble_names, preamble_members) = collect_module_names(&preamble_module);
    let (source_directives, source_body) = split_directives(normalized_source);
    let (preamble_directives, preamble_body) = split_directives(normalized_preamble);
    let full_source = join_with_newline(&[
        source_directives,
        preamble_directives,
        preamble_body,
        source_body,
    ]);
    // The body is the last fragment, plus the newline the join adds.
    let body_start = full_source.len()
        - source_body.len()
        - usize::from(!source_body.is_empty() && !source_body.ends_with('\n'));
    Ok(SplicedSource {
        preamble_names,
        preamble_members,
        full_source: Cow::Owned(full_source),
        body_start,
    })
}

/// Minify `source` end-to-end: parse (with [`Config::preamble`] prepended),
/// optimise, and emit via the custom generator, falling back to naga's
/// emitter when the generator fails or its output does not round-trip -
/// unless naga's text cannot stand in (a preamble is active, or it would
/// respell a host-visible name), when the error propagates instead.  Input
/// naga parses but rejects, and IR with no WGSL text form, ship the input
/// lexically compacted with [`Report::bailout`] set; both are typed outcomes
/// of a naga call, never a match on its message.  Fails with
/// [`Error::Parse`], [`Error::Validation`] or [`Error::Emit`] from the
/// underlying stages.
pub fn run(source: &str, config: &Config) -> Result<Output, Error> {
    let normalized_source = preprocess_source_for_naga(source);

    // Empty or whitespace-only preambles collapse to the no-preamble path,
    // one predicate downstream.  The preamble gets the body's preprocessing
    // (`f16` in it would otherwise fail at parse time where the same text
    // in the body succeeds); the normalised text is reused by the
    // post-emission self-check.
    let effective_preamble = config.preamble.as_deref().filter(|s| !s.trim().is_empty());
    let normalized_preamble: Option<Cow<'_, str>> =
        effective_preamble.map(preprocess_source_for_naga);
    let SplicedSource {
        preamble_names,
        preamble_members,
        full_source,
        body_start,
    } = splice_preamble(&normalized_source, normalized_preamble.as_deref())?;

    let mut module = match io::parse_wgsl(&full_source) {
        Ok(module) => module,
        Err(err) => {
            return Err(relocate_parse_label(
                err,
                body_start,
                &normalized_source,
                source,
            ));
        }
    };
    let mut report = Report::new(source.len());

    // Preamble names must survive rename and mangle: user-source accesses
    // resolve against the consumer's declarations.
    let mut effective_config = config.clone();
    effective_config
        .preserve_symbols
        .extend(preamble_names.iter().cloned());
    let mut preserve_members = preamble_members;
    if config.preserve_interface {
        let (symbols, members) = interface_names(&module);
        effective_config.preserve_symbols.extend(symbols);
        preserve_members.extend(members);
    }
    let preserve_members: Vec<String> = preserve_members.into_iter().collect();

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

    let pipeline::Converged {
        name_log,
        info,
        type_uses,
    } = pipeline::run_ir_passes(
        &mut module,
        info,
        &effective_config,
        &mut report,
        preamble_names.clone(),
    )?;
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
            preserve_members: preserve_members.iter().cloned().collect(),
            preamble_names,
            type_uses,
            ..GenerateOptions::from_config(&effective_config)
        },
    );

    let has_preamble = effective_preamble.is_some();
    // naga's text cannot stand in with a preamble (it embeds the preamble's
    // declarations, which the consumer re-prepends) or when it would respell
    // a host-visible name.  Evaluated only on a generator failure.
    let fallback_blocked = || -> Option<String> {
        if has_preamble {
            return Some("a preamble is active (the fallback would embed its declarations)".into());
        }
        name_map::naga_respelled_interface_name(
            &module,
            &effective_config.preserve_symbols,
            &preserve_members,
        )
        .map(|(ir, printed)| format!("naga's emitter would print `{ir}` as `{printed}`"))
    };

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
        naga_fallback,
    } = resolve_generator_output(
        gen_result,
        naga_output,
        normalized_preamble.as_deref(),
        effective_preamble,
        source,
        before_bytes,
        config.trace.enabled,
        &fallback_blocked,
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

    // The verbatim input carries neither the fallback's text nor any rename.
    let naga_fallback = naga_fallback.filter(|_| !shipped_input_verbatim);
    report.fallback = naga_fallback.clone();
    let name_map = if naga_fallback.is_some() {
        Some(name_map::NameMap::assemble_naga_spelled(&module, &name_log))
    } else if !rolled_back && !shipped_input_verbatim {
        let (structs, live_const_names) = gen_name_tables.unwrap_or_default();
        Some(name_map::NameMap::assemble(
            &module,
            &name_log,
            structs,
            Some(&live_const_names),
            &|ir, _| ir.to_owned(),
        ))
    } else {
        None
    };

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
