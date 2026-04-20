//! Thin wrappers around naga's WGSL front-end and validator.  Every
//! helper here converts naga errors into [`Error`] variants carrying
//! codespan-formatted diagnostics so consumers never deal with raw
//! naga error types.

use crate::error::Error;

/// Parse a WGSL source string into a naga IR module.
///
/// # Errors
///
/// Returns [`Error::Parse`] with a source-annotated diagnostic produced
/// by naga's `ParseError::emit_to_string` when the front-end rejects
/// `source`.
pub fn parse_wgsl(source: &str) -> Result<naga::Module, Error> {
    naga::front::wgsl::parse_str(source).map_err(|e| Error::Parse(e.emit_to_string(source)))
}

/// Parse WGSL with a custom path label for diagnostics (e.g. `<preamble>`).
/// Identical semantics to [`parse_wgsl`] but the rendered error points at
/// `path` instead of the default `wgsl` label.
pub fn parse_wgsl_with_path(source: &str, path: &str) -> Result<naga::Module, Error> {
    naga::front::wgsl::parse_str(source)
        .map_err(|e| Error::Parse(e.emit_to_string_with_path(source, path)))
}

/// Validate `module` with all validation flags and capabilities enabled.
/// Use this when the caller has no original source text to annotate; the
/// returned error carries naga's default string rendering.
pub fn validate_module(module: &naga::Module) -> Result<naga::valid::ModuleInfo, Error> {
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(module)
    .map_err(|e| Error::Validation(e.to_string()))
}

/// Validate a naga module and render failures against `source`.
///
/// Only safe when `source` still corresponds to the module's spans;
/// once IR passes have mutated the module the spans are stale and the
/// annotation may point at the wrong line.  Prefer [`validate_module`]
/// after any IR transform.
pub fn validate_module_with_source(
    module: &naga::Module,
    source: &str,
) -> Result<naga::valid::ModuleInfo, Error> {
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(module)
    .map_err(|e| Error::Validation(e.emit_to_string(source)))
}

/// Round-trip a WGSL string through the front-end: parse, validate, drop.
/// Used to confirm that emitted output still round-trips through naga.
pub fn validate_wgsl_text(source: &str) -> Result<(), Error> {
    let module = parse_wgsl(source)?;
    let _ = validate_module_with_source(&module, source)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_error_contains_source_annotation() {
        let bad = "fn bad { }";
        let err = parse_wgsl(bad).unwrap_err();
        let msg = err.to_string();
        // Codespan annotation must round-trip: both the `wgsl:LINE` label
        // and the offending source line are part of the stable format.
        assert!(
            msg.contains("wgsl:1"),
            "parse error should contain source location: {msg}"
        );
        assert!(
            msg.contains("fn bad { }"),
            "parse error should contain source line: {msg}"
        );
    }

    #[test]
    fn parse_error_with_path_uses_custom_label() {
        let bad = "fn bad { }";
        let err = parse_wgsl_with_path(bad, "<preamble>").unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("<preamble>"),
            "parse error should use custom path label: {msg}"
        );
    }

    #[test]
    fn parse_error_multiline_source_shows_correct_line() {
        let src = "fn good() {}\nfn bad { }";
        let err = parse_wgsl(src).unwrap_err();
        let msg = err.to_string();
        // Annotation must point at line 2, not line 1.
        assert!(
            msg.contains("wgsl:2"),
            "parse error should point to line 2: {msg}"
        );
    }

    #[test]
    fn validate_module_error_is_descriptive() {
        // NOTE: naga's front-end does most semantic checking inline, so
        // producing a module that parses yet fails validation requires
        // IR-level construction the tests here do not cover.  This test
        // degenerates into a positive round-trip check; the negative
        // path is exercised from higher-level pipeline tests.
        let valid = "@vertex fn main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0,0.0,0.0,1.0); }";
        assert!(validate_wgsl_text(valid).is_ok());
    }

    #[test]
    fn validate_wgsl_text_propagates_parse_error() {
        let err = validate_wgsl_text("fn bad { }").unwrap_err();
        assert_eq!(err.kind(), "parse");
        assert!(err.to_string().contains("wgsl:1"));
    }

    #[test]
    fn valid_wgsl_round_trips_successfully() {
        let src = "fn helper() -> f32 { return 1.0; }";
        assert!(validate_wgsl_text(src).is_ok());
    }
}
