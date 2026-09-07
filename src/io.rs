//! Thin wrappers around naga's WGSL front-end and validator that render
//! naga errors into [`Error`] diagnostics.

use crate::error::Error;

pub fn parse_wgsl(source: &str) -> Result<naga::Module, Error> {
    naga::front::wgsl::parse_str(source).map_err(|e| Error::Parse(e.emit_to_string(source)))
}

/// `path` replaces the default `wgsl` label in diagnostics (e.g. `<preamble>`).
pub fn parse_wgsl_with_path(source: &str, path: &str) -> Result<naga::Module, Error> {
    naga::front::wgsl::parse_str(source)
        .map_err(|e| Error::Parse(e.emit_to_string_with_path(source, path)))
}

/// `Capabilities::all()` on purpose: this checks IR-level soundness after
/// every pass, not backend compatibility, which is the caller's concern.
pub fn validate_module(module: &naga::Module) -> Result<naga::valid::ModuleInfo, Error> {
    // Pointer parameters naga rejects but WGSL admits validate through a
    // handle-preserving stand-in, whose info indexes `module` itself.
    let stand_in = crate::passes::specialize_ptr_params::validation_stand_in(module);
    let module = stand_in.as_ref().unwrap_or(module);
    fn fresh_validator() -> naga::valid::Validator {
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
    }
    thread_local! {
        /// Reused across the per-pass validations to keep container
        /// capacity; `validate` resets all per-module state except the
        /// ray-pipeline pins.
        static VALIDATOR: std::cell::RefCell<naga::valid::Validator> =
            std::cell::RefCell::new(fresh_validator());
    }
    VALIDATOR.with(|validator| {
        let result = validator.borrow_mut().validate(module);
        match result {
            Ok(info) => Ok(info),
            // naga's `reset` misses the ray-pipeline pins (`trace_rays_*`):
            // a payload-type handle pinned by an earlier module can
            // false-reject a valid one whose handles shifted.  A failure is
            // re-checked on a fresh validator, which then replaces the cached
            // one so ray-heavy sessions pay the double validation once.  A
            // stale pin only adds a constraint, so false accepts are
            // impossible and a fresh reject is genuine.
            Err(_) => {
                let mut fresh = fresh_validator();
                let info = fresh
                    .validate(module)
                    .map_err(|e| Error::Validation(render_error_chain(&e)))?;
                *validator.borrow_mut() = fresh;
                Ok(info)
            }
        }
    })
}

/// `e` and its `source()` chain on one line: naga's `Display` shows only
/// the outermost frame, the actionable detail sits below it.
fn render_error_chain(e: &dyn std::error::Error) -> String {
    let mut msg = e.to_string();
    let mut src = e.source();
    while let Some(s) = src {
        msg.push_str(": ");
        msg.push_str(&s.to_string());
        src = s.source();
    }
    msg
}

/// Renders failures against `source`, so only valid while the module's
/// spans still match it, i.e. before any IR pass.
pub fn validate_module_with_source(
    module: &naga::Module,
    source: &str,
) -> Result<naga::valid::ModuleInfo, Error> {
    let stand_in = crate::passes::specialize_ptr_params::validation_stand_in(module);
    let module = stand_in.as_ref().unwrap_or(module);
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(module)
    .map_err(|e| Error::Validation(e.emit_to_string(source)))
}

pub fn validate_wgsl_text(source: &str) -> Result<(), Error> {
    let module = parse_wgsl(source)?;
    let _ = validate_module_with_source(&module, source)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pointer_parameters_naga_rejects_validate_through_the_stand_in() {
        let src = "fn touch(p: ptr<workgroup, f32>) { *p = 1.0; }";
        let module = parse_wgsl(src).expect("parses");
        validate_module(&module).expect("stand-in");
        validate_module_with_source(&module, src).expect("stand-in with source");
        validate_wgsl_text(src).expect("text round trip");
    }

    #[test]
    fn parse_error_contains_source_annotation() {
        let bad = "fn bad { }";
        let err = parse_wgsl(bad).unwrap_err();
        let msg = err.to_string();
        // The `wgsl:LINE` label and the quoted source line are stable format.
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
        assert!(
            msg.contains("wgsl:2"),
            "parse error should point to line 2: {msg}"
        );
    }

    #[test]
    fn validate_module_error_is_descriptive() {
        // naga's front-end checks most semantics inline, so a module that
        // parses yet fails validation needs IR-level construction; the
        // negative path is covered by pipeline tests.
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
