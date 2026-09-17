//! Thin wrappers around naga's WGSL front-end and validator that render
//! naga errors into [`Error`] diagnostics.

use std::fmt::Write;

use crate::error::{Error, ParseDiagnostic};
use naga::valid::{
    CallError, ConstExpressionError, EntryPointError, ExpressionError, FunctionError,
    GlobalVariableError, OverrideError, ValidationError,
};

/// The location is naga's, relative to `source` as parsed; [`crate::run`]
/// maps it back to the caller's text.
pub fn parse_wgsl(source: &str) -> Result<naga::Module, Error> {
    naga::front::wgsl::parse_str(source).map_err(|e| {
        Error::Parse(ParseDiagnostic {
            message: e.emit_to_string(source),
            location: e.location(source),
        })
    })
}

/// `path` replaces the default `wgsl` label in diagnostics (e.g. `<preamble>`).
pub fn parse_wgsl_with_path(source: &str, path: &str) -> Result<naga::Module, Error> {
    naga::front::wgsl::parse_str(source).map_err(|e| {
        Error::Parse(ParseDiagnostic {
            message: e.emit_to_string_with_path(source, path),
            location: e.location(source),
        })
    })
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
            Ok(info) => match crate::passes::const_fold::module_static_error_slot(module) {
                Some(detail) => Err(Error::Validation(detail)),
                None => Ok(info),
            },
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
                    .map_err(|e| Error::Validation(render_validation_error(&e, None)))?;
                *validator.borrow_mut() = fresh;
                Ok(info)
            }
        }
    })
}

/// naga's `Display` is the outermost frame alone; the actionable detail
/// sits in the `source` chain, joined here on one line.  The chain is
/// walked by type: a generic walk over `dyn Error` would link, through the
/// vtables, the derived `Debug` of naga's whole error tree, which nothing
/// prints.  A future naga variant whose `source` this walk does not name
/// loses that frame from the message, nothing more.  With `source`, each
/// labelled span follows on its own ` --> wgsl:LINE:COLUMN LABEL` line.
fn render_validation_error(e: &naga::WithSpan<ValidationError>, source: Option<&str>) -> String {
    let mut msg = String::new();
    validation_frames(&mut msg, e.as_inner());
    if let Some(source) = source {
        for (span, label) in e.spans() {
            let at = span.location(source);
            let _ = write!(
                msg,
                "\n  --> wgsl:{}:{} {label}",
                at.line_number, at.line_position
            );
        }
    }
    msg
}

fn frame(msg: &mut String, text: &dyn std::fmt::Display) {
    if !msg.is_empty() {
        msg.push_str(": ");
    }
    let _ = write!(msg, "{text}");
}

/// Exhaustive at this level so a new naga variant is placed on purpose; a
/// transparent wrapper or a leaf is whole in its own `Display`.
fn validation_frames(msg: &mut String, e: &ValidationError) {
    frame(msg, e);
    match e {
        ValidationError::Type { source, .. } => frame(msg, source),
        ValidationError::ConstExpression { source, .. } => const_expression_frames(msg, source),
        ValidationError::Constant { source, .. } => frame(msg, source),
        ValidationError::Override { source, .. } => {
            frame(msg, source);
            if let OverrideError::ConstExpression { source, .. } = source {
                const_expression_frames(msg, source);
            }
        }
        ValidationError::GlobalVariable { source, .. } => {
            frame(msg, source);
            match source {
                GlobalVariableError::Alignment(_, _, why) => frame(msg, why),
                GlobalVariableError::InvalidImmediateType(why) => frame(msg, why),
                _ => {}
            }
        }
        ValidationError::Function { source, .. }
        | ValidationError::EntryPoint {
            source: EntryPointError::Function(source),
            ..
        } => function_frames(msg, source),
        ValidationError::EntryPoint { source, .. } => {
            frame(msg, source);
            if let EntryPointError::Argument(_, why) = source {
                frame(msg, why);
            }
        }
        ValidationError::InvalidHandle(_)
        | ValidationError::Layouter(_)
        | ValidationError::ArraySizeError { .. }
        | ValidationError::Corrupted => {}
    }
}

fn function_frames(msg: &mut String, e: &FunctionError) {
    frame(msg, e);
    match e {
        FunctionError::Expression { source, .. }
        | FunctionError::InvalidImageStore(source)
        | FunctionError::InvalidImageAtomic(source) => expression_frames(msg, source),
        FunctionError::LocalVariable { source, .. } => frame(msg, source),
        FunctionError::InvalidAtomic(source) => frame(msg, source),
        FunctionError::InvalidSubgroup(source) => frame(msg, source),
        FunctionError::InvalidCall { error, .. } => {
            frame(msg, error);
            if let CallError::Argument { source, .. } = error {
                expression_frames(msg, source);
            }
        }
        _ => {}
    }
}

fn expression_frames(msg: &mut String, e: &ExpressionError) {
    frame(msg, e);
    if let ExpressionError::Type(source) = e {
        frame(msg, source);
    }
}

fn const_expression_frames(msg: &mut String, e: &ConstExpressionError) {
    frame(msg, e);
    if let ConstExpressionError::Type(source) = e {
        frame(msg, source);
    }
}

/// Positions failures in `source`, so only valid while the module's spans
/// still match it, i.e. before any IR pass.
pub fn validate_module_with_source(
    module: &naga::Module,
    source: &str,
) -> Result<naga::valid::ModuleInfo, Error> {
    let stand_in = crate::passes::specialize_ptr_params::validation_stand_in(module);
    let module = stand_in.as_ref().unwrap_or(module);
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(module)
    .map_err(|e| Error::Validation(render_validation_error(&e, Some(source))))?;
    match crate::passes::const_fold::module_static_error_slot(module) {
        Some(detail) => Err(Error::Validation(detail)),
        None => Ok(info),
    }
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

    /// The typed walk shows every frame a `dyn Error` walk would, on the
    /// deepest shapes in naga's tree; the generic walk lives only here, so
    /// the release binary never links the tree's `Debug`.
    #[test]
    fn validation_chain_matches_the_error_source_chain() {
        use naga::proc::{LayoutError, LayoutErrorInner, ResolveError};
        use naga::valid::{
            Disalignment, LiteralError, LocalVariableError, TypeError, VaryingError, WidthError,
        };

        fn generic(e: &dyn std::error::Error) -> String {
            let mut msg = e.to_string();
            let mut source = e.source();
            while let Some(s) = source {
                msg.push_str(": ");
                msg.push_str(&s.to_string());
                source = s.source();
            }
            msg
        }
        let module = parse_wgsl(
            "override o: f32; var<private> g: f32; fn f() -> f32 { var l = o; return l; }",
        )
        .expect("parses");
        let function = module.functions.iter().next().expect("f").0;
        let body = &module.functions[function];
        let expression = body.expressions.iter().next().expect("an expression").0;
        let local = body.local_variables.iter().next().expect("l").0;
        let ty = module.types.iter().next().expect("f32").0;
        let over = module.overrides.iter().next().expect("o").0;
        let global = module.global_variables.iter().next().expect("g").0;
        let name = || "n".to_string();
        let resolve = || ResolveError::FunctionReturnsVoid;
        let cases = [
            ValidationError::EntryPoint {
                stage: naga::ShaderStage::Compute,
                name: name(),
                source: EntryPointError::Function(FunctionError::InvalidCall {
                    function,
                    error: CallError::Argument {
                        index: 1,
                        source: ExpressionError::Type(resolve()),
                    },
                }),
            },
            ValidationError::Function {
                handle: function,
                name: name(),
                source: FunctionError::InvalidImageStore(ExpressionError::Literal(
                    LiteralError::Width(WidthError::Abstract),
                )),
            },
            ValidationError::Function {
                handle: function,
                name: name(),
                source: FunctionError::LocalVariable {
                    handle: local,
                    name: name(),
                    source: LocalVariableError::InitializerType,
                },
            },
            ValidationError::EntryPoint {
                stage: naga::ShaderStage::Fragment,
                name: name(),
                source: EntryPointError::Argument(0, VaryingError::InvalidType(ty)),
            },
            ValidationError::Override {
                handle: over,
                name: name(),
                source: OverrideError::ConstExpression {
                    handle: expression,
                    source: ConstExpressionError::Type(resolve()),
                },
            },
            ValidationError::GlobalVariable {
                handle: global,
                name: name(),
                source: GlobalVariableError::Alignment(
                    naga::AddressSpace::Uniform,
                    ty,
                    Disalignment::NonHostShareable,
                ),
            },
            ValidationError::Type {
                handle: ty,
                name: name(),
                source: TypeError::WidthError(WidthError::Abstract),
            },
            ValidationError::Layouter(LayoutError {
                ty,
                inner: LayoutErrorInner::TooLarge,
            }),
            ValidationError::Corrupted,
        ];
        for e in cases {
            let expected = generic(&e);
            assert!(expected.contains(": ") || matches!(e, ValidationError::Corrupted));
            assert_eq!(
                render_validation_error(&naga::WithSpan::new(e), None),
                expected
            );
        }
    }

    #[test]
    fn validation_error_against_source_names_the_position() {
        let src = "@compute @workgroup_size(1) fn m() {\n  var x = 1;\n  let d = x / 0;\n}";
        let module = parse_wgsl(src).expect("parses");
        let positioned = validate_module_with_source(&module, src)
            .unwrap_err()
            .to_string();
        assert!(
            positioned.starts_with("Entry point m at Compute is invalid: ")
                && positioned.contains(": Division by zero\n  --> wgsl:3:11 "),
            "{positioned}"
        );
        let bare = validate_module(&module).unwrap_err().to_string();
        assert_eq!(bare, positioned.split('\n').next().unwrap());
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
