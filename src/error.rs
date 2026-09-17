//! Error type for the minification pipeline; the [`Display`] formats are a
//! stable part of the public surface.

use std::fmt::{Display, Formatter};

/// Errors that can occur during WGSL minification.
///
/// `Parse` carries naga's codespan rendering, which begins with `error: `
/// and quotes the source; `Validation` typically carries naga's message
/// chain on one line, followed by ` --> wgsl:LINE:COLUMN` lines when the
/// module was checked against its text.  [`Display`] adds no prefix to
/// either; `Emit` and `Io` carry bare strings shown as `emit error: {msg}`
/// and `I/O error: {msg}`.  These formats are public surface, parsed
/// verbatim downstream and locked by a snapshot test.
#[derive(Debug)]
pub enum Error {
    /// WGSL source could not be parsed.
    Parse(ParseDiagnostic),
    /// The naga IR failed validation.
    Validation(String),
    /// WGSL code generation (emit) failed.
    Emit(String),
    /// A filesystem or I/O operation failed.
    Io(String),
}

/// A parse failure: naga's rendered diagnostic plus, when it labels a span,
/// the first label's position in the text handed to [`crate::run`] (the
/// directives `run` injects and a spliced preamble are subtracted out;
/// `None` for a span inside either, or a diagnostic without a label).  A
/// preamble that fails to parse on its own is labelled `<preamble>` and
/// positioned in the preamble as `run` preprocessed it, injected directive
/// lines included.  The message renders the text naga parsed, so its own
/// line numbers count any injected directive lines; `location` is the one
/// to show.
#[derive(Debug)]
pub struct ParseDiagnostic {
    /// naga's codespan rendering.
    pub message: String,
    /// 1-based line and byte column.
    pub location: Option<naga::SourceLocation>,
}

impl Error {
    /// Source position of a [`Error::Parse`] label; see [`ParseDiagnostic`]
    /// for which text it counts in.
    pub fn location(&self) -> Option<naga::SourceLocation> {
        match self {
            Error::Parse(d) => d.location,
            _ => None,
        }
    }

    /// Stable category label for log scraping: `parse`, `validation`,
    /// `emit` or `io`.
    pub fn kind(&self) -> &'static str {
        match self {
            Error::Parse(_) => "parse",
            Error::Validation(_) => "validation",
            Error::Emit(_) => "emit",
            Error::Io(_) => "io",
        }
    }

    /// Inner message without the category prefix that [`Display`] may add.
    pub fn message(&self) -> &str {
        match self {
            Error::Parse(d) => &d.message,
            Error::Validation(m) | Error::Emit(m) | Error::Io(m) => m,
        }
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Parse(d) => f.write_str(&d.message),
            Error::Validation(msg) => f.write_str(msg),
            Error::Emit(msg) => write!(f, "emit error: {msg}"),
            Error::Io(msg) => write!(f, "I/O error: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drift in any format or label is a breaking change to the public surface.
    #[test]
    fn display_and_kind_snapshot() {
        let cases = [
            (
                Error::Parse(ParseDiagnostic {
                    message: "X".into(),
                    location: None,
                }),
                "X",
                "parse",
            ),
            (Error::Validation("X".into()), "X", "validation"),
            (Error::Emit("X".into()), "emit error: X", "emit"),
            (Error::Io("X".into()), "I/O error: X", "io"),
        ];
        for (err, display, kind) in cases {
            assert_eq!(err.to_string(), display);
            assert_eq!(err.kind(), kind);
            assert_eq!(err.message(), "X");
        }
    }

    #[test]
    fn from_io_error() {
        let e = Error::from(std::io::Error::new(std::io::ErrorKind::NotFound, "gone"));
        assert_eq!(e.kind(), "io");
        assert!(e.message().contains("gone"));
        let _: &dyn std::error::Error = &e;
    }
}
