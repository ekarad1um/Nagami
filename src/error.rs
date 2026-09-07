//! Error type for the minification pipeline; the [`Display`] formats are a
//! stable part of the public surface.

use std::fmt::{Display, Formatter};

/// Errors that can occur during WGSL minification.
///
/// `Parse` and `Validation` carry naga diagnostics, typically codespan
/// renderings that already begin with `error: ` and embed source context,
/// so [`Display`] adds no prefix; `Emit` and `Io` carry bare strings shown
/// as `emit error: {msg}` and `I/O error: {msg}`.  These formats are public
/// surface, parsed verbatim downstream and locked by a snapshot test.
#[derive(Debug)]
pub enum Error {
    /// WGSL source could not be parsed.
    Parse(String),
    /// The naga IR failed validation.
    Validation(String),
    /// WGSL code generation (emit) failed.
    Emit(String),
    /// A filesystem or I/O operation failed.
    Io(String),
}

impl Error {
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
            Error::Parse(m) | Error::Validation(m) | Error::Emit(m) | Error::Io(m) => m,
        }
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Parse(msg) | Error::Validation(msg) => f.write_str(msg),
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
            (Error::Parse("X".into()), "X", "parse"),
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
