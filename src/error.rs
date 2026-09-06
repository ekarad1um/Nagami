//! Error type for the WGSL minification pipeline.  Every public entry
//! point funnels failures into [`Error`]; the variants here are the
//! canonical taxonomy and their [`Display`] formats are a stable part
//! of the crate's public surface.

use std::fmt::{Display, Formatter};

/// Errors that can occur during WGSL minification.
///
/// Each variant stores a fully-formatted, self-describing message.
/// [`Error::Parse`] and [`Error::Validation`] may carry multi-line
/// source-annotated codespan diagnostics produced via naga's
/// `emit_to_string`; the remaining variants carry bare strings.
///
/// # Display convention
///
/// The [`Display`] impl deliberately treats variants asymmetrically:
///
/// | Variant      | `Display` output format                |
/// |--------------|----------------------------------------|
/// | `Parse`      | `{msg}` (no prefix)                    |
/// | `Validation` | `{msg}` (no prefix)                    |
/// | `Emit`       | `emit error: {msg}`                    |
/// | `Io`         | `I/O error: {msg}`                     |
///
/// `Parse` and `Validation` messages already begin with `error: ` and
/// embed source context, so adding a category prefix would produce
/// redundant noise (e.g. `parse: error: bad token ...`).  The other
/// variants carry bare strings that benefit from an explicit prefix.
///
/// NOTE: these formats are part of the public surface.  Downstream
/// consumers (the CLI's stderr printer, log scrapers) parse them
/// verbatim.  The snapshot test in this module locks each variant's
/// output so an incidental cleanup cannot silently renormalise them.
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
    /// Short, stable category label suitable for log scraping.
    /// One of `"parse"`, `"validation"`, `"emit"`, `"io"`.
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
        // Parse/Validation messages already begin with "error: " and embed
        // codespan context, so they are emitted verbatim.  Other variants
        // prepend a category prefix.  See the type-level doc for the full
        // format table and its stability guarantee.
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

    /// Snapshot lock for every `Display` format and `kind` label: drift is a
    /// breaking change to the public surface described on [`Error`].
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
