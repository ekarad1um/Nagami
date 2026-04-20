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
/// | `Config`     | `configuration error: {msg}`           |
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
    /// An invalid configuration was provided.
    Config(String),
}

impl Error {
    /// Short, stable category label suitable for log scraping.
    /// One of `"parse"`, `"validation"`, `"emit"`, `"io"`, `"config"`.
    pub fn kind(&self) -> &'static str {
        match self {
            Error::Parse(_) => "parse",
            Error::Validation(_) => "validation",
            Error::Emit(_) => "emit",
            Error::Io(_) => "io",
            Error::Config(_) => "config",
        }
    }

    /// Inner message without the category prefix that [`Display`] may add.
    pub fn message(&self) -> &str {
        match self {
            Error::Parse(m)
            | Error::Validation(m)
            | Error::Emit(m)
            | Error::Io(m)
            | Error::Config(m) => m,
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
            Error::Config(msg) => write!(f, "configuration error: {msg}"),
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

    #[test]
    fn kind_returns_correct_label() {
        assert_eq!(Error::Parse("x".into()).kind(), "parse");
        assert_eq!(Error::Validation("x".into()).kind(), "validation");
        assert_eq!(Error::Emit("x".into()).kind(), "emit");
        assert_eq!(Error::Io("x".into()).kind(), "io");
        assert_eq!(Error::Config("x".into()).kind(), "config");
    }

    #[test]
    fn message_returns_inner_string() {
        let e = Error::Parse("hello".into());
        assert_eq!(e.message(), "hello");
    }

    #[test]
    fn display_parse_is_self_describing() {
        // Parse errors carry codespan output that already starts with
        // "error: "; Display must not prepend another category prefix.
        let e = Error::Parse("error: bad token\n  ┌─ wgsl:1:1".into());
        let s = e.to_string();
        assert!(
            !s.starts_with("parse"),
            "Parse Display must not double-prefix: {s}"
        );
        assert!(s.contains("bad token"));
    }

    #[test]
    fn display_validation_is_self_describing() {
        // Validation errors, like Parse, are self-describing codespan
        // diagnostics.  Display must not add a category prefix.
        let e = Error::Validation("error: invalid type\n  ┌─ wgsl:3:5".into());
        let s = e.to_string();
        assert!(
            !s.starts_with("validation"),
            "Validation Display must not double-prefix: {s}"
        );
        assert!(s.contains("invalid type"));
    }

    #[test]
    fn display_emit_has_prefix() {
        let e = Error::Emit("something broke".into());
        assert_eq!(e.to_string(), "emit error: something broke");
    }

    #[test]
    fn display_io_has_prefix() {
        let e = Error::Io("not found".into());
        assert_eq!(e.to_string(), "I/O error: not found");
    }

    #[test]
    fn display_config_has_prefix() {
        let e = Error::Config("bad value".into());
        assert_eq!(e.to_string(), "configuration error: bad value");
    }

    /// Snapshot lock for every `Display` format.  Any drift here is a
    /// breaking change to the public surface described on [`Error`];
    /// update both the format table and downstream consumers before
    /// touching these strings.
    #[test]
    fn display_format_snapshot() {
        assert_eq!(Error::Parse("X".into()).to_string(), "X");
        assert_eq!(Error::Validation("X".into()).to_string(), "X");
        assert_eq!(Error::Emit("X".into()).to_string(), "emit error: X");
        assert_eq!(Error::Io("X".into()).to_string(), "I/O error: X");
        assert_eq!(
            Error::Config("X".into()).to_string(),
            "configuration error: X"
        );
    }

    #[test]
    fn from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "gone");
        let e = Error::from(io_err);
        assert_eq!(e.kind(), "io");
        assert!(e.message().contains("gone"));
    }

    #[test]
    fn implements_std_error() {
        let e = Error::Parse("test".into());
        let _: &dyn std::error::Error = &e;
    }
}
