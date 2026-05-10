//! Errors surfaced by the safe wrapper.  Two design rules:
//!
//! 1. **No silent failures.** Every `rknn_*` C call is checked against
//!    `RKNN_SUCC` and wrapped in [`Error::Rknn`] on failure.  (This is the
//!    concrete bug from rknpu2 we're fixing: its wrapper swallowed
//!    `rknn_inputs_set`'s -1, leading to a silent zero-buffer run.)
//!
//! 2. **Each error identifies which call failed and with what code.** The
//!    C API's integer return code is preserved along with the function name
//!    and a short context string, so on-device logs are diagnosable without
//!    a strace / gdb session.

use crate::rknn_runtime::ffi::LoadError;
use std::ffi::c_int;

/// Crate-wide result alias.
pub type Result<T> = std::result::Result<T, Error>;

/// Public error type.  Non-exhaustive so we can add variants later without
/// a breaking change.
#[non_exhaustive]
#[derive(Debug)]
pub enum Error {
    /// `libloading::Library::new(path)` failed -- file not found, wrong
    /// architecture, broken ELF, etc.
    LibraryLoad {
        path: std::path::PathBuf,
        source: libloading::Error,
    },

    /// A required symbol was missing from the loaded library.  Likely means
    /// the operator loaded the wrong library variant (`librknnrt` vs
    /// `librknnmrt`) or an older SDK without this function.
    SymbolNotFound {
        name: &'static str,
        source: libloading::Error,
    },

    /// A C call returned a non-success code.  `name` is the C function, code
    /// is the raw `RKNN_ERR_*` integer, `context` is a short description of
    /// what we were trying to do (e.g. `"query SDK version"`).
    Rknn {
        name: &'static str,
        code: c_int,
        context: &'static str,
    },

    /// Caller passed a buffer whose length doesn't match what the model
    /// declares or what we otherwise expected.  Kept separate from
    /// `Error::Rknn` because it's a precondition violation, not a C-layer
    /// error.
    ShapeMismatch {
        what: &'static str,
        expected: usize,
        got: usize,
    },

    /// Tensor name / SDK version strings were not valid UTF-8.  Very
    /// unlikely in practice (Rockchip's strings are ASCII) but preserves
    /// information instead of panicking.
    Utf8(std::str::Utf8Error),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::LibraryLoad { path, source } => {
                write!(f, "load library {}: {source}", path.display())
            }
            Error::SymbolNotFound { name, source } => {
                write!(f, "resolve symbol {name}: {source}")
            }
            Error::Rknn {
                name,
                code,
                context,
            } => {
                write!(
                    f,
                    "{name} ({context}) failed: {code} ({})",
                    rknn_error_name(*code),
                )
            }
            Error::ShapeMismatch {
                what,
                expected,
                got,
            } => {
                write!(f, "{what} shape mismatch: expected {expected}, got {got}")
            }
            Error::Utf8(e) => write!(f, "utf-8 decode: {e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::LibraryLoad { source, .. } | Error::SymbolNotFound { source, .. } => {
                Some(source)
            }
            Error::Utf8(e) => Some(e),
            _ => None,
        }
    }
}

impl From<LoadError> for Error {
    fn from(e: LoadError) -> Self {
        match e {
            LoadError::Library { path, source } => Error::LibraryLoad { path, source },
            LoadError::Symbol { name, source } => Error::SymbolNotFound { name, source },
        }
    }
}

impl From<std::str::Utf8Error> for Error {
    fn from(e: std::str::Utf8Error) -> Self {
        Error::Utf8(e)
    }
}

/// Map a known `RKNN_ERR_*` code to its header name, or `"UNKNOWN"` if it's
/// not one we recognize.  The list tracks the constants in bindgen's
/// output -- if Rockchip adds a new error code we'll just report
/// `"UNKNOWN"` plus the raw integer, which is still useful.
pub(crate) fn rknn_error_name(code: c_int) -> &'static str {
    use crate::rknn_runtime::sys;
    match code {
        x if x == sys::RKNN_SUCC as c_int => "RKNN_SUCC",
        sys::RKNN_ERR_FAIL => "RKNN_ERR_FAIL",
        sys::RKNN_ERR_TIMEOUT => "RKNN_ERR_TIMEOUT",
        sys::RKNN_ERR_DEVICE_UNAVAILABLE => "RKNN_ERR_DEVICE_UNAVAILABLE",
        sys::RKNN_ERR_MALLOC_FAIL => "RKNN_ERR_MALLOC_FAIL",
        sys::RKNN_ERR_PARAM_INVALID => "RKNN_ERR_PARAM_INVALID",
        sys::RKNN_ERR_MODEL_INVALID => "RKNN_ERR_MODEL_INVALID",
        sys::RKNN_ERR_CTX_INVALID => "RKNN_ERR_CTX_INVALID",
        sys::RKNN_ERR_INPUT_INVALID => "RKNN_ERR_INPUT_INVALID",
        sys::RKNN_ERR_OUTPUT_INVALID => "RKNN_ERR_OUTPUT_INVALID",
        sys::RKNN_ERR_DEVICE_UNMATCH => "RKNN_ERR_DEVICE_UNMATCH",
        sys::RKNN_ERR_INCOMPATILE_PRE_COMPILE_MODEL => "RKNN_ERR_INCOMPATILE_PRE_COMPILE_MODEL",
        sys::RKNN_ERR_INCOMPATILE_OPTIMIZATION_LEVEL_VERSION => {
            "RKNN_ERR_INCOMPATILE_OPTIMIZATION_LEVEL_VERSION"
        }
        sys::RKNN_ERR_TARGET_PLATFORM_UNMATCH => "RKNN_ERR_TARGET_PLATFORM_UNMATCH",
        _ => "UNKNOWN",
    }
}

/// Central check used on every C call.  If the code indicates failure,
/// returns `Error::Rknn`; otherwise `Ok(())`.
///
/// `name` is the C function name (for diagnostics); `context` is the
/// current operation (e.g. `"query input 0 attr"`).
#[inline]
pub(crate) fn check(name: &'static str, context: &'static str, code: c_int) -> Result<()> {
    if code == crate::rknn_runtime::sys::RKNN_SUCC as c_int {
        Ok(())
    } else {
        Err(Error::Rknn {
            name,
            code,
            context,
        })
    }
}
