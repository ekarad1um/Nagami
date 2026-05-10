//! Coarse error taxonomy shared by every module.
//!
//! Each domain module impls [`Categorized`] for its error type,
//! classifying every variant once at its point of definition.
//! The API layer reduces to a six-arm match on [`ErrorKind`]
//! when mapping domain errors to HTTP statuses; adding a new
//! variant in a domain module needs no edit to the API layer.
//!
//! # How to extend
//!
//! 1. Place the variant in its domain module (e.g.
//!    `file_mgr::FileError::NewVariant`).
//! 2. Update that module's `impl Categorized for FileError` to
//!    classify the variant -- exactly one new match arm.
//! 3. The API layer needs no changes; the trait dispatch is
//!    automatic.
//!
//! # Why six categories
//!
//! The six-arm enumeration is deliberately small: each value
//! has a clean HTTP-status mapping and a clear operator-facing
//! meaning.  New categories should not be added without a
//! strong reason; when in doubt, classify into the nearest
//! existing category and add a `#[error("...")]` message that
//! disambiguates.
//!
//! Rule of thumb:
//!
//! - **`UserInput`** -- operator-supplied data failed
//!   validation (malformed UUID, wrong-shape config, missing
//!   required field).  The operator can fix the request and
//!   retry.
//! - **`NotFound`** -- the operator referred to a resource by
//!   id and that id is not present (workspace, asset, job,
//!   head).  Distinguished from `UserInput` because the shape
//!   of the request was valid; the resource just doesn't
//!   exist.
//! - **`Conflict`** -- the request would change state that
//!   another request is already mutating, or that another
//!   request has already mutated to a competing form (job
//!   already running, asset name already taken at upload time,
//!   schema-version mismatch on a load).
//! - **`NotImplemented`** -- the operation is recognised but
//!   the daemon does not yet support it.  Distinct from
//!   "endpoint doesn't exist" (axum returns 404 directly):
//!   this is for known endpoints whose body indicates an
//!   unsupported mode.
//! - **`Unavailable`** -- a downstream service or device is
//!   temporarily missing or wedged (RKNN library not loaded,
//!   ALSA card unplugged).  Distinguished from `Internal`
//!   because the operator can sometimes resolve it by
//!   reconnecting / restarting; from `NotFound` because the
//!   daemon knows the thing should exist, it just can't reach
//!   it right now.
//! - **`Internal`** -- anything else.  Logic bugs, IO mid-write
//!   on a known-good path, prost decode failures on
//!   daemon-internal channels.  Operator can't usefully react;
//!   the response is "file a bug".

use std::fmt;

/// Coarse category of a domain error.
///
/// Used by the API layer to map domain errors to HTTP statuses
/// uniformly.  See module docs for the rule of thumb on
/// choosing a category.
///
/// Variant order is meaningful: roughly increasing "severity"
/// (UserInput is recoverable by the operator; Internal is not).
/// [`Ord`] / [`PartialOrd`] therefore carry useful semantics --
/// handlers chaining classified errors can prefer the most
/// severe category.
#[derive(Copy, Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ErrorKind {
    /// 400 -- operator input failed validation.
    UserInput,
    /// 404 -- referenced resource does not exist.
    NotFound,
    /// 409 -- request conflicts with existing state.
    Conflict,
    /// 501 -- recognised operation, not yet implemented.
    NotImplemented,
    /// 503 -- downstream service or device is temporarily
    /// unavailable.
    Unavailable,
    /// 500 -- internal failure (logic bug, unexpected IO).
    Internal,
}

impl ErrorKind {
    /// HTTP status numeric value for the API layer's
    /// `IntoResponse` impl.  Returning `u16` keeps `common`
    /// free of an http crate dep; the API layer wraps with
    /// `StatusCode::from_u16(...).unwrap()` (always succeeds --
    /// the returned values are exactly the canonical
    /// statuses).
    pub const fn http_status_code(self) -> u16 {
        match self {
            ErrorKind::UserInput => 400,
            ErrorKind::NotFound => 404,
            ErrorKind::Conflict => 409,
            ErrorKind::NotImplemented => 501,
            ErrorKind::Unavailable => 503,
            ErrorKind::Internal => 500,
        }
    }

    /// Short stable identifier used as the API response's
    /// `code` field (`"bad_request"` / `"not_found"` / ...).
    /// Pinned here so consumers see the same string regardless
    /// of which domain module the error originated in.
    pub const fn code_str(self) -> &'static str {
        match self {
            ErrorKind::UserInput => "bad_request",
            ErrorKind::NotFound => "not_found",
            ErrorKind::Conflict => "conflict",
            ErrorKind::NotImplemented => "not_implemented",
            ErrorKind::Unavailable => "unavailable",
            ErrorKind::Internal => "internal",
        }
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code_str())
    }
}

/// Implemented by every workspace error type so the API layer
/// can derive an [`ErrorKind`] without per-variant match arms.
///
/// One impl per error type, located in the same module as the
/// error.  Adding a variant to that error means adding exactly
/// one arm to the impl's match; the API layer needs no
/// changes.
pub trait Categorized {
    /// Classify this error.
    fn kind(&self) -> ErrorKind;
}

// MARK: Tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn http_status_codes_are_canonical() {
        assert_eq!(ErrorKind::UserInput.http_status_code(), 400);
        assert_eq!(ErrorKind::NotFound.http_status_code(), 404);
        assert_eq!(ErrorKind::Conflict.http_status_code(), 409);
        assert_eq!(ErrorKind::NotImplemented.http_status_code(), 501);
        assert_eq!(ErrorKind::Unavailable.http_status_code(), 503);
        assert_eq!(ErrorKind::Internal.http_status_code(), 500);
    }

    #[test]
    fn code_str_round_trips_through_display() {
        for k in [
            ErrorKind::UserInput,
            ErrorKind::NotFound,
            ErrorKind::Conflict,
            ErrorKind::NotImplemented,
            ErrorKind::Unavailable,
            ErrorKind::Internal,
        ] {
            assert_eq!(format!("{k}"), k.code_str());
        }
    }

    /// The severity hierarchy lets callers
    /// `.iter().max_by_key(.kind())` when chaining classified
    /// errors -- recoverable categories sort before
    /// opaque-internal ones.
    #[test]
    fn ord_reflects_severity() {
        assert!(ErrorKind::UserInput < ErrorKind::Internal);
        assert!(ErrorKind::NotFound < ErrorKind::Internal);
        assert!(ErrorKind::Conflict < ErrorKind::Unavailable);
    }
}
