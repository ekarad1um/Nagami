//! API error type + IntoResponse plumbing.
//!
//! Split out of `lib.rs` to bring the facade under the
//! 1,500-LoC layer-gate.  `ApiError` is re-exported by [`crate`]
//! so existing import paths continue to work.

use crate::common::error::Categorized;
use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;
use thiserror::Error;
use tokio::task;

/// Top-level API failure shape.  Every domain error type
/// (`FileError`, `ConfigError`, `TrainingError`,
/// `ConvertError`, ...) maps to one of these variants
/// before being rendered as an HTTP response.  See the
/// [`axum::response::IntoResponse`] impl below for the
/// status-code mapping.
#[derive(Debug, Error)]
pub enum ApiError {
    #[error("invalid request: {0}")]
    Bad(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("config: {0}")]
    Config(#[from] crate::config::ConfigError),
    #[error("mic: {0}")]
    Mic(#[from] crate::config::MicError),
    #[error("head load: {0}")]
    Head(#[from] crate::inference::HeadError),
    /// Trait-surface head failure from
    /// [`crate::common::traits::head_store::HeadStore::try_swap`].
    /// Wraps a boxed dyn error coming from the production
    /// `HotHead` impl (which is itself a `HeadError`); the
    /// taxonomy collapses to UserInput because the operator
    /// pointed at a head file that didn't load.
    #[error("head swap: {0}")]
    HeadStore(#[from] crate::common::traits::head_store::HeadStoreError),
    #[error("file: {0}")]
    File(#[from] crate::file_mgr::FileError),
    /// Type-erased error from the `FsService` trait.
    /// Concrete `FileError` variants stay reachable via
    /// `Categorized::kind` (preserved across the boxing) so the
    /// HTTP status mapping is unchanged; API handlers `?` through
    /// this variant on every trait call.
    #[error("fs: {0}")]
    Fs(#[from] crate::file_mgr::FsError),
    #[error("invalid identifier: {0}")]
    Id(#[from] crate::common::ids::IdError),
    #[error("convert: {0}")]
    Convert(#[from] crate::converter::ConvertError),
    #[error("training: {0}")]
    Training(#[from] crate::training::TrainingError),
    /// Active-head writer failure.  Categorized kind is preserved
    /// through the `Categorized` delegation below so 500/404/400
    /// routing matches the underlying `ActivationError` variant.
    #[error("activation: {0}")]
    Activation(#[from] crate::file_mgr::ActivationError),
    #[error("internal: spawn_blocking join: {0}")]
    Join(#[from] task::JoinError),
    #[error("not implemented (Phase {phase})")]
    NotImplemented { phase: &'static str },
    /// Read-your-writes.  The caller asked for a
    /// snapshot at version >= `requested` via `?min_version=N`,
    /// but the live resource is still at `current < requested`.
    /// Mapped to HTTP 425 Too Early.  Non-blocking by design;
    /// callers retry after their write's [`crate::common::version::SwapReceipt`] settles
    /// (or up to a small deadline) rather than parking the
    /// server thread.
    #[error("requested min_version={requested}, current={current}: retry after the write settles")]
    TooEarly { requested: u64, current: u64 },
    /// Wired by the router's `method_not_allowed_fallback` so 405s
    /// carry the same `{error, code}` envelope as every other
    /// failure.  The `ErrorKind` taxonomy intentionally doesn't
    /// enumerate 405; the mapping is a per-variant override, same
    /// shape as `TooEarly -> 425`.
    #[error("method not allowed: {method} {path}")]
    MethodNotAllowed { method: String, path: String },
}

impl crate::common::error::Categorized for ApiError {
    /// Each `#[from]`-wrapped domain error delegates to its own
    /// `Categorized::kind()` impl (defined in the owning crate);
    /// api-internal variants classify directly here.  Adding a
    /// variant to a domain error only touches the owning crate's
    /// impl.
    fn kind(&self) -> crate::common::error::ErrorKind {
        use crate::common::error::ErrorKind::*;
        match self {
            ApiError::Bad(_) => UserInput,
            ApiError::NotFound(_) => NotFound,
            ApiError::NotImplemented { .. } => NotImplemented,
            // Delegate to the wrapped domain error's classifier.
            ApiError::Id(e) => e.kind(),
            ApiError::Config(e) => e.kind(),
            ApiError::Mic(e) => e.kind(),
            ApiError::Head(e) => e.kind(),
            ApiError::HeadStore(e) => e.kind(),
            ApiError::File(e) => e.kind(),
            ApiError::Fs(e) => e.kind(),
            ApiError::Convert(e) => e.kind(),
            ApiError::Training(e) => e.kind(),
            ApiError::Activation(e) => e.kind(),
            // Tokio JoinError surfaces only when a spawn_blocking
            // task panicked or was cancelled mid-flight; either
            // way this is a daemon-internal failure.
            ApiError::Join(_) => Internal,
            // Read-your-writes; mapping to HTTP 425 is
            // handled in `http_status()`.  The `Categorized` arm
            // collapses to `Conflict` (the closest taxonomy fit:
            // request well-formed, current state hasn't caught up
            // with caller's expectation) so any future consumer
            // that goes through `kind()` doesn't see a panic.
            ApiError::TooEarly { .. } => Conflict,
            // 405 is a per-variant override in
            // `http_status()`; UserInput is the closest taxonomy
            // fit (request well-formed but the verb doesn't apply
            // to the resource).
            ApiError::MethodNotAllowed { .. } => UserInput,
        }
    }
}

impl ApiError {
    fn http_status(&self) -> StatusCode {
        // `TooEarly` maps to 425 and `MethodNotAllowed` to 405 via
        // per-variant overrides; every other variant routes
        // through `Categorized::kind() -> http_status_code()`,
        // which returns canonical HTTP status numbers
        // (400 / 404 / 409 / 500 / 501 / 503) all accepted by
        // `StatusCode::from_u16`.
        match self {
            ApiError::TooEarly { .. } => StatusCode::TOO_EARLY,
            ApiError::MethodNotAllowed { .. } => StatusCode::METHOD_NOT_ALLOWED,
            _ => StatusCode::from_u16(self.kind().http_status_code())
                .expect("ErrorKind::http_status_code returns canonical HTTP statuses"),
        }
    }

    fn code(&self) -> &'static str {
        // The single-train-job conflict gets a dedicated
        // discriminator so dashboards can distinguish "your
        // request raced an unrelated upload" (`conflict`) from
        // "the daemon already has a train running"
        // (`another_train_running`).  The error reaches us
        // wrapped in any of three carriers; check each.
        match self {
            ApiError::TooEarly { .. } => "too_early",
            ApiError::MethodNotAllowed { .. } => "method_not_allowed",
            ApiError::File(crate::file_mgr::FileError::AnotherTrainRunning) => {
                "another_train_running"
            }
            ApiError::Training(crate::training::TrainingError::File(
                crate::file_mgr::FileError::AnotherTrainRunning,
            )) => "another_train_running",
            ApiError::Fs(e)
                if matches!(
                    {
                        use std::error::Error as _;
                        e.source()
                            .and_then(|s| s.downcast_ref::<crate::file_mgr::FileError>())
                    },
                    Some(crate::file_mgr::FileError::AnotherTrainRunning),
                ) =>
            {
                "another_train_running"
            }
            _ => self.kind().code_str(),
        }
    }
}

#[derive(Serialize)]
struct ApiErrorBody {
    error: String,
    code: &'static str,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = ApiErrorBody {
            error: self.to_string(),
            code: self.code(),
        };
        (self.http_status(), Json(body)).into_response()
    }
}
