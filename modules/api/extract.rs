//! Envelope-preserving extractor wrappers.
//!
//! axum's stock `Json<T>` / `Query<T>` / `Path<T>` extractors emit
//! their own `IntoResponse` rejections (typically a plain-text
//! body with a `400` / `422` status) when the request body, query
//! string, or path parameter fails to deserialize.  Those default
//! rejections bypass our `ApiError` `{error, code}` envelope, so a
//! client that catches `code` for every other failure mode sees a
//! different shape on common mistakes (malformed JSON, unknown
//! fields, bad query types).
//!
//! Every route in the API crate consumes user-supplied input
//! through one of the wrappers in this module instead.  Each
//! wrapper delegates to the stock extractor for the happy path,
//! then converts the rejection into [`ApiError::Bad`] so the
//! envelope is uniform across the wire surface.
//!
//! ## Usage
//!
//! ```ignore
//! use crate::api::extract::{ApiJson, ApiQuery};
//! async fn handler(
//!     ApiQuery(q): ApiQuery<MyQuery>,
//!     ApiJson(req): ApiJson<MyRequest>,
//! ) -> Result<Json<MyResp>, ApiError> { ... }
//! ```
//!
//! Routes that consume raw body bytes
//! (`PUT /workspace/{id}/assets/{*path}`) take `axum::body::Body`
//! directly and stream via `Body::into_data_stream()`; no envelope
//! wrapper is needed because `Body`-extraction itself is
//! infallible (any failure surfaces during the in-handler stream
//! reads, where the route can wrap the `axum::Error` in its own
//! [`ApiError`] envelope).

use axum::extract::FromRequest;
use axum::extract::FromRequestParts;
use axum::extract::rejection::{JsonRejection, QueryRejection};
use axum::http::request::Parts;
use serde::de::DeserializeOwned;

use crate::api::error::ApiError;

/// JSON body extractor that maps deserialization failures into
/// [`ApiError::Bad`] (HTTP 400 with the `{error, code}` envelope)
/// instead of axum's default plain-text rejection.
///
/// Use in every handler that accepts a JSON request body in place
/// of `axum::Json<T>`.  The inner `T` is unchanged; pattern-match
/// via `ApiJson(req): ApiJson<MyDto>` exactly like the stock
/// extractor.
#[derive(Debug, Clone, Copy, Default)]
pub struct ApiJson<T>(pub T);

impl<T, S> FromRequest<S> for ApiJson<T>
where
    T: DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request(
        req: axum::http::Request<axum::body::Body>,
        state: &S,
    ) -> Result<Self, Self::Rejection> {
        // Delegate to axum's own Json extractor for the happy
        // path (same content-type checks, same buffered-bytes
        // limit) and only reshape its rejection into our
        // envelope.  Forwarding rather than re-implementing
        // keeps behaviour identical for every accept path.
        match axum::Json::<T>::from_request(req, state).await {
            Ok(axum::Json(value)) => Ok(ApiJson(value)),
            Err(rej) => Err(map_json_rejection(rej)),
        }
    }
}

fn map_json_rejection(rej: JsonRejection) -> ApiError {
    // axum's `JsonRejection` carries a typed cause + a
    // human-readable Display impl.  Surfacing the Display text
    // gives the operator the same diagnosis they would have
    // received on the plain-text path (line/column for syntax
    // errors, field name for unknown-field errors), now wrapped
    // in the envelope.
    ApiError::Bad(format!("invalid JSON body: {rej}"))
}

/// Query-string extractor that maps deserialization failures
/// into [`ApiError::Bad`].  Same envelope-preservation rationale
/// as [`ApiJson`].
#[derive(Debug, Clone, Copy, Default)]
pub struct ApiQuery<T>(pub T);

impl<T, S> FromRequestParts<S> for ApiQuery<T>
where
    T: DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        match axum::extract::Query::<T>::from_request_parts(parts, state).await {
            Ok(axum::extract::Query(value)) => Ok(ApiQuery(value)),
            Err(rej) => Err(map_query_rejection(rej)),
        }
    }
}

fn map_query_rejection(rej: QueryRejection) -> ApiError {
    ApiError::Bad(format!("invalid query string: {rej}"))
}
