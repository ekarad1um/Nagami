//! Validated identifier newtypes.
//!
//! Lifting the validators here makes them call-once at the
//! deserialization / handler boundary; downstream code receives
//! a typed value that cannot hold an unvalidated string.
//!
//! # Validation rules
//!
//! - **[`WorkspaceId`], [`JobId`], [`HeadId`]** -- strict
//!   canonical UUID-v4 form: 36 chars, lowercase hex with `-`
//!   at positions 8/13/18/23, version nibble `'4'` at byte 14,
//!   RFC 4122 variant nibble in `'8'..='b'` at byte 19.  Hex +
//!   dash is a strict subset of safe filename chars, so the type
//!   can never escape a workspace root.  Random construction via
//!   [`WorkspaceId::new`] uses [`uuid::Uuid::new_v4`].  Wire
//!   serde (`try_from = "String", into = "String"`) routes
//!   through [`WorkspaceId::parse`] so a TOML / JSON literal is
//!   validated on the way in.
//! - **[`AssetId`]** -- operator-supplied filename.
//!   Restricted to ASCII alphanumerics + `.` + `-` + `_`;
//!   non-empty; <=255 chars (filesystem `NAME_MAX` floor);
//!   leading `.` rejected to forbid `.`, `..`, and the
//!   hidden-file convention.  Subdirectory placement is owned
//!   by `AssetKind`, not by [`AssetId`].
//! - **[`MicId`]** -- operator-supplied short identifier
//!   (e.g., `"hw:1,0"`, `"mock:0"`).  Restricted to printable
//!   ASCII (`!`..`~`, no whitespace and no controls), <=128
//!   chars.  `Arc<str>`-backed so per-frame clones are
//!   refcount-cheap.  Wire serde routes through
//!   [`MicId::parse`].
//!
//! # API surface
//!
//! UUID-backed identifiers ([`WorkspaceId`], [`JobId`],
//! [`HeadId`]) expose:
//!
//! ```text
//! pub fn new() -> Self                              // fresh UUID-v4
//! pub fn parse(s: &str) -> Result<Self, IdError>
//! pub fn as_uuid(&self) -> &Uuid
//! impl Default, Display, FromStr
//! impl TryFrom<&str>, TryFrom<String>
//! ```
//!
//! String-backed identifiers ([`AssetId`], [`MicId`]) expose:
//!
//! ```text
//! pub fn parse(s: &str) -> Result<Self, IdError>
//! pub fn as_str(&self) -> &str
//! impl Display, FromStr
//! impl TryFrom<&str>, TryFrom<String>
//! impl From<Self> for String
//! ```

use crate::common::error::{Categorized, ErrorKind};
use std::fmt;
use std::sync::Arc;
use thiserror::Error;
use uuid::Uuid;

// MARK: Errors

/// Validation failure when constructing one of the identifier
/// newtypes.  Operator-facing ([`std::fmt::Display`] is the default
/// rendering in API responses); structured fields let
/// downstream code react programmatically (e.g., distinguish
/// "wrong length" from "bad character").
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum IdError {
    /// The supplied string was empty.
    #[error("identifier is empty")]
    Empty,
    /// A UUID-strict identifier ([`WorkspaceId`] / [`JobId`] /
    /// [`HeadId`]) got a string of the wrong length.
    #[error("expected UUID-v4 (36 chars hex + dashes); got {got} chars")]
    BadUuidLength {
        /// Observed length.
        got: usize,
    },
    /// A UUID-strict identifier got a non-lowercase-hex char
    /// where hex was required, or a non-`-` where `-` was
    /// required.  Uppercase hex is rejected so the canonical
    /// wire / filesystem representation stays unique.
    #[error(
        "invalid byte 0x{byte:02x} at position {index} \
         (UUID-v4 expects lowercase hex digits with `-` at positions 8/13/18/23)"
    )]
    BadUuidByte {
        /// Byte index in the input string.
        index: usize,
        /// The offending byte.
        byte: u8,
    },
    /// UUID-v4 version nibble (byte 14) was not `'4'`.
    #[error("expected UUID-v4 version nibble '4' at position 14, got 0x{byte:02x}")]
    BadUuidVersion {
        /// The offending byte.
        byte: u8,
    },
    /// RFC 4122 variant nibble (byte 19) was not in `'8'..='b'`.
    #[error("expected RFC 4122 variant nibble in '8'..='b' at position 19, got 0x{byte:02x}")]
    BadUuidVariant {
        /// The offending byte.
        byte: u8,
    },
    /// A length-bounded identifier overflowed its cap.
    #[error("{kind} too long: {got} chars > {max}")]
    TooLong {
        /// Identifier kind, for the operator-facing message.
        kind: &'static str,
        got: usize,
        max: usize,
    },
    /// A character-restricted identifier saw a forbidden char.
    #[error("{kind} contains forbidden byte 0x{byte:02x} at position {index}")]
    BadChar {
        kind: &'static str,
        index: usize,
        byte: u8,
    },
}

impl Categorized for IdError {
    /// Every [`IdError`] variant is operator-input failure.
    /// The API layer maps these to `400 Bad Request` via
    /// [`ErrorKind::UserInput`].
    fn kind(&self) -> ErrorKind {
        ErrorKind::UserInput
    }
}

// MARK: UUID-strict identifier macro

/// Validate that `s` is the canonical lowercase UUID-v4 form:
/// 36 chars, 8-4-4-4-12 lowercase hex with `-` at positions
/// 8/13/18/23, version nibble `'4'` at byte 14, RFC 4122
/// variant nibble in `'8'..='b'` at byte 19.  Lowercase-only
/// pins a single canonical representation; the version /
/// variant checks make non-v4 hex (e.g. v1 timestamp UUIDs)
/// rejectable at the boundary.
fn validate_uuid_v4_str(s: &str) -> Result<(), IdError> {
    if s.is_empty() {
        return Err(IdError::Empty);
    }
    if s.len() != 36 {
        return Err(IdError::BadUuidLength { got: s.len() });
    }
    let bytes = s.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        // The dash positions are the four hex-segment boundaries
        // (8, 13, 18, 23); every other byte is canonical lowercase
        // hex.  Inlining the literal alternation rather than a
        // `&[usize].contains(&i)` lookup keeps the comparison a
        // compile-time constant table.
        let ok = if matches!(i, 8 | 13 | 18 | 23) {
            b == b'-'
        } else {
            b.is_ascii_digit() || (b'a'..=b'f').contains(&b)
        };
        if !ok {
            return Err(IdError::BadUuidByte { index: i, byte: b });
        }
    }
    if bytes[14] != b'4' {
        return Err(IdError::BadUuidVersion { byte: bytes[14] });
    }
    // RFC 4122 variant: high two bits of byte 8 (== UUID byte
    // 19 in string form) are `10`, which in lowercase hex is
    // one of '8' / '9' / 'a' / 'b'.
    if !matches!(bytes[19], b'8' | b'9' | b'a' | b'b') {
        return Err(IdError::BadUuidVariant { byte: bytes[19] });
    }
    Ok(())
}

// A macro (not a generic `Uuid<Tag>` newtype) so each id is a
// distinct nominal type — the type system then prevents accidentally
// passing a `WorkspaceId` where a `JobId` is expected.  Validation logic
// stays shared via `validate_uuid_v4_str`; only the type-level wrapping,
// the per-type derives, and the serde `try_from`/`into` shims (which
// require the concrete name in attribute position) are duplicated here.
macro_rules! uuid_id {
    ($(#[$attr:meta])* $name:ident) => {
        $(#[$attr])*
        #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
        #[derive(serde::Serialize, serde::Deserialize)]
        // Wire serde routes through `parse` so a TOML / JSON
        // literal that misses any v4-strict invariant is
        // rejected at deserialize time, not silently wrapped.
        #[serde(try_from = "String", into = "String")]
        pub struct $name(Uuid);

        impl $name {
            /// Generate a fresh random UUID-v4.  Use this
            /// when the daemon is creating the identifier
            /// (e.g., the converter assigning a head_id at
            /// publish time).
            #[inline]
            pub fn new() -> Self {
                Self(Uuid::new_v4())
            }

            /// Validate operator-supplied input.  Accepts the
            /// canonical lowercase UUID-v4 form only (36 chars
            /// hex + dashes, version nibble `'4'`, RFC 4122
            /// variant `'8'..='b'`); anything else is rejected.
            pub fn parse(s: &str) -> Result<Self, IdError> {
                // Strict shape check first (fast,
                // allocation-free).
                validate_uuid_v4_str(s)?;
                // Shape passed -- uuid's parser cannot fail.
                Ok(Self(Uuid::parse_str(s).expect("validated UUID-v4 shape")))
            }

            /// Borrow the inner [`Uuid`] for cases that need
            /// the 16-byte representation (hashing into
            /// another wire format, etc.).
            #[inline]
            pub fn as_uuid(&self) -> &Uuid {
                &self.0
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Display::fmt(&self.0, f)
            }
        }

        impl std::str::FromStr for $name {
            type Err = IdError;
            fn from_str(s: &str) -> Result<Self, Self::Err> {
                Self::parse(s)
            }
        }

        impl TryFrom<&str> for $name {
            type Error = IdError;
            fn try_from(s: &str) -> Result<Self, Self::Error> {
                Self::parse(s)
            }
        }

        impl TryFrom<String> for $name {
            type Error = IdError;
            fn try_from(s: String) -> Result<Self, Self::Error> {
                Self::parse(&s)
            }
        }

        // `into = "String"` requires this; `Display` already
        // emits the canonical lowercase form via the inner
        // `Uuid`'s formatter.
        impl From<$name> for String {
            fn from(id: $name) -> Self {
                id.0.to_string()
            }
        }
    };
}

uuid_id! {
    /// Workspace identifier.  Names a per-workspace directory
    /// under the daemon's `file_mgr` root.  UUID-v4 strict so
    /// the identifier (and therefore the path component it
    /// becomes) is provably safe -- no `..`, no path
    /// separators, no case-sensitivity tricks.
    WorkspaceId
}

uuid_id! {
    /// Background-job identifier.  Used by `training::Registry`
    /// (fine-tune jobs) and `converter::Registry`
    /// (model-conversion jobs).
    JobId
}

uuid_id! {
    /// Inference-head identifier.  Stamped at converter-publish
    /// time onto every emitted `InferenceFrame.head_id` so
    /// downstream consumers can disambiguate which weights
    /// produced the result.
    HeadId
}

// MARK: Default runtime HeadId

/// Canonical UUID-v4 string of the daemon-bundled default head.
/// Pinned so the active-head manifest's `runtime_head_id` for
/// `origin: "default"` is reproducible across deploys, and so
/// the operator-facing fixture path
/// `misc/heads/00000000-default/` (which is **not** parsed as a
/// `HeadId`) and the runtime identity stay aligned.
pub const DEFAULT_RUNTIME_HEAD_ID_STR: &str = "00000000-0000-4000-8000-000000000000";

/// Parsed [`HeadId`] for the daemon-bundled default head.  Used
/// when activating with `POST /active {default: true}`; the
/// active manifest stamps this id on every emitted frame so
/// downstream consumers can distinguish "operator chose the
/// default" from "operator activated a trained head."
///
/// The constant is parsed (not literal) so the v4-strict
/// validator gets exercised at every call; the round-trip is
/// guaranteed to succeed by the unit test.  Cheap to call --
/// returns a `Copy` newtype.
pub fn default_runtime_head_id() -> HeadId {
    HeadId::parse(DEFAULT_RUNTIME_HEAD_ID_STR)
        .expect("default runtime head id is a hard-coded valid UUID-v4")
}

// MARK: AssetId

const ASSET_ID_MAX: usize = 255;

/// Asset identifier.  Names a single file basename inside a
/// workspace (`head_v3.mpk`, `labels.txt`, `dataset.tar.gz`).
/// Subdirectory placement is owned by `AssetKind`, not by
/// [`AssetId`]; the `/` separator is rejected here so the
/// type can never escape the workspace root.  Restricted to
/// safe-filename ASCII (alphanumerics + `.` + `_` + `-`);
/// non-empty; <=255 chars; leading `.` rejected.
///
/// Serde shape: `try_from = "String"` + `into = "String"`
/// routes JSON / TOML deserialization through [`Self::parse`]
/// (so a wire-supplied asset name is validated on the way
/// in, not silently wrapped) and serialization through
/// `From<AssetId> for String` (preserving the transparent
/// wire shape: a bare string, no `{ "0": "..." }` wrapper).
#[derive(Clone, Debug, Eq, Hash, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct AssetId(String);

impl From<AssetId> for String {
    fn from(id: AssetId) -> Self {
        id.0
    }
}

impl AssetId {
    /// Validate a filename-shaped string for use as an asset
    /// identifier.  Rejects:
    ///
    /// - empty input,
    /// - inputs over 255 chars (filesystem `NAME_MAX` floor),
    /// - any char outside `[A-Za-z0-9._-]`,
    /// - leading `.` (catches `.`, `..`, `.hidden`).
    ///   Trailing or interior `.` is fine (`weights.mpk`,
    ///   `dataset.tar.gz`).
    pub fn parse(s: &str) -> Result<Self, IdError> {
        if s.is_empty() {
            return Err(IdError::Empty);
        }
        if s.len() > ASSET_ID_MAX {
            return Err(IdError::TooLong {
                kind: "asset id",
                got: s.len(),
                max: ASSET_ID_MAX,
            });
        }
        // Leading `.` is the simplest single rule that catches
        // the path-traversal special names (`.` / `..`) and
        // the unix-hidden convention without enumerating
        // them.
        if s.as_bytes()[0] == b'.' {
            return Err(IdError::BadChar {
                kind: "asset id",
                index: 0,
                byte: b'.',
            });
        }
        for (i, &b) in s.as_bytes().iter().enumerate() {
            let ok = b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b'_');
            if !ok {
                return Err(IdError::BadChar {
                    kind: "asset id",
                    index: i,
                    byte: b,
                });
            }
        }
        Ok(Self(s.to_string()))
    }

    /// Borrow the underlying string.
    #[inline]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for AssetId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::str::FromStr for AssetId {
    type Err = IdError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

impl TryFrom<&str> for AssetId {
    type Error = IdError;
    fn try_from(s: &str) -> Result<Self, Self::Error> {
        Self::parse(s)
    }
}

impl TryFrom<String> for AssetId {
    type Error = IdError;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        Self::parse(&s)
    }
}

// Ergonomic comparisons against `&str` so call sites that
// compare `record.name == "weights/head.mpk"` continue to
// read naturally.  PartialEq is reflexive only against the
// operand types declared here; comparisons against `String`
// go through `as_str()` explicitly.
impl PartialEq<str> for AssetId {
    fn eq(&self, other: &str) -> bool {
        self.0 == other
    }
}

impl PartialEq<&str> for AssetId {
    fn eq(&self, other: &&str) -> bool {
        self.0 == *other
    }
}

impl PartialEq<AssetId> for str {
    fn eq(&self, other: &AssetId) -> bool {
        self == other.0
    }
}

impl PartialEq<AssetId> for &str {
    fn eq(&self, other: &AssetId) -> bool {
        *self == other.0
    }
}

// MARK: MicId

const MIC_ID_MAX: usize = 128;

/// Microphone identifier.  Operator-set via TOML (e.g.,
/// `"hw:1,0"`, `"mock:0"`); the arbitrator uses it as a key
/// in per-source policy / catalogue resolution.
/// `Arc<str>`-backed so the per-frame
/// `snapshot.mic_id.clone()` is refcount-cheap and
/// allocation-free.
///
/// Validation: non-empty; printable ASCII (`!`..`~`) only
/// (no whitespace, no controls, no non-ASCII); <=128 chars.
/// The narrow charset prevents accidental shell-quoting
/// confusion in error messages and rules out any
/// whitespace-trim ambiguity.
///
/// Wire serde shape (`try_from = "String"`, `into = "String"`)
/// routes JSON / TOML deserialization through [`Self::parse`]
/// so a config-supplied id is validated on the way in, not
/// silently wrapped.  Serialization remains a bare string via
/// `From<MicId> for String`.
#[derive(Clone, Debug, Eq, Hash, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct MicId(Arc<str>);

impl From<MicId> for String {
    fn from(id: MicId) -> Self {
        id.0.as_ref().to_owned()
    }
}

impl MicId {
    /// Construct from a `&'static str` literal that the
    /// caller knows to be valid.  Panics if `s` would fail
    /// [`Self::parse`].
    ///
    /// Intended for test fixtures and hardcoded mock catalogue
    /// entries (e.g., `MicId::from_static("hw:1,0")`); the
    /// `&'static str` constraint signals "compile-time
    /// literal, not operator input".  Anything coming from
    /// TOML / HTTP / disk must go through [`Self::parse`].
    pub fn from_static(s: &'static str) -> Self {
        Self::parse(s)
            .unwrap_or_else(|e| panic!("MicId::from_static({s:?}) -- invalid literal: {e}"))
    }

    /// Validate operator input.
    pub fn parse(s: &str) -> Result<Self, IdError> {
        if s.is_empty() {
            return Err(IdError::Empty);
        }
        if s.len() > MIC_ID_MAX {
            return Err(IdError::TooLong {
                kind: "mic id",
                got: s.len(),
                max: MIC_ID_MAX,
            });
        }
        for (i, &b) in s.as_bytes().iter().enumerate() {
            // Printable ASCII excluding space and DEL.
            let ok = (b'!'..=b'~').contains(&b);
            if !ok {
                return Err(IdError::BadChar {
                    kind: "mic id",
                    index: i,
                    byte: b,
                });
            }
        }
        Ok(Self(Arc::from(s)))
    }

    /// Borrow the underlying string.
    #[inline]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for MicId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::str::FromStr for MicId {
    type Err = IdError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

impl TryFrom<&str> for MicId {
    type Error = IdError;
    fn try_from(s: &str) -> Result<Self, Self::Error> {
        Self::parse(s)
    }
}

impl TryFrom<String> for MicId {
    type Error = IdError;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        Self::parse(&s)
    }
}

// MARK: Tests

#[cfg(test)]
mod tests {
    use super::*;

    // UUID-strict identifiers.

    /// Canonical lowercase UUID-v4 fixture.  Version nibble at
    /// position 14 is `4`; variant nibble at position 19 is `8`
    /// (RFC 4122).  Hand-written so tests stay deterministic.
    fn good_uuid() -> &'static str {
        "11111111-2222-4333-8444-555555555555"
    }

    #[test]
    fn uuid_id_parse_accepts_canonical() {
        let id = WorkspaceId::parse(good_uuid()).expect("valid uuid");
        assert_eq!(id.to_string(), good_uuid());
    }

    #[test]
    fn uuid_id_parse_rejects_empty() {
        assert_eq!(WorkspaceId::parse(""), Err(IdError::Empty));
    }

    #[test]
    fn uuid_id_parse_rejects_wrong_length() {
        let res = JobId::parse("abc");
        assert!(matches!(res, Err(IdError::BadUuidLength { got: 3 })));
    }

    #[test]
    fn uuid_id_parse_rejects_invalid_chars() {
        // Position 0 has 'X', not a hex digit.
        let bad = "X1111111-2222-3333-4444-555555555555";
        let res = HeadId::parse(bad);
        assert!(matches!(
            res,
            Err(IdError::BadUuidByte {
                index: 0,
                byte: b'X'
            })
        ));
    }

    #[test]
    fn uuid_id_parse_rejects_missing_dash() {
        // Position 8 should be '-' but is hex.
        let bad = "111111111222-3333-4444-555555555555";
        let res = WorkspaceId::parse(bad);
        // Length is 35 -- caught by length check first.  The
        // structured error still tells the operator the right
        // thing; verify either of the two reasonable failures.
        assert!(matches!(
            res,
            Err(IdError::BadUuidLength { .. }) | Err(IdError::BadUuidByte { index: 8, .. })
        ));
    }

    /// Random construction must always pass the validator --
    /// no version/variant bits stripped, no formatting drift.
    #[test]
    fn uuid_id_new_round_trips_through_parse() {
        for _ in 0..16 {
            let id = JobId::new();
            let s = id.to_string();
            let parsed = JobId::parse(&s).expect("self-round-trip");
            assert_eq!(id, parsed);
        }
    }

    #[test]
    fn uuid_id_try_from_works() {
        let _: WorkspaceId = good_uuid().try_into().unwrap();
        let _: WorkspaceId = good_uuid().to_string().try_into().unwrap();
        let res: Result<WorkspaceId, _> = "nope".try_into();
        assert!(res.is_err());
    }

    /// Uppercase hex MUST be rejected so the canonical wire /
    /// filesystem representation stays unique.
    #[test]
    fn uuid_id_rejects_uppercase_hex() {
        let bad = "11111111-2222-4333-8444-55555555555A";
        let res = WorkspaceId::parse(bad);
        assert!(matches!(
            res,
            Err(IdError::BadUuidByte {
                index: 35,
                byte: b'A'
            })
        ));
    }

    /// Non-v4 version nibble (e.g. v1 timestamp UUIDs) MUST be
    /// rejected; the docs and the random construction path
    /// both promise v4.
    #[test]
    fn uuid_id_rejects_non_v4_version_nibble() {
        // Position 14 is '1' (v1) instead of '4'.
        let bad = "11111111-2222-1333-8444-555555555555";
        let res = HeadId::parse(bad);
        assert!(matches!(res, Err(IdError::BadUuidVersion { byte: b'1' })));
    }

    /// RFC 4122 variant nibble must be one of `'8'..='b'`;
    /// other values would be a non-RFC4122 UUID layout.
    #[test]
    fn uuid_id_rejects_bad_variant_nibble() {
        // Position 19 is 'c' instead of '8'..='b'.
        let bad = "11111111-2222-4333-c444-555555555555";
        let res = JobId::parse(bad);
        assert!(matches!(res, Err(IdError::BadUuidVariant { byte: b'c' })));
    }

    /// Pinned: the bundled default head's runtime identity is
    /// the canonical lowercase v4 form
    /// `00000000-0000-4000-8000-000000000000`.  The `4` and `8`
    /// nibbles at positions 14 and 19 satisfy the strict
    /// validator without any non-zero bits.  Round-trips
    /// through Display.
    #[test]
    fn default_runtime_head_id_parses_and_round_trips() {
        let id = default_runtime_head_id();
        assert_eq!(id.to_string(), DEFAULT_RUNTIME_HEAD_ID_STR);
        let parsed = HeadId::parse(DEFAULT_RUNTIME_HEAD_ID_STR).unwrap();
        assert_eq!(id, parsed);
    }

    /// Wire serde routes UUID newtypes through `parse`, same
    /// shape as `AssetId` / `MicId`.  Rejects any string that
    /// fails the v4-strict check; accepts the canonical form
    /// and round-trips through Display.
    #[test]
    fn uuid_id_serde_round_trip_validates() {
        let id = WorkspaceId::parse(good_uuid()).unwrap();
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, format!("\"{}\"", good_uuid()), "transparent wire");
        let round: WorkspaceId = serde_json::from_str(&json).unwrap();
        assert_eq!(round, id);

        // Each fixture targets a distinct invariant.
        for bad in [
            "\"\"",                                     // empty
            "\"abc\"",                                  // length
            "\"11111111-2222-3333-8444-555555555555\"", // v3
            "\"11111111-2222-4333-c444-555555555555\"", // bad variant
            "\"11111111-2222-4333-8444-55555555555A\"", // uppercase
            "\"11111111_2222_4333_8444_555555555555\"", // wrong separators
        ] {
            let res: Result<WorkspaceId, _> = serde_json::from_str(bad);
            assert!(
                res.is_err(),
                "wire input {bad:?} must be rejected by serde shim"
            );
        }
    }

    // AssetId.

    #[test]
    fn asset_id_accepts_filename_shape() {
        for s in &[
            "head_v3.mpk",
            "labels.txt",
            "dataset-2026-05-01.zip",
            "tfjs_model.json",
            "a",
        ] {
            assert!(AssetId::parse(s).is_ok(), "should accept {s:?}");
        }
    }

    #[test]
    fn asset_id_rejects_empty() {
        assert_eq!(AssetId::parse(""), Err(IdError::Empty));
    }

    #[test]
    fn asset_id_rejects_path_separators() {
        for s in &["weights/head.mpk", "..", "../etc/passwd", "a\\b"] {
            assert!(
                matches!(AssetId::parse(s), Err(IdError::BadChar { .. })),
                "should reject {s:?}"
            );
        }
    }

    #[test]
    fn asset_id_rejects_oversized() {
        let s = "a".repeat(256);
        assert!(matches!(
            AssetId::parse(&s),
            Err(IdError::TooLong {
                max: 255,
                got: 256,
                ..
            })
        ));
    }

    /// Wire-supplied path traversal MUST be rejected at
    /// deserialize time, not silently wrapped.  Same for the
    /// hidden-file leading-dot rule and any non-allowlisted
    /// character.
    #[test]
    fn asset_id_serde_round_trip_validates() {
        let id = AssetId::parse("head_v3.mpk").unwrap();
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, "\"head_v3.mpk\"", "transparent wire shape");
        let round: AssetId = serde_json::from_str(&json).unwrap();
        assert_eq!(round, id);

        for bad in ["\"..\"", "\"weights/head.mpk\"", "\".hidden\"", "\"a b\""] {
            let res: Result<AssetId, _> = serde_json::from_str(bad);
            assert!(
                res.is_err(),
                "wire input {bad:?} must be rejected by serde shim",
            );
        }
    }

    // MicId.

    #[test]
    fn mic_id_accepts_canonical_shapes() {
        for s in &["hw:1,0", "mock:0", "default", "plughw:CARD=USB"] {
            assert!(MicId::parse(s).is_ok(), "should accept {s:?}");
        }
    }

    #[test]
    fn mic_id_rejects_whitespace() {
        for s in &["hw 1,0", " hw:1,0", "hw:1,0\n"] {
            assert!(
                matches!(MicId::parse(s), Err(IdError::BadChar { .. })),
                "should reject {s:?}"
            );
        }
    }

    #[test]
    fn mic_id_rejects_non_ascii() {
        // Em-dash (3-byte UTF-8); the first byte is non-ASCII.
        assert!(matches!(
            MicId::parse("mic\u{2014}1"),
            Err(IdError::BadChar { .. })
        ));
    }

    #[test]
    fn mic_id_rejects_oversized() {
        let s = "x".repeat(129);
        assert!(matches!(
            MicId::parse(&s),
            Err(IdError::TooLong {
                max: 128,
                got: 129,
                ..
            })
        ));
    }

    /// Both Arcs point to the same allocation: stable across
    /// clones, the design rationale for `Arc<str>`.
    #[test]
    fn mic_id_clone_is_refcount_cheap() {
        let original = MicId::parse("hw:1,0").unwrap();
        let cheap = original.clone();
        assert_eq!(original, cheap);
        assert!(Arc::ptr_eq(&original.0, &cheap.0));
    }

    /// Wire-supplied control / whitespace / non-ASCII MUST be
    /// rejected at deserialize time so a TOML literal cannot
    /// smuggle log-noisy or shell-quoting-ambiguous bytes
    /// through the boundary.
    #[test]
    fn mic_id_serde_rejects_invalid_strings() {
        for bad in [
            "\"\"",          // empty
            "\"hw 1,0\"",    // space
            "\"hw:1,0\\n\"", // trailing newline
            "\" hw:1,0\"",   // leading space
            // Em-dash JSON-encoded: serde decodes — to a
            // 3-byte UTF-8 sequence starting with 0xE2.
            "\"mic\\u2014\"",
        ] {
            let res: Result<MicId, _> = serde_json::from_str(bad);
            assert!(
                res.is_err(),
                "wire input {bad:?} must be rejected by serde shim"
            );
        }
    }

    /// Happy path: a valid id round-trips as a bare JSON
    /// string, preserving the wire shape inherited from the
    /// previous `transparent` form.
    #[test]
    fn mic_id_serde_round_trip_validates() {
        let id = MicId::parse("hw:1,0").unwrap();
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, "\"hw:1,0\"", "transparent wire shape");
        let round: MicId = serde_json::from_str(&json).unwrap();
        assert_eq!(round, id);
    }
}
