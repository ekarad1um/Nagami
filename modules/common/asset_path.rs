//! Validated workspace asset-tree path.
//!
//! [`AssetPath`] is the type the daemon hands to every consumer
//! that resolves a path under a workspace's daemon-owned asset
//! trees (`datasets/` and `converters/`).  Lifting validation into
//! a parsed newtype keeps the per-component allowlist call-once at
//! the deserialize / handler boundary; downstream code receives a
//! typed value that cannot escape the workspace root.
//!
//! # Validation rules
//!
//! `[AssetPath]` is a `/`-joined sequence of components.  Each
//! component reuses the prior `AssetId` byte allowlist (`[A-Za-z0-9._-]`,
//! non-empty, no leading `.`).  Aggregate bounds:
//!
//! - total length <= 256 bytes;
//! - per-component length <= 255 bytes (filesystem `NAME_MAX` floor);
//! - depth <= 8 components;
//! - empty path rejected.
//!
//! Wire serde shape (`try_from = "String"`, `into = "String"`)
//! routes JSON / TOML deserialization through [`AssetPath::parse`]
//! so a literal that fails any rule is rejected at deserialize,
//! not silently wrapped.
//!
//! # Defence in depth
//!
//! The route layer URL-decodes path captures before constructing
//! an [`AssetPath`]; even when an attacker smuggles in
//! `%2E%2E%2F`, the decoded `../` reaches the parser which
//! rejects the leading-`.` component.  Validating after
//! decoding catches both the raw and the encoded forms.

use crate::common::error::{Categorized, ErrorKind};
use std::fmt;
use thiserror::Error;

#[inline]
fn is_allowed_component_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b'_')
}

/// Maximum total byte length of a parsed [`AssetPath`].
pub const MAX_TOTAL_BYTES: usize = 256;
/// Maximum per-component byte length (filesystem `NAME_MAX` floor).
pub const MAX_COMPONENT_BYTES: usize = 255;
/// Maximum number of `/`-separated components in an [`AssetPath`].
pub const MAX_DEPTH: usize = 8;

/// Validation failure for [`AssetPath::parse`].  Operator-facing
/// ([`std::fmt::Display`] is the rendering used in API
/// responses); structured fields let downstream code react
/// programmatically.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum AssetPathError {
    /// The supplied string was empty.
    #[error("asset path is empty")]
    Empty,
    /// A `/`-separated component was empty -- catches leading
    /// `/`, trailing `/`, and `//` runs.
    #[error("asset path component {index} is empty (no leading/trailing/double `/`)")]
    EmptyComponent {
        /// Component index (0-based) of the empty slot.
        index: usize,
    },
    /// Total byte length exceeded [`MAX_TOTAL_BYTES`].
    #[error("asset path too long: {got} bytes > {max}")]
    TotalTooLong {
        /// Observed length.
        got: usize,
        /// Allowed maximum.
        max: usize,
    },
    /// A single component exceeded [`MAX_COMPONENT_BYTES`].
    #[error("asset path component {index} too long: {got} bytes > {max}")]
    ComponentTooLong {
        /// Component index (0-based) of the offending slot.
        index: usize,
        /// Observed component length.
        got: usize,
        /// Allowed maximum component length.
        max: usize,
    },
    /// Depth exceeded [`MAX_DEPTH`].
    #[error("asset path too deep: {got} components > {max}")]
    TooDeep {
        /// Observed depth.
        got: usize,
        /// Allowed maximum depth.
        max: usize,
    },
    /// A component started with `.` -- catches `.`, `..`, and the
    /// unix-hidden convention with one rule.
    #[error("asset path component {index} starts with `.` (forbidden)")]
    LeadingDot {
        /// Component index (0-based).
        index: usize,
    },
    /// A byte outside the per-component allowlist (`[A-Za-z0-9._-]`).
    /// Catches `\\`, NUL, control bytes, non-ASCII, URL-encoded
    /// percent introducers, etc.
    #[error(
        "asset path component {component_index} byte {byte_index} \
         is forbidden (0x{byte:02x}); allowed: [A-Za-z0-9._-]"
    )]
    BadByte {
        /// Component index (0-based).
        component_index: usize,
        /// Byte index inside the component (0-based).
        byte_index: usize,
        /// Offending byte.
        byte: u8,
    },
}

impl Categorized for AssetPathError {
    /// All [`AssetPathError`] variants are operator-input
    /// failures.  Maps to `400 Bad Request` via
    /// [`ErrorKind::UserInput`].
    fn kind(&self) -> ErrorKind {
        ErrorKind::UserInput
    }
}

/// Operator-supplied path identifying a file or directory inside
/// a workspace's daemon-owned `datasets/` tree.  See module docs
/// for the validation contract.
#[derive(Clone, Debug, Eq, Hash, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct AssetPath(String);

impl AssetPath {
    /// Validate operator-supplied input.  See module docs for
    /// the full rule set.
    ///
    /// The validator allocates exactly one `String` on success
    /// (the canonical owned form).  It does no normalisation:
    /// the parsed value is byte-identical to the input, so
    /// equality and hashing are well-defined and a round-trip
    /// through serde is the identity.
    pub fn parse(s: &str) -> Result<Self, AssetPathError> {
        if s.is_empty() {
            return Err(AssetPathError::Empty);
        }
        if s.len() > MAX_TOTAL_BYTES {
            return Err(AssetPathError::TotalTooLong {
                got: s.len(),
                max: MAX_TOTAL_BYTES,
            });
        }
        // `split('/')` yields one component per slash-delimited
        // slot, including empties for leading / trailing /
        // double slashes -- exactly what we want to reject.
        let mut depth = 0usize;
        for (component_index, component) in s.split('/').enumerate() {
            depth += 1;
            if depth > MAX_DEPTH {
                return Err(AssetPathError::TooDeep {
                    got: depth,
                    max: MAX_DEPTH,
                });
            }
            if component.is_empty() {
                return Err(AssetPathError::EmptyComponent {
                    index: component_index,
                });
            }
            if component.len() > MAX_COMPONENT_BYTES {
                return Err(AssetPathError::ComponentTooLong {
                    index: component_index,
                    got: component.len(),
                    max: MAX_COMPONENT_BYTES,
                });
            }
            // Leading `.` rules out `.`, `..`, `.hidden` with one
            // rule.  Interior / trailing `.` is fine
            // (`audio.tar.gz`).
            if component.starts_with('.') {
                return Err(AssetPathError::LeadingDot {
                    index: component_index,
                });
            }
            for (byte_index, &b) in component.as_bytes().iter().enumerate() {
                if !is_allowed_component_byte(b) {
                    return Err(AssetPathError::BadByte {
                        component_index,
                        byte_index,
                        byte: b,
                    });
                }
            }
        }
        Ok(Self(s.to_string()))
    }

    /// Borrow the underlying string.
    #[inline]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Number of `/`-separated components.  Always >= 1 for a
    /// valid [`AssetPath`].
    pub fn depth(&self) -> usize {
        self.0.split('/').count()
    }

    /// Iterator over the path's `/`-separated components.  Each
    /// yielded slice has been individually validated.
    pub fn components(&self) -> impl Iterator<Item = &str> {
        self.0.split('/')
    }
}

impl fmt::Display for AssetPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::str::FromStr for AssetPath {
    type Err = AssetPathError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

impl TryFrom<&str> for AssetPath {
    type Error = AssetPathError;
    fn try_from(s: &str) -> Result<Self, Self::Error> {
        Self::parse(s)
    }
}

impl TryFrom<String> for AssetPath {
    type Error = AssetPathError;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        Self::parse(&s)
    }
}

impl From<AssetPath> for String {
    fn from(p: AssetPath) -> Self {
        p.0
    }
}

impl PartialEq<str> for AssetPath {
    fn eq(&self, other: &str) -> bool {
        self.0 == other
    }
}

impl PartialEq<&str> for AssetPath {
    fn eq(&self, other: &&str) -> bool {
        self.0 == *other
    }
}

// MARK: Tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_canonical_shapes() {
        for s in &[
            "audio_dataset",
            "audio_dataset/cat",
            "audio_dataset/cat/sample.wav",
            "labels.txt",
            "model/manifest.json",
            "a",
            "a-b_c.d",
            "a/b/c/d/e/f/g/h", // depth = MAX_DEPTH
        ] {
            assert!(AssetPath::parse(s).is_ok(), "should accept {s:?}");
        }
    }

    #[test]
    fn rejects_empty() {
        assert_eq!(AssetPath::parse(""), Err(AssetPathError::Empty));
    }

    #[test]
    fn rejects_dot_and_dotdot_components() {
        for s in &[
            ".",
            "..",
            "../etc",
            "foo/.",
            "foo/..",
            "foo/../bar",
            ".hidden",
        ] {
            assert!(
                matches!(AssetPath::parse(s), Err(AssetPathError::LeadingDot { .. })),
                "should reject {s:?}"
            );
        }
    }

    /// URL-decoded traversal MUST be rejected.  When the route
    /// layer decodes `%2E%2E%2F` to `../` the parser sees the
    /// literal `..` and rejects it via `LeadingDot`.  When the
    /// raw URL-encoded form reaches the parser undecoded, the
    /// `%` byte falls outside the allowlist and is rejected via
    /// `BadByte` -- both shapes fail closed.
    #[test]
    fn rejects_url_decoded_traversal() {
        // Decoded form -- what the route layer should pass in.
        assert!(matches!(
            AssetPath::parse("../etc/passwd"),
            Err(AssetPathError::LeadingDot { index: 0 })
        ));
        // Undecoded form -- defence in depth: `%` is outside
        // the per-component allowlist.
        let res = AssetPath::parse("%2E%2E%2Fetc");
        assert!(matches!(
            res,
            Err(AssetPathError::BadByte { byte: b'%', .. })
        ));
    }

    #[test]
    fn rejects_backslash() {
        let res = AssetPath::parse("foo\\bar");
        assert!(matches!(
            res,
            Err(AssetPathError::BadByte { byte: b'\\', .. })
        ));
    }

    #[test]
    fn rejects_nul_and_control_bytes() {
        let res_nul = AssetPath::parse("foo\0bar");
        assert!(matches!(
            res_nul,
            Err(AssetPathError::BadByte { byte: 0x00, .. })
        ));
        let res_lf = AssetPath::parse("foo\nbar");
        assert!(matches!(
            res_lf,
            Err(AssetPathError::BadByte { byte: b'\n', .. })
        ));
        let res_tab = AssetPath::parse("foo\tbar");
        assert!(matches!(
            res_tab,
            Err(AssetPathError::BadByte { byte: b'\t', .. })
        ));
    }

    #[test]
    fn rejects_non_ascii_bytes() {
        // Em-dash (3-byte UTF-8 starting with 0xE2).
        let res = AssetPath::parse("caf\u{00e9}/foo");
        assert!(matches!(res, Err(AssetPathError::BadByte { .. })));
        // Multi-byte CJK char.
        let res = AssetPath::parse("\u{6f22}");
        assert!(matches!(res, Err(AssetPathError::BadByte { .. })));
    }

    #[test]
    fn rejects_leading_trailing_double_slash() {
        // Leading `/` -- first component empty.
        assert!(matches!(
            AssetPath::parse("/foo"),
            Err(AssetPathError::EmptyComponent { index: 0 })
        ));
        // Trailing `/` -- last component empty.
        assert!(matches!(
            AssetPath::parse("foo/"),
            Err(AssetPathError::EmptyComponent { index: 1 })
        ));
        // Double `/` -- middle component empty.
        assert!(matches!(
            AssetPath::parse("foo//bar"),
            Err(AssetPathError::EmptyComponent { index: 1 })
        ));
    }

    #[test]
    fn rejects_total_length_exceeded() {
        // 257-byte single component blows total-length first.
        let s = "a".repeat(MAX_TOTAL_BYTES + 1);
        assert!(matches!(
            AssetPath::parse(&s),
            Err(AssetPathError::TotalTooLong { got: 257, max: 256 })
        ));
    }

    #[test]
    fn accepts_total_length_at_cap() {
        // Total budget exactly at MAX_TOTAL_BYTES, split across
        // two components so neither exceeds MAX_COMPONENT_BYTES.
        // "a/" prefix (2 bytes) + 254-byte trailing component
        // = 256 total = MAX_TOTAL_BYTES.
        let s = format!("a/{}", "b".repeat(MAX_TOTAL_BYTES - 2));
        assert_eq!(s.len(), MAX_TOTAL_BYTES);
        assert!(AssetPath::parse(&s).is_ok());
    }

    #[test]
    fn rejects_component_length_exceeded_when_total_fits() {
        // Total <= MAX_TOTAL_BYTES but a component is 256.
        // Build "<256 a's>" -- but that's 256 chars total, equal
        // to total cap and triggers component cap (256 > 255).
        let s = "a".repeat(MAX_COMPONENT_BYTES + 1);
        // 256-byte single component: total cap accepts (== 256),
        // component cap rejects (256 > 255).
        let res = AssetPath::parse(&s);
        assert!(matches!(
            res,
            Err(AssetPathError::ComponentTooLong {
                index: 0,
                got: 256,
                max: 255
            })
        ));
    }

    #[test]
    fn accepts_component_at_max() {
        let s = "a".repeat(MAX_COMPONENT_BYTES);
        // 255-byte single component; total = 255, depth = 1.
        assert!(AssetPath::parse(&s).is_ok());
    }

    #[test]
    fn rejects_depth_exceeded() {
        // 9 components separated by `/`: "a/a/a/a/a/a/a/a/a"
        // length = 9 + 8 = 17 bytes, well under total cap;
        // depth check fires.
        let s = std::iter::repeat_n("a", MAX_DEPTH + 1)
            .collect::<Vec<_>>()
            .join("/");
        let res = AssetPath::parse(&s);
        assert!(matches!(
            res,
            Err(AssetPathError::TooDeep { got: 9, max: 8 })
        ));
    }

    #[test]
    fn accepts_depth_at_cap() {
        let s = std::iter::repeat_n("a", MAX_DEPTH)
            .collect::<Vec<_>>()
            .join("/");
        let p = AssetPath::parse(&s).expect("depth 8 accepted");
        assert_eq!(p.depth(), MAX_DEPTH);
    }

    #[test]
    fn try_from_works() {
        let _: AssetPath = "audio/dataset.wav".try_into().unwrap();
        let _: AssetPath = "audio/dataset.wav".to_string().try_into().unwrap();
        let res: Result<AssetPath, _> = "..".try_into();
        assert!(res.is_err());
    }

    /// Wire-supplied path traversal MUST be rejected at
    /// deserialize, not silently wrapped.  Same for the
    /// hidden-file leading-dot rule and any non-allowlisted
    /// character.
    #[test]
    fn serde_round_trip_validates() {
        let p = AssetPath::parse("audio_dataset/cat/sample.wav").unwrap();
        let json = serde_json::to_string(&p).unwrap();
        assert_eq!(
            json, "\"audio_dataset/cat/sample.wav\"",
            "transparent wire shape"
        );
        let round: AssetPath = serde_json::from_str(&json).unwrap();
        assert_eq!(round, p);

        for bad in [
            "\"\"",           // empty
            "\"..\"",         // leading dot
            "\"foo/../bar\"", // interior dotdot
            "\"foo\\\\bar\"", // backslash
            "\"foo bar\"",    // space (not in allowlist)
            "\"%2E%2E%2F\"",  // raw URL-encoded `..`
            "\"\\u0000\"",    // NUL
            "\"caf\\u00e9\"", // non-ASCII
            "\"/foo\"",       // leading slash
            "\"foo/\"",       // trailing slash
            "\"foo//bar\"",   // double slash
        ] {
            let res: Result<AssetPath, _> = serde_json::from_str(bad);
            assert!(
                res.is_err(),
                "wire input {bad:?} must be rejected by serde shim"
            );
        }
    }

    #[test]
    fn components_iterator_matches_depth() {
        let p = AssetPath::parse("a/b/c").unwrap();
        let comps: Vec<_> = p.components().collect();
        assert_eq!(comps, vec!["a", "b", "c"]);
        assert_eq!(p.depth(), 3);
    }

    #[test]
    fn display_matches_input() {
        let p = AssetPath::parse("audio/dataset.wav").unwrap();
        assert_eq!(p.to_string(), "audio/dataset.wav");
        assert_eq!(p.as_str(), "audio/dataset.wav");
    }

    #[test]
    fn error_kind_classification_is_user_input() {
        let err = AssetPath::parse("..").unwrap_err();
        assert_eq!(err.kind(), ErrorKind::UserInput);
        let err = AssetPath::parse("").unwrap_err();
        assert_eq!(err.kind(), ErrorKind::UserInput);
    }
}
