//! Wire-format types for the daemon.
//!
//! Three message domains share the `acoustics` proto package:
//!
//! - audio stream -- Opus-encoded mic audio
//! - inference stream -- classification results
//! - envelope -- top-level wire wrapper for every streaming
//!   write; receivers decode this first and dispatch on its
//!   `payload` oneof.
//!
//! Single-version layout: there is no v1/v2 split.  The
//! daemon's clients are limited to its own consumers (no
//! third-party peers in the wild), so re-versioning at any
//! future point is a fresh-start replacement rather than an
//! incremental upgrade.  Re-introducing a versioning surface
//! (e.g., a `schema_version` envelope field, or a sibling
//! `proto_v2/` package) is justified only if a real
//! cross-deployment compatibility need appears.
//!
//! Identifier types (workspace / job / head / mic) live in
//! [`crate::common::ids`] as validated newtypes; the wire
//! shape embeds them as their `Display` representation
//! (UUID-v4 strings, printable-ASCII for `MicId`).  There
//! is no parallel proto identifier schema -- a previous
//! `control.proto` defined untyped wrapper messages that
//! never matched the runtime contract; it has been removed
//! to avoid two competing identity models.

#![forbid(unsafe_code)]
// prost copies the comment text from `.proto` files verbatim
// into `///` doc comments on the generated Rust types.  Our
// `.proto` comments use indented bullets (`*` after spaces)
// that proto3 readers expect; clippy's
// `doc_lazy_continuation` lint flags those as needing extra
// indentation in Rust doc strings.  Suppressing the lint at
// the module level is cheaper than rewriting every `.proto`
// comment to a Rust-doc-compatible form (which would harm
// readability for proto-first reviewers and would have to be
// re-applied on every `.proto` edit).
#![allow(clippy::doc_lazy_continuation)]
#![allow(clippy::doc_overindented_list_items)]

// prost emits one Rust file per package; this `include!`
// covers every `.proto` under `proto/`.
include!(concat!(env!("OUT_DIR"), "/acoustics.rs"));

// Sync framing primitives (envelope wrap helpers + the
// envelope decoder).  The async length-prefix decoder lives
// in `stream_io::framing` because it needs `tokio::io`;
// producer modules (`opus_stream`, `inference`) import only
// this module so they don't pick up `stream_io`'s dep tree.
pub mod framing;

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    /// Round-trip every wire message: encode to bytes, decode
    /// back, check equality.  Catches accidental field-number
    /// drift between sender and receiver.
    #[test]
    fn audio_frame_round_trip() {
        use audio_frame::Codec;
        let f = AudioFrame {
            seq: 12345,
            t_us_capture_monotonic: Some(123_456_789),
            t_us_publish_unix: Some(1_700_000_000_000_000),
            sample_rate: Some(48_000),
            frame_duration_ms: Some(20),
            codec: Some(Codec::Opus(bytes::Bytes::from_static(&[
                0xDE, 0xAD, 0xBE, 0xEF,
            ]))),
        };
        let bytes = f.encode_to_vec();
        let back = AudioFrame::decode(bytes.as_slice()).expect("decode");
        assert_eq!(f, back);
    }

    #[test]
    fn inference_frame_round_trip() {
        let f = InferenceFrame {
            seq: 7,
            t_us_capture_monotonic: Some(123_456_789),
            t_us_publish_unix: Some(1_700_000_000_000_000),
            top_k: vec![
                TopK {
                    class_idx: 1,
                    label: "yes".into(),
                    prob: 0.91,
                },
                TopK {
                    class_idx: 2,
                    label: "no".into(),
                    prob: 0.07,
                },
                TopK {
                    class_idx: 0,
                    label: "bg".into(),
                    prob: 0.02,
                },
            ],
            head_id: Some("00000000-0000-0000-0000-000000000000".into()),
            head_version: Some(42),
        };
        let bytes = f.encode_to_vec();
        let back = InferenceFrame::decode(bytes.as_slice()).expect("decode");
        assert_eq!(f, back);
    }
}
