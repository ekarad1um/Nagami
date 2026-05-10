//! Sync framing primitives.
//!
//! Producer-side helpers that wrap a payload in a
//! [`crate::proto::Envelope`] and prost-encode the result to
//! `bytes::Bytes`, ready for broadcast on a
//! `tokio::sync::broadcast::Sender<Bytes>`.
//!
//! The async UDS reader counterpart
//! ([`crate::stream_io::framing::decode_length_prefixed`])
//! lives in `stream_io` because it needs `tokio::io`; every
//! primitive that fits in std (the wrap helpers + the checked
//! length-prefix encoder + the envelope decoder + the
//! protocol constants) lives here so producer modules
//! (`opus_stream`, `inference`) don't need to depend on
//! `stream_io`.

use crate::proto::{AudioFrame, Envelope, InferenceFrame, envelope::Payload as EnvelopePayload};
use bytes::{Bytes, BytesMut};
use prost::Message;
use thiserror::Error;

/// Required WebSocket subprotocol token.  The daemon's WS
/// handlers refuse upgrades whose `Sec-WebSocket-Protocol`
/// header doesn't list this exact value.
pub const WS_SUBPROTOCOL: &str = "acoustics";

/// Maximum accepted UDS frame length (server-side reader cap).
/// Comfortably above any legitimate envelope: one Opus packet
/// is <=4 KiB, one [`InferenceFrame`] is a few hundred bytes,
/// envelope overhead ~2 B.  64 KiB caps at ~16x headroom;
/// readers close on a prefix beyond this without parsing,
/// defending against a hostile peer that sends a giant prefix
/// as a DoS vector.
pub const MAX_UDS_FRAME_BYTES: u32 = 64 * 1024;

/// Wrap an [`AudioFrame`] in an [`Envelope`] and prost-encode.
/// Returns the bytes ready for `broadcast::Sender::send`.
///
/// Allocates one `Vec<u8>` per call.  Hot-path producers
/// (`opus_stream`) should prefer [`wrap_audio_into`] which
/// reuses a caller-owned `BytesMut` scratch buffer.
#[must_use = "envelope bytes are produced for a broadcast send -- ignoring them drops the frame"]
pub fn wrap_audio(frame: AudioFrame) -> Bytes {
    let env = Envelope {
        payload: Some(EnvelopePayload::Audio(frame)),
    };
    Bytes::from(env.encode_to_vec())
}

/// Wrap an [`InferenceFrame`] in an [`Envelope`] and
/// prost-encode.
///
/// Allocates one `Vec<u8>` per call.  Hot-path producers
/// (`inference::engine`) should prefer [`wrap_inference_into`]
/// which reuses a caller-owned `BytesMut` scratch buffer.
#[must_use = "envelope bytes are produced for a broadcast send -- ignoring them drops the frame"]
pub fn wrap_inference(frame: InferenceFrame) -> Bytes {
    let env = Envelope {
        payload: Some(EnvelopePayload::Inference(frame)),
    };
    Bytes::from(env.encode_to_vec())
}

/// Allocation-reusing variant of [`wrap_audio`].
///
/// `buf` is cleared on entry, the encoded envelope is
/// appended, and the head is split off as a [`Bytes`]
/// (Arc-backed, zero-copy fan-out across
/// `broadcast::Sender::send`'s clones).  `buf` retains the
/// residual capacity for the next call -- after a few frames
/// the underlying allocation is reused indefinitely, so the
/// steady-state envelope-encode allocation rate drops to
/// zero.
///
/// Hot-path producers should keep one `BytesMut` scratch
/// outside the loop and pass `&mut` here.  See
/// `opus_stream::run` for the canonical caller pattern (one
/// scratch per task).
#[must_use = "envelope bytes are produced for a broadcast send -- ignoring them drops the frame"]
pub fn wrap_audio_into(buf: &mut BytesMut, frame: AudioFrame) -> Bytes {
    buf.clear();
    let env = Envelope {
        payload: Some(EnvelopePayload::Audio(frame)),
    };
    // `BytesMut` implements `bytes::BufMut`, so prost's
    // `encode` (which calls `BufMut::put_*` under the hood)
    // grows the buffer on demand.
    //
    // The `expect` is abort-by-design, not recoverable.
    // `prost::encode` can fail here only if the allocator
    // can't grow `buf` -- the encoder buffer is a few KB, so
    // an allocator failure here means the next allocation
    // anywhere in the process will also fail.  There is no daemon-level recovery
    // path for OOM; the external supervisor restarts on
    // SIGABRT, and `BytesMut::reserve` aborts on allocator
    // failure regardless of this expect.  Documenting the
    // intent so future readers don't try to "harden" this
    // into a Result.
    env.encode(buf).expect("BytesMut grows on demand");
    buf.split().freeze()
}

/// Allocation-reusing variant of [`wrap_inference`].  See
/// [`wrap_audio_into`] for the buffer-reuse contract.
#[must_use = "envelope bytes are produced for a broadcast send -- ignoring them drops the frame"]
pub fn wrap_inference_into(buf: &mut BytesMut, frame: InferenceFrame) -> Bytes {
    buf.clear();
    let env = Envelope {
        payload: Some(EnvelopePayload::Inference(frame)),
    };
    // Same abort-by-design contract as `wrap_audio_into`; see
    // that function's `.encode()` comment for the OOM
    // rationale.
    env.encode(buf).expect("BytesMut grows on demand");
    buf.split().freeze()
}

/// Failure shapes from [`try_encode_length_prefixed`].
///
/// The encoder is the symmetric counterpart of the
/// [`crate::stream_io::framing::decode_length_prefixed`]
/// reader: both share [`MAX_UDS_FRAME_BYTES`] as the
/// per-frame cap.  An encoder that emitted a frame larger
/// than the decoder accepts would either silently truncate
/// the length prefix (the prior bug, `as u32` cast) or be
/// rejected by every conforming reader.
#[derive(Debug, Error)]
pub enum FramingEncodeError {
    /// Payload exceeds the per-frame cap.  `observed` is the
    /// real (`usize`) payload length so the operator-facing
    /// log line shows the true magnitude even on platforms
    /// where `usize` is wider than `u32`; `max` is the
    /// configured cap (`MAX_UDS_FRAME_BYTES`).
    #[error("payload too large for length-prefixed frame: {observed} bytes > {max} cap")]
    PayloadTooLarge {
        /// Observed payload length in bytes.
        observed: usize,
        /// The per-frame cap (`MAX_UDS_FRAME_BYTES`).
        max: u32,
    },
}

impl crate::common::error::Categorized for FramingEncodeError {
    /// Hitting the cap is a producer-side bug (oversize
    /// envelope being broadcast) rather than operator input;
    /// classify as `Internal` so the log triage path matches
    /// the daemon-internal failures it sits next to.
    fn kind(&self) -> crate::common::error::ErrorKind {
        crate::common::error::ErrorKind::Internal
    }
}

/// Encode `payload` as a 4-byte LE length prefix followed by
/// `payload` bytes, rejecting payloads larger than
/// [`MAX_UDS_FRAME_BYTES`].
///
/// Used by raw-UDS producers (server-side wiring deferred
/// until an in-tree raw-UDS endpoint exists).  Mirrors the
/// per-frame cap that the decoder
/// ([`crate::stream_io::framing::decode_length_prefixed`])
/// enforces, so an encoder cannot emit a frame that every
/// conforming reader will reject -- and cannot silently
/// truncate the length prefix the way an `as u32` cast on
/// `payload.len()` would for payloads above `u32::MAX`.
#[must_use = "the encoded frame must be sent or it is dropped"]
pub fn try_encode_length_prefixed(payload: &[u8]) -> Result<Bytes, FramingEncodeError> {
    // Bound check is on `usize` against `MAX_UDS_FRAME_BYTES as
    // usize`; `MAX_UDS_FRAME_BYTES` is a small `u32` (64 KiB) so
    // the cast back to `u32` for the wire prefix is guaranteed
    // lossless after this check.
    if payload.len() > MAX_UDS_FRAME_BYTES as usize {
        return Err(FramingEncodeError::PayloadTooLarge {
            observed: payload.len(),
            max: MAX_UDS_FRAME_BYTES,
        });
    }
    let len = payload.len() as u32;
    let mut buf = Vec::with_capacity(4 + payload.len());
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(payload);
    Ok(Bytes::from(buf))
}

/// Failure shapes from [`decode_envelope`].
///
/// Centralising the decode-time policy here keeps every
/// future raw-UDS / WS receiver from inventing subtly
/// different envelope handling.  Each variant is a terminal
/// framing failure: per `docs/PROTO.md`, the prefix IS the
/// synchronization point and resync is undefined, so callers
/// MUST close the connection on any of these.
#[derive(Debug, Error)]
pub enum ProtoDecodeError {
    /// The bytes are not a valid prost-encoded [`Envelope`].
    /// Either a hostile peer or wire-format drift (which the
    /// daemon does not support -- the protocol has a single
    /// version).
    #[error("envelope decode: {source}")]
    Decode {
        #[source]
        source: prost::DecodeError,
    },
    /// The envelope decoded but carries no `payload` variant.
    /// Every routed endpoint in this protocol expects a
    /// payload; an empty envelope is either a producer bug or
    /// a hostile peer.  A future control-plane endpoint that
    /// genuinely allows payload-less envelopes (heartbeat,
    /// ack) should call a sibling helper rather than relax
    /// this check.
    #[error("envelope missing payload variant")]
    MissingPayload,
}

impl crate::common::error::Categorized for ProtoDecodeError {
    /// All variants describe data sourced from a peer (the
    /// receiver's input); classify as `UserInput` so a future
    /// API surface that wraps a decode failure renders 400
    /// rather than 500.
    fn kind(&self) -> crate::common::error::ErrorKind {
        crate::common::error::ErrorKind::UserInput
    }
}

/// Decode bytes as an [`Envelope`] and validate the
/// payload-presence invariant from `docs/PROTO.md`.  Every
/// receiver that decodes envelope bytes (raw-UDS, future
/// WS-with-decode, internal integration tests) MUST go
/// through this helper rather than calling [`Envelope::decode`]
/// directly so the policy stays centralised.
///
/// Returns `Ok(envelope)` only when:
///
/// - prost decode succeeds,
/// - `payload.is_some()`.
///
/// On any failure the caller MUST close the framing channel
/// (the prefix IS the synchronization point; resync is not
/// defined per the protocol contract).
pub fn decode_envelope(bytes: &[u8]) -> Result<Envelope, ProtoDecodeError> {
    let env = Envelope::decode(bytes).map_err(|source| ProtoDecodeError::Decode { source })?;
    if env.payload.is_none() {
        return Err(ProtoDecodeError::MissingPayload);
    }
    Ok(env)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::{TopK, audio_frame::Codec};

    #[test]
    fn wrap_audio_decodes_via_envelope() {
        let frame = AudioFrame {
            seq: 7,
            t_us_capture_monotonic: Some(123),
            t_us_publish_unix: Some(456),
            sample_rate: Some(48_000),
            frame_duration_ms: Some(20),
            codec: Some(Codec::Opus(Bytes::from_static(b"\xDE\xAD\xBE\xEF"))),
        };
        let wire = wrap_audio(frame.clone());
        let env = Envelope::decode(wire.as_ref()).expect("decode envelope");
        match env.payload {
            Some(EnvelopePayload::Audio(decoded)) => assert_eq!(decoded, frame),
            other => panic!("expected Audio payload, got {other:?}"),
        }
    }

    #[test]
    fn wrap_inference_decodes_via_envelope() {
        let frame = InferenceFrame {
            seq: 11,
            t_us_capture_monotonic: Some(123),
            t_us_publish_unix: Some(456),
            top_k: vec![TopK {
                class_idx: 0,
                label: "bg".into(),
                prob: 1.0,
            }],
            head_id: Some("00000000-0000-0000-0000-000000000000".into()),
            head_version: Some(1),
        };
        let wire = wrap_inference(frame.clone());
        let env = Envelope::decode(wire.as_ref()).expect("decode envelope");
        match env.payload {
            Some(EnvelopePayload::Inference(decoded)) => assert_eq!(decoded, frame),
            other => panic!("expected Inference payload, got {other:?}"),
        }
    }

    #[test]
    fn wrap_audio_into_decodes_via_envelope() {
        let frame = AudioFrame {
            seq: 7,
            t_us_capture_monotonic: Some(123),
            t_us_publish_unix: Some(456),
            sample_rate: Some(48_000),
            frame_duration_ms: Some(20),
            codec: Some(Codec::Opus(Bytes::from_static(b"\xDE\xAD\xBE\xEF"))),
        };
        let mut buf = BytesMut::with_capacity(64);
        let wire = wrap_audio_into(&mut buf, frame.clone());
        let env = Envelope::decode(wire.as_ref()).expect("decode envelope");
        match env.payload {
            Some(EnvelopePayload::Audio(decoded)) => assert_eq!(decoded, frame),
            other => panic!("expected Audio payload, got {other:?}"),
        }
    }

    #[test]
    fn wrap_inference_into_decodes_via_envelope() {
        let frame = InferenceFrame {
            seq: 11,
            t_us_capture_monotonic: Some(123),
            t_us_publish_unix: Some(456),
            top_k: vec![TopK {
                class_idx: 0,
                label: "bg".into(),
                prob: 1.0,
            }],
            head_id: Some("00000000-0000-0000-0000-000000000000".into()),
            head_version: Some(1),
        };
        let mut buf = BytesMut::with_capacity(64);
        let wire = wrap_inference_into(&mut buf, frame.clone());
        let env = Envelope::decode(wire.as_ref()).expect("decode envelope");
        match env.payload {
            Some(EnvelopePayload::Inference(decoded)) => assert_eq!(decoded, frame),
            other => panic!("expected Inference payload, got {other:?}"),
        }
    }

    /// Reusing the same `BytesMut` across many `wrap_*_into`
    /// calls keeps the steady-state capacity bounded.  The
    /// load-bearing property: the buffer does NOT grow
    /// unboundedly across many iterations (which it would if
    /// the `_into` shape were buggy and held a reference to
    /// each emitted Bytes through an internal Arc cycle).
    #[test]
    fn wrap_inference_into_steady_state_capacity_is_bounded() {
        let frame = InferenceFrame {
            seq: 1,
            t_us_capture_monotonic: Some(123),
            t_us_publish_unix: Some(456),
            top_k: vec![TopK {
                class_idx: 0,
                label: "bg".into(),
                prob: 1.0,
            }],
            head_id: Some("00000000-0000-0000-0000-000000000000".into()),
            head_version: Some(1),
        };
        let mut buf = BytesMut::with_capacity(4096);
        for _ in 0..1000 {
            let _wire = wrap_inference_into(&mut buf, frame.clone());
        }
        assert!(
            buf.capacity() < 64 * 1024,
            "scratch capacity unexpectedly grew unbounded (cap={})",
            buf.capacity(),
        );
    }

    #[test]
    fn try_encode_length_prefixed_round_trip() {
        let payload = b"\x01\x02\x03\x04\x05".to_vec();
        let framed = try_encode_length_prefixed(&payload).expect("under cap");
        let mut len_bytes = [0u8; 4];
        len_bytes.copy_from_slice(&framed[..4]);
        assert_eq!(u32::from_le_bytes(len_bytes), payload.len() as u32);
        assert_eq!(&framed[4..], payload.as_slice());
    }

    /// At-cap is accepted (the cap is inclusive); cap+1 is
    /// rejected with the structured error.  Boundary check
    /// prevents the common off-by-one that would let a peer
    /// craft the maximum-allowed-prefix-plus-one DoS payload.
    #[test]
    fn try_encode_length_prefixed_enforces_cap() {
        let at_cap = vec![0xABu8; MAX_UDS_FRAME_BYTES as usize];
        assert!(
            try_encode_length_prefixed(&at_cap).is_ok(),
            "at-cap payload must be accepted",
        );

        let over_cap = vec![0xCDu8; MAX_UDS_FRAME_BYTES as usize + 1];
        let err = try_encode_length_prefixed(&over_cap)
            .expect_err("over-cap payload must be rejected, not silently truncated");
        match err {
            FramingEncodeError::PayloadTooLarge { observed, max } => {
                assert_eq!(observed, MAX_UDS_FRAME_BYTES as usize + 1);
                assert_eq!(max, MAX_UDS_FRAME_BYTES);
            }
        }
    }

    /// Happy path: a stamped envelope round-trips through
    /// `decode_envelope` and yields the original payload.
    /// Both audio and inference variants are exercised so a
    /// regression on either oneof tag is caught.
    #[test]
    fn decode_envelope_accepts_audio() {
        let frame = AudioFrame {
            seq: 1,
            t_us_capture_monotonic: Some(1),
            t_us_publish_unix: Some(2),
            sample_rate: Some(48_000),
            frame_duration_ms: Some(20),
            codec: Some(Codec::Opus(Bytes::from_static(b"\x01\x02"))),
        };
        let wire = wrap_audio(frame.clone());
        let env = decode_envelope(wire.as_ref()).expect("happy decode");
        match env.payload {
            Some(EnvelopePayload::Audio(decoded)) => assert_eq!(decoded, frame),
            other => panic!("expected Audio payload, got {other:?}"),
        }
    }

    #[test]
    fn decode_envelope_accepts_inference() {
        let frame = InferenceFrame {
            seq: 9,
            t_us_capture_monotonic: Some(1),
            t_us_publish_unix: Some(2),
            top_k: vec![TopK {
                class_idx: 0,
                label: "bg".into(),
                prob: 1.0,
            }],
            head_id: Some("00000000-0000-0000-0000-000000000000".into()),
            head_version: Some(1),
        };
        let wire = wrap_inference(frame.clone());
        let env = decode_envelope(wire.as_ref()).expect("happy decode");
        match env.payload {
            Some(EnvelopePayload::Inference(decoded)) => assert_eq!(decoded, frame),
            other => panic!("expected Inference payload, got {other:?}"),
        }
    }

    /// An envelope with no payload variant must be rejected.
    /// Every routed endpoint expects a payload; an empty
    /// envelope is either a producer bug or a probe.
    #[test]
    fn decode_envelope_rejects_missing_payload() {
        let env = Envelope { payload: None };
        let wire = env.encode_to_vec();
        let err = decode_envelope(&wire).expect_err("must reject");
        assert!(matches!(err, ProtoDecodeError::MissingPayload));
    }

    /// Bytes that don't decode at all surface as
    /// `ProtoDecodeError::Decode` carrying the prost source --
    /// helpful for triaging "garbage frame" vs "valid
    /// frame with bad shape".
    #[test]
    fn decode_envelope_rejects_garbage_bytes() {
        // High tag numbers + bogus wire types -- prost will
        // reject before any later checks run.
        let garbage = b"\xff\xff\xff\xff\xff\xff\xff\xff";
        let err = decode_envelope(garbage).expect_err("must reject");
        assert!(matches!(err, ProtoDecodeError::Decode { .. }));
    }
}
