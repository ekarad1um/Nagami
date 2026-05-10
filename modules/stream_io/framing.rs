//! Async UDS framing helpers.
//!
//! The sync wrap helpers + protocol constants live in
//! [`crate::proto::framing`] (so `opus_stream` / `inference` can
//! import them without picking up `stream_io`'s tokio dep
//! tree).  This module hosts the async UDS reader counterpart
//! that needs `tokio::io::AsyncRead`.
//!
//! ## UDS length-prefix framing contract
//!
//! For raw-UDS clients (a future Python CLI / on-host
//! integration that doesn't want WebSocket upgrade ceremony),
//! each envelope on the stream is prefixed with a 4-byte
//! little-endian length:
//!
//! ```text
//! [u32 LE: payload_len] [payload_len bytes: Envelope-encoded]
//! ```
//!
//! Server-side wiring is **deferred** (no in-tree raw-UDS
//! consumer today).  The helpers ship here so a future PR
//! adding a raw-UDS endpoint just wires them up.  The contract
//! is documented in [`docs/PROTO.md`](../../docs/PROTO.md).
//!
//! Readers MUST close the connection on any framing error
//! (oversized prefix, truncated payload, I/O error).  Re-syncing
//! is not defined -- the prefix IS the synchronization point.

use bytes::Bytes;

// Re-export the sync helpers + protocol constants so
// `stream_io::framing::*` remains a one-stop import for the
// daemon's WS handlers.  `try_encode_length_prefixed` is the
// checked counterpart of `decode_length_prefixed` -- both
// share the `MAX_UDS_FRAME_BYTES` cap, so an encoder cannot
// emit a frame the decoder will reject.  `decode_envelope`
// centralises the envelope-decode policy from `docs/PROTO.md`
// so future raw-UDS / WS-with-decode receivers can't invent
// their own decode handling.
pub use crate::proto::framing::{
    FramingEncodeError, MAX_UDS_FRAME_BYTES, ProtoDecodeError, WS_SUBPROTOCOL, decode_envelope,
    try_encode_length_prefixed, wrap_audio, wrap_inference,
};

/// Errors from raw-UDS length-prefix framing.  Server-side
/// wiring is deferred (no current consumer); the helpers ship
/// for a future PR.
#[derive(Debug, thiserror::Error)]
pub enum FramingError {
    /// Length prefix exceeds [`MAX_UDS_FRAME_BYTES`].  Probably
    /// a buggy or hostile peer; caller MUST close the
    /// connection.
    #[error(
        "length prefix {observed} exceeds max {max}; close connection -- \
         re-sync not defined for length-prefixed framing"
    )]
    OversizedPrefix { observed: u32, max: u32 },
    /// Stream ended before the declared `length` bytes arrived
    /// (truncated payload).  Caller MUST close the connection.
    #[error("payload truncated: declared {declared} bytes, never completed before EOF")]
    Truncated { declared: u32 },
    /// Underlying I/O failure while reading the prefix or
    /// payload.
    #[error("io: {source}")]
    Io {
        #[source]
        source: std::io::Error,
    },
}

/// Decode one length-prefixed frame from `reader`.  Returns the
/// payload bytes (without the prefix).  On any framing error
/// the caller MUST close the connection -- the prefix IS the
/// synchronization point and resync is not defined.
pub async fn decode_length_prefixed<R>(reader: &mut R) -> Result<Bytes, FramingError>
where
    R: tokio::io::AsyncRead + Unpin,
{
    use tokio::io::AsyncReadExt;
    let mut len_buf = [0u8; 4];
    reader
        .read_exact(&mut len_buf)
        .await
        .map_err(|source| FramingError::Io { source })?;
    let len = u32::from_le_bytes(len_buf);
    if len > MAX_UDS_FRAME_BYTES {
        return Err(FramingError::OversizedPrefix {
            observed: len,
            max: MAX_UDS_FRAME_BYTES,
        });
    }
    let mut payload = vec![0u8; len as usize];
    match reader.read_exact(&mut payload).await {
        Ok(_) => Ok(Bytes::from(payload)),
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
            Err(FramingError::Truncated { declared: len })
        }
        Err(source) => Err(FramingError::Io { source }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn length_prefix_round_trip() {
        let payload = b"\x01\x02\x03\x04\x05".to_vec();
        let framed = try_encode_length_prefixed(&payload).expect("under cap");
        let mut cursor = std::io::Cursor::new(framed.to_vec());
        let decoded = decode_length_prefixed(&mut cursor).await.expect("decode");
        assert_eq!(decoded.as_ref(), payload.as_slice());
    }

    #[tokio::test]
    async fn length_prefix_rejects_oversized() {
        // Length = MAX + 1, then no body -- decoder rejects on
        // the prefix without attempting to read the payload.
        let mut prefix = Vec::with_capacity(4);
        prefix.extend_from_slice(&(MAX_UDS_FRAME_BYTES + 1).to_le_bytes());
        let mut cursor = std::io::Cursor::new(prefix);
        let err = decode_length_prefixed(&mut cursor)
            .await
            .expect_err("oversized prefix must reject");
        assert!(matches!(err, FramingError::OversizedPrefix { .. }));
    }

    #[tokio::test]
    async fn length_prefix_rejects_truncated_payload() {
        // Prefix declares 10 bytes, only 4 follow.
        let mut buf = Vec::with_capacity(8);
        buf.extend_from_slice(&10u32.to_le_bytes());
        buf.extend_from_slice(b"abcd");
        let mut cursor = std::io::Cursor::new(buf);
        let err = decode_length_prefixed(&mut cursor)
            .await
            .expect_err("truncated payload must reject");
        assert!(matches!(err, FramingError::Truncated { .. }));
    }
}
