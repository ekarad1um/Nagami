//! `.mpk` head-artifact persistence header.
//!
//! # Wire layout (32 bytes, little-endian)
//!
//! ```text
//! offset  size  field
//! 0       8     magic         = b"ACSTHEAD"
//! 8       4     header_version = 1u32
//! 12      4     feature_dim   = u32 (e.g. 2000 for the canonical backbone)
//! 16      4     num_classes   = u32 (matches labels.len())
//! 20      4     reserved      = 0u32 (room for hidden_dim / num_layers; bump header_version when used)
//! 24      4     payload_len   = u32 (bytes of prost payload that follow)
//! 28      4     header_crc32  = CRC32-IEEE of bytes [0..28)
//! ```
//!
//! Total: 32 bytes.  Loaders read the header, assert
//! `feature_dim == BackboneFeatureDim::USIZE` and `num_classes
//! == labels.len()`, validate `header_crc32`, then decode
//! `payload_len` bytes of Burn-Mpk via the existing recorder.
//! The CRC catches truncation / corruption before the prost
//! decoder gets confused.
//!
//! # Why a hand-rolled CRC32 (not a crate)
//!
//! The header is 28 bytes; CRC32 is invoked exactly once on
//! write and once on load.  Importing `crc32fast` would add a
//! new workspace dep for ~10 LoC of work.  The standard IEEE
//! polynomial (`0xEDB88320`) is universally understood -- any
//! external tool that wants to verify an `ACSTHEAD` artifact
//! reaches the same value via any standard CRC-32 library.
//!
//! # Forward-compat
//!
//! `header_version=1` is the load-bearing distinction.  Older
//! daemons that don't know about a future `header_version=2`
//! return [`HeadHeaderError::SchemaTooNew`]; newer daemons
//! reading a `header_version=1` artifact treat the `reserved`
//! field as zero and proceed (it is defined as zero today).
//! There is no auto-detect of pre-header `.mpk` files -- they
//! fail with [`HeadHeaderError::BadMagic`] and the operator
//! regenerates via the converter.

use std::io::Write;

/// Magic bytes that identify an `ACSTHEAD`-prefixed `.mpk`.
pub const HEAD_MAGIC: &[u8; 8] = b"ACSTHEAD";

/// Header version stamped on every freshly-written artifact.
/// Bumped only when the header layout changes (e.g. when the
/// `reserved` slot acquires meaning).
pub const HEAD_HEADER_VERSION: u32 = 1;

/// On-disk header size in bytes.  Loaders read exactly this
/// many bytes before parsing.
pub const HEAD_HEADER_SIZE: usize = 32;

/// Decoded header fields.  Loaders cross-check `feature_dim`
/// and `num_classes` against runtime expectations (the
/// backbone + labels file) and use `payload_len` to bound the
/// prost payload read.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HeadHeader {
    pub header_version: u32,
    pub feature_dim: u32,
    pub num_classes: u32,
    pub payload_len: u32,
}

/// Failure shapes from [`parse_header`].
#[derive(Debug, thiserror::Error)]
pub enum HeadHeaderError {
    /// Input slice was shorter than [`HEAD_HEADER_SIZE`];
    /// indexing the fixed offsets would otherwise panic.
    #[error("header too short: got {got} bytes, need at least {min}")]
    TooShort { got: usize, min: usize },
    /// File doesn't begin with the `ACSTHEAD` magic,
    /// typically a pre-header `.mpk`.  Operator regenerates
    /// via the
    /// converter.
    #[error("bad magic: expected {:?}, got {got:?}", HEAD_MAGIC)]
    BadMagic { got: [u8; 8] },
    /// `header_version` is greater than this daemon
    /// understands.  Operator upgrades the daemon.
    #[error("schema too new: header_version={found} > supported max {supported}")]
    SchemaTooNew { found: u32, supported: u32 },
    /// CRC32 check failed; the header is truncated or corrupt
    /// in transit.  Loader MUST refuse to decode the payload.
    #[error("header CRC mismatch: stored=0x{stored:08x}, computed=0x{computed:08x}")]
    BadCrc { stored: u32, computed: u32 },
    /// I/O error reading the header from the file.
    #[error("io: {source}")]
    Io {
        #[source]
        source: std::io::Error,
    },
}

/// Serialize a [`HeadHeader`] to its 32-byte on-disk
/// representation.  Computes the CRC over bytes [0..28) and
/// stamps it at offset 28.
pub fn serialize_header(
    feature_dim: u32,
    num_classes: u32,
    payload_len: u32,
) -> [u8; HEAD_HEADER_SIZE] {
    let mut buf = [0u8; HEAD_HEADER_SIZE];
    buf[0..8].copy_from_slice(HEAD_MAGIC);
    buf[8..12].copy_from_slice(&HEAD_HEADER_VERSION.to_le_bytes());
    buf[12..16].copy_from_slice(&feature_dim.to_le_bytes());
    buf[16..20].copy_from_slice(&num_classes.to_le_bytes());
    // Reserved at [20..24] stays zero.
    buf[24..28].copy_from_slice(&payload_len.to_le_bytes());
    let crc = crc32_ieee(&buf[..28]);
    buf[28..32].copy_from_slice(&crc.to_le_bytes());
    buf
}

/// Parse a 32-byte header from `bytes`.  Callers typically
/// read exactly [`HEAD_HEADER_SIZE`] from the file before
/// calling, but the function is panic-free on any slice: a
/// short input returns [`HeadHeaderError::TooShort`].  The v1
/// `reserved` field at offset `[20..24]` is ignored on read
/// (the CRC already detects unintended drift); a future
/// `header_version=2` may repurpose those bytes.
///
/// # Errors
///
/// - [`HeadHeaderError::TooShort`] if `bytes.len() <
///   HEAD_HEADER_SIZE`.
/// - [`HeadHeaderError::BadMagic`] if the leading 8 bytes do
///   not match [`HEAD_MAGIC`].
/// - [`HeadHeaderError::SchemaTooNew`] if `header_version >
///   HEAD_HEADER_VERSION`.
/// - [`HeadHeaderError::BadCrc`] if the stored CRC does not
///   match a recomputation over `bytes[..28]`.
pub fn parse_header(bytes: &[u8]) -> Result<HeadHeader, HeadHeaderError> {
    if bytes.len() < HEAD_HEADER_SIZE {
        return Err(HeadHeaderError::TooShort {
            got: bytes.len(),
            min: HEAD_HEADER_SIZE,
        });
    }
    let magic: [u8; 8] = bytes[0..8].try_into().expect("8-byte slice");
    if &magic != HEAD_MAGIC {
        return Err(HeadHeaderError::BadMagic { got: magic });
    }
    let header_version = u32::from_le_bytes(bytes[8..12].try_into().expect("4-byte slice"));
    if header_version > HEAD_HEADER_VERSION {
        return Err(HeadHeaderError::SchemaTooNew {
            found: header_version,
            supported: HEAD_HEADER_VERSION,
        });
    }
    let stored_crc = u32::from_le_bytes(bytes[28..32].try_into().expect("4-byte slice"));
    let computed_crc = crc32_ieee(&bytes[..28]);
    if stored_crc != computed_crc {
        return Err(HeadHeaderError::BadCrc {
            stored: stored_crc,
            computed: computed_crc,
        });
    }
    let feature_dim = u32::from_le_bytes(bytes[12..16].try_into().expect("4-byte slice"));
    let num_classes = u32::from_le_bytes(bytes[16..20].try_into().expect("4-byte slice"));
    let payload_len = u32::from_le_bytes(bytes[24..28].try_into().expect("4-byte slice"));
    Ok(HeadHeader {
        header_version,
        feature_dim,
        num_classes,
        payload_len,
    })
}

/// Stream-write the header followed by `payload` bytes to
/// `writer`.  Used by `MpkSink::publish` to compose the
/// on-disk artifact.
///
/// `payload.len()` must fit in `u32` (the on-disk
/// `payload_len` slot); otherwise an
/// [`std::io::ErrorKind::InvalidInput`] is returned without
/// writing anything.  The `.mpk` format reserves 4 bytes for
/// the length so a silent narrowing cast would corrupt the
/// header.
pub fn write_with_payload<W: Write>(
    writer: &mut W,
    feature_dim: u32,
    num_classes: u32,
    payload: &[u8],
) -> std::io::Result<()> {
    let payload_len = payload_len_or_err(payload.len())?;
    let header = serialize_header(feature_dim, num_classes, payload_len);
    writer.write_all(&header)?;
    writer.write_all(payload)?;
    Ok(())
}

/// Narrow `usize -> u32` for the payload-length slot, surfacing
/// overflow as a typed `io::Error` instead of silently
/// truncating.  Split out so the contract is unit-testable
/// without allocating a >4 GiB payload on the host.
fn payload_len_or_err(len: usize) -> std::io::Result<u32> {
    u32::try_from(len).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "head payload too large: {len} bytes > u32::MAX ({})",
                u32::MAX
            ),
        )
    })
}

// MARK: CRC-32 IEEE 802.3

const CRC32_TABLE: [u32; 256] = {
    let mut t = [0u32; 256];
    let mut i = 0u32;
    while i < 256 {
        let mut c = i;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                0xEDB88320u32 ^ (c >> 1)
            } else {
                c >> 1
            };
            k += 1;
        }
        t[i as usize] = c;
        i += 1;
    }
    t
};

/// CRC-32 IEEE polynomial `0xEDB88320` (the same checksum used
/// by zip, gzip, ext4 journal, etc.).  Pure-Rust, no dep.
fn crc32_ieee(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in bytes {
        crc = CRC32_TABLE[((crc ^ b as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_header() {
        let feature_dim = 2000;
        let num_classes = 17;
        let payload_len = 12_345;
        let bytes = serialize_header(feature_dim, num_classes, payload_len);
        assert_eq!(bytes.len(), HEAD_HEADER_SIZE);
        let parsed = parse_header(&bytes).expect("parse");
        assert_eq!(parsed.header_version, HEAD_HEADER_VERSION);
        assert_eq!(parsed.feature_dim, feature_dim);
        assert_eq!(parsed.num_classes, num_classes);
        assert_eq!(parsed.payload_len, payload_len);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = serialize_header(2000, 2, 100);
        bytes[0] = b'Z';
        // Recompute CRC so the bad-magic arm fires (otherwise
        // BadCrc would mask BadMagic).
        let crc = crc32_ieee(&bytes[..28]);
        bytes[28..32].copy_from_slice(&crc.to_le_bytes());
        let err = parse_header(&bytes).expect_err("bad magic must reject");
        assert!(matches!(err, HeadHeaderError::BadMagic { .. }));
    }

    #[test]
    fn rejects_schema_too_new() {
        let mut bytes = serialize_header(2000, 2, 100);
        bytes[8..12].copy_from_slice(&(HEAD_HEADER_VERSION + 1).to_le_bytes());
        let crc = crc32_ieee(&bytes[..28]);
        bytes[28..32].copy_from_slice(&crc.to_le_bytes());
        let err = parse_header(&bytes).expect_err("future schema must reject");
        assert!(matches!(err, HeadHeaderError::SchemaTooNew { .. }));
    }

    #[test]
    fn rejects_bad_crc() {
        let mut bytes = serialize_header(2000, 2, 100);
        // Flip a payload-relevant byte WITHOUT updating the CRC.
        bytes[12] ^= 0xFF;
        let err = parse_header(&bytes).expect_err("bad CRC must reject");
        assert!(matches!(err, HeadHeaderError::BadCrc { .. }));
    }

    /// Sanity-check the CRC32-IEEE implementation against the
    /// canonical fixed-input vector: `b"123456789"` ->
    /// `0xCBF43926`.
    #[test]
    fn crc32_ieee_matches_canonical_test_vector() {
        assert_eq!(crc32_ieee(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn write_with_payload_round_trips() {
        let payload = b"\x01\x02\x03\x04\x05\x06\x07\x08";
        let mut buf = Vec::new();
        write_with_payload(&mut buf, 2000, 2, payload).expect("write");
        assert_eq!(buf.len(), HEAD_HEADER_SIZE + payload.len());
        let header = parse_header(&buf[..HEAD_HEADER_SIZE]).expect("parse header");
        assert_eq!(header.payload_len, payload.len() as u32);
        assert_eq!(&buf[HEAD_HEADER_SIZE..], payload);
    }

    /// Public contract: empty input must produce `TooShort`,
    /// not a panic from slice indexing.
    #[test]
    fn parse_header_rejects_empty_input() {
        let err = parse_header(&[]).expect_err("empty input must reject");
        assert!(matches!(
            err,
            HeadHeaderError::TooShort {
                got: 0,
                min: HEAD_HEADER_SIZE
            }
        ));
    }

    /// One byte short of the header size still must not
    /// trip an out-of-bounds index.
    #[test]
    fn parse_header_rejects_31_byte_input() {
        let bytes = [0u8; HEAD_HEADER_SIZE - 1];
        let err = parse_header(&bytes).expect_err("31 bytes must reject");
        assert!(matches!(
            err,
            HeadHeaderError::TooShort {
                got: 31,
                min: HEAD_HEADER_SIZE
            }
        ));
    }

    /// `payload_len_or_err` (the helper that backs
    /// [`write_with_payload`]'s overflow check) returns
    /// `InvalidInput` for any `usize > u32::MAX`.  Tested via
    /// the helper to avoid a >4 GiB allocation on the host.
    #[test]
    fn write_with_payload_rejects_oversized_length() {
        // Sanity: in-range values pass through verbatim.
        assert_eq!(payload_len_or_err(0).unwrap(), 0);
        assert_eq!(payload_len_or_err(u32::MAX as usize).unwrap(), u32::MAX);

        // Overflow path.  `u32::MAX as usize + 1` is only
        // representable on 64-bit hosts; the build target is
        // 64-bit aarch64 + dev x86_64, both qualify.
        let too_big: usize = (u32::MAX as usize) + 1;
        let err = payload_len_or_err(too_big).expect_err("must reject");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
        assert!(
            err.to_string().contains("u32::MAX"),
            "diagnostic must name the cap, got {err}"
        );

        let err2 = payload_len_or_err(usize::MAX).expect_err("must reject");
        assert_eq!(err2.kind(), std::io::ErrorKind::InvalidInput);
    }
}
