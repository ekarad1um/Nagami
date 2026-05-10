//! Lowercase-hex byte encoding.
//!
//! Lives in `common` so every layer that stamps a digest
//! (`file_mgr`'s upload validators, the converter's source-sha256,
//! the inference backbone's optional verify, the api-route upload
//! digest) shares one encoder; the `inference -> file_mgr` edge
//! is forbidden by the layer guard at
//! `tests/dependency_edge_guard.rs`, so the helper cannot live
//! in `file_mgr`.  Pure stdlib; respects `common`'s
//! `#![forbid(unsafe_code)]`.

/// Encode `bytes` as lowercase ASCII hex.
///
/// Direct nibble-to-ASCII lookup; the `core::fmt`-driven
/// `format!("{b:02x}")` spelling is roughly 5x slower per byte
/// (`core::fmt::Formatter` handles padding / sign / fill, plus
/// allocates per-call) and we are on the hot side of every
/// upload digest.
pub fn hex_lowercase(bytes: &[u8]) -> String {
    static HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = vec![0u8; bytes.len() * 2];
    for (i, &b) in bytes.iter().enumerate() {
        out[2 * i] = HEX[(b >> 4) as usize];
        out[2 * i + 1] = HEX[(b & 0x0f) as usize];
    }
    // Only ASCII hex digits were written.
    String::from_utf8(out).expect("ascii hex is utf8")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_yields_empty_string() {
        assert_eq!(hex_lowercase(&[]), "");
    }

    #[test]
    fn known_vector() {
        assert_eq!(hex_lowercase(&[0x00, 0x0f, 0xa5, 0xff]), "000fa5ff");
    }

    #[test]
    fn output_length_is_double_input() {
        let payload: Vec<u8> = (0..=255u8).collect();
        assert_eq!(hex_lowercase(&payload).len(), payload.len() * 2);
    }
}
