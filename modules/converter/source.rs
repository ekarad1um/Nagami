//! `SourceModel` trait + the production loader (`TfjsSource`).
//! Trait colocates with its `HeadWeights` and `ConvertError`
//! references; the api crate uses [`crate::converter::Pipeline`],
//! which composes `SourceModel` and [`crate::converter::ArtifactSink`]
//! over the [`crate::file_mgr::FsService`] surface.  TFJS is the
//! sole supported source format.
//!
//! The free `convert_tfjs` function in [`crate::converter`] stays as
//! a thin wrapper for one-shot callers; new code paths should drive
//! `Pipeline` directly.
//!
//! [`read_head_bytes_streaming`] walks the shard set in manifest
//! order, hashes each shard into the source SHA-256 (when requested),
//! and copies only the byte ranges that overlap the head's kernel +
//! bias entries into ~24 KB output buffers.  Peak heap is one shard
//! at a time (~50 MB) rather than the concatenated payload.

use crate::converter::{
    ConvertError, ConvertLimits, HeadWeights, SourceKind, TfjsManifest, TfjsManifestEntry,
};
use sha2::{Digest, Sha256};
use std::path::Path;

/// Out-of-band artifacts the source loader computes once and
/// the sink consumes when writing metadata.  Avoids re-hashing
/// the source from the sink + lets the sink emit
/// `ConversionMetadata` without re-reading anything from disk.
#[derive(Clone, Debug)]
pub struct LoadedSource {
    /// Extracted head weights ready for sink publish (already in
    /// Burn's `[in_dim, n_classes]` orientation).
    pub weights: HeadWeights,
    /// SHA-256 of the source-model bytes (model.json + shards in
    /// manifest order).  Lowercase hex.
    pub source_sha256: String,
}

/// Loader for one source-model format.  The production impl
/// ([`TfjsSource`]) wraps the existing `extract_head_from_*`
/// extractors plus deterministic SHA-256 hashing of the source
/// bytes.  Stateless: `&self`.
pub trait SourceModel: Send + Sync + std::fmt::Debug {
    /// Discriminator surfaced into [`crate::converter::ConversionMetadata`].
    fn kind(&self) -> SourceKind;

    /// Read the model directory from `src` and return extracted
    /// weights + the source-bytes SHA-256 for downstream metadata.
    ///
    /// The labels list is forwarded so future formats that
    /// embed labels (TFJS metadata.json) can cross-validate
    /// against the operator-supplied list.  Today's impl
    /// doesn't need it, but the trait surface accepts it for
    /// forward-compat.
    fn load(&self, src: &Path, labels: &[String]) -> Result<LoadedSource, ConvertError>;
}

/// TFJS Layers-Model directory loader (`model.json` + shards).
/// Stateless unit struct; the production daemon uses
/// [`ConvertLimits::default`] for resource caps.  Callers that
/// need a tighter or (with care) looser cap construct
/// [`TfjsSourceLimited`] instead -- both implement
/// [`SourceModel`] and plug into [`crate::converter::Pipeline`]
/// interchangeably.
#[derive(Clone, Copy, Debug, Default)]
pub struct TfjsSource;

impl SourceModel for TfjsSource {
    fn kind(&self) -> SourceKind {
        SourceKind::Tfjs
    }
    fn load(&self, tfjs_dir: &Path, labels: &[String]) -> Result<LoadedSource, ConvertError> {
        TfjsSourceLimited::new(ConvertLimits::default()).load(tfjs_dir, labels)
    }
}

/// TFJS source with explicit [`ConvertLimits`].  Used by
/// callers that want to override the production defaults
/// (e.g. tests that need a small cap to exercise rejection,
/// or future config-driven caps).
#[derive(Clone, Copy, Debug)]
pub struct TfjsSourceLimited {
    limits: ConvertLimits,
}

impl TfjsSourceLimited {
    pub fn new(limits: ConvertLimits) -> Self {
        Self { limits }
    }
}

impl SourceModel for TfjsSourceLimited {
    fn kind(&self) -> SourceKind {
        SourceKind::Tfjs
    }
    fn load(&self, tfjs_dir: &Path, _labels: &[String]) -> Result<LoadedSource, ConvertError> {
        // Read each shard exactly once, feed every
        // byte through the SHA-256 hasher, but only COPY the
        // byte ranges that overlap the head's kernel + bias
        // entries.  Prior code held the full ~250 MB concat
        // blob in RAM to slice out a ~24 KB head; the new path
        // peaks at one shard's worth (~50 MB).
        let model_json = tfjs_dir.join("model.json");
        let json_bytes = std::fs::read(&model_json)
            .map_err(|e| crate::converter::convert_read_err(model_json.display(), e))?;
        let manifest =
            crate::converter::parse_tfjs_manifest_with_limits(&json_bytes, &self.limits)?;
        let (k_entry, b_entry) = crate::converter::pick_tfjs_head_entries(&manifest)?;
        let mut hasher = Sha256::new();
        hasher.update(&json_bytes);
        let (kernel_bytes, bias_bytes) =
            read_head_bytes_streaming(tfjs_dir, &manifest, k_entry, b_entry, Some(&mut hasher))?;
        let source_sha256 = crate::common::hex::hex_lowercase(&hasher.finalize());
        let weights = crate::converter::head_weights_from_head_byte_ranges(
            &manifest,
            k_entry,
            b_entry,
            &kernel_bytes,
            &bias_bytes,
        )?;
        Ok(LoadedSource {
            weights,
            source_sha256,
        })
    }
}

/// Stream-read each shard exactly once: feed its bytes through
/// `hasher` (when supplied) AND copy out just the byte ranges
/// occupied by the head's kernel + bias entries.  Avoids holding
/// the concatenated shard set in RAM (the prior code path kept
/// a `Vec<u8>` of every shard's bytes just to slice an ~80 KB
/// head out of a ~250-300 MB blob).
///
/// Per-shard memory peak is one shard's `Vec<u8>` (<= 50 MB on
/// the canonical 5-shard Teachable-Machine layout); the
/// returned `kernel_bytes` + `bias_bytes` together are ~24 KB.
///
/// `hasher` is `Option<&mut Sha256>` because the public
/// `extract_head_from_tfjs_dir` / `extract_head_from_tfjs`
/// entry points don't need the source-SHA256 (they predate
/// the SHA contract).  Callers that DO need it (e.g.
/// `TfjsSource::load`, `convert_tfjs`) feed `model.json` in
/// before calling and finalize after.
///
/// `manifest.shards` ordering MUST match the cumulative-offset
/// math used at parse time (each entry's `offset_bytes` is the
/// absolute offset into the concatenated shard set in manifest
/// order).  The function checks that the cumulative shard length
/// equals the manifest's declared total -- that defends against
/// a manifest/shard-set mismatch that would otherwise silently
/// produce zeroed kernel bytes.
pub(crate) fn read_head_bytes_streaming(
    tfjs_dir: &Path,
    manifest: &TfjsManifest,
    k: &TfjsManifestEntry,
    b: &TfjsManifestEntry,
    mut hasher: Option<&mut Sha256>,
) -> Result<(Vec<u8>, Vec<u8>), ConvertError> {
    let declared_total: usize = manifest.entries.iter().map(|e| e.len_bytes).sum();
    let mut kernel_bytes = vec![0u8; k.len_bytes];
    let mut bias_bytes = vec![0u8; b.len_bytes];
    let mut cumulative: usize = 0;

    for shard in &manifest.shards {
        let path = tfjs_dir.join(shard);
        // Read this shard whole -- peak adds one shard's bytes
        // to the heap, then drops at end of iteration.  A
        // chunked BufReader is possible but unnecessary on
        // device-class hardware where shards are 50 MB and
        // `max_upload_bytes` is 256 MB.
        let bytes = std::fs::read(&path)
            .map_err(|e| crate::converter::convert_read_err(path.display(), e))?;
        if let Some(h) = hasher.as_deref_mut() {
            h.update(&bytes);
        }
        let shard_start = cumulative;
        let shard_end =
            cumulative
                .checked_add(bytes.len())
                .ok_or_else(|| ConvertError::TfjsShapeOverflow {
                    name: "<cumulative shard offset>".to_string(),
                    shape: vec![],
                })?;

        // For each of {kernel, bias}, find the overlap with
        // this shard's byte range and copy.  `entry` lives at
        // absolute offset [entry.offset_bytes, entry.offset_bytes +
        // entry.len_bytes); shard at [shard_start, shard_end).  The
        // overlap is [max(starts), min(ends)).
        for (entry, out) in [(k, &mut kernel_bytes), (b, &mut bias_bytes)] {
            let entry_end = entry.offset_bytes.saturating_add(entry.len_bytes);
            let overlap_start = entry.offset_bytes.max(shard_start);
            let overlap_end = entry_end.min(shard_end);
            if overlap_start >= overlap_end {
                continue;
            }
            let in_shard = overlap_start - shard_start;
            let in_buffer = overlap_start - entry.offset_bytes;
            let copy_len = overlap_end - overlap_start;
            out[in_buffer..in_buffer + copy_len]
                .copy_from_slice(&bytes[in_shard..in_shard + copy_len]);
        }

        cumulative = shard_end;
    }

    if cumulative != declared_total {
        return Err(ConvertError::TfjsBlobLength {
            have: cumulative,
            declared: declared_total,
        });
    }

    Ok((kernel_bytes, bias_bytes))
}

#[cfg(test)]
mod tests {
    // Tests stage `weights.bin` shard fixtures via direct
    // `std::fs::write`; the production constraint in the
    // workspace `clippy.toml` (writes go through file_mgr's
    // atomic writer) is for production code paths only.
    #![allow(clippy::disallowed_methods)]
    use super::*;
    use crate::converter::TfjsManifestEntry;

    fn entry(name: &str, shape: Vec<usize>, offset: usize) -> TfjsManifestEntry {
        let len_bytes: usize = shape.iter().product::<usize>() * 4;
        TfjsManifestEntry {
            name: name.to_string(),
            shape,
            offset_bytes: offset,
            len_bytes,
        }
    }

    /// Construct a 3-shard layout where the kernel straddles the
    /// shard 0/1 boundary AND a tail of it spills into shard 2;
    /// bias sits entirely in shard 2.  Verifies the byte-range
    /// overlap math handles all three cases (entirely-before,
    /// straddle, entirely-after) for one entry across one
    /// streaming pass.
    #[test]
    fn streaming_read_handles_tensor_straddling_shards() {
        let dir = tempfile::tempdir().expect("tempdir");

        // Synthetic concat blob: shard0 = bytes 0..40, shard1 =
        // 40..80, shard2 = 80..128.  Kernel = bytes 32..104 (72
        // bytes, straddling shards 0/1 AND 1/2).  Bias = bytes
        // 108..120 (12 bytes, in shard2).  All offsets / lengths
        // are 4-byte aligned to match TFJS f32 layout.  Manifest
        // entries cover the full 128 bytes contiguously so the
        // cumulative-shard-length check passes.
        let mut blob = Vec::with_capacity(128);
        for i in 0..128u8 {
            blob.push(i);
        }
        std::fs::write(dir.path().join("s0.bin"), &blob[0..40]).unwrap();
        std::fs::write(dir.path().join("s1.bin"), &blob[40..80]).unwrap();
        std::fs::write(dir.path().join("s2.bin"), &blob[80..128]).unwrap();

        let k = TfjsManifestEntry {
            name: "k".into(),
            shape: vec![18], // unused for byte-overlap test
            offset_bytes: 32,
            len_bytes: 72,
        };
        let b = TfjsManifestEntry {
            name: "b".into(),
            shape: vec![3],
            offset_bytes: 108,
            len_bytes: 12,
        };
        let manifest = TfjsManifest {
            entries: vec![
                entry("pre_k", vec![8], 0),     // 0..32   (32 bytes)
                k.clone(),                      // 32..104 (72 bytes)
                entry("between", vec![1], 104), // 104..108 (4 bytes)
                b.clone(),                      // 108..120 (12 bytes)
                entry("tail", vec![2], 120),    // 120..128 (8 bytes)
            ],
            shards: vec!["s0.bin".into(), "s1.bin".into(), "s2.bin".into()],
        };

        let (kernel_bytes, bias_bytes) =
            read_head_bytes_streaming(dir.path(), &manifest, &k, &b, None).expect("stream");
        assert_eq!(kernel_bytes, blob[32..104]);
        assert_eq!(bias_bytes, blob[108..120]);
    }

    /// Single-shard layout still works (shards.len() == 1, no
    /// straddling).  Smoke test the simple case so a future change
    /// to the boundary math doesn't silently regress the
    /// most-common path.
    #[test]
    fn streaming_read_single_shard() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut blob = vec![0u8; 200];
        for (i, b) in blob.iter_mut().enumerate() {
            *b = (i & 0xff) as u8;
        }
        std::fs::write(dir.path().join("only.bin"), &blob).unwrap();

        let k = TfjsManifestEntry {
            name: "k".into(),
            shape: vec![2, 5],
            offset_bytes: 16,
            len_bytes: 40,
        };
        let b = TfjsManifestEntry {
            name: "b".into(),
            shape: vec![5],
            offset_bytes: 56,
            len_bytes: 20,
        };
        let manifest = TfjsManifest {
            entries: vec![
                entry("pre", vec![4], 0),    // 16 bytes
                k.clone(),                   // 40 bytes -> 56
                b.clone(),                   // 20 bytes -> 76
                entry("post", vec![31], 76), // 124 bytes -> 200
            ],
            shards: vec!["only.bin".into()],
        };

        let (kb, bb) =
            read_head_bytes_streaming(dir.path(), &manifest, &k, &b, None).expect("stream");
        assert_eq!(kb, blob[16..56]);
        assert_eq!(bb, blob[56..76]);
    }

    /// SHA-256 over (model.json bytes ++ shard0 ++ shard1 ++ ...)
    /// matches a reference computed by the prior buffer-based
    /// path.  Confirms the streaming hasher path is byte-equal to
    /// the legacy implementation -- critical because
    /// `source_sha256` is persisted into `metadata.json` and
    /// downstream callers use it as a content-addressable key.
    #[test]
    fn streaming_sha256_matches_concat_hash() {
        let dir = tempfile::tempdir().expect("tempdir");
        let model_json = b"{\"weightsManifest\": []}";
        let s0 = vec![0xAAu8; 64];
        let s1 = vec![0xBBu8; 96];
        std::fs::write(dir.path().join("s0.bin"), &s0).unwrap();
        std::fs::write(dir.path().join("s1.bin"), &s1).unwrap();

        // Reference: hash (json ++ s0 ++ s1) all at once.
        let mut reference_blob = Vec::new();
        reference_blob.extend_from_slice(model_json);
        reference_blob.extend_from_slice(&s0);
        reference_blob.extend_from_slice(&s1);
        let mut h_ref = Sha256::new();
        h_ref.update(&reference_blob);
        let ref_digest = h_ref.finalize();

        // Streaming: hash json first, then walk shards.
        let manifest = TfjsManifest {
            entries: vec![entry("only", vec![64 / 4 + 96 / 4], 0)],
            shards: vec!["s0.bin".into(), "s1.bin".into()],
        };
        // Pick an entry that doesn't overlap any data we care
        // about -- we're just exercising the hasher pass.
        let dummy = TfjsManifestEntry {
            name: "dummy".into(),
            shape: vec![1],
            offset_bytes: 0,
            len_bytes: 4,
        };
        let mut h_stream = Sha256::new();
        h_stream.update(model_json);
        let _ =
            read_head_bytes_streaming(dir.path(), &manifest, &dummy, &dummy, Some(&mut h_stream))
                .expect("stream");
        let stream_digest = h_stream.finalize();

        assert_eq!(stream_digest.as_slice(), ref_digest.as_slice());
    }

    /// If the cumulative shard length disagrees with the
    /// manifest's declared total (e.g. an operator dropped a
    /// shard, or a shard was truncated mid-upload), we surface
    /// `TfjsBlobLength` rather than silently emitting zeroed
    /// kernel bytes.
    #[test]
    fn streaming_blob_length_mismatch_rejected() {
        let dir = tempfile::tempdir().expect("tempdir");
        // Write a 30-byte shard but declare a manifest that
        // expects 40 bytes total.
        let s0 = vec![0u8; 30];
        std::fs::write(dir.path().join("s0.bin"), &s0).unwrap();
        let k = TfjsManifestEntry {
            name: "k".into(),
            shape: vec![10],
            offset_bytes: 0,
            len_bytes: 40,
        };
        let b = TfjsManifestEntry {
            name: "b".into(),
            shape: vec![1],
            offset_bytes: 36,
            len_bytes: 4,
        };
        let manifest = TfjsManifest {
            entries: vec![k.clone(), b.clone()],
            shards: vec!["s0.bin".into()],
        };
        let err = read_head_bytes_streaming(dir.path(), &manifest, &k, &b, None).unwrap_err();
        assert!(
            matches!(
                err,
                ConvertError::TfjsBlobLength {
                    have: 30,
                    declared: 44
                }
            ),
            "{err:?}",
        );
    }
}
