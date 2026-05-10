// Test fixture management uses `std::fs::write` to materialise the
// .bin files when UPDATE_SNAPSHOTS=1 -- the production constraint in
// `clippy.toml` (writes go through file_mgr's atomic writer) does not
// apply to test scaffolding.
#![allow(clippy::disallowed_methods)]

//! Wire-format snapshot.
//!
//! Encodes one canonical instance of every public message type in
//! `acoustics_lab::proto::*` and asserts the bytes match a checked-in fixture.
//! The fixture's job is to make every accidental wire-format change
//! a deliberate, PR-visible byte diff:
//!
//!   * `cargo test -p acoustics-lab --test proto_snapshot_current` is green when
//!     the wire format is stable.
//!   * Setting `UPDATE_SNAPSHOTS=1` rewrites the fixtures in place.
//!     Use this when a wire-format change is intentional; the
//!     diff in the fixture file then carries the wire-format delta
//!     into the PR review.
//!
//! Single-version protocol: no v0/v1/v2 fork (the daemon's clients
//! are limited to its own consumers, no third-party peers in the
//! wild).  Re-versioning at any future point is a fresh-start
//! replacement; this fixture would be regenerated in lockstep.

use acoustics_lab::proto::{AudioFrame, InferenceFrame, TopK};
use prost::Message;
use std::path::{Path, PathBuf};

fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("proto_fixtures")
        .join(format!("{name}.bin"))
}

/// Compare `actual` bytes against the fixture file.  Auto-creates the
/// fixture if it doesn't exist or if `UPDATE_SNAPSHOTS=1` is set.
fn assert_snapshot(name: &str, actual: &[u8]) {
    let path = fixture_path(name);
    let update = std::env::var_os("UPDATE_SNAPSHOTS").is_some();

    if update || !path.exists() {
        std::fs::create_dir_all(path.parent().unwrap()).expect("mkdir fixtures");
        std::fs::write(&path, actual).expect("write fixture");
        eprintln!(
            "snapshot {} written ({} bytes) -- re-run without UPDATE_SNAPSHOTS to verify",
            path.display(),
            actual.len()
        );
        return;
    }

    let expected =
        std::fs::read(&path).unwrap_or_else(|e| panic!("read fixture {}: {e}", path.display()));
    assert_eq!(
        actual,
        expected.as_slice(),
        "wire-format snapshot mismatch for `{name}` (encoded {} bytes, fixture {} bytes). \
         If this change is intentional, re-run with UPDATE_SNAPSHOTS=1 and \
         document the wire delta in the PR.",
        actual.len(),
        expected.len(),
    );
}

// MARK: Canonical messages -- deterministic, exercise non-zero field values

fn canonical_audio_frame() -> AudioFrame {
    use acoustics_lab::proto::audio_frame::Codec;
    AudioFrame {
        seq: 0xDEAD_BEEF_CAFE_F00D,
        // Clock-domain disambiguation.  Both stamps
        // present in the canonical fixture so the snapshot
        // exercises both proto field tags.
        t_us_capture_monotonic: Some(123_456_789),
        t_us_publish_unix: Some(1_700_000_000_000_000),
        sample_rate: Some(48_000),
        frame_duration_ms: Some(20),
        // `oneof codec` discriminator.  Variant tag
        // 10 (was field 3 pre-rename); same payload bytes
        // 0..=127.
        codec: Some(Codec::Opus((0..=127u8).collect::<Vec<u8>>().into())),
    }
}

fn canonical_inference_frame() -> InferenceFrame {
    InferenceFrame {
        seq: 7,
        // Renamed/renumbered.  The pre-rename fixture
        // had `ts_ns = 2` and `window_start_ns = 3`; v1 drops
        // `window_start_ns` (load-bearing for nothing) and
        // splits the timestamp into clock-explicit pair.
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
        head_id: Some("00000000-0000-0000-0000-000000000001".into()),
        // Atomic with the head snapshot via
        // `HotHead::snapshot_with_version`.  Always present
        // in production frames.
        head_version: Some(42),
    }
}

fn canonical_top_k() -> TopK {
    TopK {
        class_idx: 42,
        label: "alpha".into(),
        prob: 0.5,
    }
}

// MARK: Snapshot tests -- one per message type

#[test]
fn audio_frame_snapshot() {
    assert_snapshot("audio_frame", &canonical_audio_frame().encode_to_vec());
}

#[test]
fn inference_frame_snapshot() {
    assert_snapshot(
        "inference_frame",
        &canonical_inference_frame().encode_to_vec(),
    );
}

#[test]
fn top_k_snapshot() {
    assert_snapshot("top_k", &canonical_top_k().encode_to_vec());
}

// MARK: Round-trip -- orthogonal to snapshot, kept here for adjacency

#[test]
fn snapshots_round_trip() {
    let orig = canonical_audio_frame();
    let bytes = orig.encode_to_vec();
    assert_eq!(AudioFrame::decode(bytes.as_slice()).unwrap(), orig);

    let orig = canonical_inference_frame();
    let bytes = orig.encode_to_vec();
    assert_eq!(InferenceFrame::decode(bytes.as_slice()).unwrap(), orig);
}
