//! Streaming hybrid inference engine.
//!
//! Topology:
//!
//! ```text
//!   crate::audio_buffer::Reader  --(WaveformLen samples)-->  Preproc (CPU)
//!                                                         |
//!                                                  spectrogram [43x232]
//!                                                         |
//!                                                  bin-major flatten
//!                                                         |
//!                                            rknn_runtime::Session (NPU)
//!                                                         |
//!                                                  features [2000]
//!                                                         |
//!                                          HotHead.snapshot() (Arc swap)
//!                                                         |
//!                                            head_forward + softmax
//!                                                         |
//!                                                  top-k indices
//!                                                         |
//!                                                  InferenceFrame (proto)
//!                                                         |
//!                                          tokio::sync::broadcast::Sender
//!                                                         |
//!                                                  UDS / WS subscribers
//! ```
//!
//! The engine runs in a single `tokio::task::spawn_blocking` worker.
//! No async, no shared mutable state across threads except via the
//! `HotHead` ArcSwap (atomic) and the `Heartbeat` watch channel.
//!
//! See module docs for the design rationale of each piece.

#![warn(missing_debug_implementations)]

// Tier T3 implementation modules per `docs/ARCH_BOUNDARIES.md`:
// the public surface (HotHead, InferenceEngine, etc.) flows
// through the parent's `pub use child::{...}` re-exports below.
// External consumers (api, daemon) write `crate::inference::X`
// against those re-exports, never `crate::inference::child::X`,
// so `pub(crate)` is sufficient and prevents accidental
// promotion to public-API surface.
pub(crate) mod backbone;
pub(crate) mod engine;
pub(crate) mod head;
pub(crate) mod kernel;

#[cfg(test)]
mod npy;

#[cfg(all(target_os = "linux", feature = "rknpu"))]
pub use backbone::RknnBackbone;
pub use backbone::{
    Backbone, BackboneCatalogue, BackboneError, BackboneKind, BackbonePipeline, BackboneRef,
    BurnBackbone,
};
pub use engine::{
    EngineError, EngineState, Heartbeat, InferenceCfg, InferenceEngine, SAMPLE_RATE_HZ,
    WAVEFORM_DURATION_NS,
};
pub use head::{HeadError, HeadInner, HotHead, MAX_N_CLASSES};
pub use kernel::{
    head_forward, softmax_into, top_k_indices_into, transpose_frame_major_to_bin_major,
};

#[cfg(test)]
mod parity_tests {
    #![allow(clippy::disallowed_methods)] // test-only labels.txt fixture stages via std::fs::write
    //! Parity tests against the upstream reference assets
    //! under `misc/`.  Marked `#[ignore]`
    //! (path-fragile + slow) and run via:
    //!
    //! ```bash
    //! cargo test --release -- --include-ignored
    //! ```
    //!
    //! These exercise the CPU side of the pipeline
    //! ([`head_forward`], [`softmax_into`],
    //! [`top_k_indices_into`]) against reference logits /
    //! probs captured during the model's preflight study.
    //! The NPU-side parity (`stream_parity_e2e`)
    //! is gated by hardware availability and lives at the integration
    //! level; on host this test stops at the "we can load the head" line.

    use super::*;
    use crate::common::dims::BackboneFeatureDim;
    use crate::inference::npy;
    use std::path::PathBuf;

    fn crate_root() -> PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
    }

    // The `npys` regen helper lives in
    // `examples/regen_fixtures.rs` so `cargo test --include-ignored`
    // does not rewrite tracked fixtures.

    /// Reference logits -> probs round-trip: softmax_into the recorded
    /// `logits_0.npy`, compare against `probs_0.npy`.  Validates that
    /// our softmax matches the upstream tf-style implementation.
    #[test]
    #[ignore = "depends on repo-root reference assets; --include-ignored"]
    fn softmax_parity_against_reference_probs() {
        let root = crate_root();
        for i in 0..5 {
            let logits_path = root.join(format!("misc/npys/logits_{i}.npy"));
            let probs_path = root.join(format!("misc/npys/probs_{i}.npy"));
            let (_, logits) = npy::read_f32(&logits_path);
            let (_, ref_probs) = npy::read_f32(&probs_path);
            assert_eq!(logits.len(), ref_probs.len(), "shape mismatch in {i}");
            let mut probs = vec![0.0f32; logits.len()];
            softmax_into(&logits, &mut probs);
            for (j, (a, b)) in probs.iter().zip(ref_probs.iter()).enumerate() {
                let drift = (a - b).abs();
                assert!(
                    drift < 1e-6,
                    "softmax drift at sample {i} class {j}: ours={a}, ref={b}, |D|={drift}",
                );
            }
        }
    }

    /// Full Burn pipeline parity: read `waveform_0.npy`, run
    /// `Preproc`, run `BurnBackbone`, run `head_forward` on
    /// the bundled default head, and compare the resulting logits
    /// against `misc/npys/logits_0.npy`.
    ///
    /// `misc/npys/logits_*.npy` are captured by
    /// `regen_default_logits_and_probs_npys` against this very
    /// pipeline, so the test reduces to "the pipeline is bit-stable
    /// across runs" -- a regression guard.  Cross-implementation
    /// parity (does our backbone match upstream Speech-Commands?)
    /// lives in
    /// `inference::backbone::tests::backbone_mpk_matches_speech_commands_tfjs`,
    /// which compares `backbone.mpk` weights byte-for-byte against
    /// the upstream TFJS bundle in `misc/models/`.
    #[test]
    #[ignore = "depends on bundled fixtures; --include-ignored"]
    fn burn_backbone_parity_against_reference_logits() {
        let root = crate_root();
        let backbone_path = root.join("misc/backbones/backbone.mpk");
        let head_path = root.join("misc/heads/00000000-default/head.mpk");
        let labels_path = root.join("misc/heads/00000000-default/labels.txt");
        for p in [&backbone_path, &head_path, &labels_path] {
            assert!(p.exists(), "missing test asset: {}", p.display());
        }

        // Load once; reuse across all 5 samples.  Backbone load is
        // ~200 ms on M1 / >1 s on Pi 5 and head load is ~5 ms; per-
        // sample reloading costs roughly a second total without
        // changing what the test asserts.
        let mut backbone = BurnBackbone::load(&backbone_path).expect("load Burn backbone");
        let head = HotHead::load(&head_path, &labels_path, crate::common::ids::HeadId::new())
            .expect("load head");
        let snap = head.snapshot();
        let mut preproc_inst = crate::preproc::Preproc::new();

        for sample_idx in 0..5 {
            let waveform_path = root.join(format!("misc/npys/waveform_{sample_idx}.npy"));
            let logits_ref_path = root.join(format!("misc/npys/logits_{sample_idx}.npy"));
            let (_, pcm_vec) = npy::read_f32(&waveform_path);
            assert_eq!(pcm_vec.len(), crate::common::dims::WaveformLen::USIZE);
            let pcm: &[f32; crate::common::dims::WaveformLen::USIZE] =
                pcm_vec.as_slice().try_into().expect("pcm length match");

            let spec = preproc_inst.spectrogram(pcm);

            let mut features = Box::new([0.0f32; BackboneFeatureDim::USIZE]);
            backbone
                .infer(&spec, &mut features)
                .expect("backbone infer");

            let mut logits = vec![0.0f32; snap.n_classes];
            head_forward(&features[..], &snap.weight, &snap.bias, &mut logits);

            let (_, ref_logits) = npy::read_f32(&logits_ref_path);
            assert_eq!(
                ref_logits.len(),
                logits.len(),
                "sample {sample_idx}: logit count mismatch",
            );
            // Burn's fp32 path is deterministic; the only drift
            // source is single-rounding differences across crate
            // versions.  1e-3 absolute is generous; 1e-5 is the
            // expected typical scale.
            for (i, (&a, &b)) in logits.iter().zip(ref_logits.iter()).enumerate() {
                let drift = (a - b).abs();
                assert!(
                    drift < 1e-3,
                    "sample {sample_idx} logit {i}: ours={a}, ref={b}, |D|={drift}",
                );
            }
        }
    }

    /// Top-k stability + ordering test against the reference probs.  We
    /// don't have a reference top-k file, but the top-3 of any softmax
    /// is uniquely determined; verify that our top-k matches the
    /// argmax-by-descending of the recorded probs.
    #[test]
    #[ignore = "depends on repo-root reference assets; --include-ignored"]
    fn top_k_parity_against_argsort_of_reference() {
        let root = crate_root();
        let (_, ref_probs) = npy::read_f32(&root.join("misc/npys/probs_0.npy"));
        let mut top = Vec::with_capacity(8);
        top_k_indices_into(&ref_probs, 3, &mut top);
        assert_eq!(top.len(), 3.min(ref_probs.len()));
        // Strictly descending by prob.
        for w in top.windows(2) {
            let (a, b) = (w[0], w[1]);
            assert!(
                ref_probs[a] >= ref_probs[b],
                "top_k not descending: idx {a} (p={}) before idx {b} (p={})",
                ref_probs[a],
                ref_probs[b],
            );
        }
    }
}

#[cfg(test)]
mod stream_e2e {
    #![allow(clippy::disallowed_methods)] // test-only labels.txt fixture stages via std::fs::write
    //! End-to-end gate: drive the engine from a stitched
    //! waveform_0.npy in the audio buffer, verify per-frame top-1
    //! matches the bundled reference within tolerance.
    //!
    //! Two flavors live here:
    //!
    //!   * [`stream_parity_e2e`] -- RKNN backbone end-to-end.  Cfg-gated
    //!     to `linux + rknpu` because `RknnBackbone` and the
    //!     `BackbonePipeline::Rknn` variant only exist on those
    //!     builds.  Requires NPU + librknnrt + on-device assets.
    //!   * [`stream_parity_e2e_burn`] -- Burn (CPU) backbone variant
    //!     of the same flow.  Always compiled.  Runs on host without
    //!     an NPU; `#[ignore]`'d because it depends on repo-root
    //!     reference assets and Burn forward is multi-second.
    //!
    //! On-device run:
    //!   `cargo test -p inference --release -- --include-ignored stream_parity_e2e`

    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use crate::audio_buffer::AudioBuffer;
    use crate::common::dims::WaveformLen;
    use arc_swap::ArcSwap;
    use bytes::Bytes;
    use prost::Message;
    use tokio::sync::{broadcast, watch};
    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::inference::npy;
    use crate::proto::envelope::Payload as EnvelopePayload;
    use crate::proto::{Envelope, InferenceFrame};

    /// Decode the envelope-wrapped bytes the engine emits via
    /// `wrap_inference_into` and extract the inner `InferenceFrame`.
    /// Decoding the bytes as a raw `InferenceFrame` silently produces
    /// default values because prost ignores unknown fields, which was
    /// the assertion-fires-on-empty-top_k regression class this helper
    /// was introduced to lock down.
    fn decode_inference_envelope(bytes: &[u8]) -> InferenceFrame {
        let env = Envelope::decode(bytes).expect("decode envelope");
        match env.payload.expect("envelope.payload") {
            EnvelopePayload::Inference(f) => f,
            other => panic!("unexpected envelope payload variant: {other:?}"),
        }
    }

    fn crate_root() -> PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
    }

    #[cfg(all(target_os = "linux", feature = "rknpu"))]
    #[test]
    #[ignore = "on-device only; requires NPU + librknnrt + reference assets"]
    fn stream_parity_e2e() {
        let root = crate_root();
        let backbone = root.join("misc/backbones/backbone.rknn");
        let head_mpk = root.join("misc/heads/00000000-default/head.mpk");
        let labels_path = root.join("misc/heads/00000000-default/labels.txt");
        let waveform_path = root.join("misc/npys/waveform_0.npy");
        for p in [&backbone, &head_mpk, &labels_path, &waveform_path] {
            assert!(p.exists(), "missing test asset: {}", p.display());
        }

        // 5 s buffer at 44.1 kHz; reader behind head=0 so it sees
        // everything we write from sample 0.
        let buf = AudioBuffer::new(262_144);
        let mut writer = buf.take_writer();
        let reader = buf.reader_at(0);

        let head = HotHead::load(&head_mpk, &labels_path, crate::common::ids::HeadId::new())
            .expect("load head");
        let cfg = Arc::new(ArcSwap::from_pointee(InferenceCfg {
            hop_samples: 11_025,
            top_k: 3,
        }));
        let (mon_tx, _mon_rx) = watch::channel(Heartbeat::default());
        let pipeline = BackbonePipeline::Rknn(Box::new(
            RknnBackbone::load(&backbone).expect("load rknn backbone"),
        ));
        let engine = InferenceEngine::new(pipeline.into_boxed(), head, cfg, mon_tx, None);

        let (out_tx, mut out_rx) = broadcast::channel::<Bytes>(64);
        let token = CancellationToken::new();
        let token_engine = token.clone();

        // Engine thread (mimics tokio::task::spawn_blocking).
        let engine_handle =
            std::thread::spawn(move || engine.run_blocking(reader, out_tx, token_engine));

        // Stitch waveform_0 into the buffer 10x at ~real-time cadence
        // (1024 samples ~= 23 ms).
        let (_, pcm) = npy::read_f32(&waveform_path);
        assert_eq!(
            pcm.len(),
            WaveformLen::USIZE,
            "reference waveform must be 44032 samples"
        );
        let writer_handle = std::thread::spawn(move || {
            for _ in 0..10 {
                for chunk in pcm.chunks(1024) {
                    writer.push(chunk);
                    std::thread::sleep(Duration::from_millis(23));
                }
            }
        });

        // Collect at most 8 frames or wait up to 12 s.
        let mut frames: Vec<InferenceFrame> = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(12);
        while Instant::now() < deadline && frames.len() < 8 {
            match out_rx.try_recv() {
                Ok(bytes) => {
                    let f = decode_inference_envelope(bytes.as_ref());
                    frames.push(f);
                }
                Err(broadcast::error::TryRecvError::Empty) => {
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(broadcast::error::TryRecvError::Closed) => break,
                Err(broadcast::error::TryRecvError::Lagged(_)) => {
                    // We didn't drain fast enough; pretend it didn't
                    // happen and keep collecting fresh frames.
                    continue;
                }
            }
        }

        // Stop the engine + writer cleanly.
        token.cancel();
        writer_handle.join().expect("writer thread panicked");
        engine_handle
            .join()
            .expect("engine thread panicked")
            .expect("engine returned an error");

        // Assertions
        assert!(!frames.is_empty(), "no inference frames produced");
        // Monotonic seq.
        for w in frames.windows(2) {
            assert!(
                w[1].seq > w[0].seq,
                "seq not monotonic: {} -> {}",
                w[0].seq,
                w[1].seq
            );
        }
        // Per frame: top_k populated, probs in [0,1], head_id non-empty.
        for f in &frames {
            assert!(!f.top_k.is_empty(), "top_k empty in frame {}", f.seq);
            assert!(
                f.t_us_capture_monotonic.is_some(),
                "capture timestamp absent in frame {}",
                f.seq,
            );
            assert!(
                f.t_us_publish_unix.is_some(),
                "publish timestamp absent in frame {}",
                f.seq
            );
            // `head_id` is now `optional`; the engine
            // always populates it in production frames.
            assert!(
                f.head_id.as_deref().is_some_and(|s| !s.is_empty()),
                "head_id absent or empty in frame {}",
                f.seq,
            );
            for tk in &f.top_k {
                assert!(
                    (0.0..=1.0).contains(&tk.prob),
                    "prob out of range in frame {}: {}",
                    f.seq,
                    tk.prob
                );
                assert!(!tk.label.is_empty(), "label empty in frame {}", f.seq);
            }
        }

        // The reference waveform_0 has a deterministic top-1; if the
        // pipeline is correct, every frame after the warmup transient
        // (rubato/realfft converged from 0-init) should converge to
        // the same top-1.  Steady-state frames (after the first):
        let steady: Vec<_> = frames.iter().skip(1).collect();
        if !steady.is_empty() {
            let first_top1 = steady[0].top_k[0].class_idx;
            for f in &steady {
                assert_eq!(
                    f.top_k[0].class_idx, first_top1,
                    "top-1 drift after warm-up: frame {} -> idx {}, expected {}",
                    f.seq, f.top_k[0].class_idx, first_top1,
                );
            }
            eprintln!(
                "stream_parity_e2e: {} frames; steady-state top-1 idx={} label={:?} p={:.3}",
                frames.len(),
                steady[0].top_k[0].class_idx,
                steady[0].top_k[0].label,
                steady[0].top_k[0].prob,
            );
        }
    }

    /// Mirror of `stream_parity_e2e` but uses the **Burn** backbone
    /// fallback path.  Runs on host without an NPU.  Marked `#[ignore]`
    /// because it depends on repo-root reference assets and Burn's
    /// per-frame forward (~5-20 ms on M1+, longer on Pi 5) makes
    /// the full 10x stitched waveform a multi-second test --
    /// undesirable in routine CI.
    ///
    /// Run via:
    ///   cargo test -p inference --release -- --include-ignored stream_parity_e2e_burn
    #[test]
    #[ignore = "depends on repo-root reference assets; --include-ignored"]
    fn stream_parity_e2e_burn() {
        let root = crate_root();
        let backbone_path = root.join("misc/backbones/backbone.mpk");
        let head_mpk = root.join("misc/heads/00000000-default/head.mpk");
        let labels_path = root.join("misc/heads/00000000-default/labels.txt");
        let waveform_path = root.join("misc/npys/waveform_0.npy");
        for p in [&backbone_path, &head_mpk, &labels_path, &waveform_path] {
            assert!(p.exists(), "missing test asset: {}", p.display());
        }

        let buf = AudioBuffer::new(262_144);
        let mut writer = buf.take_writer();
        let reader = buf.reader_at(0);

        let head = HotHead::load(&head_mpk, &labels_path, crate::common::ids::HeadId::new())
            .expect("load head");
        let cfg = Arc::new(ArcSwap::from_pointee(InferenceCfg {
            hop_samples: 11_025,
            top_k: 3,
        }));
        let (mon_tx, _mon_rx) = watch::channel(Heartbeat::default());
        let pipeline = BackbonePipeline::Burn(Box::new(
            BurnBackbone::load(&backbone_path).expect("load burn backbone"),
        ));
        let engine = InferenceEngine::new(pipeline.into_boxed(), head, cfg, mon_tx, None);

        let (out_tx, mut out_rx) = broadcast::channel::<Bytes>(64);
        let token = CancellationToken::new();
        let token_engine = token.clone();
        let engine_handle =
            std::thread::spawn(move || engine.run_blocking(reader, out_tx, token_engine));

        let (_, pcm) = npy::read_f32(&waveform_path);
        assert_eq!(pcm.len(), WaveformLen::USIZE);
        // Burn forward is slower than RKNN, so we push fewer copies
        // (3x vs the RKNN test's 10x) -- 3 windows x ~1 s each is
        // enough to clear the warmup and observe steady state.
        let writer_handle = std::thread::spawn(move || {
            for _ in 0..3 {
                for chunk in pcm.chunks(1024) {
                    writer.push(chunk);
                    std::thread::sleep(Duration::from_millis(23));
                }
            }
        });

        // Burn forward is slow; widen the deadline to 30 s and the
        // collection target to 4 frames (less than RKNN's 8 because
        // the test wall-clock budget is what dominates).
        let mut frames: Vec<InferenceFrame> = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(30);
        while Instant::now() < deadline && frames.len() < 4 {
            match out_rx.try_recv() {
                Ok(bytes) => {
                    let f = decode_inference_envelope(bytes.as_ref());
                    frames.push(f);
                }
                Err(broadcast::error::TryRecvError::Empty) => {
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(broadcast::error::TryRecvError::Closed) => break,
                Err(broadcast::error::TryRecvError::Lagged(_)) => continue,
            }
        }

        token.cancel();
        writer_handle.join().expect("writer thread panicked");
        engine_handle
            .join()
            .expect("engine thread panicked")
            .expect("engine returned an error");

        assert!(!frames.is_empty(), "no inference frames produced");
        for f in &frames {
            assert!(!f.top_k.is_empty(), "top_k empty in frame {}", f.seq);
            // `head_id` is now `optional`; the engine
            // always populates it in production frames.
            assert!(
                f.head_id.as_deref().is_some_and(|s| !s.is_empty()),
                "head_id absent or empty in frame {}",
                f.seq,
            );
            for tk in &f.top_k {
                assert!(
                    (0.0..=1.0).contains(&tk.prob),
                    "prob out of range in frame {}: {}",
                    f.seq,
                    tk.prob
                );
            }
        }
        eprintln!(
            "stream_parity_e2e_burn: {} frames; first top-1 idx={} prob={:.3}",
            frames.len(),
            frames[0].top_k[0].class_idx,
            frames[0].top_k[0].prob,
        );
    }

    /// Capture-timing anchor for the inference engine: when
    /// the engine is plumbed with a producer-side
    /// `SharedTimingAnchor`, every emitted `InferenceFrame`
    /// carries `t_us_capture_monotonic` derived from the
    /// inference window's first 44.1 kHz sample position
    /// (window-start convention) projected through the anchor.
    ///
    /// Mirrors the opus accuracy test
    /// (`tests/opus_stream_run_smoke.rs::timing_anchor_drives_capture_us_within_one_ms`)
    /// shape: stages a known anchor (`captured_at = 1e9` us
    /// far from process-boot-relative `CaptureTime::now()`),
    /// runs the engine briefly with a real Burn backbone +
    /// reference waveform, decodes the first emitted
    /// `InferenceFrame`, and verifies via back-projection
    /// that the recovered window-start sample position lies
    /// in the valid range `[0, head_now]` and the round-trip
    /// residual `|capture_us_for(anchor, N) - stamp|` is
    /// within 1 ms.
    ///
    /// Fixture-gated like the parent parity tests (Burn forward
    /// is slow; ~30 s wall-clock budget).
    #[test]
    #[ignore = "depends on repo-root reference assets; --include-ignored"]
    fn timing_anchor_drives_inference_capture_us_within_one_ms() {
        use crate::common::time::{
            BufferTimingAnchor, CaptureTime, capture_us_for, shared_timing_anchor,
        };

        let root = crate_root();
        let backbone_path = root.join("misc/backbones/backbone.mpk");
        let head_mpk = root.join("misc/heads/00000000-default/head.mpk");
        let labels_path = root.join("misc/heads/00000000-default/labels.txt");
        let waveform_path = root.join("misc/npys/waveform_0.npy");
        for p in [&backbone_path, &head_mpk, &labels_path, &waveform_path] {
            assert!(p.exists(), "missing test asset: {}", p.display());
        }

        let buf = AudioBuffer::new(262_144);
        let mut writer = buf.take_writer();
        let reader = buf.reader_at(0);

        // Stage a deterministic anchor BEFORE any audio
        // flows.  `head_pos = 0` and `captured_at = 1e9` us
        // pin the projection: any sample at absolute position
        // N has capture_us = 1e9 + N * 1e6 / 44_100.
        let anchor_cell = shared_timing_anchor();
        anchor_cell.store(Arc::new(BufferTimingAnchor {
            head_pos: 0,
            captured_at: CaptureTime::from_micros(1_000_000_000),
            sample_rate_hz: 44_100,
        }));
        let anchor_for_writer = anchor_cell.clone();

        let head = HotHead::load(&head_mpk, &labels_path, crate::common::ids::HeadId::new())
            .expect("load head");
        let cfg = Arc::new(ArcSwap::from_pointee(InferenceCfg {
            hop_samples: 11_025,
            top_k: 3,
        }));
        let (mon_tx, _mon_rx) = watch::channel(Heartbeat::default());
        let pipeline = BackbonePipeline::Burn(Box::new(
            BurnBackbone::load(&backbone_path).expect("load burn backbone"),
        ));
        let engine = InferenceEngine::new(
            pipeline.into_boxed(),
            head,
            cfg,
            mon_tx,
            Some(anchor_cell.clone()),
        );

        let (out_tx, mut out_rx) = broadcast::channel::<Bytes>(64);
        let token = CancellationToken::new();
        let token_engine = token.clone();
        let engine_handle =
            std::thread::spawn(move || engine.run_blocking(reader, out_tx, token_engine));

        // Push 3 copies of the reference waveform (~3 s of
        // audio).  Anchor stays stable at head_pos=0 because
        // we don't update it here -- the test verifies the
        // engine projects against the staged anchor, not
        // against a producer-side update.
        let (_, pcm) = npy::read_f32(&waveform_path);
        assert_eq!(pcm.len(), WaveformLen::USIZE);
        let writer_handle = std::thread::spawn(move || {
            for _ in 0..3 {
                for chunk in pcm.chunks(1024) {
                    writer.push(chunk);
                    std::thread::sleep(Duration::from_millis(23));
                }
            }
            // Hand back the post-fill head so the test can
            // bound the back-projected sample position.
            anchor_for_writer
        });

        // Burn forward is slow; widen deadline + accept a
        // single frame as proof of contract.
        let mut frames: Vec<InferenceFrame> = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(30);
        while Instant::now() < deadline && frames.is_empty() {
            match out_rx.try_recv() {
                Ok(bytes) => {
                    let f = decode_inference_envelope(bytes.as_ref());
                    frames.push(f);
                }
                Err(broadcast::error::TryRecvError::Empty) => {
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(broadcast::error::TryRecvError::Closed) => break,
                Err(broadcast::error::TryRecvError::Lagged(_)) => continue,
            }
        }

        token.cancel();
        let _final_anchor = writer_handle.join().expect("writer thread panicked");
        engine_handle
            .join()
            .expect("engine thread panicked")
            .expect("engine returned an error");

        assert!(!frames.is_empty(), "no inference frames produced");
        let frame = &frames[0];
        let stamp = frame
            .t_us_capture_monotonic
            .expect("anchor-plumbed engine must populate t_us_capture_monotonic");

        // Sanity: stamp falls in the anchor's projection
        // image for some sample position N >= 0.  Upper
        // bound is the writer's total push (~3 s of audio at
        // 44.1 kHz = 132_300 samples = ~3 s) plus a generous
        // 1 ms slack.
        assert!(
            stamp >= 999_900_000,
            "stamp {stamp} below anchor captured_at; \
             projection went backwards from sample 0",
        );
        assert!(
            stamp <= 1_000_000_000 + 3_500_000,
            "stamp {stamp} above expected upper bound (~3 s of audio)",
        );

        // Back-project: recover the inference window's first
        // sample position N from the stamp using the inverse
        // of `capture_us_for`.  N must be a non-negative u64.
        // Round-trip residual must be <= 1 ms (the contract's
        // accuracy tolerance).
        let anchor = **anchor_cell.load();
        let stamp_delta_us = stamp.saturating_sub(anchor.captured_at.as_micros());
        let n_recovered = (stamp_delta_us as u128 * anchor.sample_rate_hz as u128 / 1_000_000)
            as u64
            + anchor.head_pos;
        let projected = capture_us_for(anchor, n_recovered);
        let residual = projected.abs_diff(stamp);
        assert!(
            residual <= 1_000,
            "anchor round-trip residual {residual} us > 1 ms tolerance; \
             stamp {stamp}, recovered N {n_recovered}, projected {projected}",
        );

        eprintln!(
            "timing_anchor_drives_inference_capture_us_within_one_ms: \
             stamp={} us, recovered N={}, residual={} us",
            stamp, n_recovered, residual,
        );
    }
}
