//! P.4 baseline bench (inference): end-to-end per-window latency
//! on the Burn fp32 (CPU) backbone path.
//!
//! Per-window pipeline mirroring `engine::run_blocking_inner`:
//!
//!   PCM[44032] --Preproc.spectrogram_into--> spec[43][232]
//!              --acoustics_lab::model::Backbone::forward---> features[2000]
//!              --acoustics_lab::inference::head_forward----> logits[N_CLASSES]
//!
//! The bench skips:
//!
//!   * the audio_buffer `peek_into` (covered by `audio_buffer/benches/ring_throughput.rs`),
//!   * softmax + top-k (microseconds; not the hot path),
//!   * acoustics_lab::proto::encode + broadcast::send (covered nowhere yet --
//!     reasonable since the broadcast send is an Arc-clone).
//!
//! Backbone weights are random-initialised via Burn's default
//! initializer.  The matmul/conv shapes are unchanged from
//! production, so wall-clock time is representative; absolute
//! logits are meaningless (the bench never asserts numerical
//! correctness).
//!
//! The RKNN path is intentionally *not* benched here -- it requires
//! librknnrt + a Linux + aarch64 host with the Rockchip NPU
//! attached.  RKNN regression detection happens in the deployment
//! lab against real hardware.

use acoustics_lab::common::dims::{BackboneFeatureDim, NBins, NFrames, WaveformLen};
use acoustics_lab::inference::head_forward;
use acoustics_lab::model::Backbone;
use acoustics_lab::preproc::Preproc;
use burn::backend::ndarray::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

const N_CLASSES: usize = 10; // representative head size

type B = NdArray<f32>;

fn bench_run_window_burn(c: &mut Criterion) {
    // Deterministic 1 kHz sine + harmonics -- drives the FFT/conv
    // path through non-trivial energy bins (silence would be a
    // sub-realistic best case).
    let pcm: Box<[f32; WaveformLen::USIZE]> = {
        let mut buf = Box::new([0.0f32; WaveformLen::USIZE]);
        for (i, s) in buf.iter_mut().enumerate() {
            let t = i as f32 / 44_100.0;
            *s = 0.5 * (2.0 * std::f32::consts::PI * 1_000.0 * t).sin()
                + 0.2 * (2.0 * std::f32::consts::PI * 2_500.0 * t).sin();
        }
        buf
    };

    let mut preproc = Preproc::new();
    let device = NdArrayDevice::default();
    let backbone = Backbone::<B>::new(&device);

    // Pre-allocated scratch matching the engine's per-loop buffers.
    let mut spec = Box::new([[0.0f32; NBins::USIZE]; NFrames::USIZE]);
    let mut features = Box::new([0.0f32; BackboneFeatureDim::USIZE]);
    let head_weight: Vec<f32> = vec![0.001; BackboneFeatureDim::USIZE * N_CLASSES];
    let head_bias: Vec<f32> = vec![0.0; N_CLASSES];
    let mut logits: Vec<f32> = vec![0.0; N_CLASSES];

    let mut group = c.benchmark_group("inference/run_window_burn");
    // One window = one "inference event" emitted per HopSamples cadence.
    group.throughput(Throughput::Elements(1));
    group.bench_function("preproc+backbone+head", |b| {
        b.iter(|| {
            // 1) Spectrogram (CPU FFT) -- `spectrogram_into` writes
            //    into `spec` in place; no allocation per window.
            preproc.spectrogram_into(black_box(&pcm), &mut spec);

            // 2) Backbone forward (Burn NdArray fp32).  Unavoidable
            //    `Vec<f32>` allocation inside `BurnBackbone::infer`
            //    today (Burn TensorData consumes its Vec) -- same
            //    behaviour the engine sees in production.
            let flat: Vec<f32> = spec.as_slice().as_flattened().to_vec();
            let input = burn::tensor::Tensor::<B, 4>::from_data(
                burn::tensor::TensorData::new(flat, [1, 1, NFrames::USIZE, NBins::USIZE]),
                &device,
            );
            let output = backbone.forward(input);
            let data = output.into_data();
            let slice = data.as_slice::<f32>().expect("burn slice");
            features.copy_from_slice(slice);

            // 3) Head forward (hand-written matmul).  Same kernel
            //    the engine calls.
            head_forward(&features[..], &head_weight, &head_bias, &mut logits);
            black_box(&logits[0]);
        });
    });
    group.finish();
}

criterion_group!(benches, bench_run_window_burn);
criterion_main!(benches);
