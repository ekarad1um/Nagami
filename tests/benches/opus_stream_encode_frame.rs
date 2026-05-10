//! P.4 baseline bench (opus_stream): encode wall time per 20 ms
//! Opus frame.
//!
//! Drives `OpusEngine::process_pcm` with one `PCM_PULL_CHUNK`
//! (1024 samples @ 44.1 kHz, ~23 ms) per iteration, mirroring the
//! daemon's per-period feed rate.  The engine resamples 44.1 -> 48 kHz
//! and emits ~one 20 ms Opus packet per call (output cadence depends
//! on the resampler's buffered residue; one feed averages ~1.15
//! packets).
//!
//! Bitrate, complexity, FEC settings inherited from production
//! defaults (`BITRATE_BPS = 32_000`, `COMPLEXITY = 5`,
//! `Application::Audio`).  The encoder warms up over the first ~5
//! frames (predictor state converges); criterion's default
//! `warm_up_time = 3 s` is enough to reach steady state.

use acoustics_lab::opus_stream::{OpusEngine, PCM_PULL_CHUNK};
use bytes::Bytes;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

fn bench_encode_one_frame(c: &mut Criterion) {
    // Deterministic synthetic signal: 1 kHz sine at 44.1 kHz, mono.
    // Real Opus encode times are content-dependent (silence is
    // cheaper than music); a sine is a representative middle-ground
    // and reproducible across runs.
    let pcm: Vec<f32> = (0..PCM_PULL_CHUNK)
        .map(|i| {
            let t = i as f32 / 44_100.0;
            (2.0 * std::f32::consts::PI * 1000.0 * t).sin() * 0.5
        })
        .collect();

    let mut engine = OpusEngine::new().expect("opus encoder init");
    let mut out: Vec<Bytes> = Vec::with_capacity(2);

    // Warm up the resampler + encoder predictors.
    for _ in 0..16 {
        out.clear();
        engine.process_pcm(&pcm, &mut out).expect("warmup encode");
    }

    let mut group = c.benchmark_group("opus_stream/encode_frame");
    group.throughput(Throughput::Elements(PCM_PULL_CHUNK as u64));
    group.bench_function("1024_samples_in_per_call", |b| {
        b.iter(|| {
            out.clear();
            let n = engine
                .process_pcm(black_box(&pcm), &mut out)
                .expect("encode");
            black_box(n)
        });
    });
    group.finish();
}

criterion_group!(benches, bench_encode_one_frame);
criterion_main!(benches);
