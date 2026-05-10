//! P.4 baseline bench (audio_buffer): seqlock ring read/write
//! throughput.
//!
//! Two scenarios mirror the daemon's hot path:
//!
//!  * `push_period` -- `Writer::push` of one ALSA period (1024 f32
//!    samples = ~23 ms at 44.1 kHz).  Captures the cost of seqlock
//!    publish on the write side.
//!  * `peek_window` -- `Reader::peek_into` of one inference window
//!    (44 032 f32 samples = ~1.0 s at 44.1 kHz).  Captures the cost
//!    of seqlock-protected double-read + checksum on the read side.
//!
//! Capacity is the daemon's canonical 262 144 (2^18) so wrap-around
//! frequency matches production.  Pre-warmed: writer pushes 4x
//! window-worth of samples before measurement starts so peeks land
//! in the steady-state path (no cold-cache effect).

use acoustics_lab::audio_buffer::AudioBuffer;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

const CAPACITY: usize = 262_144; // 2^18 -- production value
const PERIOD: usize = 1_024; // ALSA period
const WINDOW: usize = 44_032; // inference window (acoustics_lab::common::dims::WaveformLen)

fn bench_push_period(c: &mut Criterion) {
    let buf = AudioBuffer::new(CAPACITY);
    let mut writer = buf.take_writer();
    let samples: Vec<f32> = (0..PERIOD).map(|i| (i as f32).sin()).collect();

    let mut group = c.benchmark_group("audio_buffer/push_period");
    group.throughput(Throughput::Elements(PERIOD as u64));
    group.bench_function("1024_samples", |b| {
        b.iter(|| {
            writer.push(black_box(&samples));
        });
    });
    group.finish();
}

fn bench_peek_window(c: &mut Criterion) {
    let buf = AudioBuffer::new(CAPACITY);
    let mut writer = buf.take_writer();
    let samples: Vec<f32> = (0..PERIOD).map(|i| (i as f32).sin()).collect();
    // Pre-warm: push enough periods that head > WINDOW + safety margin.
    // We need head >= peek-distance + window for `peek_into` to land in
    // the Ready path (lag = 0 case).
    let warmup_periods = (WINDOW * 2).div_ceil(PERIOD) + 4;
    for _ in 0..warmup_periods {
        writer.push(&samples);
    }

    let reader = buf.reader_at(WINDOW); // peek the most recent window
    let mut out = vec![0.0f32; WINDOW];

    let mut group = c.benchmark_group("audio_buffer/peek_window");
    group.throughput(Throughput::Elements(WINDOW as u64));
    group.bench_function("44032_samples", |b| {
        b.iter(|| {
            let _status = reader.peek_into(black_box(&mut out));
        });
    });
    group.finish();
}

criterion_group!(benches, bench_push_period, bench_peek_window);
criterion_main!(benches);
