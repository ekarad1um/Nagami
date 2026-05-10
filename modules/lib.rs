//! Acoustics Lab daemon library crate.
//!
//! Each top-level `pub mod` is one architectural concern; modules
//! with multi-file source carry a sibling `<name>/` directory.  The
//! `acousticsd` binary at `bin/acousticsd.rs` calls
//! [`daemon::run`].

// Contract module: dimension newtypes, validated identifiers,
// `ErrorKind` / `Categorized`, time + version primitives,
// object-safe traits.  Inner `#![forbid(unsafe_code)]` keeps the
// guardrail scoped to this subtree only.
pub mod common;

// Wire-format types (prost-generated from `proto/v1/*.proto` via
// the workspace-root `build.rs`) plus sync framing primitives.
pub mod proto;

// Single-writer / multi-reader f32 seqlock ring buffer.  Stdlib-only.
pub mod audio_buffer;

// Linux per-thread CPU affinity + SCHED_FIFO realtime priority
// helpers via direct libc syscalls; non-Linux gets no-op shims.
pub mod sched;

// Workspace + asset management: atomic upload pipeline (tempfile +
// rename + parent-dir fsync), sha256-checked file ingestion,
// `metadata.json` schema versioning, per-workspace registry.
pub mod file_mgr;

// ALSA + mock capture, RMS-arbitrated mic switch.  The
// streaming sinc resampler is in `dsp::resample`; `preproc`
// and `opus_stream` reach it there directly, not through
// capture.
pub mod audio_io;

// Neutral DSP utilities shared across capture / preproc / opus
// (today: the canonical sinc-resampler parameters).  Owns no
// runtime state; consumers wire their own resampler instance
// from the canonical config.
pub mod dsp;

// Burn fp32 acoustic-model definitions (Backbone + Head + Model +
// the `.mpk` recorder mapping).
pub mod model;

// 44.1 kHz mono to 43x232 spectrogram (realfft + rustfft) plus
// operator-side WAV ingest helpers and the bundled Blackman window
// asset.
pub mod preproc;

// Minimal safe wrapper over Rockchip's NPU runtime
// (`librknnrt.so` / `librknnmrt.so`) via hand-committed FFI
// bindings and `libloading` dispatch.
#[cfg(feature = "rknpu")]
pub mod rknn_runtime;

// Streaming hybrid CPU-preproc / NPU-or-Burn-backbone / CPU-head
// classifier engine.  Hot-swappable head, sha256-checked backbone
// catalogue, `BackbonePipeline` enum unified over Burn and RKNN
// backends.
pub mod inference;

// Continuous Opus encoder pipeline: 44.1 kHz mono PCM to 48 kHz to
// 20 ms Opus packets to a broadcast channel.
pub mod opus_stream;

// WebSocket fan-out for audio + inference streams over TCP and
// UDS, with subscriber counting that lets `opus_stream` pause
// encode work when nothing is listening.
pub mod stream_io;

// TOML on-disk config + `ArcSwap` in-memory snapshot + notify-driven
// hot reload.  `LaunchConfig` (mic + backbone catalogues, read once
// at boot) and `Config` (hot-reloadable user preferences).
pub mod config;

// Per-subsystem heartbeats + sysinfo snapshot for `/api/v1/status`,
// published as `ArcSwap<MetricsSnapshot>` at 500 ms cadence.
pub mod status;

// Head-weight extractor from operator-uploaded TFJS Layers-Model
// bundles to Burn `head.mpk` + `labels.txt` + `metadata.json`.
pub mod converter;

// In-process job registry + head-only fine-tune algorithm built on
// Burn's autodiff backend.
pub mod training;

// REST + WebSocket endpoints over axum 0.8
// (`/api/v1/{mic,inference,workspace,training,converter,status}`).
pub mod api;

// Daemon umbrella: boot sequence ([`daemon::run`]), CLI, drain
// registry, and the per-subsystem wiring that ties the rest
// together.
pub mod daemon;
