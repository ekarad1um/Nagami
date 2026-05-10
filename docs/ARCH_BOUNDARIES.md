# Architecture Boundaries

This document is the source of truth for the layered structure of
the `acoustics_lab` crate; both the dependency-edge guard and
per-module visibility decisions track against it. Sibling docs:
[ARCHITECTURE.md](ARCHITECTURE.md),
[BUILD.md](BUILD.md),
[API.md](API.md),
[PROTO.md](PROTO.md),
[LAYOUT.md](LAYOUT.md).

## Layer Graph

The crate is a single-binary monolith organized as five layers.
A module may only depend on modules in lower layers (and on the
narrow contract layer). Cross-layer dependencies upward, or
horizontal dependencies that bypass `daemon`, are forbidden.

```
                 ┌──────────────────────────┐
        L5       │           daemon         │   composition root
                 │  (run, drain_registry)   │   (only here)
                 └─────────────┬────────────┘
                               │ composes everything below
                 ┌─────────────┴────────────┐
        L4       │            api           │   control surface
                 │  (axum routes, AppState) │   (HTTP / WS)
                 └─────────────┬────────────┘
        ┌────────┬─────────────┼──────────────────┬────────────┐
        │        │             │                  │            │
        ▼        ▼             ▼                  ▼            ▼
   converter  training      inference           status       config
   (cold path  (Burn         (hybrid hot path,   (heartbeat   (TOML +
    extract)   fine-tune)    backbone + head)    + sysinfo)   ArcSwap)
        L3        L3              L3              L3              L3
        │         │               │               │               │
        ├─────────┴───────┬───────┴────────┬──────┴───────────────┤
        │                 │                │                      │
        ▼                 ▼                ▼                      ▼
     model            preproc          opus_stream            stream_io
     (Burn topology   (STFT, WAV       (Opus encode +         (WS / UDS
      + ACSTHEAD)     ingest)          broadcast)             fan-out)
        L2               L2               L2                      L2
                          │                │
                          ▼                ▼
                       audio_io       audio_buffer
                       (ALSA / mock,  (lock-free
                        mic arb)     seqlock ring)
                          L2              L2

                  rknn_runtime          sched         file_mgr
                  (FFI wrapper,         (CPU/RT       (workspace,
                   feature-gated)        shim)         atomic write)
                          L2              L2              L2

                                 dsp
                            (shared sinc-
                             resampler config;
                             consumed by
                             audio_io / preproc /
                             opus_stream)
                                 L2

                       ┌────────────────────────────────┐
        L1             │   common      proto            │   contracts
                       │ (ids, dims,  (wire format,     │   + wire
                       │  errors)     framing)          │
                       └────────────────────────────────┘
```

Concrete edge map (production-like; `#[cfg(test)]`-only edges
excluded):

| Module | Allowed direct dependencies |
|---|---|
| `common` | (none) |
| `proto` | `common` |
| `audio_buffer` | (none) |
| `sched` | (none) |
| `file_mgr` | `common` |
| `dsp` | `common` (for `Categorized` trait impl on `StreamingResampleError`) |
| `audio_io` | `audio_buffer`, `common`, `dsp`, `sched` |
| `model` | `common` |
| `preproc` | `audio_io` (capture-side caps `MAX_CHANNELS` / `MIN_SAMPLE_RATE` / `MAX_SAMPLE_RATE` for WAV ingest validation), `common`, `dsp` |
| `rknn_runtime` | (none; behind `feature = "rknpu"`) |
| `inference` | `audio_buffer`, `common`, `model`, `preproc`, `proto`, `rknn_runtime` |
| `opus_stream` | `audio_buffer`, `audio_io`, `common`, `dsp`, `proto` |
| `stream_io` | `common`, `proto` |
| `config` | `audio_io`, `common`, `file_mgr` (durability primitive only), `inference`, `stream_io` |
| `status` | `common` |
| `converter` | `common`, `file_mgr`, `model` |
| `training` | `common`, `file_mgr`, `model`, `preproc` |
| `api` | `audio_io`, `common`, `config`, `converter`, `file_mgr`, `inference`, `model`, `status`, `training` |
| `daemon` | every L2/L3/L4 module |

Resolved + remaining layering smells:

1. ~~`preproc -> audio_io` and `opus_stream -> audio_io` for
   shared sinc-resampler configuration.~~ **Resolved**: the
   resampler lives in `crate::dsp::resample`; both consumers
   and `audio_io` itself depend on `dsp` instead. The upward
   edges that existed solely for shared DSP config are gone.
2. `config -> {audio_io, inference, stream_io}` is acceptable
   (TOML must reflect runtime types); `config` types are
   classified between schema, runtime projection, and live
   store (see the taxonomy in the `config` module doc).

## Visibility Tiers

These tiers govern the Rust visibility a new symbol gets;
new public surfaces must justify themselves against this
table.

| Tier | Intended consumers | Example modules / items | Rust visibility |
|---|---|---|---|
| **T1 — Stable contract** | external callers (tests, fixtures, future library users), every layer | `common::ids`, `common::dims`, `common::traits`, `proto::{Envelope, AudioFrame, InferenceFrame, ...}`, `proto::framing`, the `audio_buffer` reader/writer types, selected `model`/`inference` DTOs | `pub` |
| **T2 — Internal contract** | sibling layers and `daemon` | `inference::engine`, `audio_io::MicArbitrator`, `config::Config`, `file_mgr::{FsService, JobRegistry}`, `status::StatusMonitor`, `stream_io::StreamRouter`, `training::TrainingRegistry`, `converter::Pipeline` | `pub` |
| **T3 — Implementation** | only the owning module | `api::routes::*` internals, `converter::{pipeline,sink,source}`, `daemon::drain_registry::Slot`, `audio_io::source::*` driver internals, `rknn_runtime::sys` | `pub(crate)` or `mod` |
| **T4 — Fixture / bench helper** | tests, benches, examples only | regen helpers, fixture builders | `#[cfg(any(test, feature = "fixture-tools"))]` once introduced |

Concrete rules:

1. **`daemon` is the only composition root.** Only `daemon` may
   construct concrete instances of every L2 / L3 / L4
   subsystem. Other modules may call into one or two siblings
   via published ports (`HeadStore`, `FsService`,
   `StatusReporter`, `LagSource`, `MicSettingsHandle`,
   `TrainingRegistry`, `ConfigHandle`), but they must not
   wire all of them together.
2. **`common` and `proto` stay dependency-light.** `common` has
   no production-edge dependencies; `proto` depends only on
   `common`. Neither layer may depend on a runtime module
   (`audio_io`, `inference`, `daemon`, ...). `common`
   subtree-forbids `unsafe` (`#![forbid(unsafe_code)]` lives
   inside `common`).
3. **`rknn_runtime::sys` is `pub(crate)`.** Raw C-shaped
   structs and `unsafe extern "C"` declarations do not leak
   through the public module graph. The unsafe FFI surface
   stays narrow (`Session::load` is `unsafe fn`; the single
   inference-side caller wraps the call with a `SAFETY:`
   comment naming deployment preconditions).
4. **No new `pub mod` without a named consumer.** Any new
   public child module must say in its module doc which crate-
   external or fixture-only consumer needs it. Otherwise it
   stays `mod` or `pub(crate) mod`.
5. **No `cfg(test)` import in production code.** Test helpers
   live behind `#[cfg(any(test, feature = "fixture-tools"))]`
   in their owning module; nothing under `modules/` outside
   tests should observe them.
6. **No backwards-compatibility re-exports.** This is a fresh
   start; renaming a type means callers update at the same
   commit, not adding a `pub use OldName = NewName;` shim.

## Product Profile

The crate ships a **single product**: a full all-in-one lab
daemon. The default Cargo features (`autodiff`, `mimalloc`)
build the production binary; the optional features
(`alsa-real`, `rknpu`, `panic-inject`) gate target-specific or
test-only code paths.

This is a deliberate decision, not an accident:

- The 4-core ARM A53 / 2 GB RAM SBC target runs all activities
  in one process: real-time audio capture, hybrid CPU/NPU
  inference, Opus broadcast, the HTTP / WS control plane,
  workspace persistence, on-device fine-tuning, and TFJS
  conversion.
- Resource caps in `audio_io`, `audio_buffer`, `preproc`,
  `converter`, and `training` are sized for that SBC; running
  the same binary in "inference only" mode would not save
  enough RAM to justify a feature-gating epic.
- Operators interact with one binary, one config schema, one
  status surface; splitting the product into multiple binaries
  (or a feature matrix per binary) would multiply operator
  complexity without removing the dependency graph that already
  pulls training + converter into the default build.
- Training and converter live behind capped boundaries (drop-
  ratio cap, manifest size cap, archive precheck, finite-data
  policy) and run as cold-path admin actions; their default-on
  status is recorded in [`BUILD.md`](BUILD.md) §Cargo features.

### When to revisit

The decision is reversible. Trigger conditions for splitting
the product into a smaller "inference-only appliance" feature
profile:

1. A real deployment requires the smaller binary footprint
   (current footprint is acceptable for the target SBC; this is
   not a hypothesis).
2. Training / converter is removed from the API contract
   altogether (`api::routes::training` and
   `api::routes::converter` become deployment-time disabled).
3. The autodiff dependency tree creates measurable cold-path
   memory pressure on the SBC (today it does not).

If any trigger fires, the minimal refactor is:

- `default = ["mimalloc"]`;
- a `training` feature gating `training`, `autodiff`,
  `rayon`, plus the `api::routes::training` module;
- a `converter` feature gating TFJS conversion and the
  `api::routes::converter` module.

The refactor is intentionally not pre-built: feature
combinatorics are expensive to maintain and we have no
deployment driving them.

## Governance

1. **Machine-checked.** [`tests/dependency_edge_guard.rs`](../tests/dependency_edge_guard.rs)
   scans every production `crate::<other>::` reference and
   fails the build if it lists an edge missing from the
   allowlist there. Update the allowlist in lockstep with
   the edge map above.
2. **Trigger condition for a workspace split.** This crate
   stays single-crate until an external library consumer or
   long-term team scale forces a split. Visibility shrinkage
   stays inside the single crate; it does not split it.

## Cross-Document Consistency

If this file changes, also update:

- `docs/ARCHITECTURE.md` if a top-level module renames or the
  composition-root rule moves;
- `docs/BUILD.md` if the deployment failure model changes (e.g.
  from "external supervision" to anything else);
- `modules/lib.rs` if a top-level module is added, removed, or
  renamed.
