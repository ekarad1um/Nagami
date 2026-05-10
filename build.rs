//! prost-build codegen for the wire-format protos.
//!
//! `prost-build` emits one Rust file per proto `package`
//! declaration into `$OUT_DIR/<package>.rs`; [`crate::proto`]
//! `include!`s each file relative to that `OUT_DIR`.
//!
//! Requires `protoc` on `PATH` (macOS: `brew install
//! protobuf`, Debian/Ubuntu: `apt install protobuf-compiler`).

use std::io::Result;

fn main() -> Result<()> {
    // Register the `audio_buffer_loom` cfg so
    // `RUSTFLAGS=--cfg audio_buffer_loom` builds (used by
    // `tests/audio_buffer_loom.rs`) compile without the
    // `unexpected_cfgs` warning.  We use a custom cfg name
    // rather than the bare `loom` cfg to avoid tripping
    // tokio's `#![cfg(not(loom))]` gates in sibling dev-deps
    // (tokio-tungstenite would otherwise fail to compile when
    // tokio's `net` module is elided under `--cfg loom`).
    // No loom symbols leak into normal builds: the test file
    // is gated by `#[cfg(audio_buffer_loom)]`.
    println!("cargo:rustc-check-cfg=cfg(audio_buffer_loom)");
    // Protos live under `modules/proto/` and declare package
    // `acoustics`.  Single-version layout per the second-round
    // refactor: there is no v1/v2 split because there is no
    // peer audience for the wire format outside the daemon's
    // own clients, and re-versioning at this scale is a
    // fresh-start replacement rather than an incremental
    // upgrade.
    let protos: &[&str] = &[
        "modules/proto/audio_stream.proto",
        "modules/proto/inference_stream.proto",
        // Top-level wire wrapper.  Imports `audio_stream` and
        // `inference_stream`; declared after them so any
        // toolchain-internal ordering remains
        // declaration-stable.
        "modules/proto/envelope.proto",
    ];
    // Generate `bytes` fields as `prost::bytes::Bytes`
    // (refcount-on-clone) instead of the default `Vec<u8>`
    // (allocate-on-clone) so the broadcast fan-out (one
    // writer to many WS receivers) doesn't pay a heap copy
    // per frame.  The `["."]` glob applies to every `bytes`
    // field in every message.
    let mut config = prost_build::Config::new();
    config.bytes(["."]);
    // `compile_protos` discovers `protoc` via `$PROTOC` then
    // `PATH`.
    config.compile_protos(protos, &["modules/proto/"])?;
    for p in protos {
        println!("cargo:rerun-if-changed={p}");
    }
    // Emitting any `rerun-if-changed` switches cargo from its default
    // (rerun on any source change) to an explicit watch list, so the script
    // itself must be on that list — otherwise edits to this file would not
    // trigger codegen.
    println!("cargo:rerun-if-changed=build.rs");
    Ok(())
}
