//! Acoustics Lab daemon (`acousticsd`): thin binary wrapper.
//!
//! Holds the two pieces that MUST live with the binary
//! rather than with the library:
//!
//! 1. `#[global_allocator] = mimalloc::MiMalloc`.  Global
//!    allocators are per-binary state; declaring one in the
//!    library would conflict with every test binary that
//!    links `acoustics_lab`.  The library's `lib.rs`
//!    intentionally omits the decl so tests use the
//!    cargo-test harness's system allocator.
//! 2. `fn main()` calls [`acoustics_lab::daemon::run`]
//!    and propagates the `Result`.  Everything else (CLI
//!    parsing, tokio runtime, boot sequence, supervisor)
//!    lives in [`acoustics_lab::daemon`].
//!
//! The binary path is `target/<profile>/acousticsd`; the
//! `CARGO_BIN_EXE_acousticsd` env var the integration test
//! harness depends on is the cargo-supplied path used to
//! spawn the binary as a child process.

// `mimalloc` is an optional dep of `acoustics_lab` gated
// by the `mimalloc` feature (default-on).  Test builds
// that disable default features pick the system allocator
// instead.
#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() -> anyhow::Result<()> {
    acoustics_lab::daemon::run()
}
