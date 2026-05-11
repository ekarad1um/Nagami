//! Acoustics Lab daemon wiring.
//!
//! Single binary.  Boots in this order:
//!
//!  1. Parse CLI (`--workspace <PATH> --config <PATH>` are required;
//!     `--mock-audio`, `--no-inference`, `--check[ -seconds]` are
//!     debug-build-only escape hatches).
//!  2. Load the launch TOML; load (or first-boot-create) the user-
//!     preference TOML at `<workspace>/config.toml`.
//!  3. Init the rolling-file log appender + `tracing-subscriber`.
//!  4. Construct the [`AudioBuffer`] +
//!     [`arc_swap::ArcSwap`]-wrapped `MicSettings`.
//!  5. Spawn the [`MicArbitrator`].  With `--mock-audio`
//!     the candidate list is force-overridden to a single
//!     mock source.
//!  6. Construct [`StatusMonitor`] and register every
//!     subsystem; run [`run_boot_recovery`] over the workspace.
//!  7. Construct [`StreamRouter`] (broadcast + watch
//!     counters).
//!  8. Optionally construct the [`InferenceEngine`]
//!     (skipped when `--no-inference`, the backbone catalogue is
//!     empty, the active head is missing, or boot recovery
//!     reports unhealthy).
//!  9. Spawn the Opus encoder pipeline.
//! 10. Mount the API router + stream router on TCP and
//!     UDS listeners.
//! 11. Wait for `ctrl_c` / `SIGTERM` (or `--check`'s timeout);
//!     cancel the shutdown token; drain handles.
//!
//! `--check` (debug builds only) boots, runs for
//! `args.check_seconds()`, prints the
//! [`crate::status::StatusSnapshot`] as JSON, and exits 0 if
//! every subsystem is healthy.

// `mimalloc::MiMalloc` is wired as `#[global_allocator]`
// on the binary (`bin/acousticsd.rs`), not here: global
// allocators are per-binary state, and declaring one on
// the library would conflict with test binaries that link
// `acoustics_lab` and rely on the test harness's system
// allocator.  ptmalloc per-arena fragmentation grows
// unboundedly under the converter / training spike +
// multi-hour idle pattern -- the difference between
// "operator never notices" and "RSS climbs over a week of
// uptime".  mimalloc returns memory to the OS more
// aggressively and (unlike jemalloc) has no background
// thread to contest with the audio capture thread.
use crate::daemon::drain_registry;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use arc_swap::ArcSwap;
use clap::Parser;
use tokio::sync::{broadcast, watch};
use tokio::task::JoinHandle;
use tokio::time::MissedTickBehavior;
use tokio_util::sync::CancellationToken;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use crate::audio_buffer::AudioBuffer;
use crate::audio_io::mic_arbitrator::{
    CandidateSource, ChannelSelection, MicArbitrator, MicArbitratorConfig, MicCandidate,
    MicCatalogue, MicPolicy, MicSelection, MicSettingsStore,
};
use crate::audio_io::mock::Waveform;
use crate::common::ids::MicId;
use crate::config::{
    Config, ConfigCell, LaunchConfig, MicSettingsCell, MicSettingsHandle,
    validate_policy_against_catalogue,
};
use crate::file_mgr::{AdmissionCfg, FsService, FsServiceImpl};
use crate::inference::{HotHead, InferenceEngine};
use crate::opus_stream as opus;
use crate::status::{Heartbeat, StatusMonitor};
use crate::stream_io::StreamRouter;
use crate::training::JobRegistry;

// Per-thread CPU topology + SCHED_FIFO priorities.
//
// Survey of which thread runs on which core, in one place.
// Core 0 stays unpinned for kernel / IRQ work.  Each call site
// reads from the constants here so a future topology change
// touches one spot rather than three.
//
// Priority ordering (higher = more privileged):
//   audio (50) > inference (30) > tokio (default SCHED_OTHER)
//
// Rationale: dropping audio captures is unrecoverable (the
// downstream pipeline runs on stale features); dropping inference
// frames is recoverable (next tick re-reads); tokio I/O wait is
// tolerant of scheduler jitter.  Capping audio at 50 keeps room
// above for kernel housekeeping (kernel RT threads typically run
// at 99); going above 80 risks preempting the kernel.
//
// Without `CAP_SYS_NICE`, every realtime call falls back to
// SCHED_OTHER + the pin succeeds anyway (pin_to_core only needs
// CAP_SYS_NICE on most kernels for cross-PID changes, not for
// `pid=0` self-pinning).  See `scripts/run_acousticsd.sh` for setcap
// guidance.

/// Pin the mic-arbitrator thread here.
const MIC_ARBITRATOR_PIN_CORE: usize = 1;
/// SCHED_FIFO priority for mic-arbitrator.
/// Highest of the app threads (losing audio capture is
/// unrecoverable; every other subsystem can drop frames).
const MIC_ARBITRATOR_RT_PRIORITY: i32 = 50;

/// Pin the inference engine's
/// `spawn_blocking` thread here.
const INFERENCE_PIN_CORE: usize = 2;
/// SCHED_FIFO priority for inference.
/// Lower than the audio source thread's 50 since dropping
/// inference frames is recoverable.
const INFERENCE_RT_PRIORITY: i32 = 30;

/// Pin every tokio worker thread here via
/// `Builder::on_thread_start`.  Multiple workers on the same core
/// time-share via the kernel scheduler -- fine for I/O-bound async
/// work, intentional for isolation from the audio + inference hot
/// threads.  Tokio workers stay at SCHED_OTHER (no realtime bump);
/// HTTP/WS request paths tolerate scheduler jitter.
const TOKIO_PIN_CORE: usize = 3;

/// Refresh interval for the supervisor's per-task `Heartbeat`
/// loops (see [`spawn_heartbeat_loop`]).  1 s is fast enough that
/// `GET /api/v1/status` observes a fresh sample within the HTTP
/// response budget, and slow enough that the per-tick `compose`
/// closures (some of which sample atomics, watch channels, and
/// the audio buffer head) do not become a measurable share of
/// the daemon's CPU at idle.
const HEARTBEAT_REFRESH_INTERVAL: Duration = Duration::from_secs(1);

#[derive(Parser, Debug)]
#[command(
    name = "acoustics_lab",
    version,
    about = "On-device audio classification daemon."
)]
struct Cli {
    /// Workspace root (required).  The daemon owns this entire
    /// directory tree -- config, backbone, workspaces, active head,
    /// logs, and the default UDS socket all live underneath.  The
    /// user-preference TOML is auto-materialized at
    /// `<workspace>/config.toml` on first boot.  Created (with all
    /// missing parents) if absent; the daemon must be able to
    /// read+write the path.
    #[arg(long)]
    workspace: PathBuf,

    /// Path to the launch-time TOML (required).  Holds the mic
    /// catalogue, backbone catalogue, stream listener binds, and
    /// `[head.default]` file pair.  Read once at boot; **edits are
    /// ignored until daemon restart**.  Lives outside the workspace
    /// tree so deployments can manage it independently of mutable
    /// state.
    #[arg(long)]
    config: PathBuf,

    /// Tokio worker thread count.  The default of 2 leaves CPU
    /// headroom for the std-thread mic arbitrator and the
    /// inference `spawn_blocking` task.
    #[arg(long, default_value_t = 2)]
    worker_threads: usize,

    /// Override `stream.tcp_bind` from the launch config TOML.
    /// Primarily used by the integration-test harness so each test
    /// can pass `127.0.0.1:0` for a kernel-assigned ephemeral port;
    /// without it two parallel test binaries would race port 8787.
    /// Operators set the bind in the TOML; this flag is a
    /// test-harness escape hatch.
    #[arg(long)]
    tcp_bind: Option<String>,

    /// Synthesize a 1 kHz tone instead of opening real audio devices.
    /// Overrides the launch catalogue with a single in-memory mock
    /// candidate (id `mock:0`) AND pins the user-pref policy at
    /// `Fixed { id: "mock:0" }`.  Useful for macOS dev + smoke-
    /// testing the rest of the wiring without a sound card.
    /// Debug builds only; production deployments specify a mock
    /// candidate in the launch TOML's `[[mic.candidates]]` table.
    #[cfg(debug_assertions)]
    #[arg(long)]
    mock_audio: bool,

    /// Skip InferenceEngine startup.  Useful on hosts without librknnrt
    /// (macOS dev) or for isolating audio/stream issues from inference.
    /// Debug builds only.
    #[cfg(debug_assertions)]
    #[arg(long)]
    no_inference: bool,

    /// Boot, run for `--check-seconds` (default 5), print one
    /// `StatusSnapshot` JSON, exit 0 if every registered subsystem
    /// is healthy.  Otherwise exit 1.  Debug builds only.
    #[cfg(debug_assertions)]
    #[arg(long)]
    check: bool,

    #[cfg(debug_assertions)]
    #[arg(long, default_value_t = 5)]
    check_seconds: u64,
}

/// In release builds the debug-only flags collapse to compile-time
/// constants so the rest of `async_main` reads them through the same
/// `args.<flag>` syntax without `#[cfg]` litter at every call site.
#[cfg(not(debug_assertions))]
impl Cli {
    const fn mock_audio(&self) -> bool {
        false
    }
    const fn no_inference(&self) -> bool {
        false
    }
    const fn check(&self) -> bool {
        false
    }
    const fn check_seconds(&self) -> u64 {
        0
    }
}

#[cfg(debug_assertions)]
impl Cli {
    fn mock_audio(&self) -> bool {
        self.mock_audio
    }
    fn no_inference(&self) -> bool {
        self.no_inference
    }
    fn check(&self) -> bool {
        self.check
    }
    fn check_seconds(&self) -> u64 {
        self.check_seconds
    }
}

/// Top-level entry point.  The thin `acousticsd` binary calls
/// this and propagates the `Result`.
pub fn run() -> Result<()> {
    let args = Cli::parse();
    // `worker_threads` defaults to 2 (CLI override): leaves cores
    // for the std::thread mic arbitrator + the inference
    // `spawn_blocking` task; raising it measurably steals from
    // the audio thread under load.
    //
    // `max_blocking_threads` and `thread_stack_size` stay at
    // tokio's defaults (512 / 2 MiB).  On 64-bit Linux a 2 MiB
    // stack reservation costs only the pages actually touched
    // (typically 8-32 KiB RSS); lowering it risks
    // SIGSEGV-without-traceback under deep recursion (Burn's
    // recorder, prost, serde, axum/tower).  Lowering
    // `max_blocking_threads` would deadlock `tokio::fs::*`
    // (config write + status sample + uploads + inference
    // already saturate a small cap).
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(args.worker_threads.max(1))
        .thread_name("al-rt")
        .enable_all()
        // Pin every tokio worker to
        // `TOKIO_PIN_CORE` (see module-level topology block).
        // Failure (host has fewer cores; missing CAP_SYS_NICE)
        // logs at WARN and ignored -- workers continue on default
        // placement, which costs scheduler-jitter determinism
        // but stays functional.
        .on_thread_start(|| {
            if let Err(e) = crate::sched::pin_to_core(TOKIO_PIN_CORE) {
                tracing::warn!(
                    target: "acoustics",
                    err = %e,
                    core = TOKIO_PIN_CORE,
                    "tokio worker pin_to_core failed; continuing on default placement",
                );
            }
        })
        .build()
        .context("build tokio runtime")?;
    runtime.block_on(async_main(args))
}

async fn async_main(args: Cli) -> Result<()> {
    // MARK: 1+2. Config (two-layer)
    //
    // Layer 1: launch config (immutable; operator-supplied via
    // `--config`).  Read once; edits ignored until restart.  Holds
    // the mic catalogue, backbone catalogue, stream binds,
    // bundled-default head file pair, training defaults, and file-
    // service admission caps.
    //
    // Layer 2: user-preference config (hot-reloadable + API-mutable;
    // lives at `<workspace>/config.toml`).  Holds only the fields
    // actually mutated at runtime: the mic policy (which mic +
    // channel to use right now) and the inference cadence.
    //
    // Cross-validation between layers happens at every boundary:
    // here at boot, in the `watch_with` callback on hot-reload of
    // the user TOML, and in `POST /mic/policy`.
    //
    // Workspace root is operator-supplied via `--workspace` and is
    // the single source of truth for every mutable byte the daemon
    // owns (config.toml, backbone/, workspaces/, active/, logs/,
    // var/run/).  It is NOT persisted in the user-pref TOML -- a
    // stored copy would only ever drift from the CLI on the next
    // boot.  Ensure it exists before any loader touches it.
    std::fs::create_dir_all(&args.workspace)
        .with_context(|| format!("create workspace dir {}", args.workspace.display()))?;
    let workspace_root = args.workspace.clone();
    let user_config_path = workspace_root.join("config.toml");
    if paths_may_alias(&user_config_path, &args.config) {
        anyhow::bail!(
            "--config (launch TOML) must not point at <workspace>/config.toml \
             (the user-pref TOML lives there); pass distinct paths",
        );
    }
    let launch = load_or_init_launch_config(&args.config)?;
    // Wrap the cell in `Arc` once so the API
    // (which holds `Arc<dyn ConfigHandle>`) and the in-crate
    // consumers (`MicSettingsCell`, watcher) share the same
    // pointer.  The cell is internally Arc-shaped already; the
    // outer Arc adds one ref-bump per clone but enables the
    // dyn-coercion at the API boundary without extra allocations.
    let config = Arc::new(load_or_init_config(&user_config_path)?);
    let snap = config.snapshot();
    let stream = launch.stream.clone();
    let default_head = launch.head.default.clone();

    // Validate user-pref policy against the launch catalogue at
    // boot.  A `Fixed { id }` referring to a missing catalogue
    // entry is a fatal config error -- fail loudly rather than
    // silently spinning the arbitrator inert with rate-limited
    // warns.
    if !args.mock_audio()
        && let Err(e) = validate_policy_against_catalogue(&snap.mic, &launch.mic, &user_config_path)
    {
        anyhow::bail!(
            "{e}; either fix the policy in {} or add the candidate in {}",
            user_config_path.display(),
            args.config.display(),
        );
    }
    // The watcher itself is installed AFTER the live ArcSwaps are
    // created (just before `bg_tasks` are spawned), at the
    // `_config_watcher` binding (search this file).  It needs to
    // capture them so reloaded configs propagate to the runtime.
    // Without the deferred install, file edits would update only
    // `config.inner` and never reach the arbitrator / inference
    // engine.
    // MARK: 3. Tracing
    // `create_dir_all` is idempotent, so an existence pre-check would
    // only race the actual create; just call it unconditionally.
    let log_dir = workspace_root.join("logs");
    std::fs::create_dir_all(&log_dir)
        .with_context(|| format!("create log dir {}", log_dir.display()))?;
    // Plaintext rolling log.  This deploy is not under systemd or
    // journald, so operators tail the file directly.  Daily
    // rotation, max 7 retained files (one week); older files are
    // auto-pruned by tracing-appender.  Filenames look like
    // `acousticsd.log.2026-05-02` (matches the binary name
    // operators see in `ps`/`pidof`/systemd unit names).
    let appender = tracing_appender::rolling::RollingFileAppender::builder()
        .rotation(tracing_appender::rolling::Rotation::DAILY)
        .filename_prefix("acousticsd")
        .filename_suffix("log")
        .max_log_files(7)
        .build(&log_dir)
        .with_context(|| format!("build rolling appender at {}", log_dir.display()))?;
    // Bounded SPSC channel + lossy-on-overflow.  The default
    // `non_blocking` builder uses an unbounded channel; under a
    // panic dump (a few hundred lines flushed at once) that grows
    // without limit.  2048 lines x ~4 KiB worst-case = ~8 MiB
    // bounded backlog.  `lossy(true)` drops on overflow rather than
    // blocking the producer -- blocking the producer can deadlock
    // if the producer is itself the panic-dump path.
    let (writer, log_guard) = tracing_appender::non_blocking::NonBlockingBuilder::default()
        .buffered_lines_limit(2048)
        .lossy(true)
        .finish(appender);
    // Stash the guard in a dedicated holder so it lives as long as
    // `async_main`; dropping it would silently kill the appender.
    let _log_guard_holder = LogGuardHolder { _guard: log_guard };

    // Per-target verbosity defaults.  Targets here MUST match the
    // strings used in `tracing::info!(target: "X", ...)` call sites
    // throughout this file; the canonical target `"acoustics"` is
    // the operator-facing log filter (decoupled from the internal
    // module path) and must stay stable across module renames.
    let env_filter = EnvFilter::try_from_env("ACOUSTICS_LOG")
        .unwrap_or_else(|_| EnvFilter::new("info,acoustics=info,inference=info,opus_stream=info"));

    let registry = tracing_subscriber::registry()
        .with(env_filter)
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(writer)
                .with_ansi(false),
        )
        .with(tracing_subscriber::fmt::layer().with_writer(std::io::stderr));
    registry.try_init().context("install tracing subscriber")?;

    // Global panic hook.  Surfaces panic location + payload via
    // `tracing::error!` (the default hook prints to stderr, and
    // the `JoinError` path the drain registry uses today loses
    // the payload).  Existing `catch_unwind` sites in
    // `crate::audio_io::mic_arbitrator` keep their explicit
    // `process::abort()` calls.
    let prior_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let location = info
            .location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_else(|| "<unknown>".into());
        let payload = info.payload();
        let msg = payload
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
            .unwrap_or("<non-string panic>");
        // Snake-case field names -- `tracing`'s macro matcher rejects
        // both dotted (`panic.location = ...`) and quoted (`"panic.location" = ...`)
        // forms in this position; flat snake_case is the portable
        // shape across tracing-subscriber's fmt + json formatters.
        tracing::error!(
            target: "acoustics",
            panic_location = %location,
            panic_payload  = %msg,
            backtrace = ?std::backtrace::Backtrace::force_capture(),
            "thread panicked",
        );
        prior_hook(info);
    }));

    tracing::info!(
        target: "acoustics",
        version = env!("CARGO_PKG_VERSION"),
        workspace = %args.workspace.display(),
        config = %args.config.display(),
        log_dir = %log_dir.display(),
        mock_audio = args.mock_audio(),
        no_inference = args.no_inference(),
        "daemon starting",
    );

    // MARK: External-supervision contract
    //
    // Logged once at boot so operators reading the log know what
    // failure model the daemon ships with (no in-process restart;
    // exit non-zero on drain timeout for systemd to pick up).
    // See `daemon/drain_registry.rs` module docs for the full
    // rationale and the considered alternatives.
    tracing::info!(
        target: "acoustics",
        model = "external-supervision",
        recommended = "systemd Type=notify or equivalent",
        "external supervision required; daemon does not self-restart subsystems \
         and exits non-zero on bounded-drain timeout",
    );

    // MARK: Trust-posture log
    //
    // The daemon does not terminate auth; production deployments
    // front it with an Nginx (or equivalent) reverse proxy.
    // Surface the resolved TCP bind so operators can confirm the
    // deployment shape; warn loudly on a non-loopback bind so an
    // unintentionally-exposed listener is visible at boot.
    {
        let resolved_tcp_bind = args.tcp_bind.as_deref().unwrap_or(stream.tcp_bind.as_str());
        let resolved_loopback = bind_is_loopback(resolved_tcp_bind);
        tracing::info!(
            target: "acoustics",
            tcp_bind = %resolved_tcp_bind,
            tcp_loopback = resolved_loopback,
            "trust posture: open (front with reverse proxy if exposed)",
        );
        if !resolved_loopback {
            tracing::warn!(
                target: "acoustics",
                tcp_bind = %resolved_tcp_bind,
                "non-loopback bind: API + stream listeners are open to the network. \
                 Front the daemon with a reverse proxy before exposing.",
            );
        }
    }

    // MARK: UDS parent dir hardening
    //
    // Defense in depth on top of `stream_io::bind_uds`'s
    // parent-confinement check: mkdir the UDS parent with mode
    // 0o700 if it doesn't exist, and warn if existing perms are
    // world-writable without the sticky bit.  Runs BEFORE
    // `try_bind_uds` so a permission-warning surfaces in the
    // boot log next to the bind attempt rather than only after
    // `bind_uds` itself fails.
    ensure_uds_parent_dir(&stream.uds_path)
        .with_context(|| format!("ensure UDS parent for {}", stream.uds_path.display()))?;

    // MARK: 4. Audio buffer + live policy ArcSwaps
    // 262_144 samples = 2^18 ~= 5.94 s of 44.1 kHz mono.  The
    // audio_buffer requires a power-of-two capacity (single-cycle
    // wrap index via `head & (cap - 1)`); we round up the original
    // 5 s = 220 500 target to the next power of two.  Plenty of
    // headroom for the longest peek window (WaveformLen = 44 032).
    let audio_buf = AudioBuffer::new(262_144);
    let writer = audio_buf.take_writer();

    // Capture-timing anchor.  One shared cell is built
    // here and threaded to (a) the mic arbitrator config,
    // which publishes a fresh anchor after each
    // `Writer::push`; (b) the opus encoder's `run`, which
    // reads it to stamp `AudioFrame.t_us_capture_monotonic`
    // with the encoded chunk's first-sample capture time;
    // (c) the inference engine's constructor, which reads
    // it to stamp `InferenceFrame.t_us_capture_monotonic`
    // with the inference window's first-sample capture
    // time.  Single cell = single source of truth; consumers
    // project their own read positions through the same
    // anchor.
    let timing_anchor = crate::common::time::shared_timing_anchor();

    // MARK: Live ArcSwaps
    //
    // The arbitrator's `MicSettings` bundles the launch catalogue
    // (read-only) and the live policy (hot-swappable).  The
    // catalogue is wrapped in an `Arc` so policy updates rebuild
    // `MicSettings` with the same Arc -- no `Vec<MicCandidate>`
    // clone per reload.
    //
    // **`--mock-audio`** hijacks BOTH the catalogue and the policy
    // at boot only: replaces the catalogue with a single in-memory
    // `mock:0` candidate, pins policy at `Fixed { id: "mock:0" }`.
    // The on-disk launch + user TOMLs are untouched.  Subsequent
    // user-config hot-reloads flow into `policy` only (per the
    // watcher's `--mock-audio`-aware logic below); the catalogue
    // stays the synthetic mock.
    let (catalogue_arc, policy) = if args.mock_audio() {
        let catalogue = MicCatalogue {
            candidates: vec![MicCandidate {
                id: MicId::from_static("mock:0"),
                source: CandidateSource::Mock {
                    waveforms: vec![Waveform::Sine {
                        freq_hz: 1_000.0,
                        amplitude: 0.25,
                    }],
                    period_size: 512,
                    sample_rate: crate::common::dims::SampleRate::VALUE,
                },
                channels: vec![0],
            }],
        };
        // Defensive: catch future code bugs in the synthesis above
        // (e.g. `waveforms: vec![]`) before the arbitrator silently
        // fails at first read.
        if let Err((id, err)) = catalogue.validate() {
            anyhow::bail!(
                "daemon-built mock-audio catalogue invalid (candidate {id}: {err}) \
                 -- this is a daemon bug, not a config issue",
            );
        }
        let policy = MicPolicy {
            mic: MicSelection::Fixed {
                id: MicId::from_static("mock:0"),
            },
            channel: ChannelSelection::Auto,
        };
        (Arc::new(catalogue), policy)
    } else {
        if launch.mic.candidates.is_empty() {
            tracing::warn!(
                target: "acoustics",
                "launch catalogue is empty; the arbitrator will run without an active source. \
                 Add at least one [[mic.candidates]] entry to {} (debug builds may also pass \
                 --mock-audio).",
                args.config.display(),
            );
        }
        (Arc::new(launch.mic.clone()), snap.mic.clone())
    };
    // `MicSettingsCell` wraps a `VersionedSwap<MicSettings>`
    // + back-handle to the `ConfigHandle` for persistence.  We keep
    // an `Arc<MicSettingsCell>` and project it into:
    //   * `Arc<dyn MicSettingsStore>` -- wait-free reads, handed to
    //     the arbitrator;
    //   * `Arc<dyn MicSettingsHandle>` -- read+write+persist, handed
    //     to the API and to the watcher's reload callback.
    // Both trait-object Arcs alias the same underlying cell, so a
    // policy mutation through the handle is observable on the next
    // store snapshot.
    let mic_settings_cell = Arc::new(MicSettingsCell::new(
        catalogue_arc.clone(),
        policy,
        config.clone(),
    ));
    let mic_settings_store: Arc<dyn MicSettingsStore> = mic_settings_cell.clone();
    let mic_settings_handle: Arc<dyn MicSettingsHandle> = mic_settings_cell.clone();
    let inference_cfg_arcswap = Arc::new(ArcSwap::from_pointee(snap.inference));

    // MARK: Config-reload propagation
    // Hot-reload of the user-pref TOML (operator edits) must
    // update the live ArcSwaps the arbitrator + inference engine
    // read on their hot paths.  Without this hook, file edits
    // would only update `config.inner` (the ConfigHandle's
    // snapshot) and never reach the runtime -- only API mutations
    // (which run their own `mutate_then` `after` closure) would.
    //
    // The callback runs on the config-reload-debounce thread, with
    // `mutate_lock` held inside the ConfigHandle.  Keep it cheap:
    // catalogue Arc clone + one ArcSwap store + one ArcSwap store.
    // Don't call back into `config.mutate*` from here -- that
    // would deadlock on the lock.
    //
    // ## Catalogue immutability
    //
    // The launch catalogue is captured once here as
    // `catalogue_for_reload` (an Arc).  The user TOML carries no
    // catalogue; reloads only update the policy.  The runtime
    // `MicSettings` is rebuilt as `{ catalogue: Arc::clone, policy:
    // new }` -- same catalogue, new policy.
    //
    // ## Cross-validation against the catalogue
    //
    // If the reloaded user TOML contains a `Fixed { id }` that
    // doesn't match the immutable catalogue, the arbitrator would
    // accept the policy and stay inert.  We log a warn here and
    // SKIP the policy update -- preserving the previous valid
    // policy.  (We don't reject the reload entirely; inference
    // cadence updates still propagate even if mic policy is
    // bogus, since they're independent concerns.)
    //
    // ## `--mock-audio`
    //
    // When this CLI flag is set, the daemon pinned `mic_settings`
    // to the synthetic `mock:0` catalogue + `Fixed { id: "mock:0" }`
    // policy at boot.  A subsequent user-config edit could install
    // `Fixed { id: "default-mock" }` from the on-disk policy --
    // which doesn't match the `mock:0` catalogue we're running.
    // So when the flag is active we DROP policy updates from the
    // reload path; inference updates still propagate.
    let mic_handle_for_reload = mic_settings_handle.clone();
    let inference_for_reload = inference_cfg_arcswap.clone();
    let mock_audio = args.mock_audio();
    // String-format the paths once so the move-closure below
    // captures owned strings, not borrowed `args` fields.
    let user_config_path_for_log = user_config_path.display().to_string();
    let launch_config_path_for_log = args.config.display().to_string();
    let _config_watcher = config
        .watch_with(
            move |cfg| -> Result<(), crate::config::ConfigValidationError> {
                // Apply the new policy via the handle's
                // `try_set_policy_no_persist` -- the on-disk TOML the
                // watcher just observed IS the source of truth, so
                // re-persisting would be redundant AND would re-enter
                // `ConfigHandle::mutate` (we're already inside its
                // callback, holding `mutate_lock` + the `IN_MUTATE`
                // sentinel; re-entry trips the `ReentrantMutate`
                // guard).  The cell's validator runs the same
                // catalogue cross-check; on Err we wrap with the
                // operator-friendly hint and return Err, which
                // discards the reload (inner + live cell untouched).
                //
                // `--mock-audio` skips the policy update because the
                // on-disk policy may legitimately disagree with the
                // runtime mock-audio override (e.g. on-disk says
                // `Fixed { id: "default-mock" }` while we're running
                // the synthetic `mock:0` catalogue).
                if !mock_audio
                    && let Err(e) = mic_handle_for_reload.try_set_policy_no_persist(cfg.mic.clone())
                {
                    return Err(crate::config::ConfigValidationError::Callback(format!(
                        "{e}; edit the policy in {} to match the catalogue, OR add the missing \
                     candidate to {} and restart the daemon",
                        user_config_path_for_log, launch_config_path_for_log,
                    )));
                }
                // Inference cfg is an independent ArcSwap consumed by
                // the inference engine; cross-store atomicity isn't
                // required for correctness.  Store is infallible.
                inference_for_reload.store(Arc::new(cfg.inference));
                Ok(())
            },
        )
        .context("install config watcher")?;

    let monitor = StatusMonitor::new();
    // Start the background sampler so `/api/v1/status` reads
    // stay wait-free.  500 ms cadence balances freshness
    // against the cost of a sysinfo refresh.  The sampler's
    // AbortHandle is owned by the StatusMonitor's `Inner` and
    // the loop is explicitly aborted by `Drop for Inner` when
    // the last clone drops at process exit.
    monitor.start_sampler(
        Some(workspace_root.clone()),
        std::time::Duration::from_millis(500),
    );

    // Install the process-wide `WorkspaceMetrics` global so the
    // `/api/v1/status` route can publish the counter snapshot.
    // Idempotent: a hot reload that re-enters this code path
    // observes the existing global (the second `set` returns
    // Err), which is correct because counters are
    // cumulative since process start.
    //
    // The `file_mgr -> common` allowlist edge in
    // `tests/dependency_edge_guard.rs` forbids file_mgr from
    // referencing `crate::status` directly, so we install
    // typed hook closures into `crate::file_mgr::metrics_hooks`
    // that forward into the `WorkspaceMetrics` instance the
    // status surface reads.
    let workspace_metrics = std::sync::Arc::new(crate::status::WorkspaceMetrics::new());
    let _ =
        crate::status::workspace_metrics::install_global(std::sync::Arc::clone(&workspace_metrics));
    {
        let m = std::sync::Arc::clone(&workspace_metrics);
        crate::file_mgr::metrics_hooks::install_workspace_core_write_hook(move |d| {
            m.record_workspace_core_write(d);
        });
    }
    {
        let m = std::sync::Arc::clone(&workspace_metrics);
        crate::file_mgr::metrics_hooks::install_head_index_write_hook(move |d| {
            m.record_head_index_write(d);
        });
    }
    {
        let m = std::sync::Arc::clone(&workspace_metrics);
        crate::file_mgr::metrics_hooks::install_upload_hook(move |bytes| {
            m.record_upload(bytes);
        });
    }
    {
        let m = std::sync::Arc::clone(&workspace_metrics);
        crate::file_mgr::metrics_hooks::install_dataset_mutation_rejected_hook(move || {
            m.record_dataset_mutation_rejected();
        });
    }
    {
        // Converter and dataset rejections land on separate
        // counters so operators can distinguish per-tree
        // admission contention.
        let m = std::sync::Arc::clone(&workspace_metrics);
        crate::file_mgr::metrics_hooks::install_converter_mutation_rejected_hook(move || {
            m.record_converter_mutation_rejected();
        });
    }
    {
        let m = std::sync::Arc::clone(&workspace_metrics);
        crate::file_mgr::metrics_hooks::install_job_events_dropped_hook(move |n| {
            m.record_job_events_dropped(n);
        });
    }
    let shutdown = CancellationToken::new();

    // Drain registry holds task handles + cancel tokens + pre-
    // drain hooks; `shutdown_and_drain` cancels every token, runs
    // hooks, then joins handles under per-tier budgets capped at
    // 10 s.  The mic arbitrator stays outside (not a `JoinHandle`)
    // and is silenced separately BEFORE the drain runs.  See
    // `daemon/drain_registry.rs` for the failure-model rationale.
    let mut drain_registry = drain_registry::DrainRegistry::new();

    // MARK: 5. Mic arbitrator (capture + intra-mic channel switch)
    //
    // One thread (`mic-arbitrator`) owns the active capture source,
    // demuxes its interleaved frames into per-channel slots, picks
    // the active channel via RMS arbitration, resamples to 44.1 k
    // if needed, and writes to the AudioBuffer.  There is no Block
    // MPSC channel anymore -- the arbitrator IS the producer.
    let arb_cfg = MicArbitratorConfig {
        hysteresis_db: 3.0,
        dwell: Duration::from_millis(250),
        rms_window: Duration::from_millis(100),
        mic_failover_after: Duration::from_secs(2),
        failover_retry_interval: Duration::from_secs(1),
        // Pin the mic-arbitrator thread to
        // `MIC_ARBITRATOR_PIN_CORE` and bump priority to
        // SCHED_FIFO `MIC_ARBITRATOR_RT_PRIORITY` (see
        // module-level topology block).  Both overrides are
        // best-effort inside the arbitrator's spawned thread;
        // failure (host has fewer cores; missing CAP_SYS_NICE)
        // logs at WARN and the thread continues on default
        // placement.
        sched_pin: Some(MIC_ARBITRATOR_PIN_CORE),
        sched_priority: Some(MIC_ARBITRATOR_RT_PRIORITY),
        // Producer-side capture-timing anchor: the
        // arbitrator publishes a fresh anchor after each
        // `Writer::push` so the opus encoder + inference
        // engine can stamp emitted frames with the actual
        // capture monotonic time of the audio they cover,
        // not the (potentially much later) emit time.
        timing_anchor: Some(timing_anchor.clone()),
    };
    // `MicArbitrator::start` self-validates the config and
    // panics on rejection, so callers cannot bypass the gate
    // by adding a new spawn site.
    let arb_handle = MicArbitrator::start(writer, mic_settings_store.clone(), arb_cfg);

    // Heartbeats: a single "audio_capture" subsystem
    // covers the arbitrator + capture pipeline.
    let capture_hb = monitor
        .register("audio_capture")
        .context("register audio_capture subsystem")?;
    {
        // Three operating regimes for the audio_capture heartbeat:
        //
        //   `mock:0 / 44.1 k` -> `--mock-audio` (operator
        //                            opted into the synthetic
        //                            tone; healthy).
        //   `no candidates configured` -> empty mic catalogue
        //                            (operator config gap; the
        //                            arbitrator runs but produces
        //                            no audio.  Surfaces as
        //                            `Heartbeat::degraded(detail,
        //                            "no_device")` so /status
        //                            shows the operator-actionable
        //                            misconfig without flipping
        //                            healthy off -- symmetric with
        //                            the no_backbone path on the
        //                            inference subsystem).
        //   `candidate-driven` -> operator supplied at least
        //                            one mic candidate; the
        //                            head-advance pump below
        //                            runs the normal healthy /
        //                            unhealthy switching.
        let no_mic_configured = !args.mock_audio()
            && mic_settings_store
                .snapshot()
                .catalogue
                .candidates
                .is_empty();
        let initial_detail: &'static str = if args.mock_audio() {
            "mock:0 / 44.1 k"
        } else if no_mic_configured {
            "no candidates configured"
        } else {
            "candidate-driven"
        };
        let initial = if no_mic_configured {
            Heartbeat::degraded(initial_detail, "no_device")
        } else {
            Heartbeat::ok(initial_detail)
        };
        capture_hb.send(initial).ok();
        let buf_for_watch = audio_buf.clone();
        let mut last_head = buf_for_watch.head();
        drain_registry.register_bg(
            "audio_capture_hb",
            spawn_heartbeat_loop(
                shutdown.clone(),
                capture_hb.clone(),
                HEARTBEAT_REFRESH_INTERVAL,
                move || {
                    let cur_head = buf_for_watch.head();
                    let advanced = cur_head > last_head;
                    let delta = cur_head.saturating_sub(last_head);
                    last_head = cur_head;
                    if advanced {
                        Heartbeat::ok(format!("{initial_detail}; head={cur_head} (+{delta}/s)"))
                    } else if no_mic_configured {
                        // Operator config gap: carry the degraded
                        // signal forward across pump ticks rather
                        // than flipping to "unhealthy" (which is
                        // the right signal for a transient outage,
                        // not a misconfig).
                        Heartbeat::degraded(
                            format!("{initial_detail}; head stuck at {cur_head}"),
                            "no_device",
                        )
                    } else {
                        Heartbeat::unhealthy(format!(
                            "no audio for >=1 s; head stuck at {cur_head}"
                        ))
                    }
                },
            ),
        );
    }

    // 7. StreamRouter -- constructed before inference so the engine
    //    can publish frames directly into its `infer_tx`.  Per-
    //    listener admission policies are sourced from
    //    `StreamCfg::tcp_policy` and `StreamCfg::uds_policy`; the
    //    router-level default fallback is the TCP policy, and the UDS
    //    listener pulls its own router via `router_with_policy(
    //    uds_policy)` below.
    let stream_router = StreamRouter::with_capacities_and_policy(
        stream.broadcast_capacity,
        64,
        stream.tcp_policy.clone(),
    );
    let opus_audio_tx: broadcast::Sender<bytes::Bytes> = stream_router.audio_tx();
    let audio_subs_rx: watch::Receiver<usize> = stream_router.audio_subscribers();
    let infer_tx_for_engine: broadcast::Sender<bytes::Bytes> = stream_router.infer_tx();

    // MARK: 8. HotHead + InferenceEngine + listener binds (parallelized)
    //
    // `boot_inference` (~80-200 ms Burn `.mpk` parse) runs
    // concurrently with the TCP + UDS listener
    // `bind` syscalls.  The bind syscalls don't depend on the
    // inference engine's output; the API router does, but the
    // router is mounted AFTER all three futures resolve.  Net
    // savings: ~bind-time off the cold-start budget -- the
    // listeners come up "for free" alongside the head load.
    // First-boot root layout + boot recovery.  `ensure_root_layout`
    // creates `<root>/{workspaces, .tmp, active, backbone}/`
    // idempotently; recovery then drains pending tombstones, sweeps
    // orphans, repairs `head_count`, and verifies the active
    // generation (with previous-generation + bundled-default
    // fallback).  Failures surface as boot-without-inference so
    // the daemon stays up for operator triage instead of wedging;
    // `boot_recovery_unhealthy` propagates the reason into the
    // inference heartbeat.
    // Stash the full report so we can hand it to the
    // `JobRegistry::record_boot_recovery` accessor; status surface
    // publishes the boot summary without re-walking the filesystem.
    // `boot_recovery_unhealthy` propagates the failure reason into
    // the `inference` heartbeat below; `boot_recovery_report` is
    // taken once into `JobRegistry::record_boot_recovery` further
    // down (consumed via `Option::take`).
    let (mut boot_recovery_report, boot_recovery_unhealthy) =
        run_boot_recovery(&workspace_root, default_head.as_ref(), &workspace_metrics);

    let mut head = synthetic_head_for_dev()?;
    // One subsystem entry per topology.  boot_inference re-uses this
    // sender for the live engine heartbeat pump (when running);
    // the skip/err arms reuse it for periodic refresh.  Avoids the
    // bug where two entries existed under different names and the
    // unrefreshed one went stale.
    let inference_hb = monitor
        .register("inference")
        .context("register inference subsystem")?;

    // Parse `tcp_bind` early so the bind future can start
    // alongside boot_inference.  A parse failure surfaces here, NOT
    // inside the bind future (where it would be wrapped in
    // join!'s tuple result and harder to error-propagate).
    //
    // `--tcp-bind` CLI flag overrides the TOML
    // value; the integration-test harness uses `127.0.0.1:0` to
    // get a kernel-assigned ephemeral port per test invocation
    // (avoids port-8787 collisions when multiple test binaries
    // run in parallel under cargo test).
    let tcp_bind_str = args.tcp_bind.as_deref().unwrap_or(stream.tcp_bind.as_str());
    let tcp_addr: std::net::SocketAddr = tcp_bind_str
        .parse()
        .with_context(|| format!("parse tcp_bind {tcp_bind_str}"))?;

    let want_inference = !args.no_inference()
        && boot_recovery_unhealthy.is_none()
        && head_files_present(&workspace_root)
        && !launch.backbone.is_empty();
    let inference_fut = async {
        if want_inference {
            boot_inference(
                &workspace_root,
                launch.backbone.clone(),
                &audio_buf,
                inference_hb.clone(),
                inference_cfg_arcswap.clone(),
                infer_tx_for_engine.clone(),
                shutdown.clone(),
                timing_anchor.clone(),
            )
            .await
            .map(Some)
        } else {
            Ok(None)
        }
    };
    let tcp_bind_fut = tokio::net::TcpListener::bind(tcp_addr);
    let uds_bind_fut = try_bind_uds(&stream.uds_path, stream.uds_mode);

    let (inference_outcome, tcp_bind_res, uds_bind_res) =
        tokio::join!(inference_fut, tcp_bind_fut, uds_bind_fut);

    // Process the inference outcome FIRST so `head` is set before
    // any consumer (opus_stream, AppState) reads it.  The Some/None
    // axis distinguishes the boot-attempted-and-succeeded path from
    // the boot-was-skipped path; an `Err` collapses to "boot failed
    // -- daemon continues without inference" the same way the
    // code did.
    match inference_outcome {
        Ok(Some((engine_handle, hb_pump_handle, real_head))) => {
            head = real_head;
            // Engine runs inside `tokio::task::spawn_blocking`;
            // the only way the blocking closure observes
            // shutdown is through the cancellation token it
            // polls between iterations (see
            // `InferenceEngine::run_blocking`).  Register the
            // token alongside the handle so the drain registry
            // cancels it before awaiting the join, giving the
            // engine a chance to exit cleanly within the major-
            // tier 5 s budget.
            drain_registry.register_major_with_token("inference", engine_handle, shutdown.clone());
            // Heartbeat-pump task observes the master shutdown
            // token; registered as a bg-tier task so the
            // shutdown drain awaits it before the tracing
            // writer guard drops.
            drain_registry.register_bg("inference_hb_pump", hb_pump_handle);
            inference_hb.send(Heartbeat::ok("engine spawned")).ok();
        }
        Err(e) => {
            tracing::error!(
                target: "acoustics",
                err = %e,
                "inference boot failed; daemon will continue without it",
            );
            let reason: Arc<str> = format!("boot failed: {e}").into();
            inference_hb
                .send(Heartbeat::unhealthy(reason.to_string()))
                .ok();
            // Refresh the failure heartbeat at 1 Hz so the entry
            // doesn't go "additionally stale" on top of unhealthy
            // -- operators reading `/api/v1/status` see the failure
            // detail with a current age, not a 4-hour-old timestamp
            // that suggests further breakage.
            let reason_arc = reason.clone();
            drain_registry.register_bg(
                "inference_status_refresh",
                spawn_heartbeat_loop(
                    shutdown.clone(),
                    inference_hb.clone(),
                    HEARTBEAT_REFRESH_INTERVAL,
                    move || Heartbeat::unhealthy(reason_arc.to_string()),
                ),
            );
        }
        Ok(None) => {
            // Inference was skipped -- one of four reasons.  The
            // synthetic head from `synthetic_head_for_dev` stays
            // in place so the API + /inference/* endpoints still
            // respond.
            //
            // Distinguish the VOLUNTARY skip (`--no-inference` is
            // operator intent) from the INVOLUNTARY skips (no
            // backbone, no head files, or boot-recovery
            // unhealthy):
            //
            // - Voluntary: `Heartbeat::ok("skipped via --no-inference")`.
            //   `--check` reports the daemon healthy.
            // - Involuntary: `Heartbeat::degraded(detail, reason)`
            //   for backbone / head missing; `Heartbeat::unhealthy(...)`
            //   for boot-recovery-unhealthy because the operator
            //   should treat that as a hard failure (the daemon
            //   has no usable head bytes at all, including the
            //   bundled default).
            //
            // The reason strings ("no_backbone" / "no_head" /
            // "boot_recovery_unhealthy") are operator-API
            // contract; tests pin the exact text.
            #[derive(Clone)]
            enum SkipKind {
                Voluntary,
                NoBackbone,
                NoHead,
                RecoveryUnhealthy(Arc<str>),
            }
            // Skip-kind precedence (operator-facing diagnostic):
            // 1. Voluntary (`--no-inference`) -- operator intent.
            // 2. NoBackbone -- launch config has no backbone
            //    catalogue; reported regardless of recovery
            //    state because it's a pure config issue
            //    addressable without touching `<root>/active/`.
            // 3. RecoveryUnhealthy -- recovery's bundled-default
            //    fallback failed (e.g. missing fixture on
            //    tempdir-cwd test runs).  Reported as
            //    `Heartbeat::unhealthy` so operators treat it as
            //    a hard failure.
            // 4. NoHead -- backbone present but no usable active
            //    generation (covered by `head_files_present`).
            let kind = if args.no_inference() {
                SkipKind::Voluntary
            } else if launch.backbone.is_empty() {
                SkipKind::NoBackbone
            } else if let Some(reason) = boot_recovery_unhealthy.clone() {
                SkipKind::RecoveryUnhealthy(reason.into())
            } else {
                SkipKind::NoHead
            };
            let detail: &'static str = match kind {
                SkipKind::Voluntary => "skipped via --no-inference",
                SkipKind::NoBackbone => {
                    "backbone catalogue is empty -- daemon running without inference"
                }
                SkipKind::NoHead => "head files missing -- daemon running without inference",
                SkipKind::RecoveryUnhealthy(_) => {
                    "boot recovery unhealthy -- daemon running without inference"
                }
            };
            tracing::info!(
                target: "acoustics",
                detail,
                "inference engine NOT started",
            );
            // Closure that produces the per-tick heartbeat.  Used
            // for the initial send AND captured into the 1 Hz
            // refresh pump (operator audit-fix: pre-cleanup the
            // closure cloned `kind` per tick; the `match &kind`
            // form below borrows instead -- same semantic, no
            // per-tick clone, single source of truth for the
            // variant->Heartbeat translation).
            let make_hb = move || match &kind {
                SkipKind::Voluntary => Heartbeat::ok(detail),
                SkipKind::NoBackbone => Heartbeat::degraded(detail, "no_backbone"),
                SkipKind::NoHead => Heartbeat::degraded(detail, "no_head"),
                SkipKind::RecoveryUnhealthy(reason) => {
                    Heartbeat::unhealthy(format!("{detail}: {reason}"))
                }
            };
            inference_hb.send(make_hb()).ok();
            // Refresh the skip-state heartbeat so it doesn't go
            // stale after 5 s.  The cheap 1 Hz pinger preserves
            // whichever variant we landed on above; matching kind
            // inside the closure keeps the degraded_reason string
            // consistent across re-stamps.
            drain_registry.register_bg(
                "inference_skip_refresh",
                spawn_heartbeat_loop(
                    shutdown.clone(),
                    inference_hb.clone(),
                    HEARTBEAT_REFRESH_INTERVAL,
                    make_hb,
                ),
            );
        }
    }

    // MARK: 9. opus_stream task
    let opus_reader = audio_buf.reader();
    let opus_token = shutdown.clone();
    let opus_hb = monitor
        .register("opus_stream")
        .context("register opus_stream subsystem")?;
    opus_hb.send(Heartbeat::ok("waiting for subscriber")).ok();
    // Stalled-encoder detection.  The encoder bumps this
    // counter once per packet it produces and hands to the
    // broadcast channel; the heartbeat task below reads the
    // delta across ticks and reports unhealthy when
    // subscribers are present but no new packets emerged for
    // >= 2 heartbeat periods (>= 2 s).  Paused-by-design
    // (subscribers == 0) still reports healthy because no
    // encoding is expected.  Counter semantics are
    // encoder-progress, not delivery -- see
    // [`crate::opus_stream::run`]'s `packets_encoded` parameter
    // doc.
    let opus_packets_encoded = Arc::new(std::sync::atomic::AtomicU64::new(0));
    {
        let mut audio_subs_rx_for_hb = stream_router.audio_subscribers();
        let opus_packets_for_hb = opus_packets_encoded.clone();
        let mut last_packets: u64 = 0;
        let mut last_advance_at = std::time::Instant::now();
        drain_registry.register_bg("opus_status_refresh", spawn_heartbeat_loop(
            shutdown.clone(),
            opus_hb.clone(),
            HEARTBEAT_REFRESH_INTERVAL,
            move || {
                let n = *audio_subs_rx_for_hb.borrow_and_update();
                let cur_packets =
                    opus_packets_for_hb.load(std::sync::atomic::Ordering::Relaxed);
                let now = std::time::Instant::now();
                if cur_packets != last_packets {
                    last_packets = cur_packets;
                    last_advance_at = now;
                }
                let stalled_for = now.saturating_duration_since(last_advance_at);
                if n == 0 {
                    // Paused-by-design: reset the stall-anchor so a
                    // long pause doesn't immediately read as stale on
                    // the first packet after resume.
                    last_advance_at = now;
                    Heartbeat::ok("paused (0 subscribers)")
                } else if stalled_for >= Duration::from_secs(2) {
                    Heartbeat::unhealthy(format!(
                        "no packets for {}ms with {n} subscriber{}; encoder stalled at packets={cur_packets}",
                        stalled_for.as_millis(),
                        if n == 1 { "" } else { "s" },
                    ))
                } else {
                    Heartbeat::ok(format!(
                        "encoding ({n} subscriber{}, packets={cur_packets})",
                        if n == 1 { "" } else { "s" },
                    ))
                }
            },
        ));
    }
    let opus_packets_for_run = opus_packets_encoded.clone();
    let opus_timing_anchor = timing_anchor.clone();
    drain_registry.register_major(
        "opus_stream",
        tokio::spawn(async move {
            opus::run(
                opus_reader,
                audio_subs_rx,
                opus_audio_tx,
                opus_token,
                opus_packets_for_run,
                // Share the producer's anchor so the encoder's
                // `AudioFrame.t_us_capture_monotonic` reflects
                // each packet's first-sample capture time
                // instead of emit time.
                Some(opus_timing_anchor),
            )
            .await
        }),
    );

    // MARK: 10. axum routers (api + stream)
    // The workspace root was created at the top of `async_main`
    // (`create_dir_all(&args.workspace)`) and `run_boot_recovery`'s
    // `ensure_root_layout` materialized the canonical sub-tree;
    // no further mkdir needed here.
    // Admission caps come from the operator-tunable `[file]`
    // block in the launch TOML; defaults are 256 MiB per upload
    // + 4 concurrent uploads.  `FileCfg` carries
    // `max_concurrent_uploads: usize` for ergonomic TOML; convert
    // to the `u32` shape that `crate::file_mgr::AdmissionCfg`
    // expects via saturating cast at the boundary.
    let admission = AdmissionCfg {
        max_upload_bytes: launch.file.max_upload_bytes,
        max_concurrent_uploads: u32::try_from(launch.file.max_concurrent_uploads)
            .unwrap_or(u32::MAX),
    };
    // Build the cross-cutting `JobRegistry` first so the
    // workspace-side admission paths (upload, dataset delete,
    // workspace delete, train, convert) and the api-side
    // `GET /jobs` / SSE routes share one instance.
    let jobs_registry = std::sync::Arc::new(crate::file_mgr::JobRegistry::new(
        crate::file_mgr::JobRegistryCfg::default(),
    ));
    // Stash the boot-recovery report on the registry so api
    // consumers can publish the boot summary through `/status` /
    // `/jobs`.  Single-shot: the registry accepts only the first
    // call.
    if let Some(report) = boot_recovery_report.take() {
        jobs_registry.record_boot_recovery(report);
    }
    let files: Arc<dyn FsService> = Arc::new(FsServiceImpl::with_admission_and_jobs(
        workspace_root.clone(),
        admission,
        jobs_registry.clone(),
    ));
    let training = JobRegistry::new();
    let training_hb = monitor
        .register("training")
        .context("register training subsystem")?;
    training_hb.send(Heartbeat::ok("idle")).ok();
    // Job-aware heartbeat: surfaces "idle" / "running N jobs" /
    // "cancelling N jobs" so `/api/v1/status` reflects what the
    // training subsystem is actually doing.  Distinguishes
    // running-during-shutdown (the operator can see we're
    // honouring the drain budget) from idle.
    {
        let training_for_hb = training.clone();
        let shutdown_for_training_hb = shutdown.clone();
        drain_registry.register_bg(
            "training_status_refresh",
            spawn_heartbeat_loop(
                shutdown.clone(),
                training_hb.clone(),
                HEARTBEAT_REFRESH_INTERVAL,
                move || {
                    let active = training_for_hb.active_count();
                    let cancelling = shutdown_for_training_hb.is_cancelled();
                    match (active, cancelling) {
                        (0, _) => Heartbeat::ok("idle"),
                        (n, false) => Heartbeat::ok(format!(
                            "running {n} job{}",
                            if n == 1 { "" } else { "s" },
                        )),
                        (n, true) => Heartbeat::ok(format!(
                            "cancelling {n} job{} (shutdown in progress)",
                            if n == 1 { "" } else { "s" },
                        )),
                    }
                },
            ),
        );
    }
    // Pre-drain hook: on shutdown, set the cancel flag on every
    // active training job BEFORE the registry awaits any
    // handle.  This is the only signal that reaches the
    // spawn_blocking workers running `finetune::run` -- the
    // daemon's master CancellationToken is async-only and does
    // not pass into the blocking closure.  The hook fires once,
    // returns the count of jobs cancelled (logged by the drain
    // sequence), and lets `JobRegistry`'s normal post-cancel
    // bookkeeping (state transition to `Cancelled`, watch
    // notification) run on the per-job tokio task.
    {
        let training_for_drain = training.clone();
        drain_registry
            .register_pre_drain_hook(move || training_for_drain.cancel_all_for_shutdown());
    }

    // Reaper: every 5 minutes, drop finished training entries whose
    // `finished_at` is more than an hour old.  The daemon never
    // deletes a running job; only the post-completion bookkeeping
    // entries grow without bound otherwise.  Cheap (one DashMap walk
    // every 300 s); short critical sections (per-entry mutex held
    // for one timestamp comparison).
    {
        let registry = training.clone();
        let shutdown_for_reap = shutdown.clone();
        drain_registry.register_bg(
            "training_reaper",
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_secs(300));
                // Skip behaviour: if the runtime stalls past the 5-min
                // tick we still only run the reap once on resume, not in
                // a backlog burst.  The reap is idempotent so a missed
                // tick is harmless.
                interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
                interval.tick().await;
                loop {
                    // CANCEL-SAFE: shutdown token (sticky once
                    // fired) and interval tick (state preserved
                    // across polls) are both idempotent.  Reap
                    // work runs after the select exits, so a
                    // cancel during reap is acted on at the next
                    // iteration.
                    tokio::select! {
                        biased;
                        _ = shutdown_for_reap.cancelled() => break,
                        _ = interval.tick() => {}
                    }
                    let n = registry.reap_finished(Duration::from_secs(3600));
                    if n > 0 {
                        tracing::info!(
                            target: "acoustics",
                            reaped = n,
                            "training: pruned finished job entries older than 1 h",
                        );
                    }
                }
            }),
        );
    }

    // Storage reaper: hourly sweep of `.tmp/` orphans across
    // `<root>/.tmp/`, `<root>/active/.tmp/`, and every
    // `<workspace>/.tmp/`, plus age-based pruning of per-workspace
    // `training_logs/` + `converter_logs/`.  Closes two gaps the
    // boot-time `recover_all` cannot cover:
    //
    // 1. A daemon that crashed hard (kill -9, OOM, power loss)
    //    leaves orphan tempfiles / staging dirs.  Boot recovery
    //    sweeps them on the *next* start; the storage reaper
    //    covers the case where the daemon kept running but a
    //    subsystem leaked (today only theoretical -- every
    //    production write path uses RAII -- but the reaper is
    //    the safety net for any future regression).
    // 2. Per-workspace job logs (`<job>.jsonl` under
    //    `training_logs/` / `converter_logs/`) have no built-in
    //    retention.  A busy operator's workspace would accumulate
    //    one entry per job indefinitely.  30 d is generous enough
    //    that operators reviewing a recent job still see it; a
    //    follow-up can promote both thresholds to the launch TOML
    //    if operators ask.
    //
    // The thresholds are conservative on purpose: 24 h on
    // `.tmp/` is orders of magnitude above any legitimate
    // in-flight operation (uploads finish in seconds), so the
    // sweep can never race a producer.  30 d on logs keeps a
    // month of training-job history per workspace.
    //
    // `<root>/logs/acousticsd.log.*` is NOT swept here -- the
    // `tracing_appender::rolling` daily rotation with
    // `max_log_files(7)` already prunes it; routing it through
    // this reaper would double the pruning logic without
    // changing the steady-state behaviour.
    {
        const STORAGE_REAP_INTERVAL: Duration = Duration::from_secs(3600);
        const TMP_AGE_THRESHOLD: Duration = Duration::from_secs(24 * 3600);
        const LOG_AGE_THRESHOLD: Duration = Duration::from_secs(30 * 24 * 3600);
        let workspace_root_for_reap = workspace_root.clone();
        let shutdown_for_storage = shutdown.clone();
        let metrics_for_storage = workspace_metrics.clone();
        drain_registry.register_bg(
            "storage_reaper",
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(STORAGE_REAP_INTERVAL);
                // Skip behaviour: as with `training_reaper`, a
                // backlog burst after a runtime stall would be
                // harmful (every tick walks the FS).
                interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
                interval.tick().await;
                loop {
                    tokio::select! {
                        biased;
                        _ = shutdown_for_storage.cancelled() => break,
                        _ = interval.tick() => {}
                    }
                    let cfg = crate::file_mgr::SweepConfig {
                        tmp_age: TMP_AGE_THRESHOLD,
                        log_age: LOG_AGE_THRESHOLD,
                    };
                    let root = workspace_root_for_reap.clone();
                    let metrics = metrics_for_storage.clone();
                    // Blocking I/O -- read_dir + remove syscalls
                    // round-trip the kernel.  Run on the
                    // spawn_blocking pool so the async runtime
                    // worker stays free for hot-path requests.
                    // `JoinError` only fires on task panic;
                    // logging the panic via the existing tracing
                    // surface preserves the operator's "did the
                    // reaper survive?" diagnostic.
                    let outcome = tokio::task::spawn_blocking(move || {
                        crate::file_mgr::sweep_once(&root, &cfg)
                    })
                    .await;
                    match outcome {
                        Ok(Ok(report)) => {
                            metrics.record_storage_sweep(
                                report.tmp_orphans_reaped,
                                report.log_files_pruned,
                                report.failures,
                            );
                            if report.did_work() || report.failures > 0 {
                                tracing::info!(
                                    target: "acoustics",
                                    tmp_orphans_reaped = report.tmp_orphans_reaped,
                                    log_files_pruned = report.log_files_pruned,
                                    workspaces_scanned = report.workspaces_scanned,
                                    failures = report.failures,
                                    "storage reaper sweep completed",
                                );
                            }
                        }
                        Ok(Err(e)) => {
                            tracing::warn!(
                                target: "acoustics",
                                err = %e,
                                "storage reaper sweep failed",
                            );
                        }
                        Err(je) => {
                            tracing::error!(
                                target: "acoustics",
                                err = %je,
                                "storage reaper blocking task panicked",
                            );
                        }
                    }
                }
            }),
        );
    }

    // `BroadcastLagCounters` implements
    // `crate::common::traits::lag_source::LagSource`, so we can hand the
    // counters directly to the API crate as `Arc<dyn LagSource>`
    // -- no closure adapter, no `Arc<dyn Fn() -> _>` indirection.
    // The API crate calls `.snapshot()` per `/api/v1/status`
    // request; loads are Relaxed atomic reads so this stays free
    // at status-poll cadence.
    let broadcast_lag_reader: std::sync::Arc<dyn crate::common::traits::lag_source::LagSource> =
        std::sync::Arc::new(stream_router.lag_counters());

    // `AppState::head` is `Arc<dyn HeadStore>`; wrap the
    // production `HotHead` (whose `HeadStore` impl delegates to
    // its backing `VersionedSwap<HeadInner>`).
    let head_store: std::sync::Arc<dyn crate::common::traits::head_store::HeadStore> =
        std::sync::Arc::new(head);
    // Wrap the concrete monitor + training registry in trait
    // objects so API code (and tests) sees only the trait
    // surface.  The daemon keeps the concrete `monitor` alive via
    // the `register()` calls above; cloning the Arc here is just
    // a refcount bump.
    let monitor_reporter: std::sync::Arc<dyn crate::status::StatusReporter> =
        std::sync::Arc::new(monitor.clone());
    let training_registry: std::sync::Arc<dyn crate::training::TrainingRegistry> =
        std::sync::Arc::new(training);
    let app_state = crate::api::AppState {
        config: config.clone(),
        head: head_store,
        mic_settings: mic_settings_handle,
        inference_cfg: inference_cfg_arcswap,
        files,
        monitor: monitor_reporter,
        training: training_registry,
        broadcast_lag_reader,
        // Shared active-head mutex for `POST /active`; held
        // entirely inside the route's `spawn_blocking` worker.
        active_mutex: std::sync::Arc::new(parking_lot::Mutex::new(())),
        default_head: default_head.clone(),
        // Hand the api the same registry the workspace-side
        // admission paths register against.
        jobs: jobs_registry,
    };
    let api_router = crate::api::router_v1_nested(app_state);
    // Per-listener WS admission policies.  Both routers
    // share the same broadcast channels + subscriber counters
    // (those live on `stream_router`), but each carries its own
    // `TransportPolicy` so the daemon can run the production-strict
    // gate on TCP and the relaxed gate on UDS without rebuilding
    // the broadcast plumbing.
    let tcp_stream_app = stream_router.router_with_policy(stream.tcp_policy.clone());
    let uds_stream_app = stream_router.router_with_policy(stream.uds_policy.clone());
    let tcp_app: axum::Router = api_router.clone().merge(tcp_stream_app);
    let uds_app: axum::Router = api_router.merge(uds_stream_app);

    // MARK: TCP listener (bind already completed in join)
    let tcp = tcp_bind_res.with_context(|| format!("bind {tcp_addr}"))?;
    let tcp_local = tcp
        .local_addr()
        .with_context(|| format!("local addr for {tcp_addr}"))?;
    tracing::info!(target: "acoustics", addr = %tcp_local, "TCP listener bound");
    let tcp_token = shutdown.clone();
    drain_registry.register_major(
        "stream_io_tcp",
        tokio::spawn(async move {
            crate::stream_io::serve_tcp(tcp, tcp_app, tcp_token)
                .await
                .map_err(|e| anyhow::anyhow!("tcp serve: {e}"))
        }),
    );

    // MARK: UDS listener (bind already completed in join)
    let uds_bound: bool = match uds_bind_res {
        Ok(uds) => {
            let uds_token = shutdown.clone();
            drain_registry.register_major(
                "stream_io_uds",
                tokio::spawn(async move {
                    crate::stream_io::serve_uds(uds, uds_app, uds_token)
                        .await
                        .map_err(|e| anyhow::anyhow!("uds serve: {e}"))
                }),
            );
            true
        }
        Err(e) => {
            tracing::warn!(
                target: "acoustics",
                err = %e,
                path = %stream.uds_path.display(),
                "uds bind failed; continuing TCP-only",
            );
            false
        }
    };
    let stream_io_hb = monitor
        .register("stream_io")
        .context("register stream_io subsystem")?;
    let initial_stream_detail = if uds_bound {
        format!("TCP {tcp_local}; UDS {}", stream.uds_path.display())
    } else {
        format!("TCP {tcp_local}; UDS unavailable")
    };
    stream_io_hb
        .send(Heartbeat::ok(initial_stream_detail.clone()))
        .ok();
    // Refresh stream_io heartbeat at 1 Hz with the combined audio +
    // infer subscriber count.
    {
        let mut audio_subs = stream_router.audio_subscribers();
        let mut infer_subs = stream_router.infer_subscribers();
        let initial = initial_stream_detail.clone();
        drain_registry.register_bg(
            "stream_io_status_refresh",
            spawn_heartbeat_loop(
                shutdown.clone(),
                stream_io_hb.clone(),
                HEARTBEAT_REFRESH_INTERVAL,
                move || {
                    let a = *audio_subs.borrow_and_update();
                    let i = *infer_subs.borrow_and_update();
                    Heartbeat::ok(format!("{initial} | audio={a} infer={i}"))
                },
            ),
        );
    }

    // MARK: 11. Wait for shutdown
    let exit_code = if args.check() {
        let res = run_check_mode(
            &monitor,
            &shutdown,
            Duration::from_secs(args.check_seconds()),
        )
        .await;
        if res.is_err() { 1 } else { 0 }
    } else {
        wait_for_signal().await;
        0
    };

    tracing::info!(target: "acoustics", "shutdown requested; cancelling tasks");
    // Silence the producer (mic arbitrator) BEFORE draining
    // consumers.  Previously the arbitrator kept appending audio for
    // up to ~22 s while inference / opus / WS shut down, which let
    // ALSA buffers fill, dropped frames, and produced log spam about
    // overruns.  `signal_stop` is non-blocking (a flag store + an
    // unpark); the run loop observes it within ~one capture period
    // (~12 ms) and stops appending to the audio + lag buffers, so
    // consumers drain into a quiet pipeline.
    arb_handle.signal_stop();
    shutdown.cancel();

    // Concurrent drain of every drain-registered task under a
    // single 10 s outer budget.  The drain registry first
    // cancels every registered cancellation token (so
    // `spawn_blocking` workers polling a token observe the
    // shutdown), then runs every pre-drain hook (so the
    // training-job cancel flags are set before any handle is
    // awaited), then awaits each handle under per-task budgets
    // (5 s for major, 1 s for background).  Outcomes are logged
    // (clean / error / cancelled / panicked) so post-mortems
    // have signal.  Outer-cap expiry returns `false`; we still
    // run the non-registered tail (mic arbitrator stop +
    // log-guard flush) before hard-exiting so logs aren't
    // truncated and the OS-thread arbitrator is joined.
    let drained_clean = drain_registry
        .shutdown_and_drain(Duration::from_secs(10))
        .await;

    // `MicArbitrator::stop` does a synchronous `thread::join()` --
    // calling it directly would block the tokio worker.  Defer to
    // the blocking pool so the runtime can finish other shutdown
    // work in parallel.  The arbitrator's run loop polled the stop
    // flag during the consumer drain (set above by `signal_stop`),
    // so by the time we get here it has already exited or is exiting
    // -- the join completes within ~one capture period (~12 ms at
    // 512 frames / 44.1 kHz).  `await.ok()` swallows a JoinError (the
    // arbitrator thread can't panic in the typical sense -- its body
    // is `run_loop` which uses `?`-free patterns -- but if it does,
    // the diagnostic is logged via the `Drop` of any pending
    // heartbeat sender, not here).
    if let Err(e) = tokio::task::spawn_blocking(move || arb_handle.stop()).await {
        tracing::warn!(target: "acoustics", err = %e, "mic arbitrator stop join error");
    }
    drop(_log_guard_holder);

    if !drained_clean {
        // Outer-cap drain expiry takes precedence over any
        // earlier `exit_code`: a partial drain is the more
        // load-bearing diagnostic for the external supervisor.
        std::process::exit(1);
    }
    if exit_code != 0 {
        std::process::exit(exit_code);
    }
    Ok(())
}

struct LogGuardHolder {
    _guard: tracing_appender::non_blocking::WorkerGuard,
}

/// Spawn a periodic refresh loop that re-publishes a `Heartbeat`
/// every `period`.  Centralizes the `interval` + `shutdown` select
/// dance used by every "fire-and-keep" pump in `async_main`.
///
/// `MissedTickBehavior::Skip` keeps a momentary runtime stall from
/// triggering a tick burst on resume -- for a status refresh pump
/// the older ticks add no information, only wasted `watch::send`
/// calls.
///
/// `compose` is `FnMut` so callers can carry per-tick state (e.g.
/// the `audio_capture` watcher's `last_head` cursor) without an
/// extra `RefCell`.  The first interval tick fires immediately on
/// construction; we discard it so the very first compose runs at
/// `period`, not at t=0.
fn spawn_heartbeat_loop<F>(
    shutdown: CancellationToken,
    sender: watch::Sender<Heartbeat>,
    period: Duration,
    mut compose: F,
) -> JoinHandle<()>
where
    F: FnMut() -> Heartbeat + Send + 'static,
{
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(period);
        interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
        interval.tick().await;
        loop {
            tokio::select! {
                biased;
                _ = shutdown.cancelled() => break,
                _ = interval.tick() => {}
            }
            let _ = sender.send(compose());
        }
    })
}

/// Load `<workspace>/config.toml` (the user-pref TOML), or write
/// defaults on first boot.  The workspace root is operator-
/// supplied via `--workspace` and lives in the daemon's runtime
/// state -- not in this TOML.  An older config file that still
/// carries a `workspace_root` key will fail `deny_unknown_fields`
/// on load; operators upgrading the daemon either let first-boot
/// re-materialize the file or strip the retired key by hand.
fn load_or_init_config(path: &std::path::Path) -> Result<ConfigCell> {
    if path.exists() {
        ConfigCell::load(path).with_context(|| format!("load config {}", path.display()))
    } else {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
            && !parent.exists()
        {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create config parent dir {}", parent.display()))?;
        }
        let cfg = Config::default_for();
        let h = ConfigCell::from_value(cfg, path.to_path_buf())
            .context("first-boot default config failed validation")?;
        h.persist().context("persist initial config")?;
        Ok(h)
    }
}

/// Load the launch-time config, or materialize defaults if the
/// file doesn't exist.  Mirrors `load_or_init_config`'s pattern but
/// produces a plain `LaunchConfig` (no ConfigHandle / watcher /
/// mutate machinery, since the launch layer is immutable).
fn load_or_init_launch_config(path: &std::path::Path) -> Result<LaunchConfig> {
    if path.exists() {
        // Repair the launch-owned UDS parent dir BEFORE
        // `LaunchConfig::load` runs `StreamCfg::validate()`, so an
        // otherwise-valid launch TOML whose socket parent was swept
        // can still boot.  Best-effort: parse failures fall through
        // to the structured launch loader diagnostic.
        if let Ok(text) = std::fs::read_to_string(path)
            && let Ok(value) = text.parse::<toml::Value>()
            && let Some(uds_str) = value
                .get("stream")
                .and_then(|s| s.get("uds_path"))
                .and_then(|p| p.as_str())
        {
            let uds_path = PathBuf::from(uds_str);
            let _ = ensure_uds_parent_dir(&uds_path);
        }
        LaunchConfig::load(path).with_context(|| format!("load launch config {}", path.display()))
    } else {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
            && !parent.exists()
        {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create launch config parent dir {}", parent.display()))?;
        }
        let cfg = LaunchConfig::default_for();
        ensure_uds_parent_dir(&cfg.stream.uds_path)?;
        cfg.persist(path).context("persist initial launch config")?;
        tracing::info!(
            target: "acoustics",
            path = %path.display(),
            "launch config absent; wrote first-boot defaults",
        );
        Ok(cfg)
    }
}

/// True when two CLI paths clearly point at the same file.  The
/// user-preference config and launch catalogue are intentionally
/// separate schemas; passing the same TOML to both flags produces a
/// confusing launch-parse error, so catch the mix-up before either
/// loader runs.
fn paths_may_alias(a: &std::path::Path, b: &std::path::Path) -> bool {
    if a == b {
        return true;
    }
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}

/// True iff `<workspace_root>/active/current.json` exists +
/// parses + the pointed generation's manifest validates.
/// Replaces the legacy `cfg.head_active.*.exists()` gate.
fn head_files_present(workspace_root: &std::path::Path) -> bool {
    matches!(resolve_active_head_paths(workspace_root), Ok(Some(_)))
}

/// Run the boot-time sweep (`ensure_root_layout` + `recover_all`)
/// and translate the outcome into the two pieces `async_main`
/// consumes downstream:
///
/// * `Option<RecoveryReport>` -- forwarded to
///   `JobRegistry::record_boot_recovery` so `/status` can publish
///   the boot summary without re-walking the filesystem.
/// * `Option<String>` -- the `boot_recovery_unhealthy` reason that
///   gates the inference engine's spawn (`SkipKind::RecoveryUnhealthy`)
///   and surfaces in the `inference` heartbeat.
///
/// All side-effects (tracing, `WorkspaceMetrics` counters) match
/// the previous in-line block byte for byte; this function exists
/// only to lift the 130-line block out of `async_main` so the boot
/// sequence reads top-to-bottom without a giant nested `match`.
fn run_boot_recovery(
    workspace_root: &std::path::Path,
    default_head: Option<&crate::config::DefaultHeadRef>,
    workspace_metrics: &crate::status::WorkspaceMetrics,
) -> (Option<crate::file_mgr::RecoveryReport>, Option<String>) {
    // The FsService lives later in `async_main`; reuse the sync
    // `WorkspaceMgr` directly for a fixed-cost layout pass that
    // runs before any FsService consumer.
    let layout_mgr = crate::file_mgr::WorkspaceMgr::new(workspace_root.to_path_buf());
    if let Err(e) = layout_mgr.ensure_root_layout() {
        tracing::error!(
            target: "acoustics",
            err = %e,
            "ensure_root_layout failed; daemon will boot without inference",
        );
        return (None, Some(format!("ensure_root_layout failed: {e}")));
    }

    // The production FsService is constructed below in `async_main`
    // with an empty DashMap; recovery's eviction hook fires against
    // this transient map and the FsService inherits the post-
    // recovery on-disk state lazily.
    let caches: dashmap::DashMap<
        crate::common::ids::WorkspaceId,
        std::sync::Arc<crate::file_mgr::WorkspaceCacheCell>,
    > = dashmap::DashMap::new();
    // Build the optional default-head source.  When `head.default`
    // is absent in the launch config, recovery still runs (root
    // staging + per-workspace sweeps); only the active-head
    // materialization step short-circuits to `Unhealthy`.
    let default_source =
        default_head.map(|h| crate::file_mgr::active_head_writer::DefaultHeadSource {
            path: &h.path,
            labels_path: &h.labels_path,
        });
    if default_source.is_none() {
        tracing::warn!(
            target: "acoustics",
            "head.default not configured in launch config; bundled-default fallback disabled \
             (workspace + staging recovery still runs)",
        );
    }
    // Captures nothing, so a stack closure suffices; `&loader`
    // coerces to `&HeadInnerLoader` (`&(dyn Fn + Send + Sync)`)
    // without allocating a Box.
    let loader = |head_mpk: &std::path::Path,
                  labels: &std::path::Path,
                  head_id: crate::common::ids::HeadId|
     -> Result<Box<dyn std::any::Any + Send>, String> {
        let head = HotHead::load(head_mpk, labels, head_id).map_err(|e| format!("{e}"))?;
        let inner = (*head.snapshot()).clone();
        Ok(Box::new(inner) as Box<dyn std::any::Any + Send>)
    };
    let report =
        match crate::file_mgr::recover_all(workspace_root, default_source, &caches, &loader) {
            Ok(r) => r,
            Err(e) => {
                tracing::error!(
                    target: "acoustics",
                    err = %e,
                    "boot recovery failed; daemon will boot without inference",
                );
                return (None, Some(format!("boot recovery failed: {e}")));
            }
        };

    tracing::info!(
        target: "acoustics",
        workspaces_scanned = report.workspaces.workspaces_scanned,
        workspace_recovery_failures = report.workspaces.workspace_recovery_failures,
        head_orphans_swept = report.workspaces.head_orphans_swept,
        head_count_repaired = report.workspaces.head_count_repaired,
        dataset_tombstones_completed = report.workspaces.dataset_tombstones_completed,
        dataset_stage_orphans_swept = report.workspaces.dataset_stage_orphans_swept,
        converter_tombstones_completed = report.workspaces.converter_tombstones_completed,
        converter_stage_orphans_swept = report.workspaces.converter_stage_orphans_swept,
        incomplete_creates_removed = report.workspaces.incomplete_creates_removed,
        workspace_tombstones_completed = report.root_staging.workspace_tombstones_completed,
        workspace_stage_orphans_swept = report.root_staging.workspace_stage_orphans_swept,
        "boot recovery completed",
    );
    // Sum the orphan-sweep totals onto `boot_orphans_swept_total`
    // and stash the full report on the `JobRegistry` so `/status`
    // can publish the boot summary without re-walking the
    // filesystem.
    let orphans = (report.workspaces.head_orphans_swept
        + report.workspaces.dataset_stage_orphans_swept
        + report.workspaces.converter_stage_orphans_swept
        + report.root_staging.workspace_stage_orphans_swept) as u64;
    workspace_metrics.record_boot_orphans_swept(orphans);
    // Surface per-workspace recovery failures (heads.json parse,
    // IO during sweep, ...) on a typed counter so operators can
    // spot `workspaces_scanned < expected` without grep-ing the
    // log.  The structured `tracing::warn!` from
    // `recover_workspaces` remains the authoritative diagnostic
    // source; this counter is the operator-dashboard aggregate.
    workspace_metrics.record_boot_workspace_recovery_failures(
        report.workspaces.workspace_recovery_failures as u64,
    );

    let unhealthy_reason = match &report.active {
        crate::file_mgr::RecoveryActiveResult::Current { activation_id, .. } => {
            tracing::info!(
                target: "acoustics",
                activation_id = %activation_id,
                "active head verified at boot",
            );
            None
        }
        crate::file_mgr::RecoveryActiveResult::PromotedPrevious { activation_id, .. } => {
            tracing::warn!(
                target: "acoustics",
                activation_id = %activation_id,
                "current generation failed verify; previous promoted",
            );
            None
        }
        crate::file_mgr::RecoveryActiveResult::DefaultedFromBundle { activation_id, .. } => {
            tracing::warn!(
                target: "acoustics",
                activation_id = %activation_id,
                "no valid generation; bundled default activated",
            );
            None
        }
        crate::file_mgr::RecoveryActiveResult::Unhealthy { reason } => {
            tracing::error!(
                target: "acoustics",
                reason = %reason,
                "boot recovery unhealthy; daemon will boot without inference",
            );
            Some(reason.clone())
        }
    };

    (Some(report), unhealthy_reason)
}

/// Resolve the live active-head directory under
/// `<workspace_root>/active/`.  Returns `None` when the daemon
/// has not yet activated anything (no `current.json`).  Surfaces
/// a corrupt pointer / missing manifest as `Err`.
///
/// On success the tuple is
/// `(generation_dir, head_mpk_path, labels_path, runtime_head_id)`.
/// The path triple is what `HotHead::load` consumes; the
/// runtime id is stamped on every emitted `InferenceFrame`.
///
/// Partial recovery: when the current generation's `head.mpk`
/// hash disagrees with the manifest's `sha256`, the resolver
/// tries the most recently published previous generation as a
/// fallback and emits a warn-level diagnostic so the operator
/// notices the rollback.  The full sweep (corrupt current +
/// corrupt previous -> bundled default) lives in boot recovery;
/// this helper covers the common single-failure case so a torn
/// current.json doesn't take inference offline.
fn resolve_active_head_paths(
    workspace_root: &std::path::Path,
) -> Result<Option<(PathBuf, PathBuf, PathBuf, crate::common::ids::HeadId)>> {
    use crate::file_mgr::schema as fm_schema;
    let root = workspace_root;
    let pointer_path = fm_schema::active_current_path(root);
    if !pointer_path.exists() {
        return Ok(None);
    }
    let pointer = fm_schema::read_active_current(root)
        .with_context(|| format!("read active pointer {}", pointer_path.display()))?;
    if let Some(triple) = try_resolve_generation(root, &pointer.activation_id)? {
        return Ok(Some(triple));
    }

    // Current generation failed verify.  Walk
    // `<root>/active/generations/` for any other generation
    // whose manifest validates + bytes hash-match.  Picking the
    // newest by mtime keeps the rollback deterministic when
    // multiple stale generations exist.
    tracing::warn!(
        target: "acoustics",
        activation_id = %pointer.activation_id,
        "active generation hash verify failed; trying previous generation",
    );
    let generations_root = fm_schema::active_generations_dir(root);
    if !generations_root.is_dir() {
        return Ok(None);
    }
    let mut candidates: Vec<(std::time::SystemTime, String)> = Vec::new();
    for entry in std::fs::read_dir(&generations_root)
        .with_context(|| format!("read {}", generations_root.display()))?
    {
        let entry = entry?;
        let name = match entry.file_name().into_string() {
            Ok(s) => s,
            Err(_) => continue,
        };
        if name == pointer.activation_id {
            continue;
        }
        let metadata = entry.metadata()?;
        if !metadata.is_dir() {
            continue;
        }
        let mtime = metadata
            .modified()
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        candidates.push((mtime, name));
    }
    candidates.sort_by_key(|candidate| std::cmp::Reverse(candidate.0));
    for (_, candidate) in candidates {
        if let Some(triple) = try_resolve_generation(root, &candidate)? {
            tracing::warn!(
                target: "acoustics",
                fallback_activation_id = %candidate,
                "fell back to previous active generation; boot recovery should rewrite current.json",
            );
            return Ok(Some(triple));
        }
    }
    Ok(None)
}

/// Try to resolve a single generation: read manifest, validate,
/// streaming-hash `head.mpk` + `labels.txt`, and return the
/// path triple on success.  Returns `Ok(None)` for any verify
/// failure so the caller can try another generation.
fn try_resolve_generation(
    root: &std::path::Path,
    activation_id: &str,
) -> Result<Option<(PathBuf, PathBuf, PathBuf, crate::common::ids::HeadId)>> {
    use crate::file_mgr::schema as fm_schema;
    let manifest = match fm_schema::read_active_manifest(root, activation_id) {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!(
                target: "acoustics",
                activation_id = %activation_id,
                err = %e,
                "active manifest read/parse failed",
            );
            return Ok(None);
        }
    };
    if let Err(e) = manifest.validate() {
        tracing::warn!(
            target: "acoustics",
            activation_id = %activation_id,
            err = %e,
            "active manifest validation failed",
        );
        return Ok(None);
    }
    let gen_dir = fm_schema::active_generation_dir(root, activation_id);
    let head_mpk = gen_dir.join(fm_schema::ACTIVE_HEAD_FILENAME);
    let labels = gen_dir.join(fm_schema::ACTIVE_LABELS_FILENAME);
    let head_bytes = match std::fs::read(&head_mpk) {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!(
                target: "acoustics",
                activation_id = %activation_id,
                path = %head_mpk.display(),
                err = %e,
                "active head.mpk read failed",
            );
            return Ok(None);
        }
    };
    if hash_hex(&head_bytes) != manifest.sha256 {
        tracing::warn!(
            target: "acoustics",
            activation_id = %activation_id,
            "active head.mpk hash mismatch",
        );
        return Ok(None);
    }
    // labels.txt is allowed to lag the manifest's
    // `labels_sha256`: boot recovery regenerates labels from
    // `manifest.labels[]` on a mismatch.  This helper only
    // verifies the head bytes (the operator-actionable failure
    // mode) and accepts labels as long as the file exists.
    if !labels.is_file() {
        tracing::warn!(
            target: "acoustics",
            activation_id = %activation_id,
            "active labels.txt missing",
        );
        return Ok(None);
    }
    Ok(Some((gen_dir, head_mpk, labels, manifest.runtime_head_id)))
}

/// Lowercase-hex SHA-256 of a byte slice.  Used by the boot-time
/// active-head verify path; mirrors the encode in
/// `file_mgr::active_head_writer` so a generation produced by the
/// writer round-trips through this verifier without drift.
fn hash_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    static HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = vec![0u8; digest.len() * 2];
    for (i, &b) in digest.iter().enumerate() {
        out[2 * i] = HEX[(b >> 4) as usize];
        out[2 * i + 1] = HEX[(b & 0x0f) as usize];
    }
    String::from_utf8(out).expect("ascii hex is utf8")
}

/// Build a synthetic 2-class head for hosts where the configured
/// head.mpk doesn't exist (host dev).  Keeps the inference engine
/// runnable so `GET /api/v1/active` and downstream consumers see a
/// populated runtime slot before a real activation lands.
///
/// Uses [`HotHead::try_from_inner`] so the synthetic head is
/// validated through the same finite/shape checks production
/// heads pass.  The body below is well-formed
/// by construction (zeros are finite, label count matches
/// `n_classes`, weight length matches `feature_dim * n_classes`),
/// so a `try_from_inner` failure here is a daemon bug rather
/// than an operator-actionable failure -- surfaced via `anyhow!`
/// rather than `expect` so the diagnostic chains cleanly into
/// the boot Result without aborting the runtime.
fn synthetic_head_for_dev() -> Result<HotHead> {
    let inner = crate::inference::HeadInner {
        weight: vec![0.0; crate::common::dims::BackboneFeatureDim::USIZE * 2],
        bias: vec![0.0; 2],
        labels: vec!["bg".into(), "voice".into()],
        head_id: crate::common::ids::HeadId::new(),
        n_classes: 2,
    };
    HotHead::try_from_inner(inner).map_err(|e| {
        anyhow::anyhow!(
            "synthetic head failed validation -- this is a daemon bug, not a config issue: {e}"
        )
    })
}

/// Build + spawn the inference engine and its heartbeat-pump task.
///
/// Inputs:
///   * `workspace_root` / `backbone_catalogue` -- the active head
///     under `<workspace_root>/active/`, plus the candidate list
///     from the immutable `LaunchConfig`.
///   * `inference_cfg` -- shared ArcSwap so API mutations of cadence
///     propagate to the engine without restart.
///   * `status_tx` -- the daemon's single `inference` heartbeat
///     entry; the pump task publishes engine state changes to it.
///   * `infer_tx` -- the `StreamRouter`'s broadcast sender so emitted
///     `InferenceFrame` bytes reach `/stream/infer` subscribers.
///   * `shutdown` -- master token; Ctrl-C cancels both engine + pump.
///
/// Returns three handles:
///   * the engine's main `spawn_blocking` JoinHandle (drained by
///     the daemon's shutdown sequence so its outcome is logged),
///   * the heartbeat-pump JoinHandle (a short-lived async task
///     that re-publishes the engine's per-iteration heartbeat as
///     a `crate::status::Heartbeat`; the caller pushes this onto
///     `bg_tasks` so it's awaited at shutdown rather than
///     orphaned),
///   * the loaded `HotHead`, which the caller threads into the
///     API state so `POST /active` can publish swaps against it.
// Daemon-internal boot helper; arg count is intentional --
// every parameter is a distinct cross-subsystem handle the
// daemon owns and must thread through.  Bundling them into a
// struct just adds a layer of indirection without making the
// call site cleaner.
#[allow(clippy::too_many_arguments)]
async fn boot_inference(
    workspace_root: &std::path::Path,
    backbone_catalogue: crate::inference::BackboneCatalogue,
    audio_buf: &AudioBuffer,
    status_tx: tokio::sync::watch::Sender<crate::status::Heartbeat>,
    inference_cfg: Arc<ArcSwap<crate::inference::InferenceCfg>>,
    infer_tx: broadcast::Sender<bytes::Bytes>,
    shutdown: CancellationToken,
    timing_anchor: crate::common::time::SharedTimingAnchor,
) -> Result<(JoinHandle<Result<()>>, JoinHandle<()>, HotHead)> {
    // Pick the backbone via the launch-config catalogue.
    // Walks `[[backbone.candidates]]` in declaration order, returning
    // the first kind+path+hash combination that loads on this build.
    // Cfg-gated kinds (e.g. `rknn` on a non-rknpu host build) are
    // silently skipped so the same launch.toml ships everywhere.
    let backbone = build_backbone_pipeline(backbone_catalogue).await?;
    tracing::info!(
        target: "acoustics",
        backbone = backbone.description(),
        "inference backbone selected",
    );

    // Load the head from the live active generation.  The
    // pre-flight gate (`head_files_present`) confirmed
    // `<root>/active/current.json` resolves to a manifest-valid
    // generation; resolve again here so the head + labels paths
    // come straight from the on-disk source of truth (the legacy
    // `cfg.head_active.*` TOML contract is retired).
    let (_gen_dir, head_mpk, labels_path, head_id) = resolve_active_head_paths(workspace_root)
        .with_context(|| "resolve active head paths for boot")?
        .ok_or_else(|| {
            anyhow::anyhow!(
                "active generation absent at boot; head_files_present must gate this call site"
            )
        })?;
    let head = tokio::task::spawn_blocking(move || HotHead::load(&head_mpk, &labels_path, head_id))
        .await??;

    // Engine's heartbeat watch: emits per-iteration liveness from
    // inside the hot loop.  Spawn a small async task that re-publishes
    // it as the daemon's `inference` status entry -- drops to
    // `Stopped`/`Failed` on the engine's final-tick.
    //
    // The pump runs at engine cadence (per-frame, ~4 Hz) PLUS a 1 Hz
    // floor: if the engine is in `Waiting` (audio underrun) and not
    // sending fresh heartbeats fast enough, the floor refreshes the
    // status entry so it doesn't go stale (>5 s).
    let (engine_hb_tx, engine_hb_rx) =
        tokio::sync::watch::channel(crate::inference::Heartbeat::default());
    let hb_pump_handle: JoinHandle<()> = {
        let mut hb_rx = engine_hb_rx.clone();
        let shutdown_for_pump = shutdown.clone();
        tokio::spawn(async move {
            let mut floor = tokio::time::interval(std::time::Duration::from_secs(1));
            // Skip behaviour: the floor tick exists only to refresh the
            // status entry when the engine is quiet.  After a runtime
            // stall we want one refresh, not a backlog burst.
            floor.set_missed_tick_behavior(MissedTickBehavior::Skip);
            floor.tick().await; // skip immediate first tick
            // Stalled-engine detection.  The engine emits a
            // heartbeat per loop iteration whose `frames_emitted`
            // monotonically increases; if it hasn't advanced for >= 2
            // heartbeat periods we report unhealthy regardless of the
            // engine's reported state.  Without this check, an engine
            // stuck in `Waiting` because the producer (mic arbitrator)
            // is silent reads as healthy on `/api/v1/status`, masking
            // upstream audio failures.
            //
            // Wedged-engine watchdog: aborts the daemon if the
            // engine stops sending heartbeats for STALE_ABORT_AFTER
            // (covers a wedged `rknn_run` FFI call).  The watchdog
            // sits here rather than around `Session::infer` because
            // the engine runs inside `tokio::task::spawn_blocking`
            // (sync context, no `tokio::time::timeout`) and a per-
            // call timeout would add hot-path overhead.
            //
            // Liveness signal: heartbeat-receipt time, not
            // `frames_emitted`.  A `Waiting` engine (mic silent)
            // still sends heartbeats every iter (~20 Hz) but does
            // not advance `frames_emitted`; the receipt-time signal
            // distinguishes wedged from idle.
            //
            // Four gates protect the abort:
            //   1. Boot grace: don't start the abort clock until the
            //      FIRST heartbeat arrives.
            //   2. Wedge threshold: silence >= STALE_ABORT_AFTER.
            //   3. Terminal-state gate: skip when the latest state
            //      is `Failed` or `Stopped` (the external supervisor
            //      owns clean exit).
            //   4. Shutdown gate: re-check
            //      `shutdown_for_pump.is_cancelled()` immediately
            //      before abort to avoid racing the drain.
            //
            // `STALE_AFTER` (unhealthy marker, 2 s) keeps using the
            // `frames_emitted` signal -- that's the right shape for
            // the operator-facing "inference is quiet" notice.
            let mut last_emitted_observed: u64 = 0;
            let mut last_advance_at = std::time::Instant::now();
            let mut last_hb_received_at: Option<std::time::Instant> = None;
            const STALE_AFTER: std::time::Duration = std::time::Duration::from_secs(2);
            const STALE_ABORT_AFTER: std::time::Duration = std::time::Duration::from_secs(5);
            loop {
                tokio::select! {
                    biased;
                    _ = shutdown_for_pump.cancelled() => break,
                    changed = hb_rx.changed() => {
                        if changed.is_err() {
                            break;
                        }
                        // Engine sent a fresh heartbeat -> it's alive
                        // (Running, Waiting, Lagged, Starting, OR a
                        // terminal Failed/Stopped on its way out;
                        // the terminal-state gate below handles
                        // those without aborting).
                        last_hb_received_at = Some(std::time::Instant::now());
                    }
                    _ = floor.tick() => {
                        // Floor tick: NO heartbeat in this period.
                        // Do NOT update `last_hb_received_at` -- the
                        // gap from the last real heartbeat is the
                        // wedge signal we're measuring.
                    }
                }
                let snap = *hb_rx.borrow_and_update();
                let now = std::time::Instant::now();
                if snap.frames_emitted != last_emitted_observed {
                    last_emitted_observed = snap.frames_emitted;
                    last_advance_at = now;
                }
                let stalled_for = now.saturating_duration_since(last_advance_at);
                let stalled = stalled_for >= STALE_AFTER;

                // Wedged-engine abort, four gates (see above):
                //   A. At least one heartbeat observed (boot grace).
                //   B. Silence >= STALE_ABORT_AFTER (the wedge signal).
                //   C. State is not terminal (external supervisor owns clean exit).
                //   D. Shutdown is not in progress.
                let engine_terminal = matches!(
                    snap.state,
                    crate::inference::EngineState::Failed | crate::inference::EngineState::Stopped
                );
                let hb_silence_for = last_hb_received_at
                    .map(|t| now.saturating_duration_since(t))
                    .unwrap_or(std::time::Duration::ZERO);
                let should_abort = last_hb_received_at.is_some()
                    && hb_silence_for >= STALE_ABORT_AFTER
                    && !engine_terminal
                    && !shutdown_for_pump.is_cancelled();
                if should_abort {
                    tracing::error!(
                        target: "acoustics",
                        hb_silence_ms = hb_silence_for.as_millis() as u64,
                        last_emitted = snap.frames_emitted,
                        last_state = ?snap.state,
                        "inference engine wedged > {:?} (no heartbeat); aborting for \
                         external supervisor restart (RKNN/NPU hang or hot-path \
                         deadlock unrecoverable in-process)",
                        STALE_ABORT_AFTER,
                    );
                    // Final unhealthy heartbeat for operators reading
                    // the in-flight log.  `let _` swallows the send
                    // error if no receivers (e.g. shutdown drain
                    // already dropped them).
                    let _ = status_tx.send(Heartbeat::unhealthy(format!(
                        "inference wedged {} ms; aborting",
                        hb_silence_for.as_millis()
                    )));
                    // Best-effort log flush before abort.
                    // tracing-appender's `non_blocking` writer flushes
                    // on `WorkerGuard::drop`; `process::abort` skips
                    // drops, so we yield briefly to let the worker
                    // drain its channel.  The 50 ms is heuristic
                    // (the writer thread isn't priority-pinned, and
                    // a busy SBC may delay scheduling) but is the
                    // best we can do without holding the
                    // `_log_guard_holder.guard` reference here.  Use
                    // `tokio::time::sleep` to yield cooperatively
                    // instead of blocking the worker thread --
                    // bg-tier task means the runtime can schedule
                    // the appender's poll meanwhile.
                    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                    std::process::abort();
                }
                let state_healthy = matches!(
                    snap.state,
                    crate::inference::EngineState::Running | crate::inference::EngineState::Waiting
                );
                let healthy = state_healthy && !stalled;
                let detail = if stalled {
                    format!(
                        "{:?} seq={} emitted={} drop_nan={} drop_lag={} (stalled {}ms; no new windows)",
                        snap.state,
                        snap.last_seq,
                        snap.frames_emitted,
                        snap.frames_dropped_nan,
                        snap.frames_dropped_lag,
                        stalled_for.as_millis(),
                    )
                } else {
                    format!(
                        "{:?} seq={} emitted={} drop_nan={} drop_lag={}",
                        snap.state,
                        snap.last_seq,
                        snap.frames_emitted,
                        snap.frames_dropped_nan,
                        snap.frames_dropped_lag,
                    )
                };
                let hb = if healthy {
                    Heartbeat::ok(detail)
                } else {
                    Heartbeat::unhealthy(detail)
                };
                let _ = status_tx.send(hb);
            }
        })
    };

    // Construct the engine.  The backbone has already been loaded;
    // engine construction is now infallible (just stitches handles).
    let head_clone = head.clone();
    // `InferenceEngine` holds `Box<dyn Backbone>`.
    // The catalogue-walk returns a `BackbonePipeline` (cfg-gated
    // arms stay localised to the inference crate); convert here.
    let engine = InferenceEngine::new(
        backbone.into_boxed(),
        head_clone,
        inference_cfg,
        engine_hb_tx,
        // Share the producer's anchor so the engine's
        // `InferenceFrame.t_us_capture_monotonic` reflects
        // each window's FIRST sample capture time
        // (window-start convention) instead of emit time
        // (which lags actual capture by the full window
        // length, ~1 s).
        Some(timing_anchor),
    );

    let reader = audio_buf.reader();
    let join: JoinHandle<Result<()>> = tokio::task::spawn_blocking(move || {
        // Pin inference to `INFERENCE_PIN_CORE`
        // and bump priority to SCHED_FIFO
        // `INFERENCE_RT_PRIORITY` (see module-level topology
        // block).  Both calls are best-effort; failure logs at
        // WARN and the engine continues on default placement
        // (which produces occasional cadence jitter under load
        // but stays functional).
        if let Err(e) = crate::sched::pin_to_core(INFERENCE_PIN_CORE) {
            tracing::warn!(
                target: "acoustics",
                err = %e,
                core = INFERENCE_PIN_CORE,
                "inference pin_to_core failed; continuing on default placement",
            );
        }
        if let Err(e) = crate::sched::set_realtime(INFERENCE_RT_PRIORITY) {
            tracing::warn!(
                target: "acoustics",
                err = %e,
                priority = INFERENCE_RT_PRIORITY,
                "inference set_realtime failed (likely missing CAP_SYS_NICE); \
                 continuing at SCHED_OTHER",
            );
        }
        engine
            .run_blocking(reader, infer_tx, shutdown)
            .map_err(|e| anyhow::anyhow!("inference run: {e}"))
    });
    Ok((join, hb_pump_handle, head))
}

/// Pick the inference backbone by walking the launch-config
/// catalogue (`[[backbone.candidates]]` in declaration order) and
/// returning the first candidate that loads on this build.  Heavy
/// work (file I/O, sha256 streaming, C-FFI session init for RKNN,
/// Burn `.mpk` parse) runs inside `spawn_blocking` so we don't
/// stall the async runtime worker.
///
/// RKNN library discovery (when the `rknpu` feature is on) is
/// owned by the inference crate's `RknnBackbone::load` -- it reads
/// `RKNN_LIB` (override) or falls through to the
/// `LD_LIBRARY_PATH` / standard system path search via
/// `rknn_runtime::utils::find_library_candidates`.  The daemon does
/// not pass the library path explicitly any more.
async fn build_backbone_pipeline(
    catalogue: crate::inference::BackboneCatalogue,
) -> Result<crate::inference::BackbonePipeline> {
    tokio::task::spawn_blocking(move || catalogue.load_first_supported())
        .await
        .map_err(|e| anyhow::anyhow!("spawn_blocking join (backbone): {e}"))?
        .map_err(|e| anyhow::anyhow!("backbone selection: {e}"))
}

async fn try_bind_uds(path: &std::path::Path, mode: u32) -> Result<tokio::net::UnixListener> {
    let listener = crate::stream_io::bind_uds(path)
        .await
        .map_err(|e| anyhow::anyhow!("uds bind {}: {e}", path.display()))?;
    crate::stream_io::set_uds_permissions(path, mode)
        .map_err(|e| anyhow::anyhow!("uds chmod {}: {e}", path.display()))?;
    Ok(listener)
}

/// True iff `bind` resolves to a loopback host (`127.0.0.0/8`,
/// `::1`, or `localhost`).  Used by the trust-posture WARN.
/// Fails closed: an unparseable bind biases toward "not
/// loopback" so operators get a louder signal rather than a
/// silent miss.  `SocketAddr::parse` on its own would silently
/// classify `localhost:8787` as "not loopback" (parse fails ->
/// fallback) -- that is the trap this helper exists to avoid.
fn bind_is_loopback(bind: &str) -> bool {
    let Some((host_raw, _port)) = bind.rsplit_once(':') else {
        return false;
    };
    let host = host_raw
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .unwrap_or(host_raw);
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    host.parse::<std::net::IpAddr>()
        .map(|ip| ip.is_loopback())
        .unwrap_or(false)
}

/// Ensure the UDS parent directory exists with safe (private)
/// permissions before any bind attempt.  Defence in depth on top
/// of `stream_io::bind_uds`'s parent-confinement check:
///
/// * If the parent dir is missing, create it with mode `0o700`.
///   That is the only sensible default for a daemon-private
///   runtime dir; an operator who needs group access must
///   pre-create the dir with the right mode + ownership before
///   starting the daemon.
/// * If the parent dir exists, validate its permissions (Linux
///   only; macOS skips the check because the on-disk perms model
///   differs and the macOS dev box is not the deploy target).
///   World-writable without the sticky bit is a hard reject at
///   the `bind_uds` layer; we WARN here at startup so the
///   operator notices the misconfig in the boot log even if
///   `bind_uds` later refuses.
///
/// Idempotent: if the directory already exists with safe perms,
/// no change is made.  Returns `Ok(())` unless the create itself
/// fails (e.g. ENOSPC, EACCES on the grandparent).
fn ensure_uds_parent_dir(uds_path: &std::path::Path) -> Result<()> {
    let parent = match uds_path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        // No parent (`"/foo.sock"`) or empty (`"foo.sock"`) is a
        // misconfiguration that `bind_uds` itself will reject with
        // a clear `UdsParentInsecure`; nothing to do here.
        _ => return Ok(()),
    };
    // `symlink_metadata` (not `Path::exists()`) so a dangling
    // symlink at the parent path is rejected rather than
    // silently followed by `create_dir_all` (which would either
    // fail with EEXIST or, worse, create the directory under
    // the symlink target if it resolves to a writable location
    // the daemon shouldn't be touching).
    let parent_present = match std::fs::symlink_metadata(parent) {
        Ok(md) => {
            if md.file_type().is_symlink() {
                anyhow::bail!(
                    "UDS parent {} is a symlink; refuse to chmod or bind through it",
                    parent.display(),
                );
            }
            true
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => false,
        Err(e) => return Err(e).with_context(|| format!("stat UDS parent {}", parent.display())),
    };
    if !parent_present {
        // Mode is applied at create time on Unix via the
        // mode bits in the path; std::fs::create_dir_all does NOT
        // expose a mode arg, so we recursively create then
        // tighten the leaf.  The grandparents inherit the umask
        // (operator-controlled), which is the right default.
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create UDS parent dir {}", parent.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            // 0o700: owner-rwx, no group, no world.  This is the
            // safest default for a daemon-private runtime dir.
            // Operators who need group access (e.g. a `audio`
            // group) should pre-create the dir with the right
            // mode + ownership before starting the daemon; we do
            // not narrow an operator-supplied dir.
            let perms = std::fs::Permissions::from_mode(0o700);
            std::fs::set_permissions(parent, perms).with_context(|| {
                format!(
                    "chmod 0o700 on freshly-created UDS parent {}",
                    parent.display()
                )
            })?;
            tracing::info!(
                target: "acoustics",
                path = %parent.display(),
                mode = "0o700",
                "uds parent dir created with private permissions",
            );
        }
        return Ok(());
    }
    // Existing parent: best-effort sanity check on Linux.  macOS
    // dev box is not the deploy target; skip the check there
    // rather than emit noisy warns about a non-issue.
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(md) = std::fs::metadata(parent) {
            let mode = md.permissions().mode();
            // World-writable bit set, sticky bit clear.
            // bind_uds will refuse this, but warn loudly at boot
            // so the operator notices the misconfig before the
            // bind itself fails.
            let world_writable = (mode & 0o002) != 0;
            let sticky = (mode & 0o1000) != 0;
            if world_writable && !sticky {
                tracing::warn!(
                    target: "acoustics",
                    path = %parent.display(),
                    mode = format!("{:o}", mode & 0o7777),
                    "uds parent dir is world-writable without the sticky bit; \
                     bind_uds will refuse the bind. Tighten with \
                     `chmod 0o700 {}` or set the sticky bit.",
                    parent.display(),
                );
            }
        }
    }
    Ok(())
}

async fn wait_for_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut term = signal(SignalKind::terminate()).expect("install SIGTERM handler");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                tracing::info!(target: "acoustics", "ctrl-c received");
            }
            _ = term.recv() => {
                tracing::info!(target: "acoustics", "SIGTERM received");
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

async fn run_check_mode(
    monitor: &StatusMonitor,
    shutdown: &CancellationToken,
    duration: Duration,
) -> Result<()> {
    tokio::select! {
        biased;
        _ = shutdown.cancelled() => {}
        _ = tokio::time::sleep(duration) => {}
    }
    // `--check` mode runs without the WS broadcast paths attached
    // so there's no real lag to report; pass a default snapshot.
    // `snapshot` is wait-free; the workspace path
    // was captured by `start_sampler` at boot.
    let snap = monitor.snapshot(crate::status::BroadcastLagSnapshot::default());
    let json = serde_json::to_string_pretty(&snap).unwrap_or_else(|_| "{}".into());
    println!("{json}");

    // Daemon healthy if every registered subsystem is healthy.  The
    // `audio_capture` heartbeat reflects audio_buffer head growth
    // (driven by the mic arbitrator); `opus_stream` + (optional)
    // `inference` round out the picture.
    let unhealthy: Vec<_> = snap
        .subsystems
        .iter()
        .filter(|(_, v)| !v.healthy)
        .map(|(k, v)| format!("{k}: {} (age {} ms)", v.detail, v.age_ms))
        .collect();
    if unhealthy.is_empty() {
        eprintln!(
            "daemon: --check OK ({} subsystems healthy)",
            snap.subsystems.len()
        );
        Ok(())
    } else {
        eprintln!("daemon: --check FAIL -- unhealthy: {unhealthy:?}");
        Err(anyhow::anyhow!("subsystems unhealthy: {unhealthy:?}"))
    }
}

#[cfg(test)]
mod bind_is_loopback_tests {
    use super::{bind_is_loopback, paths_may_alias};

    #[test]
    fn ipv4_loopback_accepts() {
        assert!(bind_is_loopback("127.0.0.1:8787"));
        assert!(bind_is_loopback("127.0.0.1:0"));
        assert!(
            bind_is_loopback("127.5.5.5:8787"),
            "all of 127.0.0.0/8 is loopback"
        );
    }

    #[test]
    fn ipv6_loopback_accepts_with_brackets() {
        assert!(bind_is_loopback("[::1]:8787"));
    }

    #[test]
    fn localhost_case_insensitive() {
        assert!(bind_is_loopback("localhost:8787"));
        assert!(bind_is_loopback("Localhost:9000"));
        assert!(bind_is_loopback("LOCALHOST:80"));
    }

    #[test]
    fn non_loopback_rejects() {
        assert!(!bind_is_loopback("0.0.0.0:8787"));
        assert!(!bind_is_loopback("[::]:8787"));
        assert!(!bind_is_loopback("192.168.1.10:8787"));
        assert!(!bind_is_loopback("8.8.8.8:8787"));
    }

    #[test]
    fn fails_closed_on_unparseable() {
        // Non-numeric host that isn't `localhost` -- can't classify
        // without DNS, biases toward "not loopback" so the trust-
        // posture WARN fires.
        assert!(!bind_is_loopback("myhost.local:8787"));
        assert!(!bind_is_loopback("example.com:443"));
        // No colon at all.
        assert!(!bind_is_loopback("garbage"));
    }

    #[test]
    fn paths_may_alias_detects_same_existing_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        // Test fixture: no concurrent reader, no atomicity required.
        #[allow(clippy::disallowed_methods)]
        std::fs::write(&path, b"fixture").expect("write fixture");

        assert!(paths_may_alias(&path, &path));
        assert!(paths_may_alias(
            &path,
            &dir.path().join(".").join("config.toml")
        ));
        assert!(!paths_may_alias(&path, &dir.path().join("launch.toml")));
    }
}
