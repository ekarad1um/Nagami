# Build

Quick reference for building, running, testing, and deploying
the `acousticsd` daemon.  Sibling docs:
[API.md](API.md), [ARCHITECTURE.md](ARCHITECTURE.md),
[LAYOUT.md](LAYOUT.md), [PROTO.md](PROTO.md),
[BENCHS.md](BENCHS.md).

## Prerequisites

| Tool | Tested version | Purpose |
|---|---|---|
| Rust | 1.94+ stable | edition 2024 |
| `protoc` | 3.21+ | invoked by `prost-build` for codegen |
| `cmake` | 3.5+ | builds the vendored `libopus` |
| Zig + `cargo-zigbuild` | 0.16.0 / latest | aarch64 cross-builds only |

### macOS

```bash
brew install protobuf cmake zig
cargo install cargo-zigbuild
rustup target add aarch64-unknown-linux-gnu
```

### Debian / Ubuntu

```bash
sudo apt install -y protobuf-compiler cmake build-essential
# Cross builds only: zig + cargo-zigbuild
```

## Build

### Host (development)

```bash
cargo check --lib --tests --benches --examples
cargo build --release --bin acousticsd
```

### Cross to aarch64-linux-gnu

```bash
export CC_aarch64_unknown_linux_gnu="zig cc -target aarch64-linux-gnu"
export CXX_aarch64_unknown_linux_gnu="zig c++ -target aarch64-linux-gnu"
cargo zigbuild --target aarch64-unknown-linux-gnu \
    --profile release-embedded \
    --features alsa-real,rknpu \
    --bin acousticsd
```

Minimum target glibc: 2.30 (Pi OS Bookworm/Bullseye, Ubuntu
20.04+, Debian 11+, Armbian Bookworm).  For older or musl
targets, swap the triple to `aarch64-unknown-linux-musl`.

### libopus + CMake notes

`audiopus_sys` vendors libopus and drives it through CMake.

- The vendored `cmake_minimum_required(VERSION 3.0)` is
  rejected by CMake 4.x; `.cargo/config.toml` sets
  `CMAKE_POLICY_VERSION_MINIMUM=3.5` for both host and cross
  builds.
- For cross-builds, the `CC_*` / `CXX_*` env vars above route
  CMake's `ar` / `ranlib` lookup through zig's aarch64
  toolchain; without them the libopus static-link step
  fails.

## Cargo features

| Feature | Default | Purpose |
|---|---|---|
| `autodiff` | yes | enables Burn autodiff + `burn-ndarray` multi-thread (training) |
| `mimalloc` | yes | global allocator for the binary |
| `alsa-real` | no | real ALSA capture; needs `libasound2-dev` on the target sysroot |
| `rknpu` | no | links Rockchip `librknnrt.so` at runtime via `libloading` |
| `panic-inject` | no | test-only panic injection points |

The default feature set is intentional: the crate ships a
**single product, a full all-in-one lab daemon**.  Training
and converter are not optional add-ons; they are part of the
same binary, capped by per-job resource boundaries, and
exposed through the same API surface.  See
[`ARCH_BOUNDARIES.md`](ARCH_BOUNDARIES.md) §Product Profile
for the decision rationale and the trigger conditions that
would justify a smaller "inference-only" feature profile in
the future.  Until one of those triggers fires there is no
inference-only or training-only build.

## Run

The daemon reads two TOML files: a hot-reloadable
user-preference `--config` (mic policy, inference cadence,
training defaults, stream binds, file caps) and a launch-time
`--launch-config` (mic catalogue + backbone catalogue).  Both
are auto-created with defaults if missing.

The active head is persisted under `<workspace_root>/active/`,
not in the user-preference TOML; both TOMLs are parsed with
`serde(deny_unknown_fields)`, so any unknown key (e.g. a
legacy `[head_active]` block) is rejected at config load.

### Smoke (`--check`)

Boot, run for `--check-seconds` (default 5), print one
`StatusSnapshot` JSON, exit 0 iff every subsystem is healthy:

```bash
cargo run --release --bin acousticsd -- \
    --config misc/etc/dev.toml \
    --launch-config misc/etc/launch.toml \
    --check --check-seconds 2
```

### Long-running

Drop `--check` to keep the daemon listening on the configured
TCP + UDS endpoints (the bundled `misc/etc/dev.toml` defaults
to `127.0.0.1:8787` + `misc/share/acousticsd.sock`):

```bash
cargo run --release --bin acousticsd -- \
    --config misc/etc/dev.toml \
    --launch-config misc/etc/launch.toml
```

### Useful flags

| Flag | Effect |
|---|---|
| `--mock-audio` | synthesise a 1 kHz tone instead of opening real audio devices |
| `--no-inference` | skip backbone + head startup (no NPU lib / no `.mpk` needed) |
| `--check` | one-shot health probe; pair with `--check-seconds N` |
| `--worker-threads N` | tokio worker count (default 2) |
| `--tcp-bind HOST:PORT` | override `stream.tcp_bind` (test-harness escape hatch) |

## Test

```bash
cargo test --lib                                  # ~10 s host suite
cargo test --lib --release -- --include-ignored   # + fixture-dependent tests
cargo test --tests --release -- --include-ignored # + integration suites
```

The fixture-dependent tests load
[`misc/{backbones,heads,npys}/`](../misc/).  The cross-impl
parity test `backbone_mpk_matches_speech_commands_tfjs`
additionally needs the upstream Speech-Commands TFJS bundle
in `misc/models/`; fetch it once via:

```bash
bash misc/models/get_tfjs_sc_model.sh
```

Hot-path benches and the baseline contract live in
[BENCHS.md](BENCHS.md).

## Regenerate fixtures

The bundled head + NPY fixtures are reproducible from the
upstream TFJS bundle.  Re-publish them via the standalone CLI:

```bash
cargo run --release --example regen_fixtures -- all
```

Subcommands: `default-head`, `npys`, `all`.  See the
example's module-doc for details.  These operations live
outside the cargo test harness so `cargo test
--include-ignored` never mutates tracked fixtures.

## Backbone selection

The daemon walks `[[backbone.candidates]]` in the launch
config in declaration order and picks the first whose `kind`
is supported, whose file exists, and whose optional `hash`
(lowercase hex sha256, 64 chars) matches.

| Path | Requirements | Cost per inference |
|---|---|---|
| RKNN (NPU) | `linux + rknpu` feature, `kind = "rknn"`, `librknnrt.so` resolvable | ~20 ms (Pi 5) |
| Burn fp32 (CPU) | `kind = "burn"` pointing at a `.mpk` | ~5-20 ms (M1+), ~50-100 ms (Pi 5) |

`librknnrt.so` (or `librknnmrt.so`) is resolved via
`$RKNN_LIB` first, then `/usr/lib`, `/usr/local/lib`,
`/usr/lib/aarch64-linux-gnu`, every directory in
`$LD_LIBRARY_PATH`, and `~/.local/lib`.

If no candidate qualifies the daemon still boots; `inference`
reports degraded with reason `no_backbone` and the rest of
the pipeline stays healthy.

## Operator-tunable caps

Operator-tunable resource budgets:

| Cap | TOML location | Default | Effect |
|---|---|---|---|
| `max_upload_bytes` | `[file]` | 256 MiB | per-request hard ceiling on upload bytes |
| `max_concurrent_uploads` | `[file]` | 4 | global in-flight upload semaphore |
| `max_workspace_core_bytes` | (constant) | 64 KiB | hard cap on `workspace.json` body size |
| `max_train_jobs` | (constant) | 1 | unfinished train jobs daemon-wide |
| `max_convert_jobs` | (constant) | 1 | concurrent convert jobs |
| `max_delete_jobs` | (constant) | 1 | concurrent delete jobs |
| `max_delete_batch_entries` | (constant) | 256 | filesystem entries removed per batch |
| `max_recent_jobs` | (constant) | 4 | retained typed JobRegistry snapshots |
| `max_job_event_ring` | (constant) | 1024 | per-job ring-buffer depth for SSE replay |
| `max_log_line_bytes` | (constant) | 8 KiB | per-line cap on JSONL log entries |

Caps marked "(constant)" are compile-time constants today; if
operator tuning becomes necessary, lift them into the
appropriate TOML block (e.g. `[file]` for storage caps,
`[jobs]` for the JobRegistry block).

## Bundled-default head fixture path

The bundled-default classifier head MUST be present and
discoverable from the daemon's CWD at boot:

```
misc/heads/00000000-default/head.mpk
misc/heads/00000000-default/labels.txt
```

The daemon resolves this path **CWD-relative**.  This is a
known limitation; future hardening should resolve via
`current_exe()` so a daemon launched from an arbitrary
directory still finds its bundled fixture.  Until then, every
operator launch script (systemd unit, runit run script, dev
shell) MUST set the working directory to the repo root before
invoking `acousticsd`.

A missing or corrupt bundled default surfaces as: boot
without inference, status unhealthy, no synthetic head.  The
daemon does not fall back to a fabricated head.

## Upgrading from an older daemon

There is no migration tool.  The on-disk schema is parsed with
`#[serde(deny_unknown_fields)]` (or, for `ActiveHeadManifest`,
structural validation via `ActiveHeadManifest::validate`), so
any older body shape parse-fails at the read boundary.
Operators upgrading from any earlier daemon MUST wipe
`<workspace_root>/workspaces/` and
`<workspace_root>/active/generations/` before launching; the
daemon's first boot recreates the layout and materializes the
bundled-default active generation from
`misc/heads/00000000-default/`.

## Deploy (Linux SBC)

```bash
sudo install -m 0755 \
    target/aarch64-unknown-linux-gnu/release-embedded/acousticsd \
    /usr/local/bin/acousticsd
sudo setcap cap_sys_nice+ep /usr/local/bin/acousticsd
```

`CAP_SYS_NICE` lets the mic arbitrator and inference engine
take SCHED_FIFO without running as root; without it the
daemon still boots at SCHED_OTHER but ALSA underruns become
likely under load.

### Failure model: external process supervision

The daemon does NOT self-restart failed subsystems.  The
runtime restart contract is delegated to an external process
supervisor; SIGTERM triggers the bounded drain in
[`daemon/drain_registry.rs`](../modules/daemon/drain_registry.rs),
and on drain-budget exhaustion the daemon exits non-zero
(`process::exit(1)`) so the supervisor restarts a fresh
process.

Recommended deployment is systemd:

```ini
# /etc/systemd/system/acousticsd.service
[Unit]
Description=Acoustics Lab daemon
After=network.target sound.target

[Service]
Type=simple                          # or Type=exec; sd_notify is not wired
WorkingDirectory=/opt/acoustics_lab  # MUST be the repo root so misc/heads/00000000-default/ resolves
ExecStart=/usr/local/bin/acousticsd \
    --config /etc/acoustics_lab/dev.toml \
    --launch-config /etc/acoustics_lab/launch.toml
Restart=on-failure
RestartSec=2
# StandardOutput=journal / StandardError=journal as desired.

[Install]
WantedBy=multi-user.target
```

`runit`, `s6`, or any equivalent supervisor that owns process
lifecycle and restart policy works the same way.  There is no
internal restart loop; do not add one.  The
`WorkingDirectory=` directive is load-bearing — see
[Bundled-default head fixture path](#bundled-default-head-fixture-path).

For environments where systemd is not available, the repo
ships a minimal exponential-backoff respawn wrapper as a
fallback only:

```bash
sudo install -m 0755 scripts/run_acousticsd.sh /usr/local/bin/run_acousticsd

cd /opt/acoustics_lab && /usr/local/bin/run_acousticsd \
    --config /etc/acoustics_lab/dev.toml \
    --launch-config /etc/acoustics_lab/launch.toml
```

Wrapper environment knobs: `ACOUSTICSD_BIN` (binary path),
`RUN_ACOUSTICSD_BACKOFF_INITIAL` (default `2`),
`RUN_ACOUSTICSD_BACKOFF_MAX` (default `60`).

### Workspace root and UDS parent ownership

The daemon assumes a **single-daemon, daemon-owned workspace
root**: only one `acousticsd` instance writes to a given
`workspace_root` at a time, and the workspace tree
(`workspaces/`, `var/run/`, `logs/`, `active/`, `.tmp/`,
`backbone/`) is owned by the daemon user.  Multi-process
workspace locking is intentionally not implemented; running
two daemons against the same root is unsupported.

The default UDS lives at
`<workspace_root>/var/run/acoustics_lab.sock`.  The daemon
creates the parent directory at `0o700` if missing and warns
on world-writable parents.  Operators preferring
`/run/acoustics_lab.sock` (systemd-tmpfiles) override
`[stream].uds_path` in `dev.toml` and provision the parent
directory themselves with the same daemon-owned,
restrictive-mode invariant.

Logs land in `${workspace_root}/logs/` with daily rotation
and a 7-file retention cap.

### Target-only smoke checks

These checks cannot be exercised on a Darwin host build and
are not covered by host CI; verify them directly on the SBC
after each deploy.  Failures here do not invalidate host
gates — they are deployment-environment validation.

- **RKNN runtime ABI.** With `--features rknpu`, run an
  inference smoke (any `--check` boot that wires the RKNN
  backbone) and confirm `rknn_init -> inputs_set -> run ->
  outputs_get -> destroy` succeeds against the resolved
  `librknnrt.so` / `librknnmrt.so`.  Layout asserts in
  `rknn_runtime` cover bindings; live FFI behaviour is
  SBC-only.
- **ALSA capture loop.** With `--features alsa-real`, run
  the daemon against a real USB or onboard mic and confirm
  benign `EAGAIN` is treated as a non-fatal poll wakeup (no
  spurious `Heartbeat::degraded`); 44.1 kHz native devices
  skip the per-slot rubato resampler.
- **Scheduler capabilities.** Run with and without
  `CAP_SYS_NICE`; with the cap set, confirm SCHED_FIFO
  promotion via `chrt -p`; without, confirm the daemon
  still boots at SCHED_OTHER and only logs a warn.  Confirm
  cpuset/cgroup constraints behave as intended for the
  deployment.
- **Systemd lifecycle.** Confirm `systemctl restart
  acousticsd` drives a clean drain (within the bounded
  budget) followed by `Restart=on-failure` on a deliberately
  injected `process::exit(1)`.  SIGTERM lifecycle is
  host-tested in
  `tests/daemon_lifecycle_row6_sigterm_clean_shutdown.rs`;
  the systemd integration is target-only.
- **UDS parent directory.** Confirm the parent directory is
  daemon-owned, mode `0o700`, and not world-writable on the
  deployed filesystem.
- **First-boot active generation.** Confirm a freshly-wiped
  `<workspace_root>/` contains, after one boot,
  `<workspace_root>/active/current.json` and a non-empty
  `<workspace_root>/active/generations/<activation_id>/` —
  proves the bundled-default fixture path resolved
  successfully.

## ALSA: prefer 44.1 kHz-native hardware

The preproc + inference graph is baked at 44.1 kHz.  USB
class-compliant mics often quantise to 48 kHz, which forces
the arbitrator's per-slot rubato resampler (~140 KB sinc
table per slot; ~140 us per period of A53 cycles).  The
daemon emits a `tracing::warn!` on the `audio_io.source.alsa`
target when a device negotiates a non-44.1 kHz rate.

`arecord -l` enumerates capture devices; `arecord
--dump-hw-params hw:0,0` reports a device's supported rate
set.
