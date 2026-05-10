//! Daemon integration-test harness.
//!
//! Boots the production `daemon` binary as a child process under
//! `--check --mock-audio --no-inference`, captures its
//! `StatusSnapshotJson` JSON from stdout, and surfaces a typed
//! `LaunchOutcome` to the test.  Designed to grow into the full
//! 7-row lifecycle smoke matrix per plan step 6 -- Row 1 (cold
//! boot) lands in this PR; Rows 2-7 land in  once the
//! supervisor restart-policy + Drop-aware HeartbeatHandle land.
//!
//! ## Why a child-process harness rather than an in-process one
//!
//! Two reasons:
//!
//! 1. **Production parity.** The child-process path exercises the
//!    real `main()` entry point -- including the panic hook
//!    installation , the mimalloc allocator init, the tokio
//!    runtime construction with `on_thread_start` sched pinning
//!    , the `tokio::join!` boot sequence , and the
//!    `--check` mode's StatusSnapshotJson serialization.  An in-process
//!    `async_main(args)` call would skip the panic-hook /
//!    allocator init steps because those mutate process-global
//!    state that's already set by the test harness.
//!
//! 2. **Lifecycle Rows 2-7 need it.** The matrix rows that land in
//!    ("RKNN absent" / "ALSA absent" / inference-task panic /
//!    opus task policy exhausted / SIGTERM mid-fine-tune / SIGKILL
//!    filesystem consistency) all need to either kill the daemon
//!    mid-flight OR observe behaviour that's only distinguishable
//!    when the daemon runs in its own process.  Building the
//!    harness around `tokio::process::Command` from Row 1 means we
//!    can extend the same primitive rather than rewrite.
//!
//! ## What the `--check` mode does
//!
//! Per `modules/bin/acousticsd.rs`'s `--check` arg: boots the daemon,
//! runs for `--check-seconds` (default 5), prints one
//! `StatusSnapshotJson` JSON to stdout, exits 0 if every registered
//! subsystem is healthy.  Otherwise exits 1.
//!
//! The harness pins `--check-seconds 3` to bound test wall-clock;
//! `--mock-audio` synthesises a 1 kHz tone instead of opening real
//! audio devices (essential for CI without ALSA / CoreAudio);
//! `--no-inference` skips the `librknnrt`-dependent path so the
//! test compiles + runs on macOS dev hosts identically to Linux
//! CI.

// Cargo compiles `tests/daemon_helpers/mod.rs` independently into EACH
// integration-test binary, so a binary that uses only
// `launch_check_mode` (Rows 1-3) flags `launch_long_running`
// (Rows 6-7) as dead, and vice versa.  Module-wide `dead_code`
// allow keeps the per-binary noise off without the
// per-item-attribute volume that would otherwise cover every
// type, method, and field.
#![allow(dead_code)]

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

/// Untyped JSON view of the daemon's `StatusSnapshotJson`.  The
/// `acoustics_lab::status::StatusSnapshotJson` type is `Serialize`-only (not
/// `Deserialize`), so the harness parses the daemon's output as
/// a generic `serde_json::Value` rather than round-tripping
/// through the typed shape -- matrix-row assertions reach into
/// the JSON via the wire field names directly.
pub type StatusSnapshotJson = serde_json::Value;

/// Outcome of a `acousticsd --check ...` invocation.  The daemon binary
/// always prints exactly one JSON-encoded `StatusSnapshotJson` line on
/// stdout AT THE END of its `--check` run (see `modules/bin/acousticsd.rs`
/// `run_check_mode`), so we capture stdout to EOF, look for the
/// last JSON-shaped line, and parse it.
#[derive(Debug)]
pub struct CheckRun {
    /// `daemon` process exit code: 0 = all subsystems healthy at
    /// the end of the check window, 1 = at least one unhealthy.
    pub exit_code: i32,
    /// The `StatusSnapshotJson` the daemon printed on its last tick
    /// before exiting.  `None` when the daemon crashed before
    /// reaching the snapshot-print step (signals a boot regression
    /// not just a subsystem-unhealthy signal).
    pub snapshot: Option<StatusSnapshotJson>,
    /// Captured stderr -- useful for diagnosing test failures.
    /// Tracing logs (the daemon's structured records) flow here
    /// through the tracing-subscriber bootstrap.
    pub stderr: String,
    /// Captured stdout (StatusSnapshotJson JSON + any other diag
    /// lines).
    pub stdout: String,
    /// Wall-clock the daemon process took from spawn to exit.
    /// Used as a soft regression guard -- the cold-boot row asserts
    /// the daemon reaches its first snapshot tick within a bounded
    /// window.
    pub elapsed: Duration,
}

/// Profile knobs for [`launch_check_mode`].  Defaults match the
/// post-merge smoke profile we ran by hand for B.3 / : `--check
/// --check-seconds 3 --mock-audio --no-inference`.  Future matrix
/// rows extend this with restart-injection toggles, ALSA-wedge
/// fixtures, etc.
#[derive(Debug, Clone)]
pub struct CheckProfile {
    /// Seconds the daemon runs before printing its
    /// `StatusSnapshotJson` and exiting.  Plan default is 5; we use 3
    /// here to keep CI under a 30 s test budget while still
    /// covering the 1 s health-stale-after window with margin.
    pub check_seconds: u64,
    /// Mock the audio source instead of opening ALSA / CoreAudio
    /// devices.  Required on CI hosts without sound hardware.
    pub mock_audio: bool,
    /// Skip the `InferenceEngine` startup.  Required on macOS dev
    /// hosts (no `librknnrt`); also lets the test focus on the
    /// non-inference subsystems for the cold-boot row.
    pub no_inference: bool,
    /// Total wall-clock budget for the daemon process.  The harness
    /// kills the child if it exceeds this -- defends against a
    /// regression that wedges the daemon in `--check` mode (which
    /// would otherwise hang the test until cargo's per-test
    /// timeout fires).
    pub timeout: Duration,
    /// `--tcp-bind` value passed to the daemon.
    /// Default `127.0.0.1:0` requests a kernel-assigned ephemeral
    /// port; without this, two parallel test binaries would race
    /// the production default port `0.0.0.0:8787` and the loser
    /// would fail boot with "Address already in use".
    /// Override with a fixed `host:port` only when a specific
    /// fixture (e.g. a future Row that asserts the wire bytes on
    /// a known port) needs determinism.
    pub tcp_bind: String,
    /// Matrix Rows 2-3 -- pre-write
    /// `<cwd>/misc/launch.toml` with this content BEFORE spawning
    /// the daemon.  `None` lets the daemon auto-create its
    /// default `misc/launch.toml` (Row 1 path).  `Some(toml)`
    /// substitutes the operator-supplied launch config -- used
    /// by Row 2 (empty backbone catalogue) and Row 3 (empty mic
    /// catalogue).
    ///
    /// The harness creates `<cwd>/misc/` and writes the file
    /// before exec, so the daemon's loader (which reads
    /// `misc/launch.toml` relative to cwd via the default
    /// `--launch-config` value) picks it up unmodified.
    pub launch_toml_override: Option<String>,
    /// Extra CLI args appended after the standard `--check
    /// --mock-audio --no-inference --tcp-bind ...` set.  Reserved
    /// for the  matrix rows that need fixture toggles --
    /// e.g. Row 4 (inference panic) appends a `--inject-panic
    /// inference` pair once the `panic-inject` feature wires the
    /// endpoint; Row 2 (RKNN absent) uses an alternative
    /// launch-config to suppress backbone candidates.  Empty by
    /// default so Row 1 matches the `Default` profile exactly.
    pub extra_args: Vec<String>,
    /// Row 7 -- reuse this directory as the daemon's
    /// cwd instead of creating a fresh `tempfile::TempDir`.
    /// `None` (the default) is the Row 1-3 path: each invocation
    /// gets its own scoped tempdir that drops at function exit.
    /// `Some(path)` is the Row 7 phase-2 path: the test wants the
    /// second `--check` invocation to read the `misc/dev.toml` +
    /// `misc/launch.toml` written by phase 1's SIGKILL'd daemon,
    /// so the cwd must outlive the first daemon's lifetime.
    /// The harness writes `misc/launch.toml` (when
    /// `launch_toml_override` is also set) and ensures `misc/`
    /// exists, but does NOT take ownership of the directory's
    /// lifetime -- the caller is responsible for cleanup.
    pub cwd_override: Option<PathBuf>,
}

impl Default for CheckProfile {
    fn default() -> Self {
        Self {
            check_seconds: 3,
            mock_audio: true,
            no_inference: true,
            timeout: Duration::from_secs(15),
            tcp_bind: "127.0.0.1:0".into(),
            launch_toml_override: None,
            extra_args: Vec::new(),
            cwd_override: None,
        }
    }
}

/// Spawn `acousticsd --check ...` with the given profile, wait for it
/// to exit, parse the `StatusSnapshotJson` from stdout, and return a
/// [`CheckRun`].
///
/// The child runs in a fresh `tempfile::TempDir` as its cwd so
/// the auto-created `misc/dev.toml` + `misc/launch.toml` stay
/// scoped to this one test invocation.  The tempdir drops at the
/// end of the function (the daemon process has already exited by
/// then, so no file is open).
pub async fn launch_check_mode(profile: CheckProfile) -> Result<CheckRun> {
    // Resolve the cwd: caller-supplied path (Row 7 phase 2) OR a
    // freshly-allocated tempdir (Rows 1-3).  The `_tmpdir_guard`
    // binding holds the `TempDir` (dropping it deletes the dir);
    // the override branch leaves it `None` so the caller-owned
    // dir survives past this function.
    let (cwd, _tmpdir_guard): (PathBuf, Option<tempfile::TempDir>) =
        match profile.cwd_override.as_ref() {
            Some(path) => (path.clone(), None),
            None => {
                let td = tempfile::tempdir().context("tempdir for daemon cwd")?;
                let path = td.path().to_path_buf();
                (path, Some(td))
            }
        };
    // Matrix Rows 2-3 -- pre-write
    // `<cwd>/misc/launch.toml` BEFORE spawn so the daemon's
    // default `--launch-config misc/launch.toml` picks up the
    // fixture instead of auto-creating a stock launch config.
    // The `misc/` dir gets created here unconditionally -- cheap
    // (one mkdir) and lets the daemon's auto-create path see
    // the dir already exists when no override is supplied.
    let misc_dir = cwd.join("misc");
    std::fs::create_dir_all(&misc_dir).context("create cwd/misc/ for daemon configs")?;
    if let Some(toml) = profile.launch_toml_override.as_deref() {
        // Workspace clippy.toml flags `std::fs::write` (production
        // writes go through `file_mgr::put_atomic`).  Test fixture
        // writes don't need atomicity -- the daemon hasn't booted
        // yet and there's no concurrent reader.  Allow the call
        // explicitly here.
        #[allow(clippy::disallowed_methods)]
        std::fs::write(misc_dir.join("launch.toml"), toml)
            .context("pre-write misc/launch.toml fixture")?;
    }
    let bin = PathBuf::from(env!("CARGO_BIN_EXE_acousticsd"));
    let mut cmd = tokio::process::Command::new(&bin);
    cmd.current_dir(&cwd)
        .arg("--check")
        .arg("--check-seconds")
        .arg(profile.check_seconds.to_string())
        .arg("--tcp-bind")
        .arg(&profile.tcp_bind)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .stdin(std::process::Stdio::null())
        // `kill_on_drop` is the safety net for the panic-unwind
        // path (test asserts before `wait_with_output().await`
        // returns); the explicit `child.kill().await + wait()`
        // below is the timeout-fired path.
        .kill_on_drop(true);
    if profile.mock_audio {
        cmd.arg("--mock-audio");
    }
    if profile.no_inference {
        cmd.arg("--no-inference");
    }
    for extra in &profile.extra_args {
        cmd.arg(extra);
    }

    let started = Instant::now();
    // Audit fix -- explicit spawn + take()
    // stdout/stderr handles so we keep ownership of `child` past
    // the timeout race.  The convenience `cmd.output()` drops the
    // `Child` on timeout and relies on `kill_on_drop` to send
    // SIGKILL -- but `Drop` doesn't BLOCK-WAIT for the kernel to
    // reap the zombie.  On a slow macOS CI host this can leave a
    // zombie until the cargo-test process exits + init reparents.
    // The shape below races `child.wait()` against the timeout
    // and (if it fires) explicitly kill()+wait() so the OS-level
    // process table is clean before this fn returns.
    let mut child = cmd
        .spawn()
        .with_context(|| format!("spawn daemon binary at {}", bin.display()))?;
    let mut stdout_pipe = child
        .stdout
        .take()
        .context("spawned daemon child missing stdout pipe (set Stdio::piped above)")?;
    let mut stderr_pipe = child
        .stderr
        .take()
        .context("spawned daemon child missing stderr pipe (set Stdio::piped above)")?;
    let stdout_handle: tokio::task::JoinHandle<std::io::Result<Vec<u8>>> =
        tokio::spawn(async move {
            use tokio::io::AsyncReadExt;
            let mut buf = Vec::new();
            stdout_pipe.read_to_end(&mut buf).await?;
            Ok(buf)
        });
    let stderr_handle: tokio::task::JoinHandle<std::io::Result<Vec<u8>>> =
        tokio::spawn(async move {
            use tokio::io::AsyncReadExt;
            let mut buf = Vec::new();
            stderr_pipe.read_to_end(&mut buf).await?;
            Ok(buf)
        });

    let status = match tokio::time::timeout(profile.timeout, child.wait()).await {
        Ok(r) => r.with_context(|| format!("wait() on daemon binary at {}", bin.display()))?,
        Err(_) => {
            // Timeout fired -- take ownership of the child kill +
            // reap path explicitly.  tokio's `Child::kill()`
            // sends SIGKILL; `wait()` then drains the zombie.
            // Both are cheap (~ms).  The reader tasks unblock as
            // soon as the kernel closes the pipe ends.
            let _ = child.kill().await;
            let _ = child.wait().await;
            // Bound the reader join too so a stuck reader (e.g.
            // mid-read on a pipe whose writer just got SIGKILL)
            // can't wedge the harness. 200 ms is generous; in
            // practice the kernel closes the pipe instantly on
            // process exit.
            let reader_budget = Duration::from_millis(200);
            let _ = tokio::time::timeout(reader_budget, stdout_handle).await;
            let _ = tokio::time::timeout(reader_budget, stderr_handle).await;
            return Err(anyhow::anyhow!(
                "acousticsd --check exceeded {} ms timeout (cwd {}); \
                 likely a boot regression that wedges in --check mode",
                profile.timeout.as_millis(),
                cwd.display(),
            ));
        }
    };
    let elapsed = started.elapsed();

    // Reader tasks drain to EOF after `child.wait()` returns
    // (the kernel closes the pipe write-ends on process exit).
    // `JoinHandle::await` returns `Result<io::Result<Vec<u8>>, JoinError>`;
    // a panic in the reader would propagate as JoinError -- surface as
    // anyhow error rather than swallowing.
    let stdout_bytes = stdout_handle
        .await
        .context("stdout-reader task join")?
        .context("stdout-reader read_to_end")?;
    let stderr_bytes = stderr_handle
        .await
        .context("stderr-reader task join")?
        .context("stderr-reader read_to_end")?;

    let stdout = String::from_utf8_lossy(&stdout_bytes).into_owned();
    let stderr = String::from_utf8_lossy(&stderr_bytes).into_owned();
    let exit_code = status.code().unwrap_or(-1);
    let snapshot = parse_last_status_snapshot(&stdout);
    Ok(CheckRun {
        exit_code,
        snapshot,
        stderr,
        stdout,
        elapsed,
    })
}

/// Pull the last balanced JSON object out of `stdout`.  The daemon
/// emits exactly one `StatusSnapshotJson` per `--check` run (printed
/// just before exit), but it's mixed with tracing-subscriber log
/// lines.  We scan for the LAST line that starts with `{` and ends
/// with `}` and try to deserialize it; on parse failure we return
/// None so the caller can decide whether the missing snapshot is
/// a test bug or a daemon regression.
///
/// `modules/bin/acousticsd.rs:run_check_mode` actually pretty-prints the
/// snapshot via `serde_json::to_string_pretty`, which spans many
/// lines.  We accumulate from the LAST `{` at column 0 to the LAST
/// `}` at column 0, mirroring the daemon's print format.
fn parse_last_status_snapshot(stdout: &str) -> Option<StatusSnapshotJson> {
    // Find the last `{` at column 0 and the last `}` at column 0
    // after it.  The pretty-print form opens with `{\n` and closes
    // with `\n}`; tracing log lines never start at column 0 with
    // `{` because they prefix with a timestamp.
    let mut start = None;
    let mut end = None;
    for (idx, line) in stdout.lines().enumerate() {
        if line == "{" {
            start = Some(idx);
        }
        if line == "}" {
            end = Some(idx);
        }
    }
    let (s, e) = (start?, end?);
    if e <= s {
        return None;
    }
    let block: String = stdout
        .lines()
        .skip(s)
        .take(e - s + 1)
        .collect::<Vec<_>>()
        .join("\n");
    serde_json::from_str(&block).ok()
}

// Rows 6 + 7 -- long-running daemon harness.
//
// Row 1 (cold boot) and Rows 2 + 3 (no_backbone, no_device) all run
// the daemon in `--check` mode (boots, prints snapshot, exits).  The
// remaining rows test PERSISTENCE under abrupt termination -- the
// daemon must run long enough to be killed by the harness, then the
// harness inspects the cwd's workspace state to verify the atomic-
// write discipline (file_mgr's `put_atomic` from ) actually
// holds across the kill.
//
// Why no HTTP client: Rows 6 + 7 in this PR exercise persistence
// without needing operator-driven workspace mutations (creating
// workspaces / posting heads via REST).  Future PRs that need to
// post a training job mid-flight (the original Row 6 plan-text)
// would extend this harness with a `reqwest`-based client + port
// discovery; until then, the lighter shape here covers the
// "daemon survives signal X" smoke gate.

use std::os::unix::process::ExitStatusExt;
use tokio::io::{AsyncBufReadExt, BufReader};

/// Long-running daemon handle.  Construct via [`launch_long_running`];
/// drop terminates the child via `kill_on_drop` if the test panics
/// before an explicit `kill_term()` / `kill_kill()` + `wait_exit()`
/// sequence completes.
#[derive(Debug)]
pub struct RunningDaemon {
    child: tokio::process::Child,
    /// Captured stderr lines collected during the boot wait -- useful
    /// for diagnostic dumps when a row asserts a clean shutdown but
    /// the daemon panicked instead.
    pub boot_stderr: String,
    /// Tempdir holding `misc/dev.toml` + `misc/launch.toml` + the
    /// auto-created `workspaces/` + `logs/` subdirs.  Owned here so
    /// the dir survives until the test explicitly drops the
    /// `RunningDaemon` (and therefore until after the test's
    /// post-shutdown filesystem inspection completes).
    tmpdir: tempfile::TempDir,
}

impl RunningDaemon {
    /// Path the daemon used as its cwd.  Tests reach the on-disk
    /// `workspaces/` + `misc/` etc. via this.
    pub fn cwd(&self) -> &std::path::Path {
        self.tmpdir.path()
    }

    /// PID of the spawned daemon child.  Used for `nix::sys::signal::kill`.
    pub fn pid(&self) -> nix::unistd::Pid {
        // `tokio::process::Child::id()` returns `Option<u32>`; under
        // normal operation it's always `Some` between spawn() and
        // wait()'s reap.  We unwrap with a clear message because a
        // missing PID here is a tokio-internal regression, not an
        // operator-actionable error.
        let raw = self
            .child
            .id()
            .expect("daemon child must have a PID before kill_*()");
        nix::unistd::Pid::from_raw(raw as i32)
    }

    /// Send `SIGTERM` to the daemon.  The daemon's signal handler at
    /// `modules/daemon/main_body.rs` traps SIGTERM + cancels the
    /// shutdown token, triggering the supervisor's drain sequence.
    /// Caller follows up with [`Self::wait_exit_within`] to confirm
    /// the drain completed.
    ///
    /// `ESRCH` (process already exited) is treated as Ok -- the
    /// daemon may have crashed between the boot marker and this
    /// call (e.g. a panic in a heartbeat refresh task); in that
    /// case the test's intent ("daemon is no longer running") is
    /// already satisfied, and the subsequent `wait_exit_within`
    /// will surface the real exit shape rather than this signal
    /// shaving over the diagnostic.
    pub fn kill_term(&self) -> anyhow::Result<()> {
        kill_tolerating_esrch(self.pid(), nix::sys::signal::Signal::SIGTERM, "SIGTERM")
    }

    /// Send `SIGKILL` to the daemon.  Bypasses the signal handler;
    /// the kernel terminates the process immediately.  Caller follows
    /// up with [`Self::wait_exit_within`] to reap the zombie.
    /// `ESRCH` is treated as Ok (same rationale as [`Self::kill_term`]).
    pub fn kill_kill(&self) -> anyhow::Result<()> {
        kill_tolerating_esrch(self.pid(), nix::sys::signal::Signal::SIGKILL, "SIGKILL")
    }

    /// Wait for the daemon to exit, bounded by `budget`.  On timeout
    /// the harness force-kills + reaps to leave the OS process table
    /// clean.  Returns the raw exit code (or signal number for a
    /// signal-terminated child) and the captured stderr.
    pub async fn wait_exit_within(mut self, budget: Duration) -> anyhow::Result<DaemonExit> {
        // Race the wait against the budget.  tokio's wait() returns
        // Result<ExitStatus> on the wake; the timeout arm explicitly
        // kills + waits to drain the zombie before returning Err.
        let outcome = tokio::time::timeout(budget, self.child.wait()).await;
        let status = match outcome {
            Ok(r) => r.map_err(anyhow::Error::from)?,
            Err(_) => {
                let _ = self.child.kill().await;
                let _ = self.child.wait().await;
                return Err(anyhow::anyhow!(
                    "daemon did not exit within {} ms; force-killed (cwd {})\n\
                     ===== BOOT STDERR =====\n{}",
                    budget.as_millis(),
                    self.tmpdir.path().display(),
                    self.boot_stderr,
                ));
            }
        };
        Ok(DaemonExit {
            exit_code: status.code(),
            terminating_signal: status.signal(),
            cwd: self.tmpdir,
            boot_stderr: self.boot_stderr,
        })
    }
}

/// Outcome of a long-running daemon run that terminated via signal
/// or natural exit.  The `cwd` TempDir is held here so the test can
/// inspect post-shutdown filesystem state before the dir auto-deletes.
#[derive(Debug)]
pub struct DaemonExit {
    /// Process exit code (`None` if terminated by signal).
    pub exit_code: Option<i32>,
    /// Signal number that terminated the process (`None` if natural exit).
    pub terminating_signal: Option<i32>,
    /// The daemon's cwd at termination -- tests inspect
    /// `cwd/workspaces/`, `cwd/misc/dev.toml`, `cwd/logs/` etc.
    pub cwd: tempfile::TempDir,
    /// Stderr captured during the boot wait.  The post-boot stderr
    /// lines (which the harness stops reading after the boot marker)
    /// land in `cwd/logs/...` via the daemon's tracing-appender.
    pub boot_stderr: String,
}

/// Spawn the daemon WITHOUT `--check` so it runs indefinitely until
/// signalled.  Blocks until either:
/// - the boot-complete marker (`"TCP listener bound"`) appears in
///   stderr -> returns `Ok(RunningDaemon)`, or
/// - the boot-budget elapses -> kills the child, returns `Err` with
///   the stderr captured so far (typically a panic backtrace).
///
/// `profile.check_seconds` is unused here (the daemon runs
/// indefinitely); `profile.timeout` bounds the BOOT wait, not the
/// total run.
pub async fn launch_long_running(profile: CheckProfile) -> anyhow::Result<RunningDaemon> {
    let tmpdir = tempfile::tempdir().context("tempdir for daemon cwd")?;
    let misc_dir = tmpdir.path().join("misc");
    std::fs::create_dir_all(&misc_dir).context("create cwd/misc/")?;
    if let Some(toml) = profile.launch_toml_override.as_deref() {
        // Same justification as the `--check`-mode harness above.
        #[allow(clippy::disallowed_methods)]
        std::fs::write(misc_dir.join("launch.toml"), toml)
            .context("pre-write misc/launch.toml fixture")?;
    }
    let bin = PathBuf::from(env!("CARGO_BIN_EXE_acousticsd"));
    let mut cmd = tokio::process::Command::new(&bin);
    cmd.current_dir(tmpdir.path())
        .arg("--tcp-bind")
        .arg(&profile.tcp_bind)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .stdin(std::process::Stdio::null())
        // Safety net for tests that panic before kill_*() -- drops
        // the child, sends SIGKILL via tokio.
        .kill_on_drop(true);
    if profile.mock_audio {
        cmd.arg("--mock-audio");
    }
    if profile.no_inference {
        cmd.arg("--no-inference");
    }
    for extra in &profile.extra_args {
        cmd.arg(extra);
    }
    // Note: NO `--check` flag here -- the daemon runs until signal.

    let mut child = cmd
        .spawn()
        .with_context(|| format!("spawn daemon binary at {}", bin.display()))?;
    let stderr_pipe = child
        .stderr
        .take()
        .context("spawned daemon child missing stderr pipe")?;

    // Read stderr line-by-line until the boot-complete marker.  The
    // daemon's `tracing::info!(target: "acoustics", "TCP listener bound")`
    // call site (modules/daemon/main_body.rs) is the canonical
    // signal that the boot sequence has reached the listening state
    // -- past this point, the daemon accepts requests + can be
    // signalled meaningfully.
    let boot_budget = profile.timeout;
    let mut reader = BufReader::new(stderr_pipe).lines();
    let mut boot_stderr = String::new();
    let bound_seen = tokio::time::timeout(boot_budget, async {
        loop {
            match reader.next_line().await {
                Ok(Some(line)) => {
                    boot_stderr.push_str(&line);
                    boot_stderr.push('\n');
                    if line.contains("TCP listener bound") {
                        return Ok::<bool, std::io::Error>(true);
                    }
                }
                Ok(None) => return Ok(false), // EOF -- daemon died
                Err(e) => return Err(e),
            }
        }
    })
    .await;
    match bound_seen {
        Ok(Ok(true)) => {}
        Ok(Ok(false)) => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            return Err(anyhow::anyhow!(
                "daemon stderr closed before boot completed (cwd {})\n\
                 ===== STDERR =====\n{}",
                tmpdir.path().display(),
                boot_stderr,
            ));
        }
        Ok(Err(e)) => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            return Err(anyhow::anyhow!(
                "stderr read error during boot wait (cwd {}): {e}",
                tmpdir.path().display(),
            ));
        }
        Err(_) => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            return Err(anyhow::anyhow!(
                "daemon boot exceeded {} ms timeout; force-killed (cwd {})\n\
                 ===== STDERR =====\n{}",
                boot_budget.as_millis(),
                tmpdir.path().display(),
                boot_stderr,
            ));
        }
    }

    // Stderr pipe was moved into the BufReader; re-attach by spawning
    // a drain task so the post-boot stderr doesn't backpressure the
    // daemon.  We don't need the post-boot lines for the
    // signal-injection rows (the `cwd/logs/` rolling appender
    // captures them anyway via the daemon's tracing-subscriber).
    tokio::spawn(async move {
        // Drain to EOF; ignore errors (the child is gone when EOF
        // arrives, or pipe-broken on kill).
        loop {
            match reader.next_line().await {
                Ok(Some(_)) => continue,
                _ => return,
            }
        }
    });

    Ok(RunningDaemon {
        child,
        boot_stderr,
        tmpdir,
    })
}

/// Helper for [`RunningDaemon::kill_term`] / `kill_kill` --
/// `ESRCH` (no such process) is silently OK because the test's
/// intent in both cases is "daemon should not be running after
/// this returns"; the subsequent `wait_exit_within` reports the
/// real exit shape (signal, exit code, captured stderr) so a
/// crashed-before-signal regression surfaces as a clear test
/// failure rather than a confusing "signal: no such process".
fn kill_tolerating_esrch(
    pid: nix::unistd::Pid,
    signal: nix::sys::signal::Signal,
    label: &'static str,
) -> anyhow::Result<()> {
    match nix::sys::signal::kill(pid, signal) {
        Ok(()) => Ok(()),
        Err(nix::errno::Errno::ESRCH) => Ok(()),
        Err(errno) => Err(anyhow::anyhow!("{label} via nix::kill: {errno}")),
    }
}
