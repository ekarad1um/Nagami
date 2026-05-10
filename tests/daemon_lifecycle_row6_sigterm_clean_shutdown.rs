//! Lifecycle Row 6 (simplified): SIGTERM triggers clean
//! shutdown within the drain budget.
//!
//! From plan step 6 Row 6 (original text):
//! > **Row 6 -- SIGTERM mid-fine-tune.** Submit a training job; mid-
//! > job SIGTERM; assert: drain budget respected (training
//! > cancellation within 5 s); workspace metadata is consistent (no
//! > orphan `tmp.*` files); on-disk state is recoverable.
//!
//! ## Scope reduction ( operator-scope amendment)
//!
//! The "submit a training job" portion needs an HTTP client +
//! real backbone `.mpk` + real dataset fixture in the test cwd --
//! infrastructure beyond what this PR ships.  The MINIMAL viable
//! Row 6 covers what the daemon's signal-handler contract
//! actually guarantees:
//!
//! 1. SIGTERM triggers the daemon's `signal::ctrl_c`-style trap
//!    handler at `modules/daemon/main_body.rs` (the `tracing::info!
//!    "SIGTERM received"` branch).
//! 2. The supervisor's drain sequence runs to completion within
//!    the 5 s budget that the supervisor declared.
//! 3. The daemon exits cleanly (exit code 0; no `process::abort`
//!    SIGABRT signal).
//! 4. The cwd's `workspaces/` dir is in a consistent state -- no
//!    orphan `.tmp/*` files at the workspace level (the per-
//!    workspace `.tmp/` subdir holds the staging tempfiles, and
//!    a clean shutdown leaves them either committed or removed
//!    by `NamedTempFile::Drop`).
//!
//! Future PRs that ship the HTTP client extension can extend this
//! row to "post a training job, wait for the model loader phase,
//! SIGTERM, assert cancellation within 5 s + no orphan
//! intermediate dataset files." For now, Row 6's clean-shutdown
//! gate alone catches the regression class "supervisor's drain
//! orchestration broke and the daemon now hangs on SIGTERM."

#[path = "daemon_helpers/mod.rs"]
mod daemon_helpers;

use std::time::Duration;

use daemon_helpers::{CheckProfile, launch_long_running};

/// **Row 6 -- SIGTERM clean shutdown.** Spawn the daemon
/// long-running, wait for the boot-complete marker, send SIGTERM,
/// assert the daemon exits cleanly within the supervisor's drain
/// budget (5 s + soft margin).
#[tokio::test]
async fn lifecycle_row6_sigterm_clean_shutdown_within_drain_budget() {
    let profile = CheckProfile {
        // `check_seconds` is unused for the long-running path
        // (the daemon runs until signal); keep the default for
        // doc consistency with Rows 1-3.
        check_seconds: 3,
        mock_audio: true,
        no_inference: true,
        // `timeout` here bounds the BOOT wait, not the run
        // duration; 15 s is the same generous cold-boot budget
        // Rows 1-3 use.
        timeout: Duration::from_secs(15),
        tcp_bind: "127.0.0.1:0".into(),
        launch_toml_override: None,
        extra_args: Vec::new(),
        cwd_override: None,
    };

    let daemon = launch_long_running(profile)
        .await
        .expect("daemon long-running launch must succeed");
    let cwd = daemon.cwd().to_path_buf();

    // The daemon advertises the external-supervision failure
    // model in its boot log so operators know they must wrap it
    // in systemd Type=notify (or equivalent) for restart.  A
    // regression that drops or renames this line is a quiet
    // contract breakage; pin it here so any future rename forces
    // an explicit update to both the docs and this assertion.
    assert!(
        daemon.boot_stderr.contains("external supervision required"),
        "boot log must advertise external-supervision contract; \
         got boot stderr:\n{}",
        daemon.boot_stderr,
    );
    // Trust-posture log: surfaces the resolved TCP bind so
    // operators can confirm the deployment shape without
    // grepping the config.  The daemon is unconditionally open
    // (production fronts it with a reverse proxy).  Pin both
    // the marker and the reverse-proxy hint so a future log
    // rewrite that drops either fails this assertion.
    assert!(
        daemon
            .boot_stderr
            .contains("trust posture: open (front with reverse proxy if exposed)"),
        "boot log must surface trust posture line; got boot stderr:\n{}",
        daemon.boot_stderr,
    );

    // Send SIGTERM -- daemon's signal handler at main_body.rs
    // traps it + cancels the shutdown token + the drain registry
    // calls `shutdown_and_drain` with a 10 s outer cap (per-task
    // `Major` tier budget is 5 s; outer cap bounds the total).
    daemon.kill_term().expect("SIGTERM must succeed");

    // Drain budget: 10 s drain-registry outer cap + 2 s soft
    // margin for the actual exit + zombie reap.  Anything past
    // this is
    // a regression in the drain orchestration (a task that
    // doesn't observe the cancellation token, an awaited future
    // that doesn't yield, etc.).
    let drain_budget = Duration::from_secs(12);
    let exit = daemon
        .wait_exit_within(drain_budget)
        .await
        .expect("daemon must exit within drain budget after SIGTERM");

    // Clean shutdown contract:
    // - exit_code == Some(0) (the trap handler caught SIGTERM,
    //   cancelled the shutdown token, the supervisor's drain
    //   returned Ok, main_body returned cleanly), OR
    // - terminating_signal == Some(SIGTERM) when the harness
    //   raced the daemon's signal-handler install and SIGTERM
    //   arrived BEFORE `tokio::signal::unix::signal(SIGTERM)`
    //   registered its handler.  Tokio's installer races the
    //   "TCP listener bound" boot marker the harness scans for --
    //   either order is observable on a busy host.  Both are
    //   clean per the daemon's contract (no resources to leak
    //   pre-listener); the FAIL case is exit_code != 0 OR any
    //   other signal (typically SIGABRT from `process::abort`).
    let clean = exit.exit_code == Some(0)
        || exit.terminating_signal == Some(nix::sys::signal::Signal::SIGTERM as i32);
    assert!(
        clean,
        "daemon SIGTERM should produce clean exit (code 0 or SIGTERM-terminated); \
         got exit_code={:?}, terminating_signal={:?}\n\
         ===== BOOT STDERR =====\n{}",
        exit.exit_code, exit.terminating_signal, exit.boot_stderr,
    );

    // Workspace consistency: no orphan files at the workspace-
    // root level.  The daemon's `workspaces/` dir is auto-created
    // empty on a fresh boot (no operator created any workspaces
    // in this test); a clean shutdown leaves it empty.
    let workspaces_root = cwd.join("workspaces");
    if workspaces_root.exists() {
        let entries: Vec<_> = std::fs::read_dir(&workspaces_root)
            .expect("workspaces/ must be readable")
            .collect();
        assert!(
            entries.is_empty(),
            "workspaces/ must be empty after clean shutdown (no operator \
             created any workspaces); got {:?}",
            entries
                .iter()
                .map(|e| e.as_ref().map(|d| d.file_name()))
                .collect::<Vec<_>>(),
        );
    }
}
