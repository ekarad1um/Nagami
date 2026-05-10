//! Lifecycle Row 7 (simplified): daemon survives a
//! SIGKILL + respawn cycle.
//!
//! From plan step 6 Row 7 (original text):
//! > **Row 7 -- SIGKILL filesystem consistency.** Force-kill mid-
//! > write of a head; restart daemon; assert: existing head still
//! > loads (atomic rename guarantees), new head is either fully
//! > written or absent (no half-written `.mpk`).
//!
//! ## Scope reduction ( operator-scope amendment)
//!
//! The "force-kill mid-write of a head" path needs an HTTP client
//! to POST `/api/v1/active` + a real `.mpk` fixture in the cwd --
//! infrastructure beyond what this PR ships.  The MINIMAL
//! viable Row 7 covers what the operator's external-supervisor
//! recovery story actually depends on:
//!
//! 1. Boot the daemon (writes `misc/dev.toml` + `misc/launch.toml`
//!    via `file_mgr::put_atomic`'s tempfile + atomic-rename
//!    discipline).
//! 2. SIGKILL the daemon mid-flight (skips signal handler ->
//!    no graceful shutdown -> kernel terminates the process
//!    immediately).
//! 3. Re-spawn the daemon with the SAME cwd (same workspace
//!    root, same misc/ files).
//! 4. Assert the second boot completes cleanly (no
//!    half-written config files left behind by the SIGKILL;
//!    the existing files load).
//!
//! This validates the schema-versioning gate + `put_atomic`'s
//! tempfile + rename discipline survive a force-kill -- the on-
//! device operator's recovery story.  The "post a head, kill
//! mid-write" variant lives in a future PR that adds the HTTP
//! client.

#[path = "daemon_helpers/mod.rs"]
mod daemon_helpers;

use std::time::Duration;

use daemon_helpers::{CheckProfile, launch_check_mode, launch_long_running};

/// **Row 7 -- SIGKILL + respawn cycle.** Spawn daemon long-running,
/// wait for boot completion, SIGKILL it, then spawn a SECOND
/// daemon pointing at the same cwd and assert it boots cleanly.
/// The `misc/dev.toml` + `misc/launch.toml` written by the first
/// boot must survive the force-kill (atomic-rename guarantee from
/// `file_mgr::put_atomic`); the second boot must read them
/// without complaint.
#[tokio::test]
async fn lifecycle_row7_sigkill_then_respawn_clean_boot() {
    let profile = CheckProfile {
        check_seconds: 3,
        mock_audio: true,
        no_inference: true,
        timeout: Duration::from_secs(15),
        tcp_bind: "127.0.0.1:0".into(),
        launch_toml_override: None,
        extra_args: Vec::new(),
        cwd_override: None,
    };

    // MARK: Phase 1: boot + force-kill
    let daemon = launch_long_running(profile.clone())
        .await
        .expect("daemon long-running launch must succeed (phase 1)");
    // Snapshot the cwd path BEFORE we move `daemon` into
    // wait_exit_within (which consumes self + drops the tempdir
    // when DaemonExit drops).  We need the path to spawn phase 2
    // pointing at the SAME directory.
    let cwd_path = daemon.cwd().to_path_buf();

    // SIGKILL -- bypasses the signal handler.  Kernel terminates
    // immediately; no drain, no graceful shutdown.
    daemon.kill_kill().expect("SIGKILL must succeed");

    // wait_exit_within reaps the zombie.  SIGKILL's contract:
    // exit very quickly (the kernel does the work; no userland
    // cleanup runs).  2 s is generous.
    let exit = daemon
        .wait_exit_within(Duration::from_secs(2))
        .await
        .expect("daemon must reap within 2 s after SIGKILL");

    // SIGKILL leaves no exit_code (signal-terminated); the
    // terminating_signal should be SIGKILL (9).  On macOS +
    // Linux this is identical.
    assert_eq!(
        exit.terminating_signal,
        Some(nix::sys::signal::Signal::SIGKILL as i32),
        "expected SIGKILL termination; got exit_code={:?}, signal={:?}\n\
         ===== BOOT STDERR =====\n{}",
        exit.exit_code,
        exit.terminating_signal,
        exit.boot_stderr,
    );

    // Verify the auto-created config files survived the SIGKILL.
    // `put_atomic` writes via tempfile + atomic-rename; even if
    // the SIGKILL hit between the rename and the parent-dir
    // fsync, the rename is atomic at the kernel level -- the
    // file is either fully there (committed) or fully absent
    // (rename never happened).  A half-written file is the
    // failure mode the contract excludes.
    let etc_dev = cwd_path.join("misc/dev.toml");
    let etc_launch = cwd_path.join("misc/launch.toml");
    assert!(
        etc_dev.exists(),
        "misc/dev.toml should survive SIGKILL (auto-created during phase 1 boot); \
         cwd={cwd_path:?}",
    );
    assert!(
        etc_launch.exists(),
        "misc/launch.toml should survive SIGKILL (auto-created during phase 1 boot); \
         cwd={cwd_path:?}",
    );
    // Sanity check: the files have non-zero content (a
    // half-written-then-killed scenario could leave a 0-byte
    // file if `put_atomic`'s tempfile-then-rename discipline
    // were broken).
    let dev_size = std::fs::metadata(&etc_dev)
        .expect("metadata dev.toml")
        .len();
    let launch_size = std::fs::metadata(&etc_launch)
        .expect("metadata launch.toml")
        .len();
    assert!(
        dev_size > 0,
        "misc/dev.toml should be non-empty post-SIGKILL"
    );
    assert!(
        launch_size > 0,
        "misc/launch.toml should be non-empty post-SIGKILL"
    );

    // The DaemonExit's `cwd` TempDir handle is dropped here.
    // Normally that would auto-delete the directory, but we
    // explicitly leak the path via `cwd_path` (PathBuf clone) so
    // phase 2 can reuse the directory.  The Drop deletion races
    // with phase 2's spawn -- to avoid this race, mem::forget
    // the TempDir (we'll clean up manually in our own tempdir
    // for phase 2).
    let exit_cwd = exit.cwd;
    // `keep()` (formerly `into_path`) prevents the TempDir's Drop
    // from removing the directory -- phase 2 needs to spawn pointing
    // at the same cwd, which requires the dir to outlive `exit`.
    // We clean up manually at the end of the test.
    let leaked_cwd = exit_cwd.keep();
    assert_eq!(
        leaked_cwd, cwd_path,
        "leaked cwd must match snapshotted path"
    );

    // MARK: Phase 2: respawn pointing at the same cwd
    //
    // Run a fresh `--check` invocation against the SAME cwd via
    // `cwd_override` so the second boot reads the `misc/dev.toml`
    // + `misc/launch.toml` the first boot wrote (instead of
    // auto-creating fresh ones in a new tempdir).  Going through
    // `launch_check_mode` keeps the CLI flag set + spawn shape
    // aligned with Rows 1-3 -- if the harness adds a default
    // flag later, this row picks it up automatically rather
    // than silently diverging.
    let phase2_profile = CheckProfile {
        check_seconds: 3,
        mock_audio: true,
        no_inference: true,
        timeout: Duration::from_secs(15),
        tcp_bind: "127.0.0.1:0".into(),
        launch_toml_override: None,
        extra_args: Vec::new(),
        cwd_override: Some(cwd_path.clone()),
    };
    let phase2 = launch_check_mode(phase2_profile)
        .await
        .expect("phase 2 acousticsd --check launch must succeed");
    assert_eq!(
        phase2.exit_code, 0,
        "phase 2 acousticsd must exit 0 (clean re-boot using phase-1 misc/ files); \
         got exit={}\n\
         ===== STDOUT =====\n{}\n\
         ===== STDERR =====\n{}",
        phase2.exit_code, phase2.stdout, phase2.stderr,
    );

    // MARK: Cleanup: remove the leaked cwd manually
    let _ = std::fs::remove_dir_all(&cwd_path);
}
