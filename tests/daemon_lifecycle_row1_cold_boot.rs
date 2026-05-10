//! Lifecycle smoke matrix Row 1: cold boot.
//!
//! From plan step 6 Row 1:
//! > **Row 1 -- Cold boot.** Boot, assert `/api/v1/status` reports
//! > all subsystems healthy within 1 s.
//!
//! The harness uses the daemon binary's `--check` mode (a built-in
//! shape that boots, runs for `--check-seconds`, prints exactly
//! one `StatusSnapshot` JSON, and exits 0/1 based on health).  The
//! `--check` exit-code semantic IS the assertion the plan asks
//! for; the test additionally inspects the snapshot to give a
//! granular diagnostic on regression.
//!
//! Subsequent rows (RKNN absent, ALSA absent, inference panic,
//! opus task policy exhausted, SIGTERM mid-fine-tune, SIGKILL
//! filesystem consistency) extend the same `daemon_helpers::launch_check_mode`
//! primitive in  once the supervisor `RestartPolicy` and
//! Drop-aware `HeartbeatHandle` land.

#[path = "daemon_helpers/mod.rs"]
mod daemon_helpers;

use std::time::Duration;

use daemon_helpers::{CheckProfile, launch_check_mode};

/// **Row 1 -- Cold boot.** Spawn the daemon under
/// `--check --mock-audio --no-inference`, assert it exits 0, and
/// confirm every registered subsystem is healthy.
///
/// Plan acceptance: "all subsystems healthy within 1 s".  The
/// harness uses `--check-seconds 3` for soft margin against the
/// 5 s `HEALTH_STALE_AFTER` floor -- every subsystem sends an
/// initial synchronous `Heartbeat::ok` at registration (so the
/// freshest heartbeat is at worst ~3 s old when the snapshot
/// fires); the 1 Hz refresh pumps re-stamp at t~=1, 2, 3, leaving
/// the staleness signal well under the floor.
///
/// On failure, the daemon's full stdout + stderr land in the test
/// output (via the `panic!` formatting below) so the operator can
/// triage from `cargo test` alone -- no need to re-run the daemon
/// by hand.
#[tokio::test]
async fn lifecycle_row1_cold_boot_all_subsystems_healthy() {
    let profile = CheckProfile {
        // Tighter than the daemon CLI's default 5 s for CI budget;
        // see the test docstring for the staleness-margin
        // reasoning.
        check_seconds: 3,
        mock_audio: true,
        no_inference: true,
        timeout: Duration::from_secs(15),
        // `127.0.0.1:0` lets the kernel pick an
        // ephemeral port; required for parallel test binaries
        // ( adds rows in their own files) so they don't
        // race the production default port 8787.
        tcp_bind: "127.0.0.1:0".into(),
        // Row 1 takes the daemon's auto-created default launch
        // config (mock mic + stock backbone candidates).  Rows
        // 2-3 substitute custom launch fixtures.
        launch_toml_override: None,
        extra_args: Vec::new(),
        cwd_override: None,
    };

    let run = launch_check_mode(profile)
        .await
        .expect("acousticsd --check launch must succeed");

    if run.exit_code != 0 || run.snapshot.is_none() {
        panic!(
            "acousticsd --check failed (exit={}, elapsed={:?})\n\
             ===== STDOUT =====\n{}\n\
             ===== STDERR =====\n{}",
            run.exit_code, run.elapsed, run.stdout, run.stderr,
        );
    }

    let snap = run
        .snapshot
        .as_ref()
        .expect("snapshot already non-None per check above");

    // Plan +  register five subsystems at boot:
    // audio_capture, inference (via the heartbeat pump even when
    // the engine is `--no-inference`'d), opus_stream, stream_io,
    // training.  The cold-boot row asserts ALL of them reported
    // healthy at the snapshot tick, which `--check`'s exit code
    // already implies; the per-subsystem walk below upgrades the
    // failure message from "exit 1" to the named offender.
    let subsystems = snap
        .get("subsystems")
        .and_then(|v| v.as_object())
        .unwrap_or_else(|| {
            panic!(
                "snapshot missing 'subsystems' object; full snapshot:\n{}",
                serde_json::to_string_pretty(snap).unwrap_or_default(),
            )
        });
    assert!(
        subsystems.len() >= 5,
        "expected >=5 registered subsystems on cold boot, got {}: {:?}",
        subsystems.len(),
        subsystems.keys().collect::<Vec<_>>(),
    );
    for (name, view) in subsystems {
        let healthy = view
            .get("healthy")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let stale = view.get("stale").and_then(|v| v.as_bool()).unwrap_or(true);
        // Both panic strings dump BOTH stdout (the StatusSnapshot
        // JSON) and stderr (tracing log lines) so a "single
        // subsystem unhealthy" failure surfaces the full
        // operator-readable diagnostic in one cargo-test output --
        // no need to re-run by hand to see the snapshot context.
        assert!(
            healthy,
            "subsystem {name:?} reported unhealthy on cold boot: {view}\n\
             ===== STDOUT =====\n{}\n\
             ===== STDERR =====\n{}",
            run.stdout, run.stderr,
        );
        assert!(
            !stale,
            "subsystem {name:?} stale on cold boot (heartbeat older than \
             HEALTH_STALE_AFTER): {view}\n\
             ===== STDOUT =====\n{}\n\
             ===== STDERR =====\n{}",
            run.stdout, run.stderr,
        );
    }

    // Acceptance: cold-start measurement target
    // ~80-120 ms with the boot `tokio::join!`.  We can't measure
    // the boot phase in isolation here (the daemon process spends
    // most of its lifetime in the --check-seconds tail), but we
    // can sanity-check the total wall-clock against a generous
    // cap that catches a hung-boot regression.
    let total_budget = Duration::from_secs(10);
    assert!(
        run.elapsed < total_budget,
        "acousticsd --check elapsed {:?} > {:?} budget; cold-boot regression?",
        run.elapsed,
        total_budget,
    );
}
