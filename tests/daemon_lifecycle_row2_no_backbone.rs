//! Lifecycle Row 2: no backbone available.
//!
//! From plan step 6 Row 2:
//! > **Row 2 -- RKNN absent.** Boot with no `.rknn` in catalogue ->
//! > daemon stays up; `/api/v1/inference/active` returns 404 with
//! > structured error; `/api/v1/status` shows
//! > `inference: Degraded { reason: "no backbone" }`.
//!
//! This test pre-writes a `launch.toml` with an empty backbone
//! catalogue (`[backbone] candidates = []`) AND keeps a working
//! mock mic so the audio pipeline still boots.  The daemon's
//! inference subsystem then takes the involuntary-skip path
//! (.b sub-step in `modules/daemon/main_body.rs`) and
//! reports `Heartbeat::degraded(detail, "no_backbone")`.
//!
//! What this row asserts:
//! - daemon `--check` exits 0 (degraded != unhealthy per B.3.1's
//!   orthogonal-axes model -- `healthy` stays true).
//! - the `inference` subsystem's snapshot carries
//!   `degraded_reason: "no_backbone"`.
//! - all OTHER subsystems remain plain healthy (no false-positive
//!   degradation cascade).
//!
//! Doesn't run with `--no-inference` -- that's the VOLUNTARY skip
//! path, which still reports plain `Heartbeat::ok` per the
//! distinction landed in the same acoustics commit.  Row 2's
//! contract is specifically the involuntary-skip case.

#[path = "daemon_helpers/mod.rs"]
mod daemon_helpers;

use std::time::Duration;

use daemon_helpers::{CheckProfile, launch_check_mode};

/// `misc/launch.toml` fixture: keeps the mock mic so audio works,
/// but `[backbone].candidates` is an empty array so the
/// `launch.backbone.is_empty()` branch in `acoustics`'s boot
/// sequence triggers.  Inline string (not a checked-in fixture
/// file) so the test is self-contained.
///
/// The mic catalogue mirrors `LaunchConfig::default_for()`'s
/// stock `default-mock` candidate; the daemon's launch loader
/// accepts arbitrary mic candidate sets so any single working
/// mic + empty backbone exercises Row 2's contract.
const NO_BACKBONE_LAUNCH_TOML: &str = r#"
[[mic.candidates]]
id = "default-mock"
channels = [0]
source = { kind = "mock", period_size = 512, sample_rate = 44100, waveforms = [{ kind = "sine", freq_hz = 1000.0, amplitude = 0.25 }] }

[backbone]
candidates = []
"#;

/// **Row 2 -- no backbone available.** Pre-write a launch config
/// with an empty backbone catalogue + working mock mic; spawn
/// daemon WITHOUT `--no-inference` (the operator wants
/// inference, but the workspace can't satisfy it); assert exit
/// 0, `inference` subsystem degraded with reason `"no_backbone"`,
/// other subsystems unaffected.
#[tokio::test]
async fn lifecycle_row2_no_backbone_inference_degraded() {
    let profile = CheckProfile {
        check_seconds: 3,
        mock_audio: false,   // Use the launch.toml's mic catalogue.
        no_inference: false, // Operator WANTS inference; the launch
        // config just can't supply a backbone.
        timeout: Duration::from_secs(15),
        tcp_bind: "127.0.0.1:0".into(),
        launch_toml_override: Some(NO_BACKBONE_LAUNCH_TOML.into()),
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
        .expect("snapshot non-None per check above");
    let subsystems = snap
        .get("subsystems")
        .and_then(|v| v.as_object())
        .unwrap_or_else(|| {
            panic!(
                "snapshot missing 'subsystems' object; full snapshot:\n{}",
                serde_json::to_string_pretty(snap).unwrap_or_default(),
            )
        });

    let inference = subsystems.get("inference").unwrap_or_else(|| {
        panic!(
            "snapshot missing 'inference' subsystem; got keys {:?}\n\
             ===== STDOUT =====\n{}\n\
             ===== STDERR =====\n{}",
            subsystems.keys().collect::<Vec<_>>(),
            run.stdout,
            run.stderr,
        )
    });

    // Plan acceptance: inference healthy=true (degraded != unhealthy)
    // + degraded_reason == "no_backbone".
    let healthy = inference
        .get("healthy")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let degraded_reason = inference.get("degraded_reason").and_then(|v| v.as_str());
    assert!(
        healthy,
        "inference subsystem must stay healthy under Row 2 (degraded != \
         unhealthy per B.3.1 orthogonal-axes): {inference}\n\
         ===== STDOUT =====\n{}\n\
         ===== STDERR =====\n{}",
        run.stdout, run.stderr,
    );
    assert_eq!(
        degraded_reason,
        Some("no_backbone"),
        "inference subsystem must carry degraded_reason=no_backbone \
         on Row 2: {inference}\n\
         ===== STDOUT =====\n{}\n\
         ===== STDERR =====\n{}",
        run.stdout,
        run.stderr,
    );

    // Defense-in-depth: the OTHER subsystems should NOT cascade-
    // degrade just because inference can't load a backbone.  The
    // daemon's audio + opus + stream + training paths are
    // independent of the inference engine; if any of them flips
    // degraded here it's a regression in the .b wiring.
    for (name, view) in subsystems {
        if name == "inference" {
            continue;
        }
        let degraded = view
            .get("degraded_reason")
            .and_then(|v| v.as_str())
            .is_some();
        assert!(
            !degraded,
            "subsystem {name:?} cascade-degraded on Row 2 (inference \
             missing a backbone shouldn't affect {name}): {view}\n\
             ===== STDOUT =====\n{}\n\
             ===== STDERR =====\n{}",
            run.stdout, run.stderr,
        );
    }
}
