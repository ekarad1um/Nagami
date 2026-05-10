//! Lifecycle Row 3: no mic device available.
//!
//! From plan step 6 Row 3:
//! > **Row 3 -- ALSA absent.** Boot with no ALSA device -> daemon
//! > stays up; `audio_capture` heartbeat is `Degraded { reason:
//! > "no device" }`; opus + inference fail-over to "waiting for
//! > source".
//!
//! This test pre-writes a `launch.toml` with an empty
//! `[[mic.candidates]]` list AND an empty backbone catalogue
//! (so the daemon spins up without expecting either).  The
//! audio_capture subsystem then takes the
//! `no_mic_configured` branch in `modules/daemon/main_body.rs`
//! (.b sub-step) and reports `Heartbeat::degraded(detail,
//! "no_device")` BOTH at registration AND on every 1 Hz pump
//! tick (the head-advance pump preserves the degraded variant
//! when no mic is configured rather than flipping to
//! "unhealthy", since "no device" is an operator config gap not
//! a transient outage).
//!
//! What this row asserts:
//! - daemon `--check` exits 0 (degraded != unhealthy per B.3.1's
//!   orthogonal-axes model -- `healthy` stays true on an
//!   operator-config gap).
//! - the `audio_capture` subsystem's snapshot carries
//!   `degraded_reason: "no_device"`.
//! - the `inference` subsystem ALSO degraded (no_backbone) since
//!   we used an empty backbone catalogue too -- defends the
//!   non-cascading independence from Row 2 by exercising both
//!   degradation paths in the same boot.

#[path = "daemon_helpers/mod.rs"]
mod daemon_helpers;

use std::time::Duration;

use daemon_helpers::{CheckProfile, launch_check_mode};

/// `misc/launch.toml` fixture: empty mic catalogue + empty
/// backbone catalogue.  The daemon's loader accepts both empties
/// (the launch validator's `is_empty()` check just emits a
/// `tracing::warn!` rather than refusing boot per
/// `modules/daemon/main_body.rs:441-447`).
const NO_MIC_NO_BACKBONE_LAUNCH_TOML: &str = r#"
[mic]
candidates = []

[backbone]
candidates = []
"#;

/// **Row 3 -- no mic device available.** Pre-write a launch
/// config with an empty mic catalogue (and empty backbone, so
/// the daemon doesn't try to load a `.rknn`); spawn daemon
/// WITHOUT `--mock-audio` (the operator wants real audio but
/// the launch config doesn't supply any candidates); assert
/// exit 0, `audio_capture` subsystem degraded with reason
/// `"no_device"`.
#[tokio::test]
async fn lifecycle_row3_no_mic_audio_capture_degraded() {
    let profile = CheckProfile {
        check_seconds: 3,
        // Operator wants real audio (no --mock-audio) but the
        // launch config supplies no candidates -- that's the
        // misconfig Row 3 exercises.
        mock_audio: false,
        // Empty backbone too so the test is self-contained
        // (no need to ship a `.rknn` fixture); the inference
        // subsystem reaches Row 2's "no_backbone" path.
        no_inference: false,
        timeout: Duration::from_secs(15),
        tcp_bind: "127.0.0.1:0".into(),
        launch_toml_override: Some(NO_MIC_NO_BACKBONE_LAUNCH_TOML.into()),
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

    let audio = subsystems.get("audio_capture").unwrap_or_else(|| {
        panic!(
            "snapshot missing 'audio_capture' subsystem; got keys {:?}\n\
             ===== STDOUT =====\n{}\n\
             ===== STDERR =====\n{}",
            subsystems.keys().collect::<Vec<_>>(),
            run.stdout,
            run.stderr,
        )
    });

    // Plan acceptance: audio_capture healthy=true (degraded !=
    // unhealthy on a misconfig) + degraded_reason == "no_device".
    let healthy = audio
        .get("healthy")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let degraded_reason = audio.get("degraded_reason").and_then(|v| v.as_str());
    assert!(
        healthy,
        "audio_capture must stay healthy under Row 3 (degraded != \
         unhealthy on operator misconfig per B.3.1 orthogonal-axes): {audio}\n\
         ===== STDOUT =====\n{}\n\
         ===== STDERR =====\n{}",
        run.stdout, run.stderr,
    );
    assert_eq!(
        degraded_reason,
        Some("no_device"),
        "audio_capture must carry degraded_reason=no_device on Row 3: {audio}\n\
         ===== STDOUT =====\n{}\n\
         ===== STDERR =====\n{}",
        run.stdout,
        run.stderr,
    );

    // Defense-in-depth: inference should also be degraded
    // (no_backbone) since we used an empty backbone catalogue.
    // Two independent degradation paths in one boot proves
    // they're independent (Row 2 + Row 3 don't cascade-trigger
    // each other; they each fire when their own producer's
    // launch-config branch flags the gap).
    let inference = subsystems.get("inference").expect("inference registered");
    assert_eq!(
        inference.get("degraded_reason").and_then(|v| v.as_str()),
        Some("no_backbone"),
        "inference must report no_backbone independently in Row 3 \
         (defends non-cascading independence): {inference}",
    );

    // The remaining subsystems (opus_stream, stream_io, training)
    // should NOT have degraded_reason set -- Row 3 is specifically
    // about the audio + inference paths, not about a
    // cascade-degradation regression elsewhere.
    for name in ["opus_stream", "stream_io", "training"] {
        let view = subsystems
            .get(name)
            .unwrap_or_else(|| panic!("{name} registered"));
        assert!(
            view.get("degraded_reason")
                .and_then(|v| v.as_str())
                .is_none(),
            "subsystem {name:?} should NOT cascade-degrade on Row 3: {view}",
        );
    }
}
