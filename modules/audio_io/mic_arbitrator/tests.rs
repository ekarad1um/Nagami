//! Mic arbitrator integration + unit tests.

#![cfg(test)]

use super::*;
use crate::audio_io::mock::Waveform;
use crate::common::ids::MicId;
use arc_swap::ArcSwap;

/// Tests construct `Arc<ArcSwap<MicSettings>>` for hot-swap
/// simulation; [`MicArbitrator::start`] takes
/// `Arc<dyn MicSettingsStore>`, so wrap via [`ArcSwapStore`].
/// The returned trait-object [`Arc`] is what the arbitrator
/// wants while the test keeps the inner [`ArcSwap`]
/// reachable for mid-run mutations.
fn arcswap_store_into_dyn(s: Arc<ArcSwap<MicSettings>>) -> Arc<dyn MicSettingsStore> {
    Arc::new(ArcSwapStore(s))
}

fn alsa_candidate(id: &str, channels: Vec<u16>) -> MicCandidate {
    MicCandidate {
        id: MicId::parse(id).expect("test mic id literal"),
        source: CandidateSource::Alsa {
            hw_spec: "hw:1,0".into(),
            period_size: 1024,
            buffer_size: 4096,
        },
        channels,
    }
}

fn mock_candidate(id: &str, channels: Vec<u16>, n_wave: usize) -> MicCandidate {
    MicCandidate {
        id: MicId::parse(id).expect("test mic id literal"),
        source: CandidateSource::Mock {
            waveforms: vec![Waveform::Silence; n_wave],
            period_size: 512,
            sample_rate: SampleRate::VALUE,
        },
        channels,
    }
}

#[test]
fn validate_accepts_well_formed_alsa() {
    assert!(alsa_candidate("a", vec![0, 1]).validate().is_ok());
}

#[test]
fn validate_accepts_well_formed_mock() {
    assert!(mock_candidate("m", vec![0, 2], 4).validate().is_ok());
}

#[test]
fn validate_rejects_empty_channels() {
    assert_eq!(
        alsa_candidate("a", vec![]).validate(),
        Err(CandidateError::EmptyChannels),
    );
}

#[test]
fn validate_rejects_duplicate_channel() {
    assert_eq!(
        alsa_candidate("a", vec![0, 1, 0]).validate(),
        Err(CandidateError::DuplicateChannel(0)),
    );
}

/// Static cap defends downstream u16 arithmetic in
/// `AlsaSource::open` against silent overflow on adversarial /
/// typo configs.
#[test]
fn validate_rejects_channel_above_static_cap() {
    let too_high = MAX_CHANNEL_INDEX + 1;
    let err = alsa_candidate("a", vec![0, too_high]).validate();
    assert_eq!(
        err,
        Err(CandidateError::ChannelIndexTooLarge {
            channel: too_high,
            cap: MAX_CHANNEL_INDEX,
        }),
    );
}

/// The per-index cap is inclusive -- `MAX_CHANNEL_INDEX`
/// itself is allowed (and the `u32 -> u16` round-trip still
/// fits).  Uses an Alsa candidate because Mock would also
/// trip [`MAX_CHANNELS`] by needing `MAX_CHANNEL_INDEX + 1`
/// waveforms; ALSA channel count is negotiated against the
/// device at open time, so the candidate doesn't have to
/// declare the device's channel inventory statically.
#[test]
fn validate_accepts_channel_at_static_cap() {
    let max = MAX_CHANNEL_INDEX;
    let c = MicCandidate {
        id: MicId::from_static("at-cap"),
        source: CandidateSource::Alsa {
            hw_spec: "hw:1,0".into(),
            period_size: 1024,
            buffer_size: 4096,
        },
        channels: vec![max],
    };
    assert_eq!(c.validate(), Ok(()));
}

#[test]
fn validate_rejects_empty_hw_spec() {
    let c = MicCandidate {
        id: MicId::from_static("a"),
        source: CandidateSource::Alsa {
            hw_spec: "".into(),
            period_size: 1024,
            buffer_size: 4096,
        },
        channels: vec![0],
    };
    assert_eq!(c.validate(), Err(CandidateError::EmptyHwSpec));
}

#[test]
fn validate_rejects_zero_period() {
    let c = MicCandidate {
        id: MicId::from_static("a"),
        source: CandidateSource::Alsa {
            hw_spec: "hw:1,0".into(),
            period_size: 0,
            buffer_size: 4096,
        },
        channels: vec![0],
    };
    assert_eq!(c.validate(), Err(CandidateError::InvalidPeriodSize(0)));
}

#[test]
fn validate_rejects_buffer_smaller_than_period() {
    let c = MicCandidate {
        id: MicId::from_static("a"),
        source: CandidateSource::Alsa {
            hw_spec: "hw:1,0".into(),
            period_size: 4096,
            buffer_size: 1024,
        },
        channels: vec![0],
    };
    assert_eq!(
        c.validate(),
        Err(CandidateError::InvalidBufferSize {
            period: 4096,
            buffer: 1024,
        }),
    );
}

#[test]
fn validate_rejects_empty_mock_waveforms() {
    let c = MicCandidate {
        id: MicId::from_static("m"),
        source: CandidateSource::Mock {
            waveforms: vec![],
            period_size: 512,
            sample_rate: 44_100,
        },
        channels: vec![0],
    };
    assert_eq!(c.validate(), Err(CandidateError::EmptyMockWaveforms));
}

#[test]
fn validate_rejects_whitelist_exceeding_mock_waveforms() {
    // Whitelist asks for channel 3 but mock has only 2 waveforms.
    let c = mock_candidate("m", vec![0, 3], 2);
    assert_eq!(
        c.validate(),
        Err(CandidateError::MockWhitelistOutOfRange {
            whitelist_max: 3,
            waveform_count: 2,
        }),
    );
}

#[test]
fn validate_rejects_zero_mock_sample_rate() {
    let c = MicCandidate {
        id: MicId::from_static("m"),
        source: CandidateSource::Mock {
            waveforms: vec![Waveform::Silence; 2],
            period_size: 512,
            sample_rate: 0,
        },
        channels: vec![0],
    };
    assert_eq!(c.validate(), Err(CandidateError::InvalidMockSampleRate(0)));
}

// MARK: Upper-bound checks

/// Oversized ALSA period_size is rejected before it can violate
/// the audio_buffer Writer::push safety margin or trigger an
/// oversize scratch allocation.
#[test]
fn validate_rejects_oversized_alsa_period_size() {
    let too_big = MAX_PERIOD_FRAMES + 1;
    let c = MicCandidate {
        id: MicId::from_static("a"),
        source: CandidateSource::Alsa {
            hw_spec: "hw:1,0".into(),
            period_size: too_big,
            buffer_size: too_big * 4,
        },
        channels: vec![0],
    };
    assert_eq!(
        c.validate(),
        Err(CandidateError::PeriodSizeTooLarge {
            period: too_big,
            cap: MAX_PERIOD_FRAMES,
        }),
    );
}

/// The cap is inclusive -- exactly `MAX_PERIOD_FRAMES`
/// frames is allowed (the safety-margin invariant still
/// holds at the canonical AudioBuffer capacity 262 144
/// where `max_push_len = 65 536` >> `MAX_PERIOD_FRAMES`).
#[test]
fn validate_accepts_alsa_period_at_static_cap() {
    let c = MicCandidate {
        id: MicId::from_static("a"),
        source: CandidateSource::Alsa {
            hw_spec: "hw:1,0".into(),
            period_size: MAX_PERIOD_FRAMES,
            buffer_size: MAX_PERIOD_FRAMES * 4,
        },
        channels: vec![0],
    };
    assert_eq!(c.validate(), Ok(()));
}

/// Oversized Mock period_size is rejected too (mock pushes
/// through the same writer as ALSA).
#[test]
fn validate_rejects_oversized_mock_period_size() {
    let too_big = MAX_PERIOD_FRAMES + 1;
    let c = MicCandidate {
        id: MicId::from_static("m"),
        source: CandidateSource::Mock {
            waveforms: vec![Waveform::Silence; 2],
            period_size: too_big,
            sample_rate: 44_100,
        },
        channels: vec![0],
    };
    assert_eq!(
        c.validate(),
        Err(CandidateError::PeriodSizeTooLarge {
            period: too_big,
            cap: MAX_PERIOD_FRAMES,
        }),
    );
}

/// ALSA buffer_size > absolute cap is rejected.
#[test]
fn validate_rejects_oversized_alsa_buffer_absolute_cap() {
    // Use a period of 1 to skip the multiplier check: 1 *
    // 16 = 16, much smaller than MAX_BUFFER_FRAMES, so the
    // multiplier-cap path fires first if buffer > 16. To
    // hit the absolute-cap path we need
    // `buffer > MAX_BUFFER_FRAMES` AND
    // `buffer <= 16 * period_size`. Pick period =
    // MAX_PERIOD_FRAMES so multiplier cap = MAX_BUFFER_FRAMES,
    // then any buffer > MAX_BUFFER_FRAMES trips both bounds
    // (we just verify the error kind).
    let too_big = MAX_BUFFER_FRAMES + 1;
    let c = MicCandidate {
        id: MicId::from_static("a"),
        source: CandidateSource::Alsa {
            hw_spec: "hw:1,0".into(),
            period_size: MAX_PERIOD_FRAMES,
            buffer_size: too_big,
        },
        channels: vec![0],
    };
    assert!(matches!(
        c.validate(),
        Err(CandidateError::BufferSizeTooLarge { .. }),
    ));
}

/// ALSA buffer_size > 16x period_size is
/// rejected (catches the samples-vs-ms typo).
#[test]
fn validate_rejects_buffer_exceeding_period_multiplier() {
    let c = MicCandidate {
        id: MicId::from_static("a"),
        source: CandidateSource::Alsa {
            hw_spec: "hw:1,0".into(),
            period_size: 1024,
            // 17x period: > 16 * period yet < absolute cap.
            buffer_size: 1024 * 17,
        },
        channels: vec![0],
    };
    assert!(matches!(
        c.validate(),
        Err(CandidateError::BufferSizeTooLarge { .. }),
    ));
}

/// Whitelist longer than `MAX_CHANNELS` is rejected.
#[test]
fn validate_rejects_oversized_channel_whitelist() {
    let c = MicCandidate {
        id: MicId::from_static("a"),
        source: CandidateSource::Alsa {
            hw_spec: "hw:1,0".into(),
            period_size: 1024,
            buffer_size: 4096,
        },
        channels: (0..(MAX_CHANNELS as u16 + 1)).collect(),
    };
    assert_eq!(
        c.validate(),
        Err(CandidateError::TooManyChannels {
            count: MAX_CHANNELS + 1,
            cap: MAX_CHANNELS,
        }),
    );
}

/// More than `MAX_CHANNELS` Mock waveforms is rejected.
#[test]
fn validate_rejects_oversized_mock_waveforms() {
    let c = MicCandidate {
        id: MicId::from_static("m"),
        source: CandidateSource::Mock {
            waveforms: vec![Waveform::Silence; MAX_CHANNELS + 1],
            period_size: 512,
            sample_rate: 44_100,
        },
        channels: vec![0],
    };
    assert_eq!(
        c.validate(),
        Err(CandidateError::TooManyChannels {
            count: MAX_CHANNELS + 1,
            cap: MAX_CHANNELS,
        }),
    );
}

/// Mock sample_rate below MIN_SAMPLE_RATE is
/// rejected (e.g. 4000 Hz would cause a pathological
/// resampler ratio).
#[test]
fn validate_rejects_sample_rate_below_min() {
    let c = MicCandidate {
        id: MicId::from_static("m"),
        source: CandidateSource::Mock {
            waveforms: vec![Waveform::Silence; 2],
            period_size: 512,
            sample_rate: MIN_SAMPLE_RATE - 1,
        },
        channels: vec![0],
    };
    assert_eq!(
        c.validate(),
        Err(CandidateError::SampleRateOutOfRange {
            rate: MIN_SAMPLE_RATE - 1,
            min: MIN_SAMPLE_RATE,
            max: MAX_SAMPLE_RATE,
        }),
    );
}

/// Mock sample_rate above MAX_SAMPLE_RATE is
/// rejected (e.g. a "128 MHz" config typo would allocate
/// multi-MB sinc tables).
#[test]
fn validate_rejects_sample_rate_above_max() {
    let c = MicCandidate {
        id: MicId::from_static("m"),
        source: CandidateSource::Mock {
            waveforms: vec![Waveform::Silence; 2],
            period_size: 512,
            sample_rate: MAX_SAMPLE_RATE + 1,
        },
        channels: vec![0],
    };
    assert_eq!(
        c.validate(),
        Err(CandidateError::SampleRateOutOfRange {
            rate: MAX_SAMPLE_RATE + 1,
            min: MIN_SAMPLE_RATE,
            max: MAX_SAMPLE_RATE,
        }),
    );
}

/// Bounds at the inclusive ends are accepted.
#[test]
fn validate_accepts_sample_rate_at_bounds() {
    for rate in [MIN_SAMPLE_RATE, MAX_SAMPLE_RATE] {
        let c = MicCandidate {
            id: MicId::from_static("m"),
            source: CandidateSource::Mock {
                waveforms: vec![Waveform::Silence; 2],
                period_size: 512,
                sample_rate: rate,
            },
            channels: vec![0],
        };
        assert_eq!(c.validate(), Ok(()), "rate {rate} should be accepted");
    }
}

#[test]
fn catalogue_validate_propagates_offending_id() {
    let catalogue = MicCatalogue {
        candidates: vec![
            alsa_candidate("good", vec![0]),
            mock_candidate("bad", vec![5], 2), // out of range
        ],
    };
    let err = catalogue.validate().expect_err("should reject");
    assert_eq!(err.0, MicId::from_static("bad"));
    assert!(matches!(
        err.1,
        CandidateError::MockWhitelistOutOfRange { .. }
    ));
}

/// Catalogue-level invariant: candidate ids must be unique.
/// The arbitrator's `Fixed { id }` resolution + `FirstAvailable`
/// walk both rely on `position()` (first match wins), so a
/// second candidate with a duplicate id is silently dead.  Almost
/// always an operator copy-paste typo -- catch at validation.
#[test]
fn catalogue_validate_rejects_duplicate_ids() {
    let catalogue = MicCatalogue {
        candidates: vec![
            alsa_candidate("front", vec![0]),
            alsa_candidate("rear", vec![0]),
            alsa_candidate("front", vec![1]), // duplicate of #0
        ],
    };
    let err = catalogue.validate().expect_err("must reject");
    assert_eq!(err.0, MicId::from_static("front"));
    assert_eq!(
        err.1,
        CandidateError::DuplicateMicId(MicId::from_static("front"))
    );
}

/// Distinct ids in any order pass -- duplicate detection must
/// not produce false positives on legitimately-different ids.
#[test]
fn catalogue_validate_accepts_distinct_ids() {
    let catalogue = MicCatalogue {
        candidates: vec![
            alsa_candidate("a", vec![0]),
            alsa_candidate("b", vec![0]),
            alsa_candidate("c", vec![0]),
        ],
    };
    assert_eq!(catalogue.validate(), Ok(()));
}

/// `MicCatalogue::find` -- O(N) lookup that backs API + arbitrator
/// resolution paths.  Tested directly because it's the
/// canonical entry point for cross-checking against a policy.
#[test]
fn catalogue_find_returns_matching_candidate() {
    let catalogue = MicCatalogue {
        candidates: vec![alsa_candidate("a", vec![0]), alsa_candidate("b", vec![1])],
    };
    assert_eq!(
        catalogue.find(&MicId::from_static("b")).map(|c| &c.id),
        Some(&MicId::from_static("b")),
    );
    assert!(catalogue.find(&MicId::from_static("missing")).is_none());
}

/// Cross-validation: `Fixed { id }` referencing an unknown mic
/// is rejected.  This is the boot/reload/API guard that prevents
/// the arbitrator from accepting a satisfiable-looking but
/// runtime-impossible policy.
#[test]
fn validate_policy_rejects_unknown_fixed_id() {
    let catalogue = MicCatalogue {
        candidates: vec![alsa_candidate("front", vec![0])],
    };
    let policy = MicPolicy {
        mic: MicSelection::Fixed {
            id: MicId::from_static("rear"),
        },
        channel: ChannelSelection::Auto,
    };
    let err = policy
        .validate_against(&catalogue)
        .expect_err("should reject");
    assert_eq!(
        err,
        PolicyValidationError::UnknownMicId(MicId::from_static("rear"))
    );
}

/// Cross-validation: `Fixed { id }` + `Fixed { channel }` with
/// channel not in the catalogue's whitelist for that mic is
/// rejected.
#[test]
fn validate_policy_rejects_channel_outside_catalogue_whitelist() {
    let catalogue = MicCatalogue {
        candidates: vec![alsa_candidate("front", vec![0, 2])],
    };
    let policy = MicPolicy {
        mic: MicSelection::Fixed {
            id: MicId::from_static("front"),
        },
        channel: ChannelSelection::Fixed { channel: 1 }, // not in [0, 2]
    };
    let err = policy
        .validate_against(&catalogue)
        .expect_err("should reject");
    match err {
        PolicyValidationError::ChannelNotAvailable {
            mic,
            channel,
            available,
        } => {
            assert_eq!(mic, MicId::from_static("front"));
            assert_eq!(channel, 1);
            assert_eq!(available, vec![0, 2]);
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

/// `FirstAvailable` policy is always valid against any
/// catalogue -- the resolved mic is unknown until runtime, so
/// the per-mic channel check defers to `pick_slot`.
#[test]
fn validate_policy_accepts_first_available_with_fixed_channel() {
    let catalogue = MicCatalogue {
        candidates: vec![alsa_candidate("a", vec![0])],
    };
    let policy = MicPolicy {
        mic: MicSelection::FirstAvailable,
        channel: ChannelSelection::Fixed { channel: 99 }, // unverifiable
    };
    assert_eq!(policy.validate_against(&catalogue), Ok(()));
}

/// `Fixed { id }` + `Auto` channel is valid as long as the mic
/// id matches the catalogue.
#[test]
fn validate_policy_accepts_fixed_mic_with_auto_channel() {
    let catalogue = MicCatalogue {
        candidates: vec![alsa_candidate("a", vec![0, 1, 2])],
    };
    let policy = MicPolicy {
        mic: MicSelection::Fixed {
            id: MicId::from_static("a"),
        },
        channel: ChannelSelection::Auto,
    };
    assert_eq!(policy.validate_against(&catalogue), Ok(()));
}

#[test]
fn policy_default_is_first_available_auto() {
    let p = MicPolicy::default();
    assert_eq!(p.mic, MicSelection::FirstAvailable);
    assert_eq!(p.channel, ChannelSelection::Auto);
}

#[test]
fn hysteresis_linear_3db_is_correct() {
    let cfg = MicArbitratorConfig::default();
    let h = cfg.hysteresis_linear();
    // 10^(3/20) ~= 1.4125.
    assert!((h - 1.4125).abs() < 1e-3, "h={h}");
}

/// TOML round-trip for the **catalogue** half (launch config).
/// `MicSettings` itself isn't a TOML type anymore -- catalogue
/// and policy persist in separate files; this test asserts
/// each persists faithfully.
#[test]
fn toml_round_trip_preserves_catalogue() {
    let original = MicCatalogue {
        candidates: vec![
            MicCandidate {
                id: MicId::from_static("front"),
                source: CandidateSource::Alsa {
                    hw_spec: "hw:1,0".into(),
                    period_size: 1024,
                    buffer_size: 4096,
                },
                channels: vec![0, 1],
            },
            MicCandidate {
                id: MicId::from_static("dev-mock"),
                source: CandidateSource::Mock {
                    waveforms: vec![
                        Waveform::Silence,
                        Waveform::Sine {
                            freq_hz: 1000.0,
                            amplitude: 0.25,
                        },
                    ],
                    period_size: 512,
                    sample_rate: 44_100,
                },
                channels: vec![1],
            },
        ],
    };
    let s = toml::to_string_pretty(&original).expect("serialize");
    let back: MicCatalogue = toml::from_str(&s).expect("deserialize");
    assert_eq!(back, original, "catalogue toml round-trip mismatch:\n{s}");
}

/// TOML round-trip for the **policy** half (user config).
#[test]
fn toml_round_trip_preserves_policy() {
    let original = MicPolicy {
        mic: MicSelection::Fixed {
            id: MicId::from_static("front"),
        },
        channel: ChannelSelection::Fixed { channel: 0 },
    };
    let s = toml::to_string_pretty(&original).expect("serialize");
    let back: MicPolicy = toml::from_str(&s).expect("deserialize");
    assert_eq!(back, original, "policy toml round-trip mismatch:\n{s}");
}

// MARK: `pick_slot` pure-function tests

fn slot_state(rms: f32) -> SlotState {
    SlotState { rms }
}

/// Empty slot list returns None.
#[test]
fn pick_slot_empty_returns_none() {
    let now = Instant::now();
    let cfg = MicArbitratorConfig::default();
    assert_eq!(
        pick_slot(
            &ChannelSelection::Auto,
            &[],
            &[],
            None,
            None,
            now,
            cfg.hysteresis_linear(),
            cfg.dwell,
        ),
        None,
    );
}

/// Auto + no current slot picks the loudest immediately.
#[test]
fn pick_slot_auto_no_current_picks_loudest() {
    let now = Instant::now();
    let cfg = MicArbitratorConfig::default();
    let slots = [slot_state(0.05), slot_state(0.30), slot_state(0.10)];
    let whitelist = [0_u16, 1, 2];
    let chosen = pick_slot(
        &ChannelSelection::Auto,
        &slots,
        &whitelist,
        None,
        None,
        now,
        cfg.hysteresis_linear(),
        cfg.dwell,
    );
    assert_eq!(chosen, Some(1));
}

/// Auto + hysteresis blocks a marginal-louder switch.
#[test]
fn pick_slot_auto_hysteresis_blocks_marginal() {
    let now = Instant::now();
    let cfg = MicArbitratorConfig {
        hysteresis_db: 3.0,
        dwell: Duration::ZERO,
        ..MicArbitratorConfig::default()
    };
    // Active slot has RMS 0.10; alt has 0.13 (only ~2.3 dB louder).
    let slots = [slot_state(0.10), slot_state(0.13)];
    let whitelist = [0_u16, 1];
    let chosen = pick_slot(
        &ChannelSelection::Auto,
        &slots,
        &whitelist,
        Some(0),
        Some(now),
        now,
        cfg.hysteresis_linear(),
        cfg.dwell,
    );
    assert_eq!(
        chosen,
        Some(0),
        "should NOT switch (insufficient hysteresis)"
    );
}

/// Auto + clear margin (>= hysteresis) + dwell satisfied -> switch.
#[test]
fn pick_slot_auto_clear_margin_after_dwell_switches() {
    let now = Instant::now();
    let cfg = MicArbitratorConfig::default();
    let slots = [slot_state(0.10), slot_state(0.30)]; // ~9.5 dB louder
    let whitelist = [0_u16, 1];
    // Last switch was 1 s ago (well past 250 ms dwell).
    let chosen = pick_slot(
        &ChannelSelection::Auto,
        &slots,
        &whitelist,
        Some(0),
        Some(now - Duration::from_secs(1)),
        now,
        cfg.hysteresis_linear(),
        cfg.dwell,
    );
    assert_eq!(chosen, Some(1), "should switch (clear margin + dwell ok)");
}

/// Auto + dwell not satisfied -> keeps the active slot even with
/// a clear margin.
#[test]
fn pick_slot_auto_dwell_blocks_recent_switch() {
    let now = Instant::now();
    let cfg = MicArbitratorConfig {
        dwell: Duration::from_millis(250),
        ..MicArbitratorConfig::default()
    };
    let slots = [slot_state(0.10), slot_state(0.50)]; // 14 dB louder
    let whitelist = [0_u16, 1];
    // Last switch was 100 ms ago -- within dwell.
    let chosen = pick_slot(
        &ChannelSelection::Auto,
        &slots,
        &whitelist,
        Some(0),
        Some(now - Duration::from_millis(100)),
        now,
        cfg.hysteresis_linear(),
        cfg.dwell,
    );
    assert_eq!(chosen, Some(0), "should NOT switch (dwell not satisfied)");
}

/// Fixed channel that's in the whitelist is honoured even when
/// another slot is louder.
#[test]
fn pick_slot_fixed_honours_named_channel() {
    let now = Instant::now();
    let cfg = MicArbitratorConfig::default();
    // Slot 0 -> channel 0; slot 1 -> channel 2 (a sparse whitelist).
    let slots = [slot_state(0.99), slot_state(0.05)];
    let whitelist = [0_u16, 2];
    let chosen = pick_slot(
        &ChannelSelection::Fixed { channel: 2 },
        &slots,
        &whitelist,
        Some(0),
        Some(now - Duration::from_secs(10)),
        now,
        cfg.hysteresis_linear(),
        cfg.dwell,
    );
    assert_eq!(chosen, Some(1)); // slot 1 is channel 2
}

/// Fixed channel NOT in the whitelist falls back to Auto rather
/// than emitting silence.  Defensive -- operator policy can refer
/// to a channel a candidate doesn't expose.
#[test]
fn pick_slot_fixed_off_whitelist_falls_back_to_auto() {
    let now = Instant::now();
    let cfg = MicArbitratorConfig::default();
    let slots = [slot_state(0.10), slot_state(0.40)];
    let whitelist = [0_u16, 2]; // no channel 1
    let chosen = pick_slot(
        &ChannelSelection::Fixed { channel: 1 },
        &slots,
        &whitelist,
        None,
        None,
        now,
        cfg.hysteresis_linear(),
        cfg.dwell,
    );
    // Fell back to Auto -> loudest slot.
    assert_eq!(chosen, Some(1));
}

// MARK: `resolve_desired_idx` stickiness + bounds

#[test]
fn resolve_first_available_no_active_picks_zero() {
    let cands = vec![alsa_candidate("a", vec![0]), alsa_candidate("b", vec![0])];
    assert_eq!(
        resolve_desired_idx(&MicSelection::FirstAvailable, &cands, None),
        Some(0),
    );
}

#[test]
fn resolve_first_available_active_present_is_sticky() {
    let cands = vec![alsa_candidate("a", vec![0]), alsa_candidate("b", vec![0])];
    let active = MicId::from_static("b");
    // Active is "b" -> return its index even though "a" is first.
    assert_eq!(
        resolve_desired_idx(&MicSelection::FirstAvailable, &cands, Some(&active)),
        Some(1),
    );
}

#[test]
fn resolve_first_available_active_gone_falls_back_to_zero() {
    let cands = vec![alsa_candidate("a", vec![0]), alsa_candidate("b", vec![0])];
    let active = MicId::from_static("c"); // not in candidates
    assert_eq!(
        resolve_desired_idx(&MicSelection::FirstAvailable, &cands, Some(&active)),
        Some(0),
    );
}

#[test]
fn resolve_first_available_empty_candidates_is_none() {
    assert_eq!(
        resolve_desired_idx(&MicSelection::FirstAvailable, &[], None),
        None,
    );
}

#[test]
fn resolve_fixed_returns_named_or_none() {
    let cands = vec![alsa_candidate("a", vec![0]), alsa_candidate("b", vec![0])];
    assert_eq!(
        resolve_desired_idx(
            &MicSelection::Fixed {
                id: MicId::from_static("b")
            },
            &cands,
            None,
        ),
        Some(1),
    );
    assert_eq!(
        resolve_desired_idx(
            &MicSelection::Fixed {
                id: MicId::from_static("missing")
            },
            &cands,
            Some(&MicId::from_static("a")),
        ),
        None,
    );
}

// MARK: `block_rms`

#[test]
fn block_rms_silence_is_zero() {
    assert_eq!(block_rms(&[0.0; 256]), 0.0);
}

#[test]
fn block_rms_constant_matches_value() {
    let r = block_rms(&[0.5_f32; 1024]);
    assert!((r - 0.5).abs() < 1e-6, "rms={r}");
}

#[test]
fn block_rms_sine_amplitude_over_sqrt2() {
    // RMS of a sine wave with amplitude A is A/sqrt(2).
    let n = 4096;
    let samples: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * 1000.0 / 44100.0 * i as f32).sin())
        .collect();
    let r = block_rms(&samples);
    let expected = 1.0 / 2f32.sqrt();
    assert!((r - expected).abs() < 1e-2);
}

#[test]
fn ema_alpha_block_eq_window_is_one_minus_e_inverse() {
    let a = ema_alpha(Duration::from_millis(100), Duration::from_millis(100));
    assert!((a - (1.0 - (-1.0_f32).exp())).abs() < 1e-6, "alpha={a}");
}

#[test]
fn ema_alpha_zero_window_is_one() {
    assert_eq!(ema_alpha(Duration::from_millis(10), Duration::ZERO), 1.0);
}

#[test]
fn alpha_for_frames_uses_cached_value_for_nominal_period() {
    let cached = 0.123_456_f32;
    assert_eq!(
        alpha_for_frames(1024, 1024, 44_100, cached, Duration::from_millis(100)),
        cached,
    );
}

/// ALSA can return `Ok(frames)` with a short, non-zero read on
/// hot-unplug / partial-read edges.  That block must not use the
/// full-period alpha; otherwise the EMA over-weights the partial
/// RMS sample.
#[test]
fn alpha_for_frames_recomputes_for_short_read() {
    let window = Duration::from_millis(100);
    let cached = ema_alpha(Duration::from_secs_f64(1024.0 / 44_100.0), window);
    let short = alpha_for_frames(256, 1024, 44_100, cached, window);
    let expected = ema_alpha(Duration::from_secs_f64(256.0 / 44_100.0), window);

    assert!((short - expected).abs() < 1e-7, "short alpha = {short}");
    assert!(short < cached, "short read should have less EMA weight");
}

/// `reset_per_channel_fir` clears FIR history + pending
/// output of every `Some` resampler and leaves `None` slots
/// untouched.  After reset, fresh input through the resampler
/// is bit-identical to a freshly-constructed resampler (i.e.
/// no phantom of pre-reset samples leaks into post-reset
/// output).  This property is what the arbitrator's xrun
/// recovery path depends on -- only reachable on Linux with
/// the `alsa-real` feature, so verifying via this
/// macOS-runnable unit test keeps the contract honest.
#[test]
fn reset_per_channel_fir_clears_fir_state() {
    // Build a 2-slot setup: slot 0 is `Some` (48 k -> 44.1 k),
    // slot 1 is `None` (native rate).  The reset must zero
    // slot 0 and ignore slot 1.
    let mut resamplers: Vec<Option<Streaming>> = vec![Some(Streaming::new(48_000, 44_100)), None];

    // Pump 2 chunks through slot 0 so it has FIR state +
    // pending output.
    let primer: Vec<f32> = (0..2 * 1024)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48_000.0).sin())
        .collect();
    if let Some(r) = &mut resamplers[0] {
        let _ = r.process(&primer).expect("primer test");
        assert!(r.pending() > 0, "primer should have produced output");
    }

    // Reset.
    reset_per_channel_fir(&mut resamplers);

    // Slot 0 should now have zero pending output.
    match &resamplers[0] {
        Some(r) => assert_eq!(r.pending(), 0, "slot 0 still has pending output"),
        None => panic!("slot 0 must remain Some"),
    }
    // Slot 1 must still be None (we don't touch native-rate
    // slots).
    assert!(resamplers[1].is_none());

    // Bit-identity check: feed a probe through the reset
    // resampler AND through a fresh one, compare outputs.
    // If the reset truly zeroed FIR + accumulators, the two
    // are bit-equal.
    let mut fresh = Streaming::new(48_000, 44_100);
    let probe: Vec<f32> = (0..1024)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48_000.0).sin())
        .collect();
    let reset_slot = resamplers[0].as_mut().unwrap();
    let _ = reset_slot.process(&probe).expect("test");
    let _ = fresh.process(&probe).expect("test");
    let out_reset = reset_slot.take_output();
    let out_fresh = fresh.take_output();
    assert_eq!(
        out_reset.len(),
        out_fresh.len(),
        "reset resampler output length differs from fresh",
    );
    for (i, (a, b)) in out_reset.iter().zip(out_fresh.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "reset resampler diverged from fresh at sample {i}: {a} vs {b}",
        );
    }
}

/// Empty-resamplers case: helper must accept `&mut []` without
/// panic.  Defensive -- the run loop sometimes calls this with
/// zero slots (rate-equal source).
#[test]
fn reset_per_channel_fir_empty_slice_is_noop() {
    let mut empty: Vec<Option<Streaming>> = Vec::new();
    reset_per_channel_fir(&mut empty);
    assert!(empty.is_empty());
}

/// `single_pass_demux_and_rms` must not propagate NaN/Inf into
/// the per-slot RMS state or per-slot scratch.  Without the
/// clamp the slot's EMA RMS would stay NaN forever and the
/// audio buffer would inherit non-finite samples.
#[test]
fn single_pass_demux_clamps_non_finite_to_zero() {
    // 4 frames, 2 channels: ch0 = [0.5, NaN, 0.5, +Inf];
    // ch1 = [0.0, 0.0, -Inf, 0.5].  Whitelist both channels.
    let interleaved: Vec<f32> = vec![
        0.5,
        0.0, // frame 0
        f32::NAN,
        0.0, // frame 1
        0.5,
        f32::NEG_INFINITY, // frame 2
        f32::INFINITY,
        0.5, // frame 3
    ];
    let whitelist = [0_u16, 1];
    const FRAMES: usize = 4;
    const STRIDE: usize = FRAMES;
    // Flat per-slot scratch laid out as `slot * stride +
    // frame`.  With `STRIDE == FRAMES` the buffer is
    // exactly `n_slots * frames`-long; production carries
    // `STRIDE == cached_period_frames >= frames` so short
    // reads leave the tail of each per-slot stride
    // untouched.
    let mut slot_scratch_flat = vec![f32::NAN; whitelist.len() * STRIDE];
    let mut sum_sq = vec![0.0_f32; whitelist.len()];

    single_pass_demux_and_rms(
        &interleaved,
        2,
        &whitelist,
        &mut slot_scratch_flat,
        &mut sum_sq,
        FRAMES,
        STRIDE,
    );

    // Each slot's `frames`-sized window should hold finite f32
    // samples (NaN/Inf clamped to 0).  The per-slot stride is
    // `STRIDE` floats wide.
    for slot_idx in 0..whitelist.len() {
        let offset = slot_idx * STRIDE;
        let dst = &slot_scratch_flat[offset..offset + FRAMES];
        for &s in dst {
            assert!(s.is_finite(), "non-finite sample leaked through: {s}");
        }
    }
    // ch0 expected: [0.5, 0.0, 0.5, 0.0] -> sum_sq = 0.5
    // ch1 expected: [0.0, 0.0, 0.0, 0.5] -> sum_sq = 0.25
    // f32 representations of these literals are exact, so a
    // tighter tolerance is justified than the prior 1e-9 (which
    // was f64-flavoured).
    assert!((sum_sq[0] - 0.5).abs() < 1e-6);
    assert!((sum_sq[1] - 0.25).abs() < 1e-6);
}

// MARK: End-to-end integration tests (Mock-only; run on macOS)

use crate::audio_buffer::AudioBuffer;
use std::sync::Arc;

fn arb_test_cfg() -> MicArbitratorConfig {
    // Dwell tightened so the test doesn't wait the default 250ms.
    MicArbitratorConfig {
        hysteresis_db: 3.0,
        dwell: Duration::from_millis(50),
        rms_window: Duration::from_millis(50),
        mic_failover_after: Duration::from_secs(1),
        failover_retry_interval: Duration::from_millis(100),
        // Tests don't exercise the SCHED_FIFO / pin path
        // (impossible without root on most CI hosts); leave
        // both at None so the spawned thread runs at
        // SCHED_OTHER on default placement.
        sched_pin: None,
        sched_priority: None,
        // Anchor not set: tests that don't assert capture-time
        // semantics keep using the no-anchor fallback
        // (consumers stamp CaptureTime::now() at emit time).
        // Tests that DO assert anchor behaviour build a config
        // via `..arb_test_cfg()` with `timing_anchor: Some(...)`.
        timing_anchor: None,
    }
}

/// Two-channel mock: ch0 silent, ch1 sine.  Auto must pick ch1
/// and the buffer's tail RMS reflects it.
#[test]
fn integration_auto_picks_loud_channel_inside_one_mic() {
    let buf = AudioBuffer::new(65_536); // ~1.49 s (next pow2 above 44_100)
    let writer = buf.take_writer();
    let reader = buf.reader_at(0);

    let candidate = MicCandidate {
        id: MicId::from_static("dual"),
        source: CandidateSource::Mock {
            waveforms: vec![
                Waveform::Silence,
                Waveform::Sine {
                    freq_hz: 1000.0,
                    amplitude: 0.5,
                },
            ],
            period_size: 256,
            sample_rate: SampleRate::VALUE,
        },
        channels: vec![0, 1],
    };
    let settings = Arc::new(ArcSwap::from_pointee(MicSettings {
        catalogue: Arc::new(MicCatalogue {
            candidates: vec![candidate],
        }),
        policy: MicPolicy {
            mic: MicSelection::FirstAvailable,
            channel: ChannelSelection::Auto,
        },
    }));
    let arb = MicArbitrator::start(writer, arcswap_store_into_dyn(settings), arb_test_cfg());

    // Let the arbitrator settle on ch1 + accumulate samples.
    std::thread::sleep(Duration::from_millis(300));

    let mut sample = vec![0.0f32; 4096];
    let st = reader.peek_into(&mut sample);
    assert_eq!(
        st,
        crate::audio_buffer::ReadStatus::Ready,
        "buffer should be ready"
    );
    let r = block_rms(&sample);
    // Sine with amplitude 0.5 -> RMS ~= 0.354.  If arbitrator
    // picked silence, RMS would be 0.
    assert!(
        r > 0.2,
        "tail RMS = {r}; expected > 0.2 (loud-channel should win)",
    );

    arb.stop();
}

/// With a [`SharedTimingAnchor`] configured, the arbitrator
/// publishes a fresh anchor after each `Writer::push`.
/// Snapshots taken at intervals must show:
///
///   1. `head_pos` and `captured_at` monotone non-decreasing
///      (`>=`; consecutive snapshots may observe the same
///      value when no push landed between them);
///   2. `sample_rate_hz == SampleRate::VALUE` (44_100);
///   3. final anchor has `head_pos > 0 && captured_at > 0`,
///      proving at least one real push replaced the boot
///      placeholder.
///
/// The `None` path is exercised by every other integration
/// test via `arb_test_cfg()`.
#[test]
fn integration_producer_publishes_monotonic_timing_anchor() {
    use crate::common::time::{BufferTimingAnchor, shared_timing_anchor};
    let buf = AudioBuffer::new(65_536);
    let writer = buf.take_writer();
    let reader = buf.reader_at(0);

    let candidate = MicCandidate {
        id: MicId::from_static("anchor-test"),
        source: CandidateSource::Mock {
            waveforms: vec![Waveform::Sine {
                freq_hz: 440.0,
                amplitude: 0.3,
            }],
            period_size: 256,
            sample_rate: SampleRate::VALUE,
        },
        channels: vec![0],
    };
    let settings = Arc::new(ArcSwap::from_pointee(MicSettings {
        catalogue: Arc::new(MicCatalogue {
            candidates: vec![candidate],
        }),
        policy: MicPolicy {
            mic: MicSelection::FirstAvailable,
            channel: ChannelSelection::Auto,
        },
    }));

    // Stage an anchor cell; the arbitrator updates it after
    // every push.  Initial state is the boot placeholder.
    let anchor_cell = shared_timing_anchor();
    assert_eq!(
        **anchor_cell.load(),
        BufferTimingAnchor::boot_placeholder(),
        "fresh cell must initialise to the boot placeholder",
    );

    let cfg = MicArbitratorConfig {
        timing_anchor: Some(anchor_cell.clone()),
        ..arb_test_cfg()
    };
    let arb = MicArbitrator::start(writer, arcswap_store_into_dyn(settings), cfg);

    // Sample the anchor at intervals so the test observes
    // monotonic progression rather than only the final value.
    // 300 ms is enough for >50 mock-period pushes at 256
    // frames / 44.1 kHz (~5.8 ms/period).
    let mut snapshots: Vec<BufferTimingAnchor> = Vec::with_capacity(8);
    let snapshot_intervals = 8;
    let interval = Duration::from_millis(300 / snapshot_intervals as u64);
    for _ in 0..snapshot_intervals {
        std::thread::sleep(interval);
        snapshots.push(**anchor_cell.load());
    }

    arb.stop();

    // 1: head_pos non-decreasing.
    for w in snapshots.windows(2) {
        assert!(
            w[1].head_pos >= w[0].head_pos,
            "head_pos went backwards: {} -> {}",
            w[0].head_pos,
            w[1].head_pos,
        );
    }
    // 2: captured_at non-decreasing.
    for w in snapshots.windows(2) {
        assert!(
            w[1].captured_at.as_micros() >= w[0].captured_at.as_micros(),
            "captured_at went backwards: {} -> {}",
            w[0].captured_at.as_micros(),
            w[1].captured_at.as_micros(),
        );
    }
    // 3: sample_rate_hz stays at canonical.
    for s in &snapshots {
        assert_eq!(
            s.sample_rate_hz,
            SampleRate::VALUE,
            "anchor must record the canonical buffer rate",
        );
    }
    // 4: at least one push observed (head_pos > 0); production
    // path actually ran rather than the cell still holding the
    // boot placeholder.
    let last = *snapshots.last().expect("at least one snapshot");
    assert!(
        last.head_pos > 0,
        "no push observed; producer never published an anchor; last={last:?}",
    );

    // After stop, the buffer's head matches the last anchor's
    // head (within one push period, since the producer may
    // have pushed once more between our last snapshot and the
    // stop signal).  Read via the reader's status path: a
    // peek that lags by exactly the head delta still returns
    // Ready.
    let mut tail = vec![0.0f32; 4096];
    let _ = reader.peek_into(&mut tail);
    // Final sanity: the last observed anchor's captured_at is
    // greater than the boot placeholder's zero, proving
    // the producer's `CaptureTime::now()` ran.
    assert!(
        last.captured_at.as_micros() > 0,
        "captured_at still at boot placeholder after stop; last={last:?}",
    );
}

/// Two-channel mock where **both** channels carry signal but at
/// different loudness and on different waveform shapes.  The
/// existing `integration_auto_picks_loud_channel_inside_one_mic`
/// test pairs `Silence` (RMS = 0) against a sine, which is a
/// degenerate special case where the noise floor is exactly
/// zero.  This test exercises the realistic shape: ch0 a loud
/// 500 Hz sine (RMS ~= 0.283), ch1 a quieter 2 kHz sine
/// (RMS ~= 0.071) with a different frequency so a frequency-
/// sensitive bug in the demux/RMS path would surface as the
/// arbitrator picking the wrong one.  With ~12 dB of margin,
/// hysteresis (3 dB) is comfortably cleared and the active slot
/// is ch0.
#[test]
fn integration_auto_picks_louder_of_two_active_channels() {
    let buf = AudioBuffer::new(65_536);
    let writer = buf.take_writer();
    let reader = buf.reader_at(0);

    let candidate = MicCandidate {
        id: MicId::from_static("dual-active"),
        source: CandidateSource::Mock {
            waveforms: vec![
                Waveform::Sine {
                    freq_hz: 500.0,
                    amplitude: 0.4,
                },
                Waveform::Sine {
                    freq_hz: 2000.0,
                    amplitude: 0.1,
                },
            ],
            period_size: 256,
            sample_rate: SampleRate::VALUE,
        },
        channels: vec![0, 1],
    };
    let settings = Arc::new(ArcSwap::from_pointee(MicSettings {
        catalogue: Arc::new(MicCatalogue {
            candidates: vec![candidate],
        }),
        policy: MicPolicy::default(),
    }));
    let arb = MicArbitrator::start(writer, arcswap_store_into_dyn(settings), arb_test_cfg());

    std::thread::sleep(Duration::from_millis(300));

    let mut sample = vec![0.0f32; 4096];
    let st = reader.peek_into(&mut sample);
    assert_eq!(st, crate::audio_buffer::ReadStatus::Ready);
    let r = block_rms(&sample);
    // ch0 RMS ~= 0.283; ch1 RMS ~= 0.071.  Picking ch1 would
    // produce ~0.07 -- well below 0.2.  Picking ch0 sits comfortably
    // above.  Tolerance allows for EMA settling at the boundary.
    assert!(
        r > 0.2,
        "tail RMS {r} suggests arbitrator picked the quieter channel",
    );

    arb.stop();
}

/// End-to-end channel auto-switch: two channels' loudness
/// alternates every 0.5 s in opposite phase via
/// [`Waveform::PingPongSine`].  The arbitrator must follow the
/// flips -- picking ch0 during `[0, 0.5 s)`, switching to ch1
/// during `[0.5 s, 1.0 s)`, switching back to ch0 during
/// `[1.0 s, 1.5 s)` -- across the EMA-window x hysteresis-margin
/// x dwell-timer interaction.  If any of those gates the switch
/// indefinitely, the arbitrator latches on one channel and the
/// buffer's tail goes quiet during that channel's low half.
///
/// Timing arithmetic (with `arb_test_cfg`: dwell = 50 ms,
/// rms_window = 50 ms; period 256 frames @ 44.1 k = 5.8 ms):
///   * alpha per period ~= 1 - exp(-5.8 / 50) ~= 0.110.
///   * Post-flip, the new-loud channel's EMA exceeds the old by
///     >1.41x at K=8 periods ~= 46 ms.
///   * Initial pick happens within the first window.  Switch at
///     t ~= 546 ms (first flip) and t ~= 1046 ms (second flip),
///     with the dwell timer comfortably cleared by then.
///
/// Sampling at t = 800 ms catches a "never switches" regression
/// (ch0 would be in its low half then).  Sampling again at
/// t = 1300 ms catches a "switches once, then latches" regression.
#[test]
fn integration_auto_switches_when_loudness_alternates_between_channels() {
    let buf = AudioBuffer::new(131_072);
    let writer = buf.take_writer();
    let mut reader = buf.reader_at(0);

    let half = 22_050; // 0.5 s @ 44.1 k
    let candidate = MicCandidate {
        id: MicId::from_static("ping-pong"),
        source: CandidateSource::Mock {
            waveforms: vec![
                Waveform::PingPongSine {
                    freq_hz: 500.0,
                    high_amp: 0.5,
                    low_amp: 0.02,
                    half_period_samples: half,
                    inverted: false,
                },
                Waveform::PingPongSine {
                    freq_hz: 2000.0,
                    high_amp: 0.5,
                    low_amp: 0.02,
                    half_period_samples: half,
                    inverted: true,
                },
            ],
            period_size: 256,
            sample_rate: SampleRate::VALUE,
        },
        channels: vec![0, 1],
    };
    let settings = Arc::new(ArcSwap::from_pointee(MicSettings {
        catalogue: Arc::new(MicCatalogue {
            candidates: vec![candidate],
        }),
        policy: MicPolicy::default(),
    }));
    let arb = MicArbitrator::start(writer, arcswap_store_into_dyn(settings), arb_test_cfg());

    // After ~800 ms wall clock: mock flipped at 500 ms -> ch1 is
    // now in its high half; arbitrator should have switched to
    // ch1 at ~546 ms.  Tail of 2048 samples (~46 ms) lies entirely
    // after the switch.
    std::thread::sleep(Duration::from_millis(800));
    reader.seek_latest(2048);
    let mut sample = vec![0.0f32; 2048];
    let st = reader.peek_into(&mut sample);
    assert_eq!(st, crate::audio_buffer::ReadStatus::Ready);
    let r1 = block_rms(&sample);
    assert!(
        r1 > 0.2,
        "first flip: expected arbitrator to follow ch0->ch1 switch; \
             tail RMS {r1} suggests it latched on ch0 (now in its low half)",
    );

    // After ~1300 ms total: mock flipped again at 1000 ms -> ch0
    // is loud again; arbitrator should have switched ch1->ch0 at
    // ~1046 ms.
    std::thread::sleep(Duration::from_millis(500));
    reader.seek_latest(2048);
    let st = reader.peek_into(&mut sample);
    assert_eq!(st, crate::audio_buffer::ReadStatus::Ready);
    let r2 = block_rms(&sample);
    assert!(
        r2 > 0.2,
        "second flip: expected arbitrator to follow ch1->ch0 switch-back; \
             tail RMS {r2} suggests it latched on ch1 (now in its low half)",
    );

    arb.stop();
}

/// End-to-end: phase 1 runs on a candidate where ch0 is the
/// loud channel; we then hot-swap [`MicSettings`] to point at a
/// **different** candidate id whose ch1 is loud, with a Fixed
/// policy forcing the swap.  The arbitrator must tear down
/// the old source, open the new one, and converge on the new
/// loud channel -- recent buffer tail RMS reflects the active
/// (still-loud) sine.
///
/// Note: this exercises the **mic-swap + post-swap initial
/// channel selection** path, not the in-mic channel-switch
/// dwell/hysteresis logic (which the unit-level `pick_slot_*`
/// tests cover precisely).  End-to-end channel-switch testing
/// would need a Mock that can mutate its waveform amplitude
/// at runtime -- out of scope here.
#[test]
fn integration_post_mic_swap_picks_loud_channel_of_new_mic() {
    let buf = AudioBuffer::new(131_072);
    let writer = buf.take_writer();
    let mut reader = buf.reader_at(0);

    let make_candidate = |amp_ch0: f32, amp_ch1: f32| MicCandidate {
        id: MicId::from_static("dual"),
        source: CandidateSource::Mock {
            waveforms: vec![
                Waveform::Sine {
                    freq_hz: 500.0,
                    amplitude: amp_ch0,
                },
                Waveform::Sine {
                    freq_hz: 2000.0,
                    amplitude: amp_ch1,
                },
            ],
            period_size: 256,
            sample_rate: SampleRate::VALUE,
        },
        channels: vec![0, 1],
    };

    let settings = Arc::new(ArcSwap::from_pointee(MicSettings {
        catalogue: Arc::new(MicCatalogue {
            candidates: vec![make_candidate(0.5, 0.05)], // ch0 loud
        }),
        policy: MicPolicy::default(),
    }));
    let arb = MicArbitrator::start(
        writer,
        arcswap_store_into_dyn(settings.clone()),
        arb_test_cfg(),
    );

    // Phase 1: ch0 is loud on the only candidate.  Wait for the
    // arbitrator to open it and converge.
    std::thread::sleep(Duration::from_millis(200));

    // Phase 2: install a new candidate with a different id
    // (`dual-flipped`) whose ch1 is loud, and pin policy at
    // that new id.  This forces the arbitrator to tear down
    // the old source and re-open against the new candidate --
    // a *mic switch*, not a channel switch.
    settings.store(Arc::new(MicSettings {
        catalogue: Arc::new(MicCatalogue {
            candidates: vec![
                make_candidate(0.5, 0.05),
                MicCandidate {
                    id: MicId::from_static("dual-flipped"),
                    source: CandidateSource::Mock {
                        waveforms: vec![
                            Waveform::Sine {
                                freq_hz: 500.0,
                                amplitude: 0.05,
                            },
                            Waveform::Sine {
                                freq_hz: 2000.0,
                                amplitude: 0.5,
                            },
                        ],
                        period_size: 256,
                        sample_rate: SampleRate::VALUE,
                    },
                    channels: vec![0, 1],
                },
            ],
        }),
        policy: MicPolicy {
            mic: MicSelection::Fixed {
                id: MicId::from_static("dual-flipped"),
            },
            channel: ChannelSelection::Auto,
        },
    }));

    // Wait for the arbitrator to re-open + RMS to settle on
    // ch1, plus dwell.
    std::thread::sleep(Duration::from_millis(400));

    // Recent buffer tail should be dominated by the loud sine
    // (regardless of which channel was loud -- both phases had
    // amplitude 0.5).
    reader.seek_latest(2048);
    let mut sample = vec![0.0f32; 2048];
    let st = reader.peek_into(&mut sample);
    assert_eq!(st, crate::audio_buffer::ReadStatus::Ready);
    let r = block_rms(&sample);
    assert!(r > 0.2, "post-switch RMS = {r}; expected > 0.2");

    arb.stop();
}

/// Whitelist-of-1: even with multiple device channels, only the
/// whitelisted one ever participates.  Fixed policy + a single-
/// entry whitelist proves the demux is honouring the whitelist
/// (not all device channels).
#[test]
fn integration_whitelist_filters_to_subset() {
    let buf = AudioBuffer::new(65_536);
    let writer = buf.take_writer();
    let reader = buf.reader_at(0);

    let candidate = MicCandidate {
        id: MicId::from_static("triple"),
        // Three device channels: silent, loud, silent.
        source: CandidateSource::Mock {
            waveforms: vec![
                Waveform::Silence,
                Waveform::Sine {
                    freq_hz: 1000.0,
                    amplitude: 0.5,
                },
                Waveform::Silence,
            ],
            period_size: 256,
            sample_rate: SampleRate::VALUE,
        },
        // Whitelist EXCLUDES the loud channel; only ch0 + ch2
        // (both silent) participate.
        channels: vec![0, 2],
    };
    let settings = Arc::new(ArcSwap::from_pointee(MicSettings {
        catalogue: Arc::new(MicCatalogue {
            candidates: vec![candidate],
        }),
        policy: MicPolicy::default(),
    }));
    let arb = MicArbitrator::start(writer, arcswap_store_into_dyn(settings), arb_test_cfg());

    std::thread::sleep(Duration::from_millis(300));

    let mut sample = vec![0.0f32; 4096];
    let st = reader.peek_into(&mut sample);
    assert_eq!(st, crate::audio_buffer::ReadStatus::Ready);
    let r = block_rms(&sample);
    assert!(
        r < 0.05,
        "whitelist excluded the loud channel; tail RMS = {r}; expected < 0.05",
    );

    arb.stop();
}

/// `MicSelection::Fixed` opens the named candidate even when an
/// earlier-listed candidate would also be openable.  No
/// FirstAvailable preference for "earlier".
#[test]
fn integration_fixed_selection_opens_named_mic() {
    let buf = AudioBuffer::new(65_536);
    let writer = buf.take_writer();
    let reader = buf.reader_at(0);

    let cand_quiet = MicCandidate {
        id: MicId::from_static("quiet"),
        source: CandidateSource::Mock {
            waveforms: vec![Waveform::Silence],
            period_size: 256,
            sample_rate: SampleRate::VALUE,
        },
        channels: vec![0],
    };
    let cand_loud = MicCandidate {
        id: MicId::from_static("loud"),
        source: CandidateSource::Mock {
            waveforms: vec![Waveform::Sine {
                freq_hz: 1000.0,
                amplitude: 0.5,
            }],
            period_size: 256,
            sample_rate: SampleRate::VALUE,
        },
        channels: vec![0],
    };
    let settings = Arc::new(ArcSwap::from_pointee(MicSettings {
        catalogue: Arc::new(MicCatalogue {
            candidates: vec![cand_quiet, cand_loud],
        }),
        policy: MicPolicy {
            mic: MicSelection::Fixed {
                id: MicId::from_static("loud"),
            },
            channel: ChannelSelection::Auto,
        },
    }));
    let arb = MicArbitrator::start(writer, arcswap_store_into_dyn(settings), arb_test_cfg());

    std::thread::sleep(Duration::from_millis(300));

    let mut sample = vec![0.0f32; 4096];
    let st = reader.peek_into(&mut sample);
    assert_eq!(st, crate::audio_buffer::ReadStatus::Ready);
    let r = block_rms(&sample);
    // Should be the loud sine's RMS, not silence.
    assert!(r > 0.2, "Fixed selection didn't pick named loud mic; r={r}");

    arb.stop();
}

/// Stop is honoured promptly even when the arbitrator is in a
/// long mock sleep.  End-to-end teardown should complete within
/// a few periods.
#[test]
fn integration_stop_is_prompt() {
    let buf = AudioBuffer::new(65_536);
    let writer = buf.take_writer();
    let cand = MicCandidate {
        id: MicId::from_static("paced"),
        source: CandidateSource::Mock {
            waveforms: vec![Waveform::Silence],
            // Long-ish period (~50 ms) so a stop signal during
            // mock pacing is meaningful to test.
            period_size: 2048,
            sample_rate: SampleRate::VALUE,
        },
        channels: vec![0],
    };
    let settings = Arc::new(ArcSwap::from_pointee(MicSettings {
        catalogue: Arc::new(MicCatalogue {
            candidates: vec![cand],
        }),
        policy: MicPolicy::default(),
    }));
    let arb = MicArbitrator::start(writer, arcswap_store_into_dyn(settings), arb_test_cfg());

    // Let it run a couple of periods to enter the read+pace
    // hot loop.
    std::thread::sleep(Duration::from_millis(120));

    let t = Instant::now();
    arb.stop();
    let elapsed = t.elapsed();
    // 2 ms slice + slop.  Allow up to 50 ms (one period
    // duration) on a heavily-loaded CI box.
    assert!(
        elapsed < Duration::from_millis(50),
        "stop took too long: {elapsed:?}",
    );
}

/// [`MicArbitrator::signal_stop`] returns immediately without joining the
/// thread, and a subsequent `stop()` joins promptly because the
/// run loop has already observed the cancel between the two
/// calls.  The daemon's shutdown sequence relies on this: it
/// signals the producer first so consumers drain into a quiet
/// pipeline, then joins the producer last.  Calling `stop()`
/// without prior `signal_stop` is unaffected (covered by
/// `integration_stop_is_prompt`).
#[test]
fn integration_signal_stop_returns_without_joining() {
    let buf = AudioBuffer::new(65_536);
    let writer = buf.take_writer();
    let cand = MicCandidate {
        id: MicId::from_static("paced"),
        source: CandidateSource::Mock {
            waveforms: vec![Waveform::Silence],
            period_size: 2048,
            sample_rate: SampleRate::VALUE,
        },
        channels: vec![0],
    };
    let settings = Arc::new(ArcSwap::from_pointee(MicSettings {
        catalogue: Arc::new(MicCatalogue {
            candidates: vec![cand],
        }),
        policy: MicPolicy::default(),
    }));
    let arb = MicArbitrator::start(writer, arcswap_store_into_dyn(settings), arb_test_cfg());

    // Let it enter the read+pace hot loop so signal_stop hits
    // a thread that is actively producing audio.
    std::thread::sleep(Duration::from_millis(120));

    let t_signal = Instant::now();
    arb.signal_stop();
    let signal_elapsed = t_signal.elapsed();
    // Signal alone is a flag store + thread::unpark; both are
    // sub-microsecond.  The bound is loose to absorb scheduler
    // jitter on CI but tight enough that a join slipping into
    // signal_stop would be caught -- `arb.stop()` typically
    // takes 5-50 ms, far beyond this 5 ms ceiling.
    assert!(
        signal_elapsed < Duration::from_millis(5),
        "signal_stop blocked for {signal_elapsed:?}; must not join",
    );

    // Idempotent: calling again is a no-op (just stores `true`
    // again into an already-`true` flag and unparks an already-
    // exiting thread).
    arb.signal_stop();

    // The thread has been winding down since the first signal --
    // by now the run loop has observed `stop` and exited, so the
    // join inside `stop(self)` is essentially a no-wait.
    let t_join = Instant::now();
    arb.stop();
    let join_elapsed = t_join.elapsed();
    assert!(
        join_elapsed < Duration::from_millis(50),
        "post-signal stop took too long: {join_elapsed:?}",
    );
}

/// Stop is prompt even when the run loop is in the no-source
/// `park_timeout` branch (empty catalogue -> `state.active.is_none()`
/// -> park for `failover_retry_interval`).  Regression test: the
/// previous `thread::sleep` form held teardown back by up to one
/// `failover_retry_interval` (1 s default).
#[test]
fn integration_stop_is_prompt_with_no_source_open() {
    let buf = AudioBuffer::new(65_536);
    let writer = buf.take_writer();
    // Empty catalogue -> arbitrator never opens a source -> run loop
    // sits in the no-source branch.
    let settings = Arc::new(ArcSwap::from_pointee(MicSettings {
        catalogue: Arc::new(MicCatalogue::default()),
        policy: MicPolicy::default(),
    }));
    // Long failover interval so a regression (sleep instead of
    // park) would clearly exceed the threshold.
    let cfg = MicArbitratorConfig {
        failover_retry_interval: Duration::from_secs(1),
        ..arb_test_cfg()
    };
    let arb = MicArbitrator::start(writer, arcswap_store_into_dyn(settings), cfg);

    // Give the loop time to enter the park_timeout branch.
    std::thread::sleep(Duration::from_millis(50));

    let t = Instant::now();
    arb.stop();
    let elapsed = t.elapsed();
    // Without the unpark, this would be ~1 s (failover_retry_interval).
    // With it: store(stop) -> unpark -> loop top sees stop -> exit.
    assert!(
        elapsed < Duration::from_millis(50),
        "stop took too long with no source open: {elapsed:?} \
             (expected unpark to wake park_timeout immediately)",
    );
}

/// FirstAvailable failover: the first candidate is broken
/// (Alsa-not-compiled-in on macOS); the second is a working
/// mock.  The arbitrator falls through to the second.
#[test]
fn integration_first_available_falls_through_broken_candidate() {
    let buf = AudioBuffer::new(65_536);
    let writer = buf.take_writer();
    let reader = buf.reader_at(0);

    let broken = MicCandidate {
        id: MicId::from_static("broken-alsa"),
        source: CandidateSource::Alsa {
            hw_spec: "hw:99,99".into(),
            period_size: 1024,
            buffer_size: 4096,
        },
        channels: vec![0],
    };
    let working = MicCandidate {
        id: MicId::from_static("working-mock"),
        source: CandidateSource::Mock {
            waveforms: vec![Waveform::Sine {
                freq_hz: 1000.0,
                amplitude: 0.5,
            }],
            period_size: 256,
            sample_rate: SampleRate::VALUE,
        },
        channels: vec![0],
    };
    let settings = Arc::new(ArcSwap::from_pointee(MicSettings {
        catalogue: Arc::new(MicCatalogue {
            candidates: vec![broken, working],
        }),
        policy: MicPolicy::default(),
    }));
    let arb = MicArbitrator::start(writer, arcswap_store_into_dyn(settings), arb_test_cfg());

    std::thread::sleep(Duration::from_millis(300));

    let mut sample = vec![0.0f32; 4096];
    let st = reader.peek_into(&mut sample);
    assert_eq!(st, crate::audio_buffer::ReadStatus::Ready);
    let r = block_rms(&sample);
    assert!(r > 0.2, "expected fallback to working-mock; tail RMS = {r}",);

    arb.stop();
}

/// [`MicArbitratorConfig::validate`] accepts the hardcoded
/// defaults and rejects each independently-bad field.
#[test]
fn arbitrator_config_validate_rejects_bad_fields() {
    // Default is by construction valid.
    MicArbitratorConfig::default()
        .validate()
        .expect("default must validate");

    // Negative hysteresis_db.
    let bad = MicArbitratorConfig {
        hysteresis_db: -1.0,
        ..MicArbitratorConfig::default()
    };
    let err = bad.validate().expect_err("negative hysteresis must reject");
    assert!(err.contains("hysteresis_db"), "{err}");

    // NaN hysteresis_db.
    let bad = MicArbitratorConfig {
        hysteresis_db: f32::NAN,
        ..MicArbitratorConfig::default()
    };
    bad.validate().expect_err("NaN hysteresis must reject");

    // Zero rms_window.
    let bad = MicArbitratorConfig {
        rms_window: Duration::ZERO,
        ..MicArbitratorConfig::default()
    };
    let err = bad.validate().expect_err("zero rms_window must reject");
    assert!(err.contains("rms_window"), "{err}");

    // Zero mic_failover_after.
    let bad = MicArbitratorConfig {
        mic_failover_after: Duration::ZERO,
        ..MicArbitratorConfig::default()
    };
    let err = bad.validate().expect_err("zero failover must reject");
    assert!(err.contains("mic_failover_after"), "{err}");

    // Zero failover_retry_interval.
    let bad = MicArbitratorConfig {
        failover_retry_interval: Duration::ZERO,
        ..MicArbitratorConfig::default()
    };
    let err = bad.validate().expect_err("zero retry_interval must reject");
    assert!(err.contains("failover_retry_interval"), "{err}");
}

/// `MicArbitrator::start` self-validates the config and panics
/// on rejection so a fresh call site cannot bypass the gate by
/// forgetting an upstream `validate()`.  Pre-fix the daemon's
/// call site held the only validate() guard; today the gate is
/// intrinsic.
#[test]
#[should_panic(expected = "invalid MicArbitratorConfig")]
fn start_self_validates_invalid_config_panics() {
    let buf = AudioBuffer::new(65_536);
    let writer = buf.take_writer();
    let candidate = MicCandidate {
        id: MicId::from_static("dual"),
        source: CandidateSource::Mock {
            waveforms: vec![Waveform::Silence],
            period_size: 256,
            sample_rate: SampleRate::VALUE,
        },
        channels: vec![0],
    };
    let settings = Arc::new(ArcSwap::from_pointee(MicSettings {
        catalogue: Arc::new(MicCatalogue {
            candidates: vec![candidate],
        }),
        policy: MicPolicy {
            mic: MicSelection::FirstAvailable,
            channel: ChannelSelection::Auto,
        },
    }));
    // Negative hysteresis_db is rejected by
    // `MicArbitratorConfig::validate`; pre-fix this would have
    // spawned a thread that immediately misbehaved.
    let bad = MicArbitratorConfig {
        hysteresis_db: -1.0,
        ..arb_test_cfg()
    };
    let _ = MicArbitrator::start(writer, arcswap_store_into_dyn(settings), bad);
}

/// The launch TOML; the same defaults apply.
#[test]
fn toml_defaults_fill_missing_alsa_fields() {
    let toml_text = r#"
            candidates = [
              { id = "a", source = { kind = "alsa", hw_spec = "hw:1,0" }, channels = [0] },
            ]
        "#;
    let s: MicCatalogue = toml::from_str(toml_text).expect("deserialize");
    match &s.candidates[0].source {
        CandidateSource::Alsa {
            period_size,
            buffer_size,
            ..
        } => {
            assert_eq!(*period_size, 1024_usize);
            assert_eq!(*buffer_size, 4096_usize);
        }
        _ => panic!("expected Alsa source"),
    }
}
