//! Mic-arbitrator data model: candidate descriptions,
//! catalogue, policy + selection enums, plus the
//! per-candidate / per-policy validation predicates.
//!
//! The validation predicates ([`MicCandidate::validate`],
//! [`MicCatalogue::validate`],
//! [`MicPolicy::validate_against`]) live alongside the types
//! they operate on so a future change to a type's invariants
//! and the validator that enforces them stays in one file.

use crate::audio_io::mock::Waveform;
use crate::common::ids::MicId;
use std::sync::Arc;

// MARK: Candidate description

/// Developer-curated description of a microphone the daemon
/// is willing to use, plus which of its channels participate
/// in arbitration.
///
/// A "candidate" is a willingness, not an open device.  The
/// arbitrator opens at most one candidate at a time (the
/// active mic); the rest are warm-standby.  Adding a
/// candidate to the whitelist costs nothing at runtime: it
/// just makes the mic available to
/// [`MicSelection::Fixed`] and to the
/// [`MicSelection::FirstAvailable`] failover walk.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MicCandidate {
    /// Stable name distinct from device path, e.g.
    /// `"front-array"`.  Surfaced on
    /// `/api/v1/mic/candidates`; the live policy refers to
    /// candidates by this id (NOT by `hw_spec`, since
    /// `hw_spec` can churn across reboots on USB hot-plug).
    pub id: MicId,
    /// How to open the device.
    pub source: CandidateSource,
    /// Channel-index whitelist.  Indices are into the
    /// device's interleaved frame; only these channels
    /// participate in RMS arbitration.
    ///
    /// Validated statically (non-empty, no duplicates) by
    /// [`MicCandidate::validate`]; intersected with the
    /// device's actual channel count at open time so an
    /// out-of-range entry can never cause an OOB read on the
    /// interleaved buffer.
    pub channels: Vec<u16>,
}

/// How to open the candidate's audio source.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CandidateSource {
    /// ALSA hardware.  Compiles only on Linux with the
    /// `alsa-real` feature; on macOS or test builds without
    /// the feature, attempting to open this candidate fails
    /// at open time and the arbitrator falls through to the
    /// next candidate (per [`MicSelection::FirstAvailable`])
    /// or stays inert (per [`MicSelection::Fixed`]).
    Alsa {
        /// For example `"hw:1,0"` or `"plughw:1,0"`.  The `plug`
        /// variant adds kernel-side conversion latency but is
        /// more permissive on quirky USB devices.
        hw_spec: String,
        /// Frames per `readi` call.  1024 (~23 ms at 44.1
        /// kHz) is the canonical sweet spot; configurable
        /// for hardware that snaps to specific values
        /// (480 / 960 / 1920 etc.).
        #[serde(default = "default_period_size")]
        period_size: usize,
        /// Capture buffer size in frames.  4x `period_size`
        /// is the canonical ALSA default and keeps latency
        /// bounded.
        #[serde(default = "default_buffer_size")]
        buffer_size: usize,
    },
    /// Synthetic source.  Cross-platform; used by tests, the
    /// daemon's `--mock-audio` path, and macOS dev iteration.
    Mock {
        /// One waveform per device-channel.  The candidate's
        /// `channels` whitelist indexes into this Vec;
        /// entries outside the whitelist are still
        /// synthesized (to keep the "device" mental model
        /// intact and to preserve LCG determinism) but their
        /// output is dropped by the arbitrator before
        /// reaching the audio buffer.
        waveforms: Vec<Waveform>,
        /// Synthesis chunk in frames.  The mock source paces
        /// in real time: `read_*` sleeps until the next
        /// chunk is due before returning.
        #[serde(default = "default_mock_period")]
        period_size: usize,
        /// Synthesis sample rate.  44_100 is canonical;
        /// non-canonical values exercise the arbitrator's
        /// per-channel resampler.
        #[serde(default = "default_mock_rate")]
        sample_rate: u32,
    },
}

fn default_period_size() -> usize {
    1024
}
fn default_buffer_size() -> usize {
    1024 * 4
}
fn default_mock_period() -> usize {
    512
}
fn default_mock_rate() -> u32 {
    crate::common::dims::SampleRate::VALUE
}

// MARK: Live policy

/// Launch-time deployment manifest: which mics are available
/// and, for each, which channels are usable.  **Immutable
/// for the lifetime of the daemon process**: operators edit
/// it via the launch-config TOML and restart the daemon to
/// apply changes.
///
/// The catalogue is the developer's curated source of truth.
/// The user-preference policy ([`MicPolicy`]) references
/// catalogue entries by id; the arbitrator drives RMS
/// arbitration across the per-candidate `channels` whitelist
/// defined here.
///
/// Hierarchy expressed by this type:
///
/// ```text
///   MicCatalogue
///     +-- candidates[0]: MicCandidate { id, source, channels: [0, 1, 2, 3] }
///     +-- candidates[1]: MicCandidate { id, source, channels: [0] }
///     \-- ...
/// ```
///
/// Each candidate's `channels` is the developer-curated subset of
/// the underlying device's actually-available channels (the device
/// may expose more, but only those listed here participate in
/// arbitration).
#[derive(Clone, Debug, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct MicCatalogue {
    pub candidates: Vec<MicCandidate>,
}

impl MicCatalogue {
    /// Validate every candidate's static well-formedness
    /// AND the catalogue-level invariant that ids are
    /// unique.  Reported to the operator at launch-config
    /// load time.
    ///
    /// Per-candidate errors short-circuit on the first
    /// failure and return the offending id.  Duplicate-id
    /// errors return one of the two colliding ids (the
    /// second one encountered).
    pub fn validate(&self) -> Result<(), (MicId, CandidateError)> {
        // Per-candidate static validation first; surfaces the
        // most actionable error per offending entry.
        for c in &self.candidates {
            if let Err(e) = c.validate() {
                return Err((c.id.clone(), e));
            }
        }
        // Inter-candidate id uniqueness via `HashSet`: reads
        // as "first time we've seen this id" without indexing
        // arithmetic.  Catalogues are tiny (operator-curated,
        // <=10 entries typical), so the absolute cost
        // difference vs the nested-loop O(N^2) form is
        // irrelevant; this is a readability call.
        let mut seen = std::collections::HashSet::with_capacity(self.candidates.len());
        for c in &self.candidates {
            if !seen.insert(&c.id) {
                return Err((c.id.clone(), CandidateError::DuplicateMicId(c.id.clone())));
            }
        }
        Ok(())
    }

    /// Look up a candidate by id.  O(N) scan; catalogues are
    /// small (operator-curated, typically <=10 entries) so a
    /// linear search is the natural representation.
    pub fn find(&self, id: &MicId) -> Option<&MicCandidate> {
        self.candidates.iter().find(|c| &c.id == id)
    }
}

/// Runtime bundle of (catalogue + live policy) read by the
/// arbitrator's hot loop.  Carried via
/// `Arc<arc_swap::ArcSwap<MicSettings>>`.
///
/// The catalogue is wrapped in an `Arc` so the watcher's
/// callback can rebuild [`MicSettings`] on a policy change
/// WITHOUT cloning the catalogue's `Vec<MicCandidate>`: only
/// the policy field changes, and the same catalogue
/// [`Arc`] is reused across snapshots.
///
/// [`MicSettings`] is the runtime data shape; the
/// persistence layer stores catalogue and policy in separate
/// TOML files (see [`crate::config`]'s
/// [`crate::config::LaunchConfig`] for the launch-immutable
/// side).
#[derive(Clone, Debug, PartialEq)]
pub struct MicSettings {
    pub catalogue: Arc<MicCatalogue>,
    pub policy: MicPolicy,
}

impl Default for MicSettings {
    fn default() -> Self {
        Self {
            catalogue: Arc::new(MicCatalogue::default()),
            policy: MicPolicy::default(),
        }
    }
}

/// Two-level mic policy: which mic, and within that mic, which
/// channel.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MicPolicy {
    pub mic: MicSelection,
    pub channel: ChannelSelection,
}

impl Default for MicPolicy {
    fn default() -> Self {
        Self {
            mic: MicSelection::FirstAvailable,
            channel: ChannelSelection::Auto,
        }
    }
}

/// Mic-level selection.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MicSelection {
    /// Walk `candidates` in declaration order; the first
    /// that opens successfully wins.  On hot-unplug of the
    /// active mic, walk again from the top.
    FirstAvailable,
    /// Always pick the named mic.  If it disappears, the
    /// arbitrator retries opening it forever (with
    /// rate-limited logging) and does **not** fail over to
    /// another candidate.
    Fixed { id: MicId },
}

/// Channel-level selection within the active mic.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ChannelSelection {
    /// RMS-arbitrated across the active mic's channel
    /// whitelist with hysteresis + dwell.  Default; matches
    /// the user-facing "loudest channel wins" semantic.
    Auto,
    /// Always pick the named channel index.  If the index
    /// isn't in the active mic's whitelist (e.g.
    /// operator-set policy refers to a channel the
    /// candidate's whitelist doesn't include), the
    /// arbitrator falls back to [`Self::Auto`] for THIS mic
    /// rather than emitting silence.
    Fixed { channel: u16 },
}

// MARK: Validation

/// Errors produced by [`MicCandidate::validate`]
/// (per-candidate well-formedness) and
/// [`MicCatalogue::validate`] (catalogue-level invariants
/// like id uniqueness).  Open-time errors that depend on the
/// device's actual channel count surface separately in the
/// arbitrator's source-open path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CandidateError {
    /// `channels` is empty: nothing for arbitration to
    /// consider.
    EmptyChannels,
    /// Same channel index appears twice within one
    /// candidate.  Distinct indices are required for the
    /// arbitrator's per-slot RMS state.
    DuplicateChannel(u16),
    /// Two catalogue entries share a [`MicId`].  Only the
    /// first is reachable via `Fixed { id }` resolution (and
    /// via the `FirstAvailable` walk's id-based stickiness
    /// check); the second is silently dead.  Almost always
    /// an operator copy-paste typo, caught at catalogue load.
    DuplicateMicId(MicId),
    /// Channel index is above the static sanity cap
    /// [`MAX_CHANNEL_INDEX`].  Defends downstream `u16 ->
    /// u32 + 1` arithmetic in the ALSA open path against
    /// silent overflow, and flags obvious typos (channel 999
    /// on a small mic array is almost certainly a mistake).
    ChannelIndexTooLarge { channel: u16, cap: u16 },
    /// `Alsa { hw_spec }` is empty.
    EmptyHwSpec,
    /// `period_size == 0` would produce zero-frame reads.
    InvalidPeriodSize(usize),
    /// `buffer_size < period_size`; ALSA would refuse this
    /// anyway.
    InvalidBufferSize { period: usize, buffer: usize },
    /// `Mock { waveforms }` is empty.
    EmptyMockWaveforms,
    /// `Mock { waveforms }` doesn't have enough entries to
    /// cover the channel whitelist's max index.
    MockWhitelistOutOfRange {
        whitelist_max: u16,
        waveform_count: u16,
    },
    /// `Mock { sample_rate == 0 }`.
    InvalidMockSampleRate(u32),
    /// `period_size` exceeds [`MAX_PERIOD_FRAMES`].  Bounded by
    /// the [`crate::audio_buffer::Writer::push`] safety-margin
    /// invariant.
    PeriodSizeTooLarge { period: usize, cap: usize },
    /// `buffer_size` exceeds [`MAX_BUFFER_FRAMES`] or
    /// `16 * period_size`.  Real ALSA drivers use a handful
    /// of periods deep; an oversized buffer is almost always
    /// a samples-vs-ms typo.
    BufferSizeTooLarge {
        buffer: usize,
        period: usize,
        absolute_cap: usize,
        period_multiplier_cap: usize,
    },
    /// Whitelist length OR `Mock { waveforms }.len()` exceeds
    /// [`MAX_CHANNELS`].
    TooManyChannels { count: usize, cap: usize },
    /// Source `sample_rate` outside
    /// [`MIN_SAMPLE_RATE`]..=[`MAX_SAMPLE_RATE`] -- the
    /// resampler ratio either wedges (very small from_sr) or
    /// thrashes (multi-MB sinc tables).
    SampleRateOutOfRange { rate: u32, min: u32, max: u32 },
    /// `period_size * channels` overflows `usize`.  Defence
    /// in depth: per-field caps make it unreachable
    /// (8192 * 8 = 65_536), but `checked_mul` survives a
    /// future cap change.
    ScratchSizeOverflow { period: usize, channels: usize },
}

/// Static sanity cap on individual whitelist channel
/// indices.
///
/// Real audio hardware rarely exposes more than a handful of
/// channels (typical mic arrays 2-8; pro audio interfaces
/// <=64).  Capping at 1023 is comfortably above any realistic
/// device while keeping `whitelist_max + 1` well within
/// [`u16::MAX`] so the `u16 -> u32 -> u16` round-trip in
/// `AlsaSource::open`'s channel-negotiation path cannot
/// silently wrap.  Plain code rather than an intra-doc link
/// because the ALSA source is feature-gated and the link
/// wouldn't resolve in default-feature doc builds.
///
/// Enforced at the static-validation entry points
/// ([`MicCandidate::validate`] / [`MicCatalogue::validate`]);
/// the ALSA source path additionally `debug_assert!`s the
/// negotiated count fits in `u16` to catch any future ALSA
/// backend that returns an implausible value.
pub const MAX_CHANNEL_INDEX: u16 = 1023;

/// Static cap on the number of channels in a candidate's
/// whitelist OR the number of waveforms in a `Mock` candidate.
/// Real mic arrays expose 1-8 channels; a catalogue asking for
/// more is almost certainly an operator typo, and the per-slot
/// arbitrator state + per-channel resamplers (~140 KB sinc
/// table each) make even modest excesses expensive.
///
/// Enforced at the static-validation entry points
/// ([`MicCandidate::validate`] /
/// [`MicCatalogue::validate`]).
pub const MAX_CHANNELS: usize = 8;

/// Static cap on `period_size` (frames per
/// `read_interleaved` call) for both ALSA and Mock
/// sources.
///
/// Bounded by the
/// [`crate::audio_buffer::Writer::push`] safety-margin
/// invariant: a single push must be `<=
/// audio_buffer::AudioBuffer::max_push_len() = capacity /
/// 4`.  At the daemon's canonical capacity (262 144) the
/// safety margin is 65 536 samples, so 8 192 frames keeps
/// us 8x below the production bound.  The cap is
/// expressed as a static constant rather than read off a
/// live `AudioBuffer` because validation runs at config
/// load time, before the buffer exists.
///
/// 8 192 frames at 44.1 kHz is ~186 ms per period -- well
/// beyond any real-time audio configuration (typical
/// production: 1024 frames = ~23 ms).
pub const MAX_PERIOD_FRAMES: usize = 8192;

/// Static absolute cap on `buffer_size` (frames in the
/// ALSA capture buffer).  Real ALSA drivers operate with
/// a few periods of buffer depth; the second
/// `period_multiplier_cap` check
/// (16x `period_size`) catches the more common
/// "operator confused samples with milliseconds" typo.
/// Both bounds are enforced; the absolute cap defends
/// against an oversized period that individually passed
/// the per-field cap.
pub const MAX_BUFFER_FRAMES: usize = MAX_PERIOD_FRAMES * 16;

/// Lower bound on `Mock { sample_rate }`.  8 kHz is the
/// telephony-grade voice rate; below this the resampler
/// ratio (`44 100 / from_sr`) explodes and the sinc
/// kernel produces audible aliasing.
pub const MIN_SAMPLE_RATE: u32 = 8_000;

/// Upper bound on `Mock { sample_rate }`.  192 kHz is the
/// canonical studio "high-res" rate; above that the resampler
/// ratio shrinks and per-period sample counts balloon, and
/// such rates are studio-equipment-only.
pub const MAX_SAMPLE_RATE: u32 = 192_000;

impl std::fmt::Display for CandidateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CandidateError::EmptyChannels => f.write_str("channels whitelist is empty"),
            CandidateError::DuplicateChannel(c) => {
                write!(f, "duplicate channel {c} in whitelist")
            }
            CandidateError::DuplicateMicId(id) => {
                write!(f, "duplicate mic id '{id}' in catalogue")
            }
            CandidateError::ChannelIndexTooLarge { channel, cap } => write!(
                f,
                "channel index {channel} exceeds the static cap of {cap} (likely a config typo)",
            ),
            CandidateError::EmptyHwSpec => f.write_str("alsa hw_spec is empty"),
            CandidateError::InvalidPeriodSize(n) => {
                write!(f, "period_size must be > 0, got {n}")
            }
            CandidateError::InvalidBufferSize { period, buffer } => {
                write!(f, "buffer_size {buffer} must be >= period_size {period}",)
            }
            CandidateError::EmptyMockWaveforms => f.write_str("mock waveforms is empty"),
            CandidateError::MockWhitelistOutOfRange {
                whitelist_max,
                waveform_count,
            } => write!(
                f,
                "whitelist asks for channel {whitelist_max} but mock has only {waveform_count} waveforms",
            ),
            CandidateError::InvalidMockSampleRate(r) => {
                write!(f, "mock sample_rate must be > 0, got {r}")
            }
            CandidateError::PeriodSizeTooLarge { period, cap } => write!(
                f,
                "period_size {period} exceeds cap {cap} frames; reduce period_size \
                 in the mic catalogue",
            ),
            CandidateError::BufferSizeTooLarge {
                buffer,
                period,
                absolute_cap,
                period_multiplier_cap,
            } => write!(
                f,
                "buffer_size {buffer} exceeds cap (absolute {absolute_cap} or \
                 {period_multiplier_cap}x period_size = {}); shrink buffer_size or \
                 check for a samples-vs-ms typo",
                period.saturating_mul(*period_multiplier_cap),
            ),
            CandidateError::TooManyChannels { count, cap } => {
                write!(f, "channels/waveforms count {count} exceeds cap {cap}",)
            }
            CandidateError::SampleRateOutOfRange { rate, min, max } => write!(
                f,
                "sample_rate {rate} outside supported range [{min}, {max}] Hz",
            ),
            CandidateError::ScratchSizeOverflow { period, channels } => write!(
                f,
                "period_size {period} * channels {channels} overflows usize",
            ),
        }
    }
}

impl std::error::Error for CandidateError {}

impl MicCandidate {
    /// Static validation; catches operator typos at config
    /// load.  Does **not** validate against the device's
    /// actual channel count -- that requires opening the
    /// device and is performed by the source-open path.
    pub fn validate(&self) -> Result<(), CandidateError> {
        if self.channels.is_empty() {
            return Err(CandidateError::EmptyChannels);
        }
        // Whitelist length cap.  Checked before per-index
        // sanity so a runaway whitelist (operator pasted a
        // 1024-channel TOML by mistake) reports the count
        // rather than a misleading per-index cap error.
        if self.channels.len() > MAX_CHANNELS {
            return Err(CandidateError::TooManyChannels {
                count: self.channels.len(),
                cap: MAX_CHANNELS,
            });
        }
        // Static cap check before duplicate detection: both
        // are cheap, but reporting the offending index
        // (rather than "duplicate at 65535") is more
        // actionable for the operator.
        for &ch in &self.channels {
            if ch > MAX_CHANNEL_INDEX {
                return Err(CandidateError::ChannelIndexTooLarge {
                    channel: ch,
                    cap: MAX_CHANNEL_INDEX,
                });
            }
        }
        // O(n log n) duplicate detection on a clone, so `self`
        // stays untouched and the original ordering is preserved
        // for the consumer. n <= MAX_CHANNELS (8); cheap.
        let mut sorted = self.channels.clone();
        sorted.sort_unstable();
        for w in sorted.windows(2) {
            if w[0] == w[1] {
                return Err(CandidateError::DuplicateChannel(w[0]));
            }
        }

        match &self.source {
            CandidateSource::Alsa {
                hw_spec,
                period_size,
                buffer_size,
            } => {
                if hw_spec.is_empty() {
                    return Err(CandidateError::EmptyHwSpec);
                }
                if *period_size == 0 {
                    return Err(CandidateError::InvalidPeriodSize(*period_size));
                }
                // Upper-bound period_size before any product
                // arithmetic so the scratch-overflow check
                // below operates on bounded values only.
                if *period_size > MAX_PERIOD_FRAMES {
                    return Err(CandidateError::PeriodSizeTooLarge {
                        period: *period_size,
                        cap: MAX_PERIOD_FRAMES,
                    });
                }
                if *buffer_size < *period_size {
                    return Err(CandidateError::InvalidBufferSize {
                        period: *period_size,
                        buffer: *buffer_size,
                    });
                }
                // Two upper bounds on buffer_size: an
                // absolute frame cap AND a multiplier of
                // period_size.  Real ALSA drivers use a few
                // periods of buffer depth; either bound
                // catches a samples-vs-ms typo where an
                // operator wrote 44_100 thinking "1 second"
                // but the ALSA period-snap then refuses to
                // negotiate.
                let multiplier_cap = period_size.saturating_mul(16);
                if *buffer_size > MAX_BUFFER_FRAMES || *buffer_size > multiplier_cap {
                    return Err(CandidateError::BufferSizeTooLarge {
                        buffer: *buffer_size,
                        period: *period_size,
                        absolute_cap: MAX_BUFFER_FRAMES,
                        period_multiplier_cap: 16,
                    });
                }
                // Scratch overflow guard.  In practice
                // unreachable once the per-field caps fire
                // (8192 * 1024 << usize::MAX even on 32-bit),
                // but the explicit `checked_mul` is the
                // right belt-and-braces shape: if any
                // per-field cap is later relaxed, this stays
                // sound.  ALSA scratch is sized as
                // `period_size * actual_channels` at open
                // time; we don't know `actual_channels` here
                // (it's negotiated against the device), so
                // bound against the whitelist's
                // worst-case (`max + 1`).
                let worst_case_channels =
                    self.channels.iter().copied().max().unwrap_or(0) as usize + 1;
                if period_size.checked_mul(worst_case_channels).is_none() {
                    return Err(CandidateError::ScratchSizeOverflow {
                        period: *period_size,
                        channels: worst_case_channels,
                    });
                }
            }
            CandidateSource::Mock {
                waveforms,
                period_size,
                sample_rate,
            } => {
                if waveforms.is_empty() {
                    return Err(CandidateError::EmptyMockWaveforms);
                }
                // Waveforms-as-channels cap (mock's "device
                // channel count" equals `waveforms.len()`).
                // Checked before the whitelist-out-of-range
                // pass so the more diagnostic error fires
                // when both are wrong.
                if waveforms.len() > MAX_CHANNELS {
                    return Err(CandidateError::TooManyChannels {
                        count: waveforms.len(),
                        cap: MAX_CHANNELS,
                    });
                }
                let max_ch = *self.channels.iter().max().expect("non-empty above");
                if max_ch as usize >= waveforms.len() {
                    return Err(CandidateError::MockWhitelistOutOfRange {
                        whitelist_max: max_ch,
                        waveform_count: waveforms.len() as u16,
                    });
                }
                if *period_size == 0 {
                    return Err(CandidateError::InvalidPeriodSize(*period_size));
                }
                if *period_size > MAX_PERIOD_FRAMES {
                    return Err(CandidateError::PeriodSizeTooLarge {
                        period: *period_size,
                        cap: MAX_PERIOD_FRAMES,
                    });
                }
                if *sample_rate == 0 {
                    return Err(CandidateError::InvalidMockSampleRate(*sample_rate));
                }
                if *sample_rate < MIN_SAMPLE_RATE || *sample_rate > MAX_SAMPLE_RATE {
                    return Err(CandidateError::SampleRateOutOfRange {
                        rate: *sample_rate,
                        min: MIN_SAMPLE_RATE,
                        max: MAX_SAMPLE_RATE,
                    });
                }
                // Mock scratch is `period_size *
                // waveforms.len()`.  Same belt-and-braces
                // shape as the ALSA arm.
                if period_size.checked_mul(waveforms.len()).is_none() {
                    return Err(CandidateError::ScratchSizeOverflow {
                        period: *period_size,
                        channels: waveforms.len(),
                    });
                }
            }
        }
        Ok(())
    }
}

/// Cross-validation error between a [`MicPolicy`] and a
/// [`MicCatalogue`].  Distinct from [`CandidateError`]
/// (which is per-candidate static validation): these errors
/// only arise when the user-pref policy references a
/// catalogue entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PolicyValidationError {
    /// `MicSelection::Fixed { id }` references a candidate that
    /// isn't in the catalogue.
    UnknownMicId(MicId),
    /// `MicSelection::Fixed { id }` +
    /// `ChannelSelection::Fixed { channel }` where `channel`
    /// isn't in that candidate's `channels` whitelist.  Only
    /// checkable when the mic is also Fixed: for
    /// `FirstAvailable`, channel validity depends on
    /// whichever mic wins and is knowable only at runtime.
    ChannelNotAvailable {
        mic: MicId,
        channel: u16,
        available: Vec<u16>,
    },
}

impl std::fmt::Display for PolicyValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PolicyValidationError::UnknownMicId(id) => write!(
                f,
                "policy.mic.id = '{id}' does not match any catalogue candidate",
            ),
            PolicyValidationError::ChannelNotAvailable {
                mic,
                channel,
                available,
            } => write!(
                f,
                "policy.channel = {channel} is not in catalogue candidate '{mic}'s available channels {available:?}",
            ),
        }
    }
}

impl std::error::Error for PolicyValidationError {}

impl MicPolicy {
    /// Verify the policy is satisfiable against
    /// `catalogue`.  Run at boot, on user-config
    /// hot-reload, and on `POST /mic/policy`.
    ///
    /// [`MicSelection::FirstAvailable`] is always valid (the
    /// mic gets resolved at runtime).
    /// [`MicSelection::Fixed`] must match a catalogue
    /// candidate.  If [`ChannelSelection`] is also
    /// `Fixed { channel }` AND the mic is fixed, the
    /// channel must be in that candidate's `channels`
    /// whitelist.  Mic = `FirstAvailable` + Channel =
    /// `Fixed` leaves channel validity for runtime: the
    /// arbitrator silently falls back to Auto if the channel
    /// isn't usable on the resolved mic.
    pub fn validate_against(&self, catalogue: &MicCatalogue) -> Result<(), PolicyValidationError> {
        let MicSelection::Fixed { id } = &self.mic else {
            return Ok(());
        };
        let cand = catalogue
            .find(id)
            .ok_or_else(|| PolicyValidationError::UnknownMicId(id.clone()))?;
        if let ChannelSelection::Fixed { channel } = &self.channel
            && !cand.channels.contains(channel)
        {
            return Err(PolicyValidationError::ChannelNotAvailable {
                mic: id.clone(),
                channel: *channel,
                available: cand.channels.clone(),
            });
        }
        Ok(())
    }
}
