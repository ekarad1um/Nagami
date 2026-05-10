//! Capture sources: ALSA (production, Linux only) and Mock
//! (cross-platform, dev/test).
//!
//! # `MicSource` sealed trait
//!
//! [`MicSource`] is a sealed trait (only `audio_io` adds
//! impls) that unifies the read surface across [`MockSource`]
//! and `AlsaSource`.  The arbitrator continues to
//! pattern-match on [`ActiveSource`] for source-specific
//! recovery (ALSA's xrun `try_recover` has no Mock analogue),
//! but the read step dispatches through the trait so test
//! mocks can substitute the source type without faking ALSA.
//! The `dyn`-friendly shape also lets a future per-source
//! supervisor task hold `Box<dyn MicSource>` without the
//! cfg-gated enum machinery.
//!
//! # `ReadOutcome` disambiguation
//!
//! [`ReadOutcome`] names each post-read state explicitly:
//!
//! - `Frames(n)` -- `n` interleaved frames written; the
//!   arbitrator processes them.
//! - `Timeout` -- ALSA's bounded poll elapsed before `readi`;
//!   re-check stop and continue with the same source.
//! - `StopRequested` -- the mock source observed a
//!   cooperative stop signal during its pacing sleep.
//! - `EndOfStream` -- ALSA's `readi` returned 0 frames (the
//!   PCM hit EOF on a closed device); the source is dead and
//!   must be torn down rather than spun on.
//!
//! # Variants
//!
//! - [`MockSource`] -- cross-platform synthetic source
//!   (always compiled).
//! - `AlsaSource` (multi-channel `alsa::pcm::PCM` wrapper)
//!   -- production capture; compiled iff `target_os =
//!   "linux"` AND the `alsa-real` crate feature is enabled.
//!   Plain text rather than an intra-doc link so
//!   default-feature doc builds don't warn.
//!
//! macOS dev builds + Linux test builds without `alsa-real`
//! route ALSA candidates through [`open_source`] to the
//! [`OpenError::AlsaNotCompiledIn`] error arm, exactly the
//! same shape the arbitrator already handles.

#[cfg(all(target_os = "linux", feature = "alsa-real"))]
pub mod alsa;
pub mod mock;

#[cfg(all(target_os = "linux", feature = "alsa-real"))]
pub use alsa::AlsaSource;
pub use mock::MockSource;

use crate::audio_io::mic_arbitrator::{CandidateError, CandidateSource, MicCandidate};
use crate::common::ids::MicId;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

/// Sealed marker so external crates can't add [`MicSource`]
/// impls.  Keeps the trait surface internal to `audio_io`
/// while letting callers hold `Box<dyn MicSource>`.
mod sealed {
    pub trait Sealed {}
}

/// Outcome of one [`MicSource::read_interleaved`] call.
//
// `Timeout` and `EndOfStream` are constructed only by the
// `audio_io::source::alsa` impl, which is itself gated on
// `cfg(all(target_os = "linux", feature = "alsa-real"))`.  On
// host builds without that feature the compiler sees the
// variants as unconstructed; `allow(dead_code)` records that
// the dead-code analysis is correct only under the host
// configuration.  Removing the variants would break the ALSA
// impl on Linux + alsa-real builds.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[allow(dead_code)]
pub enum ReadOutcome {
    /// `n` interleaved frames written into
    /// `out[..n * channels]`.  The arbitrator processes them.
    Frames(NonZeroUsize),
    /// Source's bounded poll / sleep elapsed without
    /// delivering data.  The arbitrator re-checks its stop
    /// flag and re-enters the read loop on the same source.
    /// `out` is untouched.
    Timeout,
    /// Source observed a cooperative stop signal during its
    /// pacing / poll phase.  Same arbitrator behaviour as
    /// [`Self::Timeout`]; emitted as a distinct variant so
    /// logs and tests can disambiguate.  `out` is untouched.
    StopRequested,
    /// Source determined it cannot produce more data (ALSA
    /// short-read on a closed PCM, mock exhausted a
    /// fixed-length stream).  The arbitrator tears down and
    /// re-resolves per the active policy.  `out` is
    /// untouched.
    EndOfStream,
}

/// Unified per-period read error.  Variants are cfg-gated to
/// the same set of impls that conditionally compile.
#[derive(Debug)]
#[allow(dead_code)] // `Mock` wraps `Infallible`; `Alsa` is cfg-gated.
pub enum ReadError {
    #[cfg(all(target_os = "linux", feature = "alsa-real"))]
    Alsa(::alsa::Error),
    /// Reserved for sources that genuinely cannot fail
    /// mid-stream (today: [`MockSource`]).  Construction is
    /// uninhabited, so a `match` exhausting other arms
    /// doesn't need a wildcard.
    Mock(std::convert::Infallible),
}

impl std::fmt::Display for ReadError {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(all(target_os = "linux", feature = "alsa-real"))]
            ReadError::Alsa(e) => std::fmt::Display::fmt(e, _f),
            ReadError::Mock(infallible) => match *infallible {},
        }
    }
}

impl std::error::Error for ReadError {}

/// Sealed trait the arbitrator's read step dispatches
/// through.  Per-source recovery (ALSA's `try_recover`,
/// mock's none-needed) lives on the concrete impls behind a
/// match on [`ActiveSource`]; only the read itself goes
/// through the trait.
///
/// `Send` is required (the arbitrator owns the source on a
/// `std::thread`); `Sync` is NOT (no concurrent access).
/// Returning the unified [`ReadError`] enum directly (no
/// associated error type) keeps the trait `dyn`-compatible.
// `id`/`channels`/`rate`/`period_size`/`effective_whitelist`
// are consumed only by the alsa-real impl + tests; on host
// builds without alsa-real they read as unused.  Allow the
// dead-code lint since deleting the methods would break the
// trait surface ALSA uses.
#[allow(dead_code)]
pub trait MicSource: sealed::Sealed + Send + std::fmt::Debug {
    fn id(&self) -> &MicId;
    fn channels(&self) -> u16;
    fn rate(&self) -> u32;
    fn period_size(&self) -> usize;
    fn effective_whitelist(&self) -> &[u16];
    /// Read up to one period of interleaved frames.  Sources
    /// own their pacing (mock's wall-clock sleep, ALSA's
    /// poll-with-bounded-timeout); see [`ReadOutcome`] for
    /// the per-variant semantics.
    fn read_interleaved(&mut self, out: &mut [f32]) -> Result<ReadOutcome, ReadError>;
}

impl sealed::Sealed for MockSource {}
impl sealed::Sealed for ActiveSource {}
#[cfg(all(target_os = "linux", feature = "alsa-real"))]
impl sealed::Sealed for AlsaSource {}

/// Currently-active capture source.  The arbitrator owns at
/// most one at a time.
///
/// Variants are gated by platform + feature so a build
/// without `alsa-real` doesn't carry vtable space for an
/// unreachable variant.  `non_exhaustive` would be excessive
/// here -- the enum is a module-internal detail and changes
/// in lockstep with the arbitrator.
#[derive(Debug)]
pub enum ActiveSource {
    Mock(MockSource),
    #[cfg(all(target_os = "linux", feature = "alsa-real"))]
    Alsa(AlsaSource),
}

impl ActiveSource {
    /// Stable id of the candidate this source was opened
    /// for.  The arbitrator compares against the
    /// desired-id-from-policy each loop iteration to detect
    /// mic-change requests.
    pub fn id(&self) -> &MicId {
        match self {
            ActiveSource::Mock(s) => s.id(),
            #[cfg(all(target_os = "linux", feature = "alsa-real"))]
            ActiveSource::Alsa(s) => s.id(),
        }
    }

    /// Total number of interleaved channels the source
    /// produces.  May be larger than the candidate's
    /// whitelist length: non-whitelisted indices are demuxed
    /// and discarded by the arbitrator, never written to the
    /// audio buffer.
    pub fn channels(&self) -> u16 {
        match self {
            ActiveSource::Mock(s) => s.channels(),
            #[cfg(all(target_os = "linux", feature = "alsa-real"))]
            ActiveSource::Alsa(s) => s.channels(),
        }
    }

    /// Source-native sample rate.  The arbitrator constructs
    /// a resampler iff this differs from
    /// [`crate::common::dims::SampleRate::VALUE`].
    pub fn rate(&self) -> u32 {
        match self {
            ActiveSource::Mock(s) => s.rate(),
            #[cfg(all(target_os = "linux", feature = "alsa-real"))]
            ActiveSource::Alsa(s) => s.rate(),
        }
    }

    /// Frames per `read_interleaved` call (the per-variant
    /// inner method on [`MockSource`] / `AlsaSource`).  Used
    /// by the arbitrator to size its per-channel scratch
    /// buffers exactly once at open time.  No intra-doc link
    /// because the method lives on the inner variants, not
    /// on [`ActiveSource`] itself.
    pub fn period_size(&self) -> usize {
        match self {
            ActiveSource::Mock(s) => s.period_size(),
            #[cfg(all(target_os = "linux", feature = "alsa-real"))]
            ActiveSource::Alsa(s) => s.period_size(),
        }
    }

    /// Whitelist usable on this source, after open-time
    /// intersection with the device's actual channel count.
    /// The
    /// [`crate::audio_io::mic_arbitrator::MicArbitrator`]
    /// reads this once at boot (caches it on its state) and
    /// drives all subsequent per-slot RMS bookkeeping off the
    /// cached copy.
    ///
    /// [`MockSource`] returns the candidate's `channels`
    /// verbatim (validated to be in-range at
    /// `MicCandidate::validate`); `AlsaSource` may have
    /// shrunk it if the operator listed indices the device
    /// doesn't expose.
    pub fn effective_whitelist(&self) -> &[u16] {
        match self {
            ActiveSource::Mock(s) => s.effective_whitelist(),
            #[cfg(all(target_os = "linux", feature = "alsa-real"))]
            ActiveSource::Alsa(s) => s.effective_whitelist(),
        }
    }
}

/// Errors produced by [`open_source`] at mic-open time.
///
/// Per-period read errors live on the source itself (only
/// the ALSA variant produces them; mock can't fail
/// mid-stream).
#[derive(Debug)]
pub enum OpenError {
    /// Static [`MicCandidate::validate`] rejected the
    /// candidate.  Surfaces operator typos at open time
    /// even when the config loader skipped validation.
    InvalidCandidate(MicId, CandidateError),
    /// `CandidateSource::Alsa` was specified but the crate
    /// was built without the `alsa-real` feature (or this
    /// is macOS).  [`crate::audio_io::mic_arbitrator::MicSelection::FirstAvailable`] callers
    /// fall through to the next candidate; `Fixed` callers
    /// stay inert.
    // Constructed only when `alsa-real` is disabled or on non-Linux builds.
    #[allow(dead_code)]
    AlsaNotCompiledIn(MicId),
    /// Source-specific runtime failure at open.  The
    /// `String` payload is operator-readable diagnostic
    /// text; the ALSA variant's underlying `alsa::Error` is
    /// captured here as a string so callers don't need to
    /// depend on `alsa-rs` directly.
    SourceUnavailable(MicId, String),
}

impl std::fmt::Display for OpenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpenError::InvalidCandidate(id, e) => {
                write!(f, "candidate {id} invalid: {e}")
            }
            OpenError::AlsaNotCompiledIn(id) => write!(
                f,
                "candidate {id} requires the `alsa-real` feature (built without it)",
            ),
            OpenError::SourceUnavailable(id, msg) => {
                write!(f, "candidate {id} unavailable: {msg}")
            }
        }
    }
}

impl std::error::Error for OpenError {}

/// Open a capture source for `candidate`, validating it
/// statically first.  On success returns an [`ActiveSource`]
/// ready for [`MicSource::read_interleaved`] calls.
///
/// `stop` is borrowed by [`MockSource`] so its real-time
/// pacing sleep can be interrupted promptly when the
/// arbitrator is asked to shut down.  ALSA sources don't
/// take it -- their `readi` is fast enough that the
/// per-period stop check inside the arbitrator loop is
/// sufficient.
pub fn open_source(
    candidate: &MicCandidate,
    stop: Arc<AtomicBool>,
) -> Result<ActiveSource, OpenError> {
    candidate
        .validate()
        .map_err(|e| OpenError::InvalidCandidate(candidate.id.clone(), e))?;

    match &candidate.source {
        CandidateSource::Mock { .. } => MockSource::open(candidate, stop)
            .map(ActiveSource::Mock)
            .map_err(|msg| OpenError::SourceUnavailable(candidate.id.clone(), msg)),
        CandidateSource::Alsa { .. } => open_alsa(candidate),
    }
}

/// Branchpoint isolating the `alsa-real` cfg gate.  Two arms
/// with the same signature so the dispatcher above stays
/// cfg-free.
#[cfg(all(target_os = "linux", feature = "alsa-real"))]
fn open_alsa(candidate: &MicCandidate) -> Result<ActiveSource, OpenError> {
    AlsaSource::open(candidate)
        .map(ActiveSource::Alsa)
        .map_err(|msg| OpenError::SourceUnavailable(candidate.id.clone(), msg))
}

#[cfg(not(all(target_os = "linux", feature = "alsa-real")))]
fn open_alsa(candidate: &MicCandidate) -> Result<ActiveSource, OpenError> {
    Err(OpenError::AlsaNotCompiledIn(candidate.id.clone()))
}

// Object-safety + sealed-trait smoke.  Forces a compile-time
// check that `MicSource` stays dyn-compatible AND that the
// `Sealed` supertrait keeps external impls out.
#[cfg(test)]
const _: fn() = || {
    fn assert_obj_safe<T: ?Sized>() {}
    assert_obj_safe::<dyn MicSource>();
};

// `ActiveSource` impls `MicSource` by delegating each method
// to its inner arm.  The arbitrator's read hot path goes
// through this trait method (one match-arm dispatch, same
// monomorphization as the open-coded match).
impl MicSource for ActiveSource {
    fn id(&self) -> &MicId {
        ActiveSource::id(self)
    }
    fn channels(&self) -> u16 {
        ActiveSource::channels(self)
    }
    fn rate(&self) -> u32 {
        ActiveSource::rate(self)
    }
    fn period_size(&self) -> usize {
        ActiveSource::period_size(self)
    }
    fn effective_whitelist(&self) -> &[u16] {
        ActiveSource::effective_whitelist(self)
    }
    fn read_interleaved(&mut self, out: &mut [f32]) -> Result<ReadOutcome, ReadError> {
        match self {
            ActiveSource::Mock(s) => s.read_interleaved(out),
            #[cfg(all(target_os = "linux", feature = "alsa-real"))]
            ActiveSource::Alsa(s) => s.read_interleaved(out),
        }
    }
}
