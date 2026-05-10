//! Dimension newtypes for the audio + ML pipeline.
//!
//! Every cross-module sample-rate / window-length / feature-dim
//! constant lives here as a typed newtype rather than a raw
//! `pub const N: usize`.  The `u32` representation is the
//! smallest type that fits every value today (`SampleRate =
//! 44_100` is the largest); array-dim contexts use the
//! `USIZE` associated constant for a cast-free path.
//!
//! # API per newtype
//!
//! For a generated newtype `FooDim`:
//!
//! | Item                          | Purpose                                         |
//! |-------------------------------|-------------------------------------------------|
//! | `FooDim::VALUE: u32`          | canonical literal for arithmetic + comparisons  |
//! | `FooDim::USIZE: usize`        | same value as `usize`, for array dims (no cast) |
//! | `FooDim::default()`           | construct an instance carrying the canonical    |
//! | `FooDim::new(v: u32)`         | construct a non-canonical value (tests)         |
//! | `FooDim::get(self) -> u32`    | extract the wrapped u32                         |
//!
//! # Why u32 rather than usize
//!
//! 1. Every constant fits comfortably in `u32`, and several
//!    are naturally `u32`-typed downstream (sample rates flow
//!    through ALSA / Opus / rubato as `u32`).  A `usize`
//!    representation would require `as u32` casts on those
//!    paths.
//! 2. `usize` is target-dependent (32 vs 64 bit).  Wire
//!    formats and config values must round-trip identically
//!    across target architectures, which `usize` cannot
//!    guarantee.
//!
//! Consumers that need a `usize` (array dims,
//! `Vec::with_capacity`, slice indexing) use `FooDim::USIZE`.
//!
//! # Adding a new dimension
//!
//! Append one `u32_newtype!(Name, value);` line below.  Drop
//! any pre-existing `pub const Name` shim at the original
//! call site so this stays the single source of truth.

// MARK: Macro

/// Generate a `u32`-backed dimension newtype with the canonical
/// API (see module docs).
macro_rules! u32_newtype {
    ($(#[$attr:meta])* $name:ident, $value:expr) => {
        $(#[$attr])*
        #[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
        #[derive(serde::Serialize, serde::Deserialize)]
        #[serde(transparent)]
        pub struct $name(u32);

        impl $name {
            /// The canonical wire/config value.
            pub const VALUE: u32 = $value;

            /// `VALUE` as `usize`, for array dims and `Vec`
            /// capacities where a cast at every use site would
            /// be noise.
            pub const USIZE: usize = $value as usize;

            /// Wrap a `u32` as a non-canonical instance.  Most
            /// production code wants [`Self::default`]; `new`
            /// exists so tests can construct divergent
            /// instances without reaching past the type.
            #[inline]
            pub const fn new(v: u32) -> Self {
                Self(v)
            }

            /// Unwrap the inner `u32`.
            #[inline]
            pub const fn get(self) -> u32 {
                self.0
            }

            /// Unwrap as `usize`.  Convenience for sites that
            /// have a runtime instance rather than the `USIZE`
            /// const.
            #[inline]
            pub const fn as_usize(self) -> usize {
                self.0 as usize
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self($value)
            }
        }

        impl ::core::fmt::Display for $name {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                ::core::fmt::Display::fmt(&self.0, f)
            }
        }
    };
}

// MARK: Dimensions

u32_newtype! {
    /// Microphone capture sample rate (44.1 kHz).
    SampleRate, 44_100
}

u32_newtype! {
    /// Inference window length in samples (~1.0 s at
    /// [`SampleRate`]).
    WaveformLen, 44_032
}

u32_newtype! {
    /// Number of spectrogram time-frames per inference window.
    NFrames, 43
}

u32_newtype! {
    /// Number of spectrogram frequency bins per frame.
    NBins, 232
}

u32_newtype! {
    /// Hop size between adjacent spectrogram frames, in
    /// samples.
    HopSamples, 1_024
}

u32_newtype! {
    /// Backbone output / head input feature-vector length.
    BackboneFeatureDim, 2_000
}

// MARK: Soft caps
//
// These values are not dimensions of the canonical pipeline; they
// cap operator-supplied or on-disk shapes so a malformed / hostile
// input cannot allocate unbounded memory before the validators
// catch it.  Centralised here so the cold-path validators in
// `inference::head` and `model` agree on the same ceiling.

/// Hard upper bound on `n_classes` for any loaded head.  Real
/// deployments are O(10)–O(100) classes; the cap defends the
/// allocator against a corrupt or adversarial `head.mpk` that
/// would otherwise drag in
/// `BACKBONE_FEATURE_DIM × MAX_N_CLASSES × 4 ≈ 800 MB` of
/// f32 weights at load time.  Mirrored at `model::MAX_N_CLASSES`
/// and `inference::head::MAX_N_CLASSES` (both re-export this
/// constant) so the two cold-path validators cannot drift.
pub const MAX_N_CLASSES: usize = 100_000;

/// `usize` alias for [`BackboneFeatureDim::USIZE`] so the
/// numerics call sites that mostly want a plain integer
/// (`vec![0.0; BACKBONE_FEATURE_DIM * n]`, slice asserts) read
/// without the newtype scaffold.  Kept in lock-step with
/// `BackboneFeatureDim::USIZE` by construction; the canonical
/// dimension still lives in the newtype.
pub const BACKBONE_FEATURE_DIM: usize = BackboneFeatureDim::USIZE;

// MARK: Tests

#[cfg(test)]
mod tests {
    use super::*;

    /// If a value here changes, every consumer needs review.
    #[test]
    fn canonical_values_are_pinned() {
        assert_eq!(SampleRate::VALUE, 44_100);
        assert_eq!(WaveformLen::VALUE, 44_032);
        assert_eq!(NFrames::VALUE, 43);
        assert_eq!(NBins::VALUE, 232);
        assert_eq!(HopSamples::VALUE, 1_024);
        assert_eq!(BackboneFeatureDim::VALUE, 2_000);
    }

    #[test]
    fn usize_matches_value() {
        assert_eq!(SampleRate::USIZE, SampleRate::VALUE as usize);
        assert_eq!(WaveformLen::USIZE, WaveformLen::VALUE as usize);
        assert_eq!(
            BackboneFeatureDim::USIZE,
            BackboneFeatureDim::VALUE as usize
        );
    }

    #[test]
    fn default_is_canonical() {
        assert_eq!(SampleRate::default().get(), SampleRate::VALUE);
        assert_eq!(WaveformLen::default().get(), WaveformLen::VALUE);
        assert_eq!(
            BackboneFeatureDim::default().as_usize(),
            BackboneFeatureDim::USIZE
        );
    }

    /// `new` builds non-canonical instances; tests rely on
    /// this to construct divergent values without bypassing
    /// encapsulation.
    #[test]
    fn new_constructs_arbitrary_value() {
        let half = WaveformLen::new(WaveformLen::VALUE / 2);
        assert_eq!(half.get(), 22_016);
        assert_ne!(half, WaveformLen::default());
    }

    /// `[f32; FooDim::USIZE]` works at array-dim position --
    /// the whole point of the cast-free `USIZE` const.
    #[test]
    fn usize_works_in_array_dim_position() {
        let _: [f32; WaveformLen::USIZE] = [0.0; 44_032];
        let _: [f32; BackboneFeatureDim::USIZE] = [0.0; 2_000];
    }

    /// `Display` writes the inner number, used in error
    /// messages like `"expected {WaveformLen::default()} samples"`.
    #[test]
    fn display_writes_inner_number() {
        assert_eq!(format!("{}", WaveformLen::default()), "44032");
        assert_eq!(format!("{}", NFrames::new(7)), "7");
    }
}
