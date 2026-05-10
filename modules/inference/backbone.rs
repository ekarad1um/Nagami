//! Backbone abstraction: NPU (RKNN) or CPU (Burn).
//!
//! The inference engine produces a 9 976-element spectrogram per
//! window and needs a 2 000-element feature vector back.  Two
//! implementations satisfy this contract:
//!
//! * `RknnBackbone` (Linux + `rknpu` feature) wraps a
//!   `crate::rknn_runtime::Session` over a pre-built `backbone.rknn`.
//!   ~20 ms per inference on Pi 5; the primary deployment target.
//! * [`BurnBackbone`] runs `crate::model::Backbone<NdArray<f32>>` on
//!   CPU.  ~50-100 ms per inference on Pi 5, ~5-20 ms on M1+.  Used
//!   when librknnrt isn't available (host development) or for
//!   correctness debugging (Burn's fp32 path is the canonical
//!   reference; RKNN runs fp16 internally and has ~1e-3 numerical
//!   drift relative to Burn).
//!
//! Selected via [`BackbonePipeline`] -- a small enum-dispatch
//! wrapper.  Concurrent calls would need external synchronization
//! (the engine owns it via `&mut self`); both impls store internal
//! scratch which mutates per call.
//!
//! ## Memory layout
//!
//! `Preproc::spectrogram` returns a `Box<[[f32; NBins::USIZE]; NFrames::USIZE]>`
//! -- frame-major row-major.  Two backbones consume it differently:
//!
//! * RKNN expects bin-major flat (librknnrt reports
//!   `dims=[1, 232, 1, 43]`).  The `RknnBackbone` impl holds a
//!   transpose scratch (`spec_flat`) and runs
//!   [`crate::inference::kernel::transpose_frame_major_to_bin_major`] before
//!   `session.infer`.
//! * Burn's `Backbone::forward` expects NCHW `[1, 1, 43, 232]`
//!   row-major.  The natural row-major flatten of `spec[h][w]`
//!   matches this layout EXACTLY (since C=1 the channel dim is a
//!   trivial 1-stride).  No transpose needed; we just flatten +
//!   build a `Tensor<B, 4>::from_data`.

#![allow(missing_debug_implementations)]

use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::model::Backbone as BurnNet;
use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};

use crate::common::dims::{BackboneFeatureDim, NBins, NFrames};
use crate::common::hex::hex_lowercase;
#[cfg(all(target_os = "linux", feature = "rknpu"))]
use crate::rknn_runtime::{InputSlice, OutputSlice, Session, TensorFormat};

#[cfg(all(target_os = "linux", feature = "rknpu"))]
use crate::inference::kernel::transpose_frame_major_to_bin_major;

/// Burn backend pinned to NdArray fp32 -- the single CPU choice
/// used both by the daemon's fallback path and by the upstream
/// `classify` parity reference.
type B = NdArray<f32>;

/// Failure shapes from backbone load + per-window
/// inference.  Mapped to [`crate::common::error::ErrorKind`]
/// via the [`crate::common::error::Categorized`] impl
/// below.
#[derive(Debug, Error)]
pub enum BackboneError {
    #[error("read backbone {path}: {source}")]
    Read {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[cfg(all(target_os = "linux", feature = "rknpu"))]
    #[error("rknn: {0}")]
    Rknn(#[from] crate::rknn_runtime::Error),
    #[error("burn: {0}")]
    Burn(String),
    #[error(
        "backbone i/o counts: must be 1 input + 1 output; \
         got {n_input} inputs / {n_output} outputs"
    )]
    IoCount { n_input: u32, n_output: u32 },
    #[error("backbone input has {got} elements; expected {expected}")]
    InputDim { got: usize, expected: usize },
    #[error("backbone output has {got} elements; expected {expected}")]
    OutputDim { got: usize, expected: usize },
    #[error("no usable backbone candidate; {summary}")]
    NoUsableCandidate { summary: String },
}

impl crate::common::error::Categorized for BackboneError {
    fn kind(&self) -> crate::common::error::ErrorKind {
        use crate::common::error::ErrorKind::*;
        match self {
            // RKNN library / Burn / FS -- the backbone is a
            // daemon-internal runtime resource; failures here
            // surface as `Internal` for an HTTP request that
            // accidentally surfaces this enum (rare; the
            // daemon-internal hot path doesn't go through HTTP
            // at all).  The catalogue-walk failure path
            // (`NoUsableCandidate`) is `Unavailable` because
            // it's transient -- re-plug the device and retry.
            BackboneError::Read { .. } => Internal,
            #[cfg(all(target_os = "linux", feature = "rknpu"))]
            BackboneError::Rknn(_) => Internal,
            BackboneError::Burn(_) => Internal,
            // Backbone metadata mismatch: malformed or wrong-shape
            // file; analogous to HeadError::WeightShape.
            BackboneError::IoCount { .. }
            | BackboneError::InputDim { .. }
            | BackboneError::OutputDim { .. } => UserInput,
            BackboneError::NoUsableCandidate { .. } => Unavailable,
        }
    }
}

/// Shorthand for `BackboneError::Read { path: path.to_string(), source }`.
/// `path` is `impl Display` so call sites can pass a `Path::display()`
/// adapter or a literal sentinel like `"librknnrt.so"`.  Cfg-gated
/// to mirror the only callers (`RknnBackbone::load` and
/// `resolve_rknn_library`) so a non-rknpu host does not flag it dead.
#[cfg(all(target_os = "linux", feature = "rknpu"))]
fn read_err(path: impl std::fmt::Display, source: std::io::Error) -> BackboneError {
    BackboneError::Read {
        path: path.to_string(),
        source,
    }
}

// MARK: RknnBackbone (Linux + rknpu only)

/// Rockchip NPU backbone.  Available only when the inference crate
/// is built with `--features rknpu` AND targeting Linux.  On any
/// other host or feature combination this whole block is cfg-gated
/// out, the `rknn_runtime` dep is omitted, and
/// `BackboneKind::Rknn::is_supported()` returns `false` so the
/// catalogue loader silently skips RKNN candidates.
#[cfg(all(target_os = "linux", feature = "rknpu"))]
pub struct RknnBackbone {
    session: Session,
    /// Bin-major scratch (`spec_flat[bin * NFrames + frame]`).
    /// Allocated once at load; reused across all inferences.
    spec_flat: Vec<f32>,
    description: String,
}

#[cfg(all(target_os = "linux", feature = "rknpu"))]
impl std::fmt::Debug for RknnBackbone {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RknnBackbone")
            .field("description", &self.description)
            .finish_non_exhaustive()
    }
}

#[cfg(all(target_os = "linux", feature = "rknpu"))]
impl RknnBackbone {
    /// Load the backbone from `backbone_rknn`.  The runtime library
    /// (`librknnrt.so` / `librknnmrt.so`) is discovered in this
    /// order:
    ///
    ///   1. `RKNN_LIB` env var (explicit operator override),
    ///   2. `crate::rknn_runtime::utils::find_library_candidates()`, which
    ///      walks `/usr/lib`, `/usr/local/lib`,
    ///      `/usr/lib/aarch64-linux-gnu`, `$LD_LIBRARY_PATH`, and
    ///      `$HOME/.local/lib`.
    ///
    /// The library path is no longer threaded through the daemon
    /// CLI / catalogue -- operators set `LD_LIBRARY_PATH` the
    /// standard way (or `RKNN_LIB` for a one-off) and the inference
    /// crate finds it.
    ///
    /// Validates 1 input + 1 output and the expected `NFrames x
    /// NBins -> BackboneFeatureDim` shapes.
    pub fn load(backbone_rknn: &Path) -> Result<Self, BackboneError> {
        let lib = resolve_rknn_library()?;
        let mut bytes =
            std::fs::read(backbone_rknn).map_err(|e| read_err(backbone_rknn.display(), e))?;
        // SAFETY (`Session::load` obligations):
        //   1. Trusted lib: file permissions + immutable rootfs +
        //      operator-restricted `RKNN_LIB` env keep the resolved
        //      path pinned at the vendored `librknnrt.so`.
        //   2. ABI match: `rknn_runtime/bindings.rs` is generated
        //      from that exact vendor build (verified by host
        //      `layout_*` tests; bumping the lib needs a re-gen).
        //   3. Path tampering: deployment chain prevents post-boot
        //      swap; `resolve_rknn_library` only confirms existence.
        let session = unsafe { Session::load(&lib, &mut bytes)? };

        let io = session.io_count()?;
        if io.n_input != 1 || io.n_output != 1 {
            return Err(BackboneError::IoCount {
                n_input: io.n_input,
                n_output: io.n_output,
            });
        }
        let in_attr = session.input_attr(0)?;
        let in_elems = in_attr.n_elems as usize;
        if in_elems != NFrames::USIZE * NBins::USIZE {
            return Err(BackboneError::InputDim {
                got: in_elems,
                expected: NFrames::USIZE * NBins::USIZE,
            });
        }
        let out_attr = session.output_attr(0)?;
        let out_elems = out_attr.n_elems as usize;
        if out_elems != BackboneFeatureDim::USIZE {
            return Err(BackboneError::OutputDim {
                got: out_elems,
                expected: BackboneFeatureDim::USIZE,
            });
        }
        Ok(Self {
            session,
            spec_flat: vec![0.0; NFrames::USIZE * NBins::USIZE],
            description: format!("RKNN: {} (lib: {})", backbone_rknn.display(), lib.display()),
        })
    }

    /// Run one inference; writes features in place.  Bin-major transpose
    /// is applied here for librknnrt's reported NHWC layout (see
    /// [`transpose_frame_major_to_bin_major`]).
    pub fn infer(
        &mut self,
        spec: &[[f32; NBins::USIZE]; NFrames::USIZE],
        features: &mut [f32; BackboneFeatureDim::USIZE],
    ) -> Result<(), BackboneError> {
        transpose_frame_major_to_bin_major::<{ NFrames::USIZE }, { NBins::USIZE }>(
            spec,
            &mut self.spec_flat,
        );
        self.session.infer(
            InputSlice::f32(0, &mut self.spec_flat).with_format(TensorFormat::Nhwc),
            OutputSlice::f32_preallocated(0, features),
        )?;
        Ok(())
    }

    /// Operator-readable description (kind + path).
    pub fn description(&self) -> &str {
        &self.description
    }
}

/// Pick the librknnrt path: explicit `RKNN_LIB` env override, else
/// the first hit from `crate::rknn_runtime::utils::find_library_candidates`.
/// Returns `BackboneError::Read` shaped as a "library not found"
/// when neither yields a path -- same error variant the rest of
/// the load path already uses for missing files.
#[cfg(all(target_os = "linux", feature = "rknpu"))]
fn resolve_rknn_library() -> Result<std::path::PathBuf, BackboneError> {
    if let Some(p) = std::env::var_os("RKNN_LIB") {
        let pb = std::path::PathBuf::from(p);
        if !pb.exists() {
            return Err(read_err(
                pb.display(),
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "RKNN_LIB points at a non-existent file",
                ),
            ));
        }
        return Ok(pb);
    }
    crate::rknn_runtime::utils::find_library_candidates()
        .into_iter()
        .next()
        .ok_or_else(|| {
            read_err(
                "librknnrt.so",
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "librknnrt.so / librknnmrt.so not found in standard locations \
                     (set RKNN_LIB=/path/to/librknnrt.so or LD_LIBRARY_PATH)",
                ),
            )
        })
}

// MARK: BurnBackbone

/// CPU (NdArray fp32) backbone; host-dev fallback and the
/// canonical fp32 reference for parity tests.
pub struct BurnBackbone {
    backbone: BurnNet<B>,
    device: burn::tensor::Device<B>,
    description: String,
}

impl std::fmt::Debug for BurnBackbone {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BurnBackbone")
            .field("description", &self.description)
            .finish_non_exhaustive()
    }
}

impl BurnBackbone {
    /// Load `backbone.mpk` from disk via [`crate::model::Backbone::load_mpk`].
    /// Allocates one `BurnNet<B>` (~5.7 MB of weights) on the
    /// `NdArray<f32>` backend's default device (CPU).
    pub fn load(backbone_mpk: &Path) -> Result<Self, BackboneError> {
        let device: burn::tensor::Device<B> = Default::default();
        let backbone = BurnNet::<B>::load_mpk(backbone_mpk, &device)
            .map_err(|e| BackboneError::Burn(format!("{e}")))?;
        Ok(Self {
            backbone,
            device,
            description: format!("Burn (NdArray fp32): {}", backbone_mpk.display()),
        })
    }

    /// Run one inference; writes features in place.
    pub fn infer(
        &mut self,
        spec: &[[f32; NBins::USIZE]; NFrames::USIZE],
        features: &mut [f32; BackboneFeatureDim::USIZE],
    ) -> Result<(), BackboneError> {
        // Flatten frame-major NCHW [1, 1, 43, 232].  The natural
        // row-major flatten of `spec[h][w]` is exactly the NCHW
        // layout because C=1; no transpose needed (RKNN, in
        // contrast, wants bin-major).
        //
        // The Vec is unavoidable: Burn's `TensorData::new`
        // owns its data, and `From<&[f32]>` for TensorData
        // re-allocates internally.  A pre-allocated scratch field
        // would not help because TensorData consumes the Vec on
        // construction.  `as_flattened().to_vec()` lowers to a
        // single memcpy of the contiguous `[NFrames][NBins]`.
        let flat: Vec<f32> = spec.as_slice().as_flattened().to_vec();
        let input = Tensor::<B, 4>::from_data(
            TensorData::new(flat, [1, 1, NFrames::USIZE, NBins::USIZE]),
            &self.device,
        );
        let output = self.backbone.forward(input);
        // Output shape `[1, 2000]`.  Validate as a defensive check --
        // shouldn't fire for a correctly-shaped backbone.mpk.
        let dims = output.dims();
        if dims != [1, BackboneFeatureDim::USIZE] {
            return Err(BackboneError::OutputDim {
                got: dims.iter().product::<usize>(),
                expected: BackboneFeatureDim::USIZE,
            });
        }
        // `as_slice::<f32>()` borrows directly from TensorData's
        // contiguous bytes (zero alloc).  TensorData must outlive
        // the slice borrow, hence the explicit `let data = ...`.
        let data = output.into_data();
        let slice = data
            .as_slice::<f32>()
            .map_err(|e| BackboneError::Burn(format!("as_slice: {e:?}")))?;
        if slice.len() != BackboneFeatureDim::USIZE {
            // Defensive: dims-check above already validated the
            // tensor shape, so a length mismatch here would mean
            // Burn's TensorData byte-count diverged from its declared
            // shape -- a Burn bug, not our input.  Surface it cleanly
            // rather than panicking via copy_from_slice.
            return Err(BackboneError::OutputDim {
                got: slice.len(),
                expected: BackboneFeatureDim::USIZE,
            });
        }
        features.copy_from_slice(slice);
        Ok(())
    }

    /// Operator-readable description (kind + path).
    pub fn description(&self) -> &str {
        &self.description
    }
}

// MARK: BackbonePipeline

/// Enum-dispatch wrapper so the inference engine can hold
/// a single concrete type.  Hot path resolves to a
/// static-monomorphized `match` (no vtable).  Cheap to add
/// new variants when future work explores e.g. wgpu-backed
/// Burn or a different RKNN flavor.
///
/// The variants are heap-boxed because [`BurnBackbone`]
/// carries a large Burn module (~5.7 MB of weight tensors
/// via `Arc` internally, but the `Param` wrappers +
/// pool / relu state inflate the stack
/// footprint by tens of KB).  Boxing keeps `BackbonePipeline`'s
/// stack size at one pointer per variant.
pub enum BackbonePipeline {
    #[cfg(all(target_os = "linux", feature = "rknpu"))]
    Rknn(Box<RknnBackbone>),
    Burn(Box<BurnBackbone>),
}

impl std::fmt::Debug for BackbonePipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(all(target_os = "linux", feature = "rknpu"))]
            Self::Rknn(r) => r.fmt(f),
            Self::Burn(b) => b.fmt(f),
        }
    }
}

impl BackbonePipeline {
    /// Run one inference; dispatches to the active variant.
    pub fn infer(
        &mut self,
        spec: &[[f32; NBins::USIZE]; NFrames::USIZE],
        features: &mut [f32; BackboneFeatureDim::USIZE],
    ) -> Result<(), BackboneError> {
        match self {
            #[cfg(all(target_os = "linux", feature = "rknpu"))]
            Self::Rknn(r) => r.infer(spec, features),
            Self::Burn(b) => b.infer(spec, features),
        }
    }

    /// Operator-readable description of the active variant.
    pub fn description(&self) -> &str {
        match self {
            #[cfg(all(target_os = "linux", feature = "rknpu"))]
            Self::Rknn(r) => r.description(),
            Self::Burn(b) => b.description(),
        }
    }

    /// Convert into a trait-object form.  The engine
    /// holds `Box<dyn Backbone>`; this method bridges from
    /// the load-time enum (which keeps cfg-gated arms local)
    /// to the runtime trait object (which lets tests
    /// substitute a mock backbone).  The enum stays as the
    /// catalogue-walk return type because cfg-gating boxed
    /// trait objects across crate boundaries is more
    /// invasive than gating one match arm here.
    pub fn into_boxed(self) -> Box<dyn Backbone> {
        match self {
            #[cfg(all(target_os = "linux", feature = "rknpu"))]
            Self::Rknn(r) => r,
            Self::Burn(b) => b,
        }
    }
}

// MARK: Backbone trait

/// Producer of a fixed-size feature vector from one preprocessed
/// spectrogram.  The two production impls live in this module
/// ([`BurnBackbone`] always available; `RknnBackbone` Linux +
/// `rknpu` feature only).  The engine holds one as
/// `Box<dyn Backbone>` so tests substitute mocks without going
/// through the cfg-gated [`BackbonePipeline`] enum.
///
/// ## Layering note
///
/// Trait colocates with its impls: pulling it
/// into `crate::common::traits` would force `common` to either depend
/// on `preproc` (for the eventual `Spectrogram` newtype --
/// Invariant 4 violation) or to host a `BackboneError` shape
/// that has no business in a base crate.  The api crate doesn't
/// import `Backbone` directly today (it goes through
/// `InferenceEngine::new`), so the consumer-pulls-from-common
/// pressure isn't there.
///
/// ## Threading
///
/// `&mut self` is mandatory -- RKNN sessions are stateful
/// (`rknn_inputs_set` mutates session state) and Burn impls
/// hold per-call scratch.  Callers hold one `Box<dyn Backbone>`
/// per inference worker thread; concurrent inference would
/// need either per-thread sessions or a `Mutex` (which defeats
/// the point on multi-core CPU fallback).
///
/// ## Hot-path shape
///
/// `infer` writes the feature vector in-place via
/// `&mut [f32; ...]`.  The engine pre-allocates `features`
/// exactly to avoid the per-call ~8 KB heap churn at inference
/// cadence (~4 Hz default).  Returning `FeatureVec` by value
/// would force the allocation on every call.
pub trait Backbone: Send + std::fmt::Debug + 'static {
    /// Output feature dimensionality.  Pinned to
    /// [`BackboneFeatureDim`] today; returned as a method (not
    /// an associated const) so the trait stays object-safe.
    fn feature_dim(&self) -> BackboneFeatureDim {
        BackboneFeatureDim::default()
    }

    /// Operator-readable description, currently used in heartbeat +
    /// backbone-failure log lines.
    fn description(&self) -> &str;

    /// Run one inference.  Writes into `features` in-place; the
    /// caller pre-allocates to avoid per-call heap churn at
    /// inference cadence.
    fn infer(
        &mut self,
        spec: &[[f32; NBins::USIZE]; NFrames::USIZE],
        features: &mut [f32; BackboneFeatureDim::USIZE],
    ) -> Result<(), BackboneError>;
}

#[cfg(all(target_os = "linux", feature = "rknpu"))]
impl Backbone for RknnBackbone {
    fn description(&self) -> &str {
        RknnBackbone::description(self)
    }
    fn infer(
        &mut self,
        spec: &[[f32; NBins::USIZE]; NFrames::USIZE],
        features: &mut [f32; BackboneFeatureDim::USIZE],
    ) -> Result<(), BackboneError> {
        RknnBackbone::infer(self, spec, features)
    }
}

impl Backbone for BurnBackbone {
    fn description(&self) -> &str {
        BurnBackbone::description(self)
    }
    fn infer(
        &mut self,
        spec: &[[f32; NBins::USIZE]; NFrames::USIZE],
        features: &mut [f32; BackboneFeatureDim::USIZE],
    ) -> Result<(), BackboneError> {
        BurnBackbone::infer(self, spec, features)
    }
}

// Object-safety smoke.  Forces a compile-time check
// that `Backbone` stays dyn-compatible: a future `fn clone(&self)
// -> Self` that breaks the contract would fail this line at
// compile time rather than surface as a confusing trait-object
// construction error in the engine.
#[cfg(test)]
const _: fn() = || {
    fn assert_obj_safe<T: ?Sized>() {}
    assert_obj_safe::<dyn Backbone>();
};

// MARK: BackboneCatalogue

/// On-disk type tag for a backbone candidate.  Drives the loader's
/// per-kind dispatch and the cfg-gated support filter.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BackboneKind {
    /// Rockchip NPU (`*.rknn`).  Loaded via `librknnrt.so` /
    /// `librknnmrt.so` discovered through `RKNN_LIB` (env override)
    /// or `LD_LIBRARY_PATH` / standard system paths.  Only supported
    /// when the inference crate is built with the `rknpu` feature
    /// on a Linux target.
    Rknn,
    /// Burn fp32 (`*.mpk`).  CPU.  Always supported.
    Burn,
}

impl BackboneKind {
    /// `true` when this build of the inference crate can actually
    /// load this kind of backbone.  Used by the loader to skip
    /// unsupported kinds during catalogue traversal without erroring
    /// the whole boot.
    pub const fn is_supported(self) -> bool {
        match self {
            BackboneKind::Burn => true,
            BackboneKind::Rknn => cfg!(all(target_os = "linux", feature = "rknpu")),
        }
    }
}

/// A single backbone candidate as declared in the launch config.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BackboneRef {
    pub kind: BackboneKind,
    pub path: PathBuf,
    /// Optional sha256 digest, bare hex (64 chars; either case
    /// accepted, comparison via `eq_ignore_ascii_case`).  When
    /// present, the loader streams the file through SHA-256 and
    /// compares before instantiating the backbone.  Mismatch is
    /// non-fatal: the candidate is skipped with `tracing::warn!`,
    /// and the next candidate is tried.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hash: Option<String>,
}

impl BackboneRef {
    /// Static well-formedness check.  Catches catalogue typos at
    /// `LaunchConfig::load` time so operators see the diagnostic in
    /// systemd logs instead of debugging "why didn't my backbone
    /// load?" at runtime.
    pub fn validate(&self) -> Result<(), String> {
        if self.path.as_os_str().is_empty() {
            return Err("path must not be empty".into());
        }
        if let Some(h) = &self.hash {
            let h = h.trim();
            if h.len() != 64 || !h.chars().all(|c| c.is_ascii_hexdigit()) {
                return Err(format!(
                    "hash must be 64 hex chars, case-insensitive (got {} chars: {h:?})",
                    h.len(),
                ));
            }
        }
        Ok(())
    }
}

/// Ordered list of backbone candidates.  The loader consults this
/// list in declaration order, returning the first candidate that:
///
/// 1. has a kind supported by this build (cfg-gated; see
///    [`BackboneKind::is_supported`]),
/// 2. exists on disk,
/// 3. matches the optional `hash` field (sha256 hex, case-insensitive), and
/// 4. instantiates without error.
///
/// Empty catalogue is legal and means "no backbone configured" --
/// the daemon's caller decides whether to start inference or skip
/// it.  Mirrors the [`crate::inference::engine::InferenceCfg`] /
/// `audio_io::mic_arbitrator::MicCatalogue` pattern: a small
/// declarative struct with a `validate()` method, serde-gated for
/// TOML round-trips through the `config` crate.
#[derive(Clone, Debug, Default, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BackboneCatalogue {
    #[serde(default)]
    pub candidates: Vec<BackboneRef>,
}

impl BackboneCatalogue {
    /// True iff the catalogue has zero candidates.
    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }

    /// Validate every candidate.  Returns the first failing
    /// candidate's index + diagnostic.
    pub fn validate(&self) -> Result<(), (usize, String)> {
        for (i, c) in self.candidates.iter().enumerate() {
            if let Err(e) = c.validate() {
                return Err((i, e));
            }
        }
        Ok(())
    }

    /// Walk the candidate list in declaration order; return the
    /// first successfully loaded backbone.  Skips unsupported kinds
    /// and hash mismatches via `tracing::warn!`.  If no candidate
    /// succeeds, returns [`BackboneError::NoUsableCandidate`] with
    /// a rendered summary of every attempt.
    ///
    /// Blocking: file I/O + (for RKNN) C-FFI session-create.  Call
    /// from a non-async context (the daemon does this inside
    /// `tokio::task::spawn_blocking`).
    pub fn load_first_supported(&self) -> Result<BackbonePipeline, BackboneError> {
        let mut summaries: Vec<String> = Vec::with_capacity(self.candidates.len());
        for cand in &self.candidates {
            match try_load_candidate(cand) {
                Ok(pipeline) => {
                    tracing::info!(
                        target: "inference",
                        kind = ?cand.kind,
                        path = %cand.path.display(),
                        "backbone candidate loaded",
                    );
                    return Ok(pipeline);
                }
                Err(reason) => {
                    tracing::warn!(
                        target: "inference",
                        kind = ?cand.kind,
                        path = %cand.path.display(),
                        reason = %reason,
                        "backbone candidate skipped",
                    );
                    summaries.push(format!(
                        "[{:?} {}]: {}",
                        cand.kind,
                        cand.path.display(),
                        reason,
                    ));
                }
            }
        }
        let summary = if summaries.is_empty() {
            "catalogue is empty".to_string()
        } else {
            summaries.join("; ")
        };
        Err(BackboneError::NoUsableCandidate { summary })
    }
}

fn try_load_candidate(cand: &BackboneRef) -> Result<BackbonePipeline, String> {
    if !cand.kind.is_supported() {
        return Err(format!(
            "kind {:?} not supported in this build (linux + `rknpu` feature required)",
            cand.kind,
        ));
    }
    if !cand.path.exists() {
        return Err(format!("file does not exist: {}", cand.path.display()));
    }
    if let Some(expected) = &cand.hash {
        verify_sha256(&cand.path, expected)?;
    }
    match cand.kind {
        BackboneKind::Rknn => load_rknn(&cand.path),
        BackboneKind::Burn => BurnBackbone::load(&cand.path)
            .map(|b| BackbonePipeline::Burn(Box::new(b)))
            .map_err(|e| format!("{e}")),
    }
}

#[cfg(all(target_os = "linux", feature = "rknpu"))]
fn load_rknn(path: &Path) -> Result<BackbonePipeline, String> {
    RknnBackbone::load(path)
        .map(|b| BackbonePipeline::Rknn(Box::new(b)))
        .map_err(|e| format!("{e}"))
}

#[cfg(not(all(target_os = "linux", feature = "rknpu")))]
fn load_rknn(_path: &Path) -> Result<BackbonePipeline, String> {
    // Should be unreachable: `BackboneKind::is_supported` returned
    // false above and `try_load_candidate` short-circuited.  Surface
    // it as a hard error here too in case someone calls this helper
    // directly in the future.
    Err("rknn backbone not supported in this build (requires linux + `rknpu` feature)".into())
}

/// Streaming SHA-256 of `path`, compared against `expected_hex`
/// (bare hex, 64 chars; case-insensitive comparison via
/// `eq_ignore_ascii_case`; surrounding whitespace trimmed).  Reads
/// in 64 KiB chunks so a multi-MB backbone doesn't pin memory.
fn verify_sha256(path: &Path, expected_hex: &str) -> Result<(), String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let expected = expected_hex.trim();
    let mut file =
        std::fs::File::open(path).map_err(|e| format!("open {} for hash: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| format!("read {} for hash: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let got_hex = hex_lowercase(&hasher.finalize());
    if got_hex.eq_ignore_ascii_case(expected) {
        Ok(())
    } else {
        Err(format!(
            "sha256 mismatch: expected {expected}, got {got_hex}"
        ))
    }
}

#[cfg(test)]
mod tests {
    // Test code: writes RKNN/labels fixtures via `std::fs::write` for
    // round-trip tests; the production constraint in `clippy.toml` does
    // not apply here.
    #![allow(clippy::disallowed_methods)]
    use super::*;
    use std::path::PathBuf;

    fn crate_root() -> PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
    }

    /// Burn backbone loads cleanly from the shipped reference .mpk
    /// and produces a finite, non-trivial feature vector on a
    /// zero-input spectrogram (biases dominate, so the output isn't
    /// identically zero).
    #[test]
    #[ignore = "depends on repo-root reference assets"]
    fn burn_backbone_loads_and_runs_on_zero_input() {
        let path = crate_root().join("misc/backbones/backbone.mpk");
        assert!(path.exists(), "missing test asset: {}", path.display());
        let mut bb = BurnBackbone::load(&path).expect("load");
        let spec = Box::new([[0.0f32; NBins::USIZE]; NFrames::USIZE]);
        let mut features = Box::new([0.0f32; BackboneFeatureDim::USIZE]);
        bb.infer(&spec, &mut features).expect("infer");
        assert!(features.iter().all(|v| v.is_finite()));
    }

    /// Validates that the bundled `misc/backbones/backbone.mpk` carries
    /// the same conv + dense_1 weights as the upstream Speech-Commands
    /// TFJS model at `misc/models/`.  Skips silently if the upstream
    /// bundle has not been fetched (`misc/models/get_tfjs_sc_model.sh`
    /// downloads it on demand).
    ///
    /// TFJS conv kernels are stored HWIO (Keras channels-last); Burn
    /// `Conv2d` weights are OIHW (PyTorch channels-first).  We re-
    /// index the TFJS flat tensor through Burn's layout and compare
    /// element-wise.  The port should be exact (no fp16 quantisation),
    /// so a tight 1e-7 tolerance is appropriate; any drift is a
    /// regression in the porting tool.
    ///
    /// `dense_1` is `[d_in, d_out]` in both Keras Dense and Burn
    /// `Linear`, so no orientation transform is needed there.
    ///
    /// Run via:
    ///   cargo test --release -- --include-ignored \
    ///     backbone_mpk_matches_speech_commands_tfjs
    #[test]
    #[ignore = "depends on bundled fixtures + upstream TFJS model; --include-ignored"]
    fn backbone_mpk_matches_speech_commands_tfjs() {
        use crate::model::Backbone as BurnNet;

        let root = crate_root();
        let model_json = root.join("misc/models/model.json");
        if !model_json.exists() {
            eprintln!(
                "skipping: {} not present (run misc/models/get_tfjs_sc_model.sh)",
                model_json.display(),
            );
            return;
        }

        let manifest_bytes = std::fs::read(&model_json).expect("read model.json");
        let manifest =
            crate::converter::parse_tfjs_manifest(&manifest_bytes).expect("parse manifest");

        let model_dir = model_json.parent().unwrap();
        let mut blob: Vec<u8> = Vec::new();
        for shard in &manifest.shards {
            let p = model_dir.join(shard);
            let mut bytes =
                std::fs::read(&p).unwrap_or_else(|e| panic!("read {}: {}", p.display(), e));
            blob.append(&mut bytes);
        }

        let entry = |suffix: &str| -> &crate::converter::TfjsManifestEntry {
            manifest
                .entries
                .iter()
                .find(|e| e.name.ends_with(suffix))
                .unwrap_or_else(|| panic!("manifest missing entry ending in {suffix:?}"))
        };
        let tfjs_f32 = |suffix: &str| -> Vec<f32> {
            let e = entry(suffix);
            blob[e.offset_bytes..e.offset_bytes + e.len_bytes]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };

        let device: burn::tensor::Device<B> = Default::default();
        let backbone_path = root.join("misc/backbones/backbone.mpk");
        let backbone: BurnNet<B> =
            BurnNet::load_mpk(&backbone_path, &device).expect("load backbone");

        fn assert_close(name: &str, got: &[f32], expected: &[f32], tol: f32) {
            assert_eq!(
                got.len(),
                expected.len(),
                "{name}: length mismatch -- burn={}, tfjs={}",
                got.len(),
                expected.len(),
            );
            let mut max_diff = 0.0f32;
            let mut max_at = 0usize;
            for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
                let d = (g - e).abs();
                if d > max_diff {
                    max_diff = d;
                    max_at = i;
                }
            }
            assert!(
                max_diff <= tol,
                "{name}: max |D|={max_diff} at flat idx {max_at} exceeds tol={tol}; \
                 burn={}, tfjs={}",
                got[max_at],
                expected[max_at],
            );
        }

        // Permute TFJS HWIO [kH, kW, in, out] -> Burn OIHW [out, in, kH, kW].
        fn hwio_to_oihw(hwio: &[f32], kh: usize, kw: usize, ic: usize, oc: usize) -> Vec<f32> {
            assert_eq!(hwio.len(), kh * kw * ic * oc);
            let mut oihw = vec![0.0f32; hwio.len()];
            for h in 0..kh {
                for w in 0..kw {
                    for i in 0..ic {
                        for o in 0..oc {
                            let src = ((h * kw + w) * ic + i) * oc + o;
                            let dst = ((o * ic + i) * kh + h) * kw + w;
                            oihw[dst] = hwio[src];
                        }
                    }
                }
            }
            oihw
        }

        // Conv layers: (Burn name, TFJS kernel suffix, TFJS bias suffix,
        // [kH, kW, in, out]).  Matches the Keras topology declared in
        // `crate::model::Backbone::new`.
        let conv_layers: [(&str, &str, &str, [usize; 4]); 4] = [
            ("conv1", "conv2d_1/kernel", "conv2d_1/bias", [2, 8, 1, 8]),
            ("conv2", "conv2d_2/kernel", "conv2d_2/bias", [2, 4, 8, 32]),
            ("conv3", "conv2d_3/kernel", "conv2d_3/bias", [2, 4, 32, 32]),
            ("conv4", "conv2d_4/kernel", "conv2d_4/bias", [2, 4, 32, 32]),
        ];

        for (idx, (name, k_suffix, b_suffix, [kh, kw, ic, oc])) in conv_layers.iter().enumerate() {
            let conv = match idx {
                0 => &backbone.conv1,
                1 => &backbone.conv2,
                2 => &backbone.conv3,
                3 => &backbone.conv4,
                _ => unreachable!(),
            };
            let burn_w: Vec<f32> = conv
                .weight
                .val()
                .clone()
                .into_data()
                .to_vec()
                .expect("to_vec weight");
            let tfjs_w_oihw = hwio_to_oihw(&tfjs_f32(k_suffix), *kh, *kw, *ic, *oc);
            assert_close(&format!("{name}.weight"), &burn_w, &tfjs_w_oihw, 1e-7);

            let burn_b: Vec<f32> = conv
                .bias
                .as_ref()
                .expect("conv has bias")
                .val()
                .clone()
                .into_data()
                .to_vec()
                .expect("to_vec bias");
            let tfjs_b = tfjs_f32(b_suffix);
            assert_close(&format!("{name}.bias"), &burn_b, &tfjs_b, 1e-7);
        }

        // Dense_1: TFJS [d_in, d_out] = Burn `Linear` weight [d_in, d_out].
        let dense1_w_burn: Vec<f32> = backbone
            .dense1
            .weight
            .val()
            .clone()
            .into_data()
            .to_vec()
            .expect("to_vec dense1.weight");
        let dense1_w_tfjs: Vec<f32> = tfjs_f32("dense_1/kernel");
        assert_close("dense1.weight", &dense1_w_burn, &dense1_w_tfjs, 1e-7);

        let dense1_b_burn: Vec<f32> = backbone
            .dense1
            .bias
            .as_ref()
            .expect("dense1 has bias")
            .val()
            .clone()
            .into_data()
            .to_vec()
            .expect("to_vec dense1.bias");
        let dense1_b_tfjs: Vec<f32> = tfjs_f32("dense_1/bias");
        assert_close("dense1.bias", &dense1_b_burn, &dense1_b_tfjs, 1e-7);
    }

    /// Loading a non-existent backbone.mpk surfaces a `BackboneError::Burn`
    /// with the underlying Burn recorder error, not a panic.  Runs
    /// in routine CI with no external dependencies.
    #[test]
    fn burn_backbone_load_missing_file_returns_err() {
        let path = std::path::Path::new("/nonexistent/.acoustics_lab/missing-backbone.mpk");
        let res = BurnBackbone::load(path);
        let err = res.unwrap_err();
        // The error should be Burn(_) since the recorder layer
        // catches the missing file.
        match err {
            BackboneError::Burn(msg) => {
                assert!(
                    !msg.is_empty(),
                    "Burn error should carry a non-empty message",
                );
            }
            other => panic!("expected BackboneError::Burn, got {other:?}"),
        }
    }

    /// Loading a non-existent backbone.rknn surfaces a
    /// `BackboneError::Read`.  Cfg-gated to a build where
    /// `RknnBackbone` is actually compiled in; on host-dev macOS
    /// builds the catalogue-level test
    /// `rknn_candidate_unsupported_on_non_rknpu_build` covers the
    /// equivalent path.
    #[test]
    #[cfg(all(target_os = "linux", feature = "rknpu"))]
    fn rknn_backbone_load_missing_file_returns_err() {
        // Override RKNN_LIB to a path we control so library
        // resolution doesn't surface as the error before the
        // backbone read fails.  The bogus value is non-existent;
        // resolve_rknn_library returns Read pointing at it.  The
        // backbone path read won't even run, but the error variant
        // we expect is the same.
        // SAFETY: tests run in a single thread by default; setting
        // an env var here is fine for the single-process test.
        unsafe {
            std::env::set_var(
                "RKNN_LIB",
                "/nonexistent/.acoustics_lab/missing-librknnrt.so",
            );
        }
        let bb = std::path::Path::new("/nonexistent/.acoustics_lab/missing-backbone.rknn");
        let res = RknnBackbone::load(bb);
        unsafe {
            std::env::remove_var("RKNN_LIB");
        }
        let err = res.unwrap_err();
        match err {
            BackboneError::Read { path, .. } => {
                assert!(
                    path.contains("missing-backbone") || path.contains("missing-librknnrt"),
                    "path mismatch: {path}",
                );
            }
            other => panic!("expected BackboneError::Read, got {other:?}"),
        }
    }

    // MARK: BackboneCatalogue

    /// `BackboneRef::validate` accepts a 64-char lowercase hex hash
    /// and rejects malformed shapes with diagnostics that name the
    /// problem.  Catches catalogue typos at boot rather than at
    /// hash-time.
    #[test]
    fn backbone_ref_validate_hash_format() {
        let mut r = BackboneRef {
            kind: BackboneKind::Burn,
            path: PathBuf::from("/tmp/x.mpk"),
            hash: None,
        };
        assert!(r.validate().is_ok(), "no-hash candidate must validate");

        // Valid: exactly 64 hex chars.
        r.hash = Some("a".repeat(64));
        assert!(r.validate().is_ok(), "64 hex chars must validate");

        // Reject: too short.
        r.hash = Some("a".repeat(63));
        assert!(r.validate().is_err());

        // Reject: contains non-hex.
        r.hash = Some("g".repeat(64));
        assert!(r.validate().is_err());

        // Reject: empty path.
        r.hash = None;
        r.path = PathBuf::new();
        assert!(r.validate().is_err());
    }

    /// An empty catalogue surfaces `NoUsableCandidate` with the
    /// "catalogue is empty" summary -- operators shouldn't see a
    /// generic `attempts.len() == 0` panic-shaped error.
    #[test]
    fn empty_catalogue_returns_no_usable_candidate() {
        let cat = BackboneCatalogue::default();
        let err = cat.load_first_supported().expect_err("must be err");
        match err {
            BackboneError::NoUsableCandidate { summary } => {
                assert!(
                    summary.contains("empty"),
                    "summary should mention emptiness, got: {summary}"
                );
            }
            other => panic!("expected NoUsableCandidate, got {other:?}"),
        }
    }

    /// A non-existent file is reported in the catalogue summary so
    /// operators can grep their TOML for typos.
    #[test]
    fn missing_file_reported_in_summary() {
        let cat = BackboneCatalogue {
            candidates: vec![BackboneRef {
                kind: BackboneKind::Burn,
                path: PathBuf::from("/nonexistent/.acoustics_lab/missing.mpk"),
                hash: None,
            }],
        };
        let err = cat.load_first_supported().expect_err("must be err");
        let msg = err.to_string();
        assert!(
            msg.contains("missing.mpk") && msg.contains("does not exist"),
            "summary should name the missing file: {msg}",
        );
    }

    /// Hash mismatch is reported in the candidate summary; the
    /// candidate is skipped (not fatal at the candidate level --
    /// fatal at the catalogue level here because we only have one).
    #[test]
    fn hash_mismatch_reported_in_summary() {
        let dir = tempfile::tempdir().expect("tempdir");
        let p = dir.path().join("dummy.mpk");
        std::fs::write(&p, b"hello world").expect("write");
        // SHA-256 of "hello world" is well-known; pass a clearly
        // wrong digest to force a mismatch.
        let cat = BackboneCatalogue {
            candidates: vec![BackboneRef {
                kind: BackboneKind::Burn,
                path: p,
                hash: Some("0".repeat(64)),
            }],
        };
        let err = cat.load_first_supported().expect_err("must be err");
        let msg = err.to_string();
        assert!(
            msg.contains("sha256 mismatch"),
            "summary should call out hash mismatch: {msg}",
        );
    }

    /// On a non-rknpu build, an `Rknn` candidate is reported as
    /// unsupported, not as a missing file (even if the file does
    /// exist -- we never reach the load stage).  This exercises the
    /// cfg-gated `is_supported` short-circuit.
    #[test]
    #[cfg(not(all(target_os = "linux", feature = "rknpu")))]
    fn rknn_candidate_unsupported_on_non_rknpu_build() {
        let dir = tempfile::tempdir().expect("tempdir");
        let p = dir.path().join("backbone.rknn");
        std::fs::write(&p, b"fake rknn bytes").expect("write");
        let cat = BackboneCatalogue {
            candidates: vec![BackboneRef {
                kind: BackboneKind::Rknn,
                path: p,
                hash: None,
            }],
        };
        let err = cat.load_first_supported().expect_err("must be err");
        let msg = err.to_string();
        assert!(
            msg.contains("not supported"),
            "summary should call out unsupported kind: {msg}",
        );
    }
}
