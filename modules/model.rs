//! Burn fp32 acoustic-model definitions.
//!
//! Topology (from `tm-my-audio-model/model.json`):
//!
//! ```text
//! input   [N, 1, 43, 232]  (NCHW; preproc emits NHWC-equivalent and we
//!                           reshape at the Rust boundary)
//! conv2d_1(8, k=2x8, valid)  + ReLU  -> [N, 8,  42, 225]
//! max_pool(2x2, s=2x2)              -> [N, 8,  21, 112]
//! conv2d_2(32, k=2x4, valid) + ReLU  -> [N, 32, 20, 109]
//! max_pool(2x2, s=2x2)              -> [N, 32, 10,  54]
//! conv2d_3(32, k=2x4, valid) + ReLU  -> [N, 32,  9,  51]
//! max_pool(2x2, s=2x2)              -> [N, 32,  4,  25]
//! conv2d_4(32, k=2x4, valid) + ReLU  -> [N, 32,  3,  22]
//! max_pool(2x2, s=1x2)              -> [N, 32,  2,  11]
//! flatten                            -> [N, 704]
//! dense_1(2000) + ReLU               -> [N, 2000]
//! NewHeadDense(n_classes)            -> [N, n_classes]  (no activation here;
//!                                         softmax applied externally so we
//!                                         have pre-softmax logits for loss)
//! ```
//!
//! # Scope
//!
//! This module owns the network definition AND its Burn
//! `.mpk` mapping: [`Backbone::load_mpk`] and
//! [`Head::load_mpk`] read; head writes go through
//! `save_mpk_atomic` (only the converter writes).  Other I/O
//! lives elsewhere: WAV + sinc resample in
//! [`crate::preproc::wav_io`]; NumPy reading is a test-only
//! `#[cfg(test)]` helper.
//!
//! Inference-side scaffolding (slice-shaped `infer`, RKNN
//! adapters, hot-path kernels) lives in
//! [`crate::inference`]; training's batched
//! feature-extraction + autodiff loop lives in
//! [`crate::training`].

use crate::common::dims::BackboneFeatureDim;
use crate::common::head_header::write_with_payload;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::nn::{Linear, LinearConfig, Relu};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkBytesRecorder, NamedMpkFileRecorder, Recorder};
use std::io::Write;
use std::path::Path;
use thiserror::Error;

/// Errors from the Burn `.mpk` to in-memory mapping
/// helpers.  Burn's recorder errors don't compose cleanly
/// with `thiserror`'s `#[source]` (the type isn't always
/// `'static + Send + Sync`); we render to a string at
/// construction so the operator-facing message is preserved
/// without dragging Burn's domain error into every
/// consumer's error tree.
#[derive(Debug, Error)]
pub enum Error {
    #[error("load .mpk {path}: {message}")]
    Load { path: String, message: String },
    #[error("save .mpk {path}: {message}")]
    Save { path: String, message: String },
    #[error("invalid head: n_classes = {got} (must be in 1..={max})")]
    BadClassCount { got: usize, max: usize },
}

/// Shorthand for `Error::Load { path: path.to_string(), message: msg.into() }`.
/// `path` is `impl Display` so callers can pass either a `&Path::display()`
/// adapter or a literal sentinel like `"<bytes>"` (used by
/// [`Head::load_mpk_bytes`]).
fn load_err(path: impl std::fmt::Display, message: impl Into<String>) -> Error {
    Error::Load {
        path: path.to_string(),
        message: message.into(),
    }
}

/// Shorthand for `Error::Save { path: path.to_string(), message: msg.into() }`.
/// Same `Display` shape as [`load_err`]; the seven call sites in
/// [`Head::save_mpk_atomic`] use it with `final_path.display()` or
/// `parent.display()` depending on which step failed.
fn save_err(path: impl std::fmt::Display, message: impl Into<String>) -> Error {
    Error::Save {
        path: path.to_string(),
        message: message.into(),
    }
}

/// Re-export of [`crate::common::dims::MAX_N_CLASSES`] so the
/// cold-path validator in [`Head::try_new`] reads the same
/// ceiling as `inference::head::HeadInner::validate` (which also
/// re-exports the central constant).  Drift is impossible by
/// construction; the rationale (and the ~800 MB headroom
/// derivation) lives at the canonical site.
pub use crate::common::dims::MAX_N_CLASSES;

/// Single recorder factory; every load and save below
/// routes through the same [`FullPrecisionSettings`] pin so
/// consumers cannot pick a drifting variant.  The recorder
/// itself is backend-agnostic; the backend `B` shows up in
/// the record type at the call site.
#[inline]
fn recorder() -> NamedMpkFileRecorder<FullPrecisionSettings> {
    NamedMpkFileRecorder::<FullPrecisionSettings>::new()
}

/// Frozen embedding backbone: 4 conv2d + maxpool stages
/// followed by a 2000-dim ReLU dense projection.  Outputs
/// the feature vector consumed by [`Head`].
#[derive(Module, Debug)]
pub struct Backbone<B: Backend> {
    pub conv1: Conv2d<B>,
    pub pool1: MaxPool2d,
    pub conv2: Conv2d<B>,
    pub pool2: MaxPool2d,
    pub conv3: Conv2d<B>,
    pub pool3: MaxPool2d,
    pub conv4: Conv2d<B>,
    pub pool4: MaxPool2d,
    pub dense1: Linear<B>,
    pub relu: Relu,
}

impl<B: Backend> Backbone<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            conv1: Conv2dConfig::new([1, 8], [2, 8]).init(device),
            pool1: MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),
            conv2: Conv2dConfig::new([8, 32], [2, 4]).init(device),
            pool2: MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),
            conv3: Conv2dConfig::new([32, 32], [2, 4]).init(device),
            pool3: MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),
            conv4: Conv2dConfig::new([32, 32], [2, 4]).init(device),
            pool4: MaxPool2dConfig::new([2, 2]).with_strides([1, 2]).init(),
            dense1: LinearConfig::new(704, BackboneFeatureDim::USIZE).init(device),
            relu: Relu::new(),
        }
    }

    /// Forward pass.  Input `[N, 1, 43, 232]` NCHW.  Output:
    /// 2000-dim ReLU'd features.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.relu.forward(self.conv1.forward(x));
        let x = self.pool1.forward(x);
        let x = self.relu.forward(self.conv2.forward(x));
        let x = self.pool2.forward(x);
        let x = self.relu.forward(self.conv3.forward(x));
        let x = self.pool3.forward(x);
        let x = self.relu.forward(self.conv4.forward(x));
        let x = self.pool4.forward(x);
        // [N, 32, 2, 11] -> Keras-order flatten
        // (channels_last): NCHW -> NHWC -> [N, 704].
        let [n, c, h, w] = x.dims();
        debug_assert_eq!([c, h, w], [32, 2, 11]);
        let x = x.permute([0, 2, 3, 1]).reshape([n, h * w * c]);
        self.relu.forward(self.dense1.forward(x))
    }

    /// Load backbone weights from a Burn `.mpk` file.
    /// Equivalent to
    /// `Backbone::new(device).load_record(recorder.load(path))`,
    /// folded so callers don't need the recorder API
    /// surface.
    ///
    /// Blocking: file I/O.  Call from
    /// [`tokio::task::spawn_blocking`] or a non-async
    /// context.
    pub fn load_mpk(path: &Path, device: &B::Device) -> Result<Self, Error> {
        let record: BackboneRecord<B> = recorder()
            .load(path.to_path_buf(), device)
            .map_err(|e| load_err(path.display(), format!("{e}")))?;
        Ok(Self::new(device).load_record(record))
    }
}

/// Hot-swappable classifier head: a single
/// `Linear[BackboneFeatureDim, n_classes]` layer producing
/// pre-softmax logits.
#[derive(Module, Debug)]
pub struct Head<B: Backend> {
    pub linear: Linear<B>,
}

impl<B: Backend> Head<B> {
    /// Trusted-input constructor.  Panics inside Burn's
    /// `LinearConfig::init` if `n_classes` is `0` (zero
    /// out-features triggers a backend allocation panic).
    /// Prefer [`Self::try_new`] for any caller fed by config,
    /// converter manifests, or training inputs that aren't
    /// already validated.
    pub fn new(n_classes: usize, device: &B::Device) -> Self {
        Self {
            linear: LinearConfig::new(BackboneFeatureDim::USIZE, n_classes).init(device),
        }
    }

    /// Validating constructor: rejects `n_classes == 0` and
    /// `n_classes > MAX_N_CLASSES` before reaching Burn's
    /// allocator.  Use this for any caller wiring untrusted or
    /// config-driven class counts (training, converter,
    /// inference cold path); reserve [`Self::new`] for trusted
    /// internal/test use.
    pub fn try_new(n_classes: usize, device: &B::Device) -> Result<Self, Error> {
        if n_classes == 0 || n_classes > MAX_N_CLASSES {
            return Err(Error::BadClassCount {
                got: n_classes,
                max: MAX_N_CLASSES,
            });
        }
        Ok(Self::new(n_classes, device))
    }

    /// Pre-softmax logits; softmax applied externally.
    pub fn forward(&self, feat: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(feat)
    }

    /// Load head weights from a Burn `.mpk` file.  The
    /// file's recorded `n_classes` overrides whatever
    /// placeholder shape the `Head::new(1, ...)`
    /// initializer started with: `load_record` swaps the
    /// entire `Linear` parameter wholesale.  Callers that
    /// require a specific class count must verify it
    /// themselves (e.g. `head.linear.weight.val().dims()[1]`).
    ///
    /// Blocking: file I/O.  Call from
    /// [`tokio::task::spawn_blocking`] or a non-async
    /// context.
    pub fn load_mpk(path: &Path, device: &B::Device) -> Result<Self, Error> {
        let record: HeadRecord<B> = recorder()
            .load(path.to_path_buf(), device)
            .map_err(|e| load_err(path.display(), format!("{e}")))?;
        // The placeholder n_classes (1) is overwritten by load_record.
        Ok(Self::new(1, device).load_record(record))
    }

    /// In-memory variant of [`Self::load_mpk`]: deserialise
    /// the Named-MessagePack payload bytes directly through
    /// Burn's [`burn::record::NamedMpkBytesRecorder`],
    /// skipping the round-trip through the filesystem.
    /// Used by `inference::head::load_inner`, which already
    /// has the prost payload sitting in a `Vec<u8>` after
    /// stripping the `ACSTHEAD` header.
    ///
    /// `payload` is consumed because Burn's `Recorder::load`
    /// is `&mut LoadArgs` (the recorder may write through
    /// the buffer during decode); callers that need the
    /// original bytes should `payload.clone()` before the
    /// call.
    ///
    /// The recorder format MUST match `Self::save_mpk` and
    /// `Self::load_mpk` (Named MessagePack with
    /// [`FullPrecisionSettings`]); see the inline `recorder()`
    /// factory above for the file-recorder counterpart.
    pub fn load_mpk_bytes(payload: Vec<u8>, device: &B::Device) -> Result<Self, Error> {
        use burn::record::NamedMpkBytesRecorder;
        let recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::new();
        let record: HeadRecord<B> = recorder
            .load(payload, device)
            .map_err(|e| load_err("<bytes>", format!("{e}")))?;
        // Same placeholder pattern as [`Self::load_mpk`]:
        // `load_record` overwrites `Linear`'s shape from
        // the recorded weights.
        Ok(Self::new(1, device).load_record(record))
    }

    /// Persist head weights to a raw Burn `.mpk` file (no
    /// `ACSTHEAD` wrapper).  Crate-private because the only
    /// permitted readers are `Head::load_mpk` /
    /// `Head::load_mpk_bytes` themselves: the workspace's
    /// published head artifacts go through
    /// [`Self::save_mpk_atomic`], which prepends the 32-byte
    /// header that `inference::head::load_inner` validates.
    ///
    /// Used internally by `training::finetune::train_head` for
    /// best-epoch snapshots that live in a private tempdir and
    /// are reloaded via `Head::load_mpk` before publication, and
    /// by `model`'s own round-trip tests; do NOT reach for this
    /// from new code -- prefer `save_mpk_atomic`.
    ///
    /// Blocking: file I/O.  Call from
    /// [`tokio::task::spawn_blocking`] or a non-async context.
    pub(crate) fn save_mpk(self, path: &Path) -> Result<(), Error> {
        recorder()
            .record(self.into_record(), path.to_path_buf())
            .map_err(|e| save_err(path.display(), format!("{e}")))
    }

    /// Persist head weights to `final_path` as the workspace's
    /// `ACSTHEAD`-wrapped Burn `.mpk` artifact -- i.e. byte-for-byte
    /// the format that `inference::head::load_inner` reads and that
    /// `converter::convert_tfjs` writes via `FsService::put_atomic`.
    /// Consumes `self` because Burn's `into_record` is owning;
    /// callers that need to keep using the head should `clone()`
    /// before calling.
    ///
    /// On-disk shape: 32-byte header (magic + feature_dim +
    /// n_classes + payload_len + CRC32) followed by the
    /// `NamedMpkBytesRecorder` payload.  The `head_id` lives in
    /// caller-side metadata (e.g. `WorkspaceMgr`'s
    /// `metadata.json`); a future header v2 may stamp it inline,
    /// at which point the signature gains a `head_id` parameter.
    ///
    /// # Atomicity
    ///
    /// Mirrors `file_mgr::fs_atomic::put_atomic`:
    ///
    /// 1. Stage the full blob in a tempfile under
    ///    `final_path.parent()` (intra-FS guarantee for the rename).
    /// 2. `sync_all` the tempfile so its data reaches stable storage
    ///    before the rename publishes the new name.
    /// 3. `tempfile::NamedTempFile::persist` -- atomic POSIX rename.
    /// 4. fsync the parent directory so the rename's directory-entry
    ///    update is itself durable.
    ///
    /// On any failure `final_path` is unchanged: the tempfile is
    /// dropped without persisting and no partial bytes appear
    /// under the final name.  Inlined (not delegated to
    /// `file_mgr::fs_atomic::put_atomic`) because `model` is a
    /// lower layer than `file_mgr`.
    ///
    /// Blocking: file I/O + fsyncs.  Call from
    /// [`tokio::task::spawn_blocking`] or a non-async context.
    pub fn save_mpk_atomic(self, final_path: &Path) -> Result<(), Error> {
        // Stamp the real `n_classes` from the live tensor BEFORE
        // `into_record()` consumes `self`; matches converter's
        // header so training- and converter-published heads are
        // byte-identical at the header AND payload boundary.
        let n_classes = self.linear.weight.val().dims()[1] as u32;
        let recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::new();
        let payload = recorder
            .record(self.into_record(), ())
            .map_err(|e| save_err(final_path.display(), format!("{e}")))?;
        let header_blob = {
            let mut blob: Vec<u8> =
                Vec::with_capacity(crate::common::head_header::HEAD_HEADER_SIZE + payload.len());
            // `write_with_payload` stamps magic + version + CRC.
            write_with_payload(
                &mut blob,
                BackboneFeatureDim::USIZE as u32,
                n_classes,
                &payload,
            )
            .map_err(|e| save_err(final_path.display(), format!("compose ACSTHEAD: {e}")))?;
            blob
        };

        // Atomic write: tmp -> fsync(file) -> rename -> fsync(parent).
        let parent = final_path.parent().ok_or_else(|| {
            save_err(
                final_path.display(),
                "save_mpk_atomic: path has no parent directory",
            )
        })?;
        let mut tmp = tempfile::NamedTempFile::new_in(parent)
            .map_err(|e| save_err(parent.display(), format!("create tempfile: {e}")))?;
        tmp.write_all(&header_blob)
            .map_err(|e| save_err(final_path.display(), format!("write tempfile: {e}")))?;
        tmp.flush()
            .map_err(|e| save_err(final_path.display(), format!("flush tempfile: {e}")))?;
        // Durability barrier: the rename below can otherwise become
        // visible BEFORE the file's data reaches stable storage.
        tmp.as_file()
            .sync_all()
            .map_err(|e| save_err(final_path.display(), format!("fsync tempfile: {e}")))?;
        tmp.persist(final_path)
            .map_err(|e| save_err(final_path.display(), format!("persist (rename): {e}")))?;
        // fsync the parent dir so the rename's directory-entry update
        // also reaches stable storage; without this a power loss
        // after `persist` returns can revert the rename.
        std::fs::File::open(parent)
            .and_then(|f| f.sync_all())
            .map_err(|e| save_err(parent.display(), format!("fsync parent dir: {e}")))?;
        Ok(())
    }
}

/// Composed [`Backbone`] + [`Head`].  Used by training and
/// the parity reference path; the streaming inference
/// engine drives the two stages independently.
#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    pub backbone: Backbone<B>,
    pub head: Head<B>,
}

impl<B: Backend> Model<B> {
    pub fn new(n_classes: usize, device: &B::Device) -> Self {
        Self {
            backbone: Backbone::new(device),
            head: Head::new(n_classes, device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        self.head.forward(self.backbone.forward(x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestB = NdArray<f32>;

    #[test]
    fn forward_shape_smoke() {
        let device: burn::tensor::Device<TestB> = Default::default();
        let model = Model::<TestB>::new(3, &device);
        let x = Tensor::<TestB, 4>::zeros([2, 1, 43, 232], &device);
        let y = model.forward(x);
        assert_eq!(y.dims(), [2, 3]);
    }

    /// [`Head::save_mpk_atomic`] writes the workspace's
    /// `ACSTHEAD`-wrapped artifact: 32-byte header + Burn
    /// `NamedMpkBytesRecorder` payload.  Verify the on-disk
    /// shape (header magic + CRC) and that the payload bytes
    /// round-trip through [`Head::load_mpk_bytes`] back to the
    /// original weights -- the same path
    /// `inference::head::load_inner` exercises in production.
    ///
    /// Cheap (Head is one `Linear[2000, n]` = ~272 KB at
    /// n=34), so it runs in routine CI.
    #[test]
    fn head_save_atomic_round_trip() {
        use crate::common::head_header::{HEAD_HEADER_SIZE, parse_header};

        let device: burn::tensor::Device<TestB> = Default::default();
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("rt_head.mpk");

        // Build a head with a non-default `n_classes` so the
        // load path can't trivially "succeed" by leaving an
        // n=1 placeholder.
        const N: usize = 7;
        let saved = Head::<TestB>::new(N, &device);
        let saved_w = saved.linear.weight.val();
        let saved_dims = saved_w.dims();
        let saved_data: Vec<f32> = saved_w.into_data().to_vec().expect("to_vec");
        // `save_mpk_atomic` consumes the original head; we
        // snapshotted its weights above for the comparison.
        saved.save_mpk_atomic(&path).expect("save_mpk_atomic");

        // 1. The on-disk file MUST begin with a valid ACSTHEAD
        //    header (magic + matching feature_dim + valid CRC).
        let bytes = std::fs::read(&path).expect("read saved");
        assert!(
            bytes.len() >= HEAD_HEADER_SIZE,
            "file too short for header: {} bytes",
            bytes.len()
        );
        let header = parse_header(&bytes[..HEAD_HEADER_SIZE]).expect("parse header");
        assert_eq!(
            header.feature_dim as usize,
            BackboneFeatureDim::USIZE,
            "header feature_dim mismatch",
        );
        assert_eq!(
            header.num_classes as usize, N,
            "header num_classes must self-describe",
        );
        assert_eq!(
            bytes.len() - HEAD_HEADER_SIZE,
            header.payload_len as usize,
            "header.payload_len disagrees with file tail length",
        );

        // 2. The payload tail MUST decode through Burn's
        //    bytes-recorder back to the original weights.  This
        //    is the path `inference::head::load_inner` walks.
        let payload = bytes[HEAD_HEADER_SIZE..].to_vec();
        let loaded = Head::<TestB>::load_mpk_bytes(payload, &device).expect("load_mpk_bytes");
        let loaded_w = loaded.linear.weight.val();
        let loaded_dims = loaded_w.dims();
        let loaded_data: Vec<f32> = loaded_w.into_data().to_vec().expect("to_vec");

        assert_eq!(saved_dims, loaded_dims, "shape drift across round-trip");
        assert_eq!(
            loaded_dims,
            [BackboneFeatureDim::USIZE, N],
            "loaded n_classes != saved n_classes"
        );
        assert_eq!(
            saved_data.len(),
            loaded_data.len(),
            "weight buffer length drift",
        );
        for (i, (a, b)) in saved_data.iter().zip(loaded_data.iter()).enumerate() {
            assert!(
                (a - b).abs() < f32::EPSILON,
                "weight drift at idx {i}: saved={a}, loaded={b}",
            );
        }
    }

    /// `Head::try_new` rejects `n_classes = 0` and
    /// `n_classes > MAX_N_CLASSES` before reaching Burn's
    /// allocator.  Mirrors the inference cold-path validator
    /// so the cap is enforced at every Head construction site
    /// the converter / training / API can reach.
    #[test]
    fn head_try_new_rejects_pathological_class_counts() {
        let device: burn::tensor::Device<TestB> = Default::default();

        let err_zero = Head::<TestB>::try_new(0, &device).expect_err("n=0 must reject");
        assert!(
            matches!(
                err_zero,
                Error::BadClassCount {
                    got: 0,
                    max: MAX_N_CLASSES
                }
            ),
            "expected BadClassCount, got {err_zero:?}",
        );

        let err_huge =
            Head::<TestB>::try_new(MAX_N_CLASSES + 1, &device).expect_err("n>MAX must reject");
        assert!(
            matches!(
                err_huge,
                Error::BadClassCount { got, max }
                    if got == MAX_N_CLASSES + 1 && max == MAX_N_CLASSES
            ),
            "expected BadClassCount, got {err_huge:?}",
        );

        // A small in-bounds n must succeed and produce the
        // same Linear shape as `Head::new(n, ...)`.
        let h = Head::<TestB>::try_new(3, &device).expect("n=3 must accept");
        let dims = h.linear.weight.val().dims();
        assert_eq!(dims, [BackboneFeatureDim::USIZE, 3]);
    }

    /// [`Head::load_mpk`] on a non-existent path returns
    /// [`Error::Load`], not a panic, with the
    /// operator-facing path embedded.
    #[test]
    fn head_load_mpk_missing_file_returns_err() {
        let device: burn::tensor::Device<TestB> = Default::default();
        let bad = std::path::Path::new("/nonexistent/.acoustics_lab/missing-head.mpk");
        let err = Head::<TestB>::load_mpk(bad, &device).expect_err("must fail");
        match err {
            Error::Load { path, .. } => {
                assert!(
                    path.contains("missing-head"),
                    "diagnostic should name the missing path: {path}",
                );
            }
            Error::Save { .. } => panic!("got Save variant on a load error"),
            Error::BadClassCount { .. } => panic!("got BadClassCount variant on a load error"),
        }
    }
}
