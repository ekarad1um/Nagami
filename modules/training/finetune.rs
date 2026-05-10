//! Library API for on-device fine-tuning of the classifier head.
//!
//! This is the extracted form of the original `finetune` CLI algorithm:
//! scan a Speech-Commands-style dataset, compute frozen-backbone
//! features once, train only the classifier head, then save `head.mpk`
//! plus sibling `labels.txt`.

use crate::common::dims::{BACKBONE_FEATURE_DIM as FEATURE_DIM, NBins, NFrames};
use crate::common::ids::HeadId;
use crate::model::{Backbone, Head};
use crate::preproc::Preproc;
use crate::preproc::wav_io::{self, ResamplerCache};
use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{GradientsParams, Optimizer, SgdConfig};
use burn::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use thiserror::Error;

const DEFAULT_PROGRESS_EVERY: usize = 500;

static RAYON_POOL_INIT: OnceLock<()> = OnceLock::new();

type InnerB = NdArray<f32>;
type AutoB = Autodiff<InnerB>;
type Example = (PathBuf, usize);
type DatasetScan = (Vec<String>, Vec<Example>);
/// Boxed row-major spectrogram, the shape `Preproc::spectrogram`
/// returns.  Aliased so the rayon collect signature stays readable.
type Spectrogram = Box<[[f32; NBins::USIZE]; NFrames::USIZE]>;

/// Configuration for a single fine-tune run.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FinetuneConfig {
    pub data: PathBuf,
    pub backbone: PathBuf,
    pub init_head: Option<PathBuf>,
    pub out: PathBuf,
    pub epochs: usize,
    pub batch: usize,
    pub lr: f32,
    pub val_split: f32,
    pub seed: u64,
}

impl FinetuneConfig {
    /// Lexical validation: epochs/batch >= 1, lr finite and > 0,
    /// `val_split` in `[0, 1)`.  Filesystem existence is not checked.
    pub fn validate(&self) -> Result<(), FinetuneError> {
        if self.epochs == 0 {
            return Err(FinetuneError::InvalidConfig("epochs must be >= 1".into()));
        }
        if self.batch == 0 {
            return Err(FinetuneError::InvalidConfig("batch must be >= 1".into()));
        }
        if !(self.lr.is_finite() && self.lr > 0.0) {
            return Err(FinetuneError::InvalidConfig(format!(
                "lr must be finite and > 0; got {}",
                self.lr
            )));
        }
        if !(self.val_split.is_finite() && self.val_split >= 0.0 && self.val_split < 1.0) {
            return Err(FinetuneError::InvalidConfig(format!(
                "val_split must be finite and in [0, 1); got {}",
                self.val_split
            )));
        }
        Ok(())
    }
}

/// Coarse-grained phase for user-facing progress.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Loading,
    FeatureExtract,
    Train,
    Saving,
    Done,
}

/// Per-epoch metrics attached to `Phase::Train` progress events.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct EpochMetrics {
    pub epoch: usize,
    pub epochs: usize,
    pub train_loss: f64,
    pub train_acc: f32,
    pub val_acc: f32,
    pub best_val_acc: f32,
}

/// Training progress snapshot.  Designed to be forwarded through a
/// `tokio::sync::watch` channel by the daemon-side training crate.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Progress {
    pub phase: Phase,
    pub current: usize,
    pub total: usize,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics: Option<EpochMetrics>,
}

impl Progress {
    fn new(phase: Phase, current: usize, total: usize, message: impl Into<String>) -> Self {
        Self {
            phase,
            current,
            total,
            message: message.into(),
            metrics: None,
        }
    }

    fn with_metrics(message: impl Into<String>, metrics: EpochMetrics) -> Self {
        Self {
            phase: Phase::Train,
            current: metrics.epoch,
            total: metrics.epochs,
            message: message.into(),
            metrics: Some(metrics),
        }
    }
}

/// Final result of one successful fine-tune run.
///
/// `final_train_acc` and `final_val_acc` describe the
/// *published* head -- the head with the highest observed
/// `val_acc` across all epochs.  When `val_split == 0.0`
/// the validation set is empty, `best_val` never updates,
/// and the published head falls back to the last-epoch
/// head; the metrics then describe that head.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FinetuneOutput {
    pub head_mpk: PathBuf,
    pub labels_txt: PathBuf,
    pub final_train_acc: f32,
    pub final_val_acc: f32,
    pub classes: Vec<String>,
    /// Identifier minted at `save_mpk_atomic` time and surfaced
    /// to API clients via `TrainingResult::head_id` so callers
    /// don't have to invent one after a successful job.  Mirrors
    /// the converter's `head_id` response symmetry.
    pub head_id: HeadId,
}

/// Failure shapes from the head fine-tune algorithm.
#[derive(Debug, Error)]
pub enum FinetuneError {
    #[error("invalid config: {0}")]
    InvalidConfig(String),
    #[error("cancelled")]
    Cancelled,
    #[error("io {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    /// Burn `.mpk` load/save error from `model`'s helpers.  The
    /// `Display` impl of `crate::model::Error` already includes the path
    /// and the underlying message; no extra wrapping needed.
    #[error(transparent)]
    Model(#[from] crate::model::Error),
    #[error("training panicked: {0}")]
    Panic(String),
    /// Dataset shape rejection at scan time.  Emitted for: no
    /// class folders under `<workspace>/datasets/`; duplicate
    /// class labels under ASCII case-insensitive comparison;
    /// unreadable class directories; empty class folders (no
    /// non-hidden regular sample files); non-hidden non-directory
    /// root entries (stray files / symlinks / devices).  400
    /// because the operator can fix the upload layout.
    #[error("bad dataset {path}: {reason}")]
    BadDataset {
        /// Path under `<workspace>/datasets/` where the rejection
        /// was detected (the dataset root for "no classes" /
        /// "duplicate label"; the offending child for
        /// "empty class" / "stray file").
        path: String,
        /// Operator-readable diagnostic.
        reason: String,
    },
    /// A previously-discovered dataset file disappeared or became
    /// unreadable during the scan / extract pass.  Uploads and
    /// deletes are allowed during training; mid-walk read failures
    /// surface here so the outer `TrainingError::DatasetRead`
    /// propagates unchanged.  500 because the operator can't
    /// recover the in-flight job.
    #[error("dataset read failure {path}: {reason}")]
    DatasetRead {
        /// Path the scan / extract pass tried to read.
        path: String,
        /// Operator-readable diagnostic.
        reason: String,
    },
    /// One or more classes have zero usable examples after dataset
    /// scan.  The `class` is the first offender; `per_class_kept` is
    /// the post-scan kept count per class label so the operator can
    /// see which directories are starved.  Distinct from
    /// [`Self::EmptyClassAfterExtract`] -- this fires before any
    /// preproc work, so the cause is "the directory has no `.wav`
    /// files at all", not "every wav failed to decode".
    ///
    /// [`Self::BadDataset`] is the canonical emission for this
    /// scenario; this variant survives as a belt-and-suspenders
    /// catch in the post-scan accounting path.
    #[error(
        "class {class:?} has no usable .wav examples in the dataset \
         (per-class kept counts: {per_class_kept:?})"
    )]
    EmptyClassAfterScan {
        class: String,
        per_class_kept: Vec<(String, usize)>,
    },
    /// One or more classes lost every example to preproc failures
    /// (decode/resample/non-finite spectrogram).  `class` is the
    /// first offender; the kept/dropped counts give the operator the
    /// shape of the loss without pulling individual file paths into
    /// the error.
    #[error(
        "class {class:?} lost every example to preproc failures \
         (per-class kept: {per_class_kept:?}; per-class dropped: {per_class_dropped:?})"
    )]
    EmptyClassAfterExtract {
        class: String,
        per_class_kept: Vec<(String, usize)>,
        per_class_dropped: Vec<(String, usize)>,
    },
    /// Aggregate preproc drop ratio crossed
    /// [`MAX_DROP_RATIO`].  Fails the job rather than silently
    /// training on a deeply degraded subset; per-class counts
    /// surface in the error so the operator can see which class is
    /// degraded without grep-ing the daemon log.
    #[error(
        "preproc drop ratio {dropped}/{total} = {ratio:.3} exceeds \
         max {max_ratio:.3} (per-class kept: {per_class_kept:?}; \
         per-class dropped: {per_class_dropped:?})"
    )]
    DropRatioExceeded {
        dropped: usize,
        total: usize,
        ratio: f32,
        max_ratio: f32,
        per_class_kept: Vec<(String, usize)>,
        per_class_dropped: Vec<(String, usize)>,
    },
    /// Stratified train/val split could not give every class a
    /// non-empty training partition (e.g. a single-example class
    /// landed entirely in val).  `class` is the first offender;
    /// the kept counts surface to the operator so they can see
    /// which class is undersized.
    #[error(
        "stratified split would leave class {class:?} with zero \
         training examples (kept per class: {per_class_kept:?}, \
         val_split={val_split})"
    )]
    StratifiedSplitImpossible {
        class: String,
        per_class_kept: Vec<(String, usize)>,
        val_split: f32,
    },
}

impl crate::common::error::Categorized for FinetuneError {
    fn kind(&self) -> crate::common::error::ErrorKind {
        use crate::common::error::ErrorKind::*;
        match self {
            FinetuneError::InvalidConfig(_) => UserInput,
            // Cancellation is operator-driven; mirror
            // TrainingError::Cancelled.
            FinetuneError::Cancelled => Conflict,
            // Dataset-quality rejections originate in the operator's
            // upload contents.  Surface as 4xx so the operator
            // sees an actionable diagnostic rather than a generic
            // 500.  Same taxonomy as `InvalidConfig`.
            FinetuneError::BadDataset { .. }
            | FinetuneError::EmptyClassAfterScan { .. }
            | FinetuneError::EmptyClassAfterExtract { .. }
            | FinetuneError::DropRatioExceeded { .. }
            | FinetuneError::StratifiedSplitImpossible { .. } => UserInput,
            // Mid-walk dataset read failures are daemon-internal
            // because `datasets/` is daemon-owned; a missing or
            // unreadable file mid-job indicates external tamper
            // or storage fault that the operator cannot recover
            // via request retry.
            FinetuneError::DatasetRead { .. } => Internal,
            // Everything else is daemon-internal.
            FinetuneError::Io { .. } | FinetuneError::Model(_) | FinetuneError::Panic(_) => {
                Internal
            }
        }
    }
}

/// Shorthand for `FinetuneError::Io { path: path.to_string(), source }`.
/// `path` is `impl Display` so call sites can pass `Path::display()`
/// adapters directly without per-site `to_string()` boilerplate.
fn finetune_io_err(path: impl std::fmt::Display, source: std::io::Error) -> FinetuneError {
    FinetuneError::Io {
        path: path.to_string(),
        source,
    }
}

/// Hard ceiling on the fraction of dataset examples
/// allowed to drop during preproc (decode/resample/non-finite
/// spectrogram).  10 % is the operator-visible "your dataset is
/// degraded" line: above it the trained head's metrics no longer
/// describe the dataset the operator submitted, so the job fails
/// loudly instead of completing on a tiny survivor subset.  Tied
/// to the on-device contract; not currently a config knob (the
/// `simplify` review can promote it later if a use case appears).
pub const MAX_DROP_RATIO: f32 = 0.10;

/// Run a complete fine-tune job.
///
/// `progress` is called synchronously from the training thread.  During
/// feature extraction it may be called by rayon workers, so the callback
/// must be `Sync`.
///
/// `cancel` is polled between expensive units of work.  It intentionally
/// cannot preempt a single wav/resample/spectrogram operation.
pub fn run(
    cfg: &FinetuneConfig,
    progress: &(dyn Fn(&Progress) + Sync),
    cancel: &(dyn Fn() -> bool + Sync),
) -> Result<FinetuneOutput, FinetuneError> {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_inner(cfg, progress, cancel)
    }));
    match result {
        Ok(r) => r,
        Err(payload) => Err(FinetuneError::Panic(panic_payload_to_string(payload))),
    }
}

fn run_inner(
    cfg: &FinetuneConfig,
    progress: &(dyn Fn(&Progress) + Sync),
    cancel: &(dyn Fn() -> bool + Sync),
) -> Result<FinetuneOutput, FinetuneError> {
    cfg.validate()?;
    install_rayon_pool_cap();
    check_cancel(cancel)?;

    let device_inner: burn::tensor::Device<InnerB> = Default::default();
    let device_auto: burn::tensor::Device<AutoB> = Default::default();

    // Seed the backend RNG so `Head::<AutoB>::new`'s
    // weight initialisation (`LinearConfig::init`) draws
    // deterministically from `cfg.seed`.  Without this, two runs
    // with identical configs and seeds produced different trained
    // heads because Burn's global RNG was thread-local default.
    // The shuffle path already used `cfg.seed` directly; this
    // closes the other half of the determinism contract.  Both
    // backends share `cfg.seed` (`AutoB = Autodiff<InnerB>` --
    // their backend RNG is the same `NdArray` global).
    <InnerB as Backend>::seed(&device_inner, cfg.seed);
    <AutoB as Backend>::seed(&device_auto, cfg.seed);

    progress(&Progress::new(
        Phase::Loading,
        0,
        0,
        format!("scan dataset: {}", cfg.data.display()),
    ));
    let (classes, examples) = scan_dataset(&cfg.data)?;
    let n_classes = classes.len();
    if n_classes < 2 {
        return Err(FinetuneError::InvalidConfig(format!(
            "need at least 2 classes; found {n_classes} in {}",
            cfg.data.display()
        )));
    }
    if examples.is_empty() {
        return Err(FinetuneError::InvalidConfig(format!(
            "dataset {} contains no .wav examples",
            cfg.data.display()
        )));
    }
    // Reject post-scan empty classes BEFORE
    // any feature extraction or optimizer step.  `scan_dataset`
    // accepts every immediate subdirectory as a class; one with
    // zero `.wav` files would otherwise be encoded as a label
    // with no training examples and silently degrade the
    // published head.
    let pre_scan_counts = per_class_counts_from_examples(&classes, &examples);
    if let Some((idx, _)) = pre_scan_counts.iter().enumerate().find(|(_, n)| **n == 0) {
        return Err(FinetuneError::EmptyClassAfterScan {
            class: classes[idx].clone(),
            per_class_kept: classes
                .iter()
                .cloned()
                .zip(pre_scan_counts.iter().copied())
                .collect(),
        });
    }
    progress(&Progress::new(
        Phase::Loading,
        examples.len(),
        examples.len(),
        format!("classes: {:?}  total examples: {}", classes, examples.len()),
    ));
    check_cancel(cancel)?;

    progress(&Progress::new(
        Phase::Loading,
        0,
        1,
        format!("load backbone: {}", cfg.backbone.display()),
    ));
    let backbone = Backbone::<InnerB>::load_mpk(&cfg.backbone, &device_inner)?;
    check_cancel(cancel)?;

    progress(&Progress::new(
        Phase::FeatureExtract,
        0,
        examples.len(),
        "extract features...",
    ));
    let preproc = Preproc::new();
    let total_in = examples.len();
    let (feats, labels) = extract_features(&backbone, &preproc, &examples, progress, cancel)?;
    if feats.is_empty() {
        return Err(FinetuneError::InvalidConfig(
            "all examples were dropped during feature extraction".into(),
        ));
    }
    assert_eq!(feats.len(), labels.len());
    check_cancel(cancel)?;

    // Post-extract per-class accounting + drop-ratio gate.
    // Extracted into `validate_post_extract_quality` so the unit
    // tests can exercise the rejection paths without standing up
    // a full Backbone load + preproc pipeline.
    validate_post_extract_quality(&classes, &pre_scan_counts, n_classes, &labels, total_in)?;

    // Stratified split.  Replaces the prior
    // global random shuffle that could leave one or more classes
    // entirely absent from the training or validation partition
    // (small / imbalanced datasets).  Deterministic via
    // `cfg.seed` so test runs reproduce.
    let split =
        stratified_split_indices(&labels, n_classes, cfg.val_split, cfg.seed, Some(&classes))?;
    let train_idx = split.train;
    let val_idx = split.val;
    if train_idx.is_empty() {
        // Defensive: `stratified_split_indices` already errors
        // when any class would land with zero training examples.
        return Err(FinetuneError::InvalidConfig(format!(
            "empty training split after val_split={} over {} examples",
            cfg.val_split,
            labels.len(),
        )));
    }
    progress(&Progress::new(
        Phase::Train,
        0,
        cfg.epochs,
        format!(
            "split: train={} val={} (stratified per-class)",
            train_idx.len(),
            val_idx.len()
        ),
    ));

    let head_auto: Head<AutoB> = if let Some(p) = &cfg.init_head {
        // The .mpk file's recorded shape overrides the placeholder
        // from `Head::new` via `load_record`.  If the operator pointed
        // `init_head` at a head trained for a different class count,
        // the mismatch would silently propagate and the loss would
        // compute against the wrong target dimension; reject early
        // with an actionable message instead.
        let loaded = Head::<AutoB>::load_mpk(p, &device_auto)?;
        let loaded_n = loaded.linear.weight.val().dims()[1];
        if loaded_n != n_classes {
            return Err(FinetuneError::InvalidConfig(format!(
                "init_head {} has {loaded_n} classes; dataset has {n_classes}",
                p.display()
            )));
        }
        loaded
    } else {
        Head::<AutoB>::new(n_classes, &device_auto)
    };

    let t_train = Instant::now();
    let train_data = TrainData {
        n_classes,
        feats: &feats,
        labels: &labels,
        train_idx: &train_idx,
        val_idx: &val_idx,
    };
    let train_settings = TrainSettings {
        epochs: cfg.epochs,
        batch: cfg.batch,
        lr: cfg.lr,
        seed: cfg.seed,
    };
    let head_auto = train_head(head_auto, train_data, train_settings, progress, cancel)?;
    progress(&Progress::new(
        Phase::Train,
        cfg.epochs,
        cfg.epochs,
        format!("train wall: {:.2?}", t_train.elapsed()),
    ));
    check_cancel(cancel)?;

    progress(&Progress::new(
        Phase::Saving,
        0,
        2,
        format!("save head: {}", cfg.out.display()),
    ));
    // Compute final metrics from the in-memory head first;
    // `save_mpk_atomic` consumes `head_inner` via Burn's owning
    // `into_record`, and the saved bytes are byte-for-byte
    // identical at the payload boundary (verified by
    // `crate::model::tests::head_save_atomic_round_trip`), so
    // re-loading from disk would only add I/O.
    let head_inner = head_auto.valid();
    let final_train = evaluate(&head_inner, n_classes, &feats, &labels, &train_idx);
    let final_val = evaluate(&head_inner, n_classes, &feats, &labels, &val_idx);

    if let Some(parent) = cfg.out.parent() {
        std::fs::create_dir_all(parent).map_err(|e| finetune_io_err(parent.display(), e))?;
    }
    // Mint a fresh `HeadId` for this trained artifact and surface
    // it via `FinetuneOutput` so API clients see the same shape
    // converter responses already carry.  Header v1 has no slot
    // for the id; today it lives in caller-side metadata only.
    let head_id = HeadId::new();

    // `with_file_name(format!("{stem}.labels.txt"))` rather than
    // `with_extension("labels.txt")`: the latter only replaces the
    // FINAL extension, so `cfg.out = "v1.2.mpk"` would yield
    // `v1.2.labels.txt` (silently dropping `.mpk`).  Compute the
    // stem-relative sibling name explicitly.
    let labels_path = {
        let stem = cfg
            .out
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "head".to_string());
        cfg.out.with_file_name(format!("{stem}.labels.txt"))
    };

    // Write labels FIRST, then the head.  Both go through
    // `fs_atomic::put_atomic` (tempfile + sync_all + rename + parent
    // fsync) so each is individually crash-consistent.  Order
    // matters: a crash between the two leaves an orphan
    // `*.labels.txt` next to no `head.mpk` -- downstream loaders
    // ignore the labels file in that case (no head to read).  The
    // reverse order would leave a head with no labels, which is the
    // load-bearing failure the converter goes out of its way to
    // avoid (see `converter::write_head_artifacts`'s metadata-last
    // ordering rationale).
    let labels_blob = format!("{}\n", classes.join("\n"));
    crate::file_mgr::fs_atomic::put_atomic(&labels_path, labels_blob.as_bytes()).map_err(|e| {
        finetune_io_err(labels_path.display(), std::io::Error::other(format!("{e}")))
    })?;
    head_inner.save_mpk_atomic(&cfg.out)?;

    let out = FinetuneOutput {
        head_mpk: cfg.out.clone(),
        labels_txt: labels_path,
        final_train_acc: final_train,
        final_val_acc: final_val,
        classes,
        head_id,
    };
    progress(&Progress::new(
        Phase::Done,
        cfg.epochs,
        cfg.epochs,
        format!(
            "final train_acc={:.4}  val_acc={:.4}",
            out.final_train_acc, out.final_val_acc
        ),
    ));
    Ok(out)
}

fn install_rayon_pool_cap() {
    RAYON_POOL_INIT.get_or_init(|| {
        // `available_parallelism` reflects the cgroup / cpuset
        // constraints the deployment may have applied (LXC,
        // Kubernetes, taskset); fall back to 1 if the platform
        // declines to answer (rare).  Reserve one core for the
        // mic / inference / tokio threads the daemon already
        // pins.
        let total = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1);
        let n = total.saturating_sub(1).max(1);
        if let Err(e) = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
        {
            // Another crate may have already installed the global pool.
            // Training remains correct; we just lose this daemon-friendly cap.
            eprintln!("finetune: rayon global pool already configured ({e}); using existing pool");
        }
    });
}

fn check_cancel(cancel: &(dyn Fn() -> bool + Sync)) -> Result<(), FinetuneError> {
    if cancel() {
        Err(FinetuneError::Cancelled)
    } else {
        Ok(())
    }
}

/// Test-only handle on the dataset scanner so the parent
/// training module can pin the class-walk behaviour
/// (`class_file_discovery_walks_two_levels`,
/// `lazy_fd_bounded_no_open_during_scan`) without standing up a
/// full backbone + preproc pipeline.  Mirrors the production
/// `scan_dataset` semantics exactly; it just unwraps to a
/// `(Vec<String>, Vec<(PathBuf, usize)>)` for ergonomic use.
#[cfg(test)]
pub(crate) fn scan_dataset_for_test(data_dir: &Path) -> (Vec<String>, Vec<(PathBuf, usize)>) {
    scan_dataset(data_dir).expect("scan_dataset")
}

/// Dataset discovery: walk `<workspace>/datasets/` and build the
/// `(classes, examples)` pair, enforcing:
///
/// - non-hidden direct child directories are class folders;
/// - hidden root entries (leading `.`) are ignored;
/// - non-hidden non-directory root entries (stray files / symlinks
///   / devices) are rejected as [`FinetuneError::BadDataset`];
/// - class directories must be readable -- a `read_dir` failure
///   surfaces as `BadDataset` with `reason = "unreadable"`;
/// - duplicate class labels under ASCII case-insensitive
///   comparison are rejected as `BadDataset`;
/// - empty class folders (no non-hidden regular `.wav` sample
///   files anywhere under the class subtree) are rejected as
///   `BadDataset`;
/// - the resulting class list is sorted by canonical byte order
///   (`Vec::sort` on the borrowed name) so the published head's
///   labels are deterministic across hosts.
///
/// The walk runs at job-spawn time; per-batch FD usage stays
/// bounded by the post-scan extract path which opens / reads /
/// closes per wav.
fn scan_dataset(data_dir: &Path) -> Result<DatasetScan, FinetuneError> {
    use std::collections::BTreeMap;

    let entries = std::fs::read_dir(data_dir).map_err(|source| FinetuneError::BadDataset {
        path: data_dir.display().to_string(),
        reason: format!("read datasets root: {source}"),
    })?;

    // Class folder discovery + duplicate-label gate.
    // `BTreeMap` keyed by the lowercase form so case-insensitive
    // duplicate detection happens at insert.  The value carries
    // the original-case name + the class directory path.
    let mut classes_by_lower: BTreeMap<String, (String, PathBuf)> = BTreeMap::new();
    for entry in entries {
        let entry = entry.map_err(|source| FinetuneError::BadDataset {
            path: data_dir.display().to_string(),
            reason: format!("read entry: {source}"),
        })?;
        let raw_name = entry.file_name();
        let name = raw_name.to_string_lossy();
        // Hidden root entries are silently skipped (matches the
        // upload-time `AssetPath` leading-dot rule).
        if name.starts_with('.') {
            continue;
        }
        let path = entry.path();
        // `symlink_metadata` does not follow links; symlinks are
        // rejected here even when the target is a regular dir.
        let md = std::fs::symlink_metadata(&path).map_err(|source| FinetuneError::BadDataset {
            path: path.display().to_string(),
            reason: format!("stat: {source}"),
        })?;
        if !md.is_dir() {
            return Err(FinetuneError::BadDataset {
                path: path.display().to_string(),
                reason: "non-hidden root entry is not a directory \
                        (only class folders may live under datasets/)"
                    .into(),
            });
        }
        let original = name.into_owned();
        let lower = original.to_ascii_lowercase();
        if let Some((existing_name, existing_path)) =
            classes_by_lower.insert(lower.clone(), (original.clone(), path.clone()))
        {
            return Err(FinetuneError::BadDataset {
                path: data_dir.display().to_string(),
                reason: format!(
                    "duplicate class label under ASCII case-insensitive comparison: \
                     {original:?} (at {}) and {existing_name:?} (at {})",
                    path.display(),
                    existing_path.display(),
                ),
            });
        }
    }

    if classes_by_lower.is_empty() {
        return Err(FinetuneError::BadDataset {
            path: data_dir.display().to_string(),
            reason: "no class folders under datasets/ \
                    (each non-hidden direct subdirectory is a class)"
                .into(),
        });
    }

    // Empty class folders surface as `BadDataset`; an unreadable
    // subdir surfaces the same way.  The class list ends up sorted
    // by canonical byte order of the original-case names so the
    // published head's label order is deterministic across hosts.
    // The BTreeMap iterates in lowercase order; for the ASCII-only
    // allowlist names match, but mixed case can differ, so we
    // re-sort by original-case below.
    let mut classes: Vec<String> = Vec::with_capacity(classes_by_lower.len());
    let mut by_class_idx: Vec<Vec<PathBuf>> = Vec::with_capacity(classes_by_lower.len());
    for (original, class_dir) in classes_by_lower.values() {
        let mut wavs = Vec::<PathBuf>::new();
        collect_wavs_recursive(class_dir, &mut wavs)?;
        if wavs.is_empty() {
            return Err(FinetuneError::BadDataset {
                path: class_dir.display().to_string(),
                reason: format!("class folder {original:?} has no non-hidden .wav sample files"),
            });
        }
        wavs.sort();
        classes.push(original.clone());
        by_class_idx.push(wavs);
    }

    // Re-sort by original-case byte order; reorder `by_class_idx`
    // in lockstep so each class's example list stays paired.
    let mut zipped: Vec<(String, Vec<PathBuf>)> = classes.into_iter().zip(by_class_idx).collect();
    zipped.sort_by(|a, b| a.0.cmp(&b.0));
    let classes: Vec<String> = zipped.iter().map(|(name, _)| name.clone()).collect();
    let mut examples: Vec<(PathBuf, usize)> = Vec::new();
    for (i, (_name, wavs)) in zipped.into_iter().enumerate() {
        for p in wavs {
            examples.push((p, i));
        }
    }
    Ok((classes, examples))
}

/// Recursively collect non-hidden `.wav` files.  Hidden subdirs and
/// files are skipped; non-`.wav` regular files are silently skipped
/// (the trainer's loader is the per-file admission gate).  An
/// unreadable subdir surfaces as [`FinetuneError::BadDataset`].
fn collect_wavs_recursive(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), FinetuneError> {
    let entries = std::fs::read_dir(dir).map_err(|source| FinetuneError::BadDataset {
        path: dir.display().to_string(),
        reason: format!("unreadable: {source}"),
    })?;
    for entry in entries {
        let entry = entry.map_err(|source| FinetuneError::BadDataset {
            path: dir.display().to_string(),
            reason: format!("read entry: {source}"),
        })?;
        let raw_name = entry.file_name();
        let name = raw_name.to_string_lossy();
        if name.starts_with('.') {
            continue;
        }
        let path = entry.path();
        let md = std::fs::symlink_metadata(&path).map_err(|source| FinetuneError::BadDataset {
            path: path.display().to_string(),
            reason: format!("stat: {source}"),
        })?;
        if md.is_dir() {
            // Recursion depth is bounded by `AssetPath::MAX_DEPTH`
            // at the upload boundary.
            collect_wavs_recursive(&path, out)?;
        } else if md.is_file() {
            if path.extension().is_some_and(|e| e == "wav") {
                out.push(path);
            }
            // Non-.wav files are skipped here; the trainer's loader
            // bumps a drop counter for unsupported encodings.
        } else {
            // Symlinks, devices, FIFOs reject explicitly.
            return Err(FinetuneError::BadDataset {
                path: path.display().to_string(),
                reason: "unsupported file type (only regular .wav files are accepted)".into(),
            });
        }
    }
    Ok(())
}

/// Count examples per class from the post-scan `examples` vec.
/// Returns a parallel `Vec<usize>` indexed the same way as
/// `classes` (which `scan_dataset` derives from
/// `BTreeMap::keys()`, i.e. lexicographic order).  Used by the
/// empty-class precondition check before any preproc work runs.
fn per_class_counts_from_examples(classes: &[String], examples: &[(PathBuf, usize)]) -> Vec<usize> {
    let mut counts = vec![0usize; classes.len()];
    for (_, label) in examples {
        if *label < counts.len() {
            counts[*label] += 1;
        }
    }
    counts
}

/// Count surviving examples per class from a flat
/// `labels` vec (`label = class index`).  Used after
/// `extract_features` to surface per-class drop totals to the
/// operator and to detect classes that lost every example.
fn per_class_counts_from_labels(n_classes: usize, labels: &[usize]) -> Vec<usize> {
    let mut counts = vec![0usize; n_classes];
    for &label in labels {
        if label < counts.len() {
            counts[label] += 1;
        }
    }
    counts
}

/// Run the post-extract dataset-quality gates: empty class
/// detection + aggregate drop-ratio cap.  Returns
/// `Err(EmptyClassAfterExtract)` when any class lost every
/// example to preproc, or `Err(DropRatioExceeded)` when the
/// global drop ratio crosses [`MAX_DROP_RATIO`].  Otherwise
/// returns `Ok(())` and the caller proceeds to splitting +
/// training.
///
/// Extracted from `run_inner` so unit tests can exercise the
/// rejection paths without standing up a real `Backbone` +
/// `Preproc` pipeline.  The signature takes the inputs the
/// caller already has on hand: the post-scan class names and
/// pre-scan counts, the post-extract `labels` (one entry per
/// surviving example), and `total_in` (the original example
/// count before preproc drops).
fn validate_post_extract_quality(
    classes: &[String],
    pre_scan_counts: &[usize],
    n_classes: usize,
    labels: &[usize],
    total_in: usize,
) -> Result<(), FinetuneError> {
    let post_extract_counts = per_class_counts_from_labels(n_classes, labels);
    let per_class_kept: Vec<(String, usize)> = classes
        .iter()
        .cloned()
        .zip(post_extract_counts.iter().copied())
        .collect();
    let per_class_dropped: Vec<(String, usize)> = classes
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let dropped = pre_scan_counts
                .get(i)
                .copied()
                .unwrap_or(0)
                .saturating_sub(post_extract_counts[i]);
            (c.clone(), dropped)
        })
        .collect();

    // Empty class after preproc: every wav for some class failed
    // to decode/resample/produced a non-finite spectrogram.
    if let Some((idx, _)) = post_extract_counts
        .iter()
        .enumerate()
        .find(|(_, n)| **n == 0)
    {
        return Err(FinetuneError::EmptyClassAfterExtract {
            class: classes[idx].clone(),
            per_class_kept,
            per_class_dropped,
        });
    }

    // Aggregate drop-ratio gate.  Surfaces datasets where
    // > MAX_DROP_RATIO of files silently failed preproc -- the
    // remaining survivors are no longer representative of the
    // archive the operator submitted.  `total_in` includes
    // dropped+kept so the ratio is the operator-meaningful
    // "fraction of submitted dataset that didn't make training".
    let total_dropped = total_in.saturating_sub(labels.len());
    if total_in > 0 {
        let ratio = total_dropped as f32 / total_in as f32;
        if ratio > MAX_DROP_RATIO {
            return Err(FinetuneError::DropRatioExceeded {
                dropped: total_dropped,
                total: total_in,
                ratio,
                max_ratio: MAX_DROP_RATIO,
                per_class_kept,
                per_class_dropped,
            });
        }
    }
    Ok(())
}

/// Stratified train/val split output.  `train` and `val` are
/// disjoint index lists into `feats`/`labels` covering every kept
/// example; both are pre-shuffled with the same deterministic
/// LCG seeded by `cfg.seed` so test runs reproduce.
#[derive(Debug)]
struct SplitIndices {
    train: Vec<usize>,
    val: Vec<usize>,
}

/// Stratified train/val split.  For each class:
///
/// 1. Gather the kept-example indices for that class.
/// 2. Deterministically shuffle them (`shuffle_in_place` with a
///    per-class seed derived from `seed` so two classes don't
///    receive the same permutation).
/// 3. Compute `n_val = round(class_n * val_split)`, clamped to
///    `[v_min, class_n - 1]` where `v_min = 1` when
///    `val_split > 0 && class_n >= 2` (every class is represented
///    in val when validation is enabled) and `v_min = 0` otherwise.
///    The upper clamp guarantees `class_n - n_val >= 1` so every
///    class lands at least one example in train.
/// 4. Prepend the per-class val/train indices to the global pools.
///
/// If a class's `class_n` is 1 and `val_split > 0`, the per-class
/// minimum cannot be satisfied without leaving the class out of
/// train; we error with [`FinetuneError::StratifiedSplitImpossible`]
/// so the operator sees the singleton class in the error rather
/// than discovering it via NaN val_acc later.
fn stratified_split_indices(
    labels: &[usize],
    n_classes: usize,
    val_split: f32,
    seed: u64,
    classes: Option<&[String]>,
) -> Result<SplitIndices, FinetuneError> {
    // Bucket kept-example indices by class.
    let mut by_class: Vec<Vec<usize>> = vec![Vec::new(); n_classes];
    for (i, &label) in labels.iter().enumerate() {
        if label < n_classes {
            by_class[label].push(i);
        }
    }

    // Resolve the operator-facing class label for diagnostic
    // messages.  `classes` carries the directory names (the same
    // strings the operator sees in `labels.txt`); falling back to
    // a synthetic `class#i` label keeps unit tests that pass a
    // raw labels vec self-contained.
    let label_for = |i: usize| -> String {
        classes
            .and_then(|cs| cs.get(i).cloned())
            .unwrap_or_else(|| format!("class#{i}"))
    };
    let per_class_kept_pairs: Vec<(String, usize)> = by_class
        .iter()
        .enumerate()
        .map(|(i, v)| (label_for(i), v.len()))
        .collect();

    let val_enabled = val_split > 0.0;
    let mut train: Vec<usize> = Vec::with_capacity(labels.len());
    let mut val: Vec<usize> = Vec::with_capacity(labels.len());
    for (class_idx, mut bucket) in by_class.into_iter().enumerate() {
        let class_n = bucket.len();
        // Per-class deterministic shuffle.  XOR the class index
        // into the seed so two classes don't share the same
        // permutation -- the canonical "different stream per
        // class" pattern for stratified splits.
        let class_seed = seed ^ ((class_idx as u64).wrapping_mul(0xA5A5_A5A5_5A5A_5A5A));
        shuffle_in_place(&mut bucket, class_seed);

        // Per-class val count: round(class_n * val_split), then
        // clamp so train always has >= 1 example AND val has >= 1
        // when validation is enabled and class_n >= 2.
        let raw_val = (class_n as f32 * val_split).round() as usize;
        let v_min = if val_enabled && class_n >= 2 { 1 } else { 0 };
        // Upper bound: leave at least 1 for train.
        let v_max = class_n.saturating_sub(1);
        let n_val = raw_val.clamp(v_min, v_max.max(v_min));
        // If validation is enabled but class_n == 1 we cannot
        // satisfy "at least 1 in train AND at least 1 in val" --
        // surface the singleton class to the operator instead of
        // silently leaving the val partition empty for this label.
        if val_enabled && class_n == 1 {
            return Err(FinetuneError::StratifiedSplitImpossible {
                class: label_for(class_idx),
                per_class_kept: per_class_kept_pairs,
                val_split,
            });
        }
        let (val_part, train_part) = bucket.split_at(n_val);
        val.extend_from_slice(val_part);
        train.extend_from_slice(train_part);
    }

    // Final shuffle of the merged splits so per-class blocks
    // don't leak into batch ordering during training (the
    // per-epoch shuffle in `train_head` re-randomises anyway, but
    // a clean post-merge shuffle keeps the eval indices well-mixed
    // for the validation pass too).  Use distinct sub-seeds so
    // `train` and `val` don't collide.
    shuffle_in_place(&mut train, seed.wrapping_add(0xC0FFEE));
    shuffle_in_place(&mut val, seed.wrapping_add(0xBADF00D));
    Ok(SplitIndices { train, val })
}

/// Examples per chunk in the fused preproc + backbone pipeline.
///
/// Each chunk runs (parallel preproc) -> (sequential batched backbone
/// forward), so this constant balances three concerns:
///
/// 1. **Backbone throughput.** Larger batches amortize Burn's per-call
///    overhead and let burn-ndarray's `multi-threads` BLAS path win
///    on the dense Linear(704->2000); near-peak past ~256.
/// 2. **Per-worker init amortization.** Each chunk re-runs `map_init`,
///    which lazy-initializes the rubato `SincResampler` (~6 ms to
///    precompute the 256x512 sinc table) on first non-44.1 kHz file.
///    With W workers and N examples, total inits ~= W . ceil(N/B); raising
///    B by 8x cuts that cost by 8x.
/// 3. **Memory and cancel latency.** Peak per-chunk spectrogram
///    memory is `B . 43 . 232 . 4 B ~= B . 40 KB`, and cancel can only
///    take effect at chunk boundaries.
///
/// 512 keeps peak at ~20 MB per chunk and cancel latency well under
/// 100 ms while reducing init overhead from ~80 s to ~10 s on a
/// 50 k-example resampled dataset.  The total feature buffer is
/// independent: `O(N . 8 KB)` regardless of B.
const BACKBONE_BATCH: usize = 512;

/// Hard ceiling on the feature buffer's resident bytes.  The
/// dense `Vec<[f32; FEATURE_DIM]>` allocated by
/// [`extract_features`] is `examples.len() . 8 KB` at
/// FEATURE_DIM = 2000 (8000 bytes); 100 k examples = 800 MB,
/// well past the daemon's RAM budget.
///
/// At 256 MiB worth of `[f32; FEATURE_DIM]` rows (= ~33 k
/// examples), peak resident fits inside the training-job
/// envelope after the backbone + autodiff workspace.  Not an
/// operator-tunable knob: the failure mode is OOM, not a tuning
/// preference.
const MAX_FEATURE_BYTES: usize = 256 * 1024 * 1024;

/// Derived ceiling on `examples.len()` for
/// [`extract_features`].  Factored out so the unit test can
/// exercise the cap without standing up a full backbone +
/// preproc pipeline.
const MAX_EXAMPLES_FEATURE_BUFFER: usize =
    MAX_FEATURE_BYTES / std::mem::size_of::<[f32; FEATURE_DIM]>();

/// Return `Err(InvalidConfig)` if `total` would
/// overflow the [`MAX_FEATURE_BYTES`] feature-buffer ceiling.
/// Used by [`extract_features`]; exposed at module scope so the
/// test below can assert the diagnostic shape without reaching
/// for a real backbone.
fn check_feature_buffer_cap(total: usize) -> Result<(), FinetuneError> {
    if total > MAX_EXAMPLES_FEATURE_BUFFER {
        return Err(FinetuneError::InvalidConfig(format!(
            "dataset has {} examples; on-device cap is {} \
             (~= {} MB feature buffer); use a developer host \
             for larger datasets",
            total,
            MAX_EXAMPLES_FEATURE_BUFFER,
            MAX_FEATURE_BYTES / (1024 * 1024),
        )));
    }
    Ok(())
}

fn extract_features(
    backbone: &Backbone<InnerB>,
    preproc: &Preproc,
    examples: &[(PathBuf, usize)],
    progress_cb: &(dyn Fn(&Progress) + Sync),
    cancel: &(dyn Fn() -> bool + Sync),
) -> Result<(Vec<[f32; FEATURE_DIM]>, Vec<usize>), FinetuneError> {
    let device: burn::tensor::Device<InnerB> = Default::default();
    let t0 = Instant::now();
    let dropped_nan = AtomicUsize::new(0);
    // Files that fail anywhere in read/resample/spectrogram surface
    // as `PreprocError` and drop the single example (counted here)
    // rather than abort the whole job.
    let dropped_io = AtomicUsize::new(0);
    let total = examples.len();

    // Fail fast if `total` would exceed the
    // 256 MiB feature-buffer ceiling.  Without this gate, the
    // `Vec::with_capacity(total)` below would request an
    // `examples.len() * 8 KB` allocation up front (no incremental
    // growth -- `with_capacity` is exact), tripping the kernel OOM
    // killer on a 1.5 GB-available box at ~190 k examples.  We
    // raise the cap *before* the alloc so the operator sees a
    // structured error (`InvalidConfig`) rather than the daemon
    // dying mid-training.  See `check_feature_buffer_cap` for the
    // exact diagnostic; the unit test exercises that helper
    // without needing a real backbone + preproc.
    check_feature_buffer_cap(total)?;
    let mut feats: Vec<[f32; FEATURE_DIM]> = Vec::with_capacity(total);
    let mut labels: Vec<usize> = Vec::with_capacity(total);
    let mut processed: usize = 0;

    for chunk in examples.chunks(BACKBONE_BATCH) {
        check_cancel(cancel)?;

        // Parallel preproc within the chunk.  `map_init` runs the init
        // once per worker per chunk; the resampler slot starts as
        // `None` and lazy-initializes on first non-44.1 kHz file
        // (paying the sinc-table precompute), and `Preproc::clone`
        // shares the FFT plan via `Arc` while allocating fresh scratch
        // buffers.  See `BACKBONE_BATCH`'s rationale for the per-chunk
        // init cost trade-off.
        let specs: Vec<Option<(Spectrogram, usize)>> = chunk
            .par_iter()
            .map_init(
                || (ResamplerCache::empty(), preproc.clone()),
                |(resampler, worker_preproc), (path, label)| {
                    // Per-worker fast path: once `cancel()` flips, the
                    // remaining files in this chunk drop straight to
                    // `None` rather than paying for I/O.  The next loop
                    // iteration's `check_cancel` then returns `Err`,
                    // bounding cancel latency to one chunk's preproc.
                    if cancel() {
                        return None;
                    }
                    // `read_wav_mono` and `to_waveform` return
                    // `Result<_, PreprocError>`; the `?`-style chain
                    // below drops a single bad file as `None`
                    // (counted in `dropped_io`) without unwinding
                    // the whole batch.  `Preproc::spectrogram`
                    // itself does not error.
                    //
                    // Resampler hygiene: a `Resample(_)` error means
                    // the resampler may have partial FIR history;
                    // null it out so the next file lazy-initialises
                    // a fresh one.  Other errors leave it untouched.
                    let spec = match wav_io::read_wav_mono(path)
                        .and_then(|(sr, mono)| wav_io::to_waveform(sr, mono, resampler))
                    {
                        Ok(pcm) => worker_preproc.spectrogram(&pcm),
                        Err(e) => {
                            if matches!(e, wav_io::PreprocError::Resample(_)) {
                                resampler.clear();
                            }
                            dropped_io.fetch_add(1, Ordering::Relaxed);
                            return None;
                        }
                    };
                    // `[[f32; NBins::USIZE]; NFrames::USIZE]` is contiguous f32
                    // memory by Rust's array layout guarantees; flat
                    // it once and run the NaN scan over the same
                    // slice we'd otherwise re-walk.
                    if spec[..].as_flattened().iter().any(|v| !v.is_finite()) {
                        dropped_nan.fetch_add(1, Ordering::Relaxed);
                        return None;
                    }
                    Some((spec, *label))
                },
            )
            .collect();
        // If cancel flipped during preproc, skip this chunk's backbone
        // forward; the next loop iteration's `check_cancel` will surface
        // the `Cancelled` error.
        check_cancel(cancel)?;

        // Assemble a single batched NCHW input for the chunk's kept
        // examples.  Dropped (NaN/IO) examples shrink the batch but do
        // not stop the run; an entirely-dropped chunk just skips the
        // forward.
        let kept: Vec<(Spectrogram, usize)> = specs.into_iter().flatten().collect();
        if !kept.is_empty() {
            let n = kept.len();
            // Exact-size alloc: `TensorData::new` consumes the Vec,
            // and `kept.len()` may be smaller than `chunk.len()` after
            // drops, so over-allocating to `BACKBONE_BATCH` would just
            // waste address space.
            let mut batched: Vec<f32> = Vec::with_capacity(n * NFrames::USIZE * NBins::USIZE);
            for (spec, _) in &kept {
                batched.extend_from_slice(spec[..].as_flattened());
            }
            let x = Tensor::<InnerB, 4>::from_data(
                TensorData::new(batched, [n, 1, NFrames::USIZE, NBins::USIZE]),
                &device,
            );
            let f = backbone.forward(x);
            let out_data = f.into_data().to_vec::<f32>().unwrap();
            // Hard assert (not `debug_assert`): a shape mismatch here
            // means the backbone produced the wrong output size, which
            // would silently corrupt downstream training.  Cheap to keep
            // in release builds.
            assert_eq!(
                out_data.len(),
                n * FEATURE_DIM,
                "backbone returned {} floats for batch {n}; expected {}",
                out_data.len(),
                n * FEATURE_DIM,
            );
            for ((_, label), feat_chunk) in kept.into_iter().zip(out_data.chunks_exact(FEATURE_DIM))
            {
                let mut arr = [0f32; FEATURE_DIM];
                arr.copy_from_slice(feat_chunk);
                feats.push(arr);
                labels.push(label);
            }
        }

        let prev_processed = processed;
        processed += chunk.len();
        // Emit when `processed` crosses a `DEFAULT_PROGRESS_EVERY`
        // boundary (`is_multiple_of(500)` would rarely land true at
        // chunk steps that aren't divisors of 500).  Comparing milestone
        // indices fires roughly once per chunk while skipping noise on
        // very small datasets.  The final summary outside the loop
        // handles the closing message.
        let prev_milestone = prev_processed / DEFAULT_PROGRESS_EVERY;
        let cur_milestone = processed / DEFAULT_PROGRESS_EVERY;
        if cur_milestone > prev_milestone && processed != total {
            progress_cb(&Progress::new(
                Phase::FeatureExtract,
                processed,
                total,
                format!(
                    "feature extract [{processed:>5}/{total:>5}] kept={} dropped_nan={} dropped_io={} elapsed={:.1?}",
                    feats.len(),
                    dropped_nan.load(Ordering::Relaxed),
                    dropped_io.load(Ordering::Relaxed),
                    t0.elapsed(),
                ),
            ));
        }
    }

    progress_cb(&Progress::new(
        Phase::FeatureExtract,
        feats.len(),
        total,
        format!(
            "feature extraction: {} kept, {} dropped (NaN), {} dropped (IO); total={:.1?}",
            feats.len(),
            dropped_nan.load(Ordering::Relaxed),
            dropped_io.load(Ordering::Relaxed),
            t0.elapsed()
        ),
    ));
    Ok((feats, labels))
}

/// Index of the maximum element in `xs`.  NaN compares as
/// less-than-everything via the `if v > acc.1` shortcut, which
/// matches the engine's NaN-tolerant `top_k_indices_into`.  Empty
/// slices return 0 (callers always pass non-empty rows).
#[inline]
fn argmax(xs: &[f32]) -> usize {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > best_v {
            best_i = i;
            best_v = v;
        }
    }
    best_i
}

/// Deterministic Fisher-Yates shuffle of `slice`.  Reused
/// across dataset partitioning and per-epoch training
/// shuffles so the final split + training order are
/// reproducible from a single root seed.
fn shuffle_in_place<T>(slice: &mut [T], seed: u64) {
    let n = slice.len();
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    for i in (1..n).rev() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (state >> 33) as usize % (i + 1);
        slice.swap(i, j);
    }
}

fn evaluate(
    head: &Head<InnerB>,
    n_classes: usize,
    feats: &[[f32; FEATURE_DIM]],
    labels: &[usize],
    indices: &[usize],
) -> f32 {
    if indices.is_empty() {
        return f32::NAN;
    }
    let device: burn::tensor::Device<InnerB> = Default::default();
    let batch = 256;
    let mut correct = 0usize;
    for chunk in indices.chunks(batch) {
        let mut flat = Vec::with_capacity(chunk.len() * FEATURE_DIM);
        for &i in chunk {
            flat.extend_from_slice(&feats[i]);
        }
        let x = Tensor::<InnerB, 2>::from_data(
            TensorData::new(flat, [chunk.len(), FEATURE_DIM]),
            &device,
        );
        let logits = head.forward(x);
        let pred: Vec<f32> = logits.into_data().to_vec::<f32>().unwrap();
        debug_assert_eq!(pred.len(), chunk.len() * n_classes);
        for (row, &i) in chunk.iter().enumerate() {
            let start = row * n_classes;
            if argmax(&pred[start..start + n_classes]) == labels[i] {
                correct += 1;
            }
        }
    }
    correct as f32 / indices.len() as f32
}

struct TrainData<'a> {
    n_classes: usize,
    feats: &'a [[f32; FEATURE_DIM]],
    labels: &'a [usize],
    train_idx: &'a [usize],
    val_idx: &'a [usize],
}

#[derive(Clone, Copy, Debug)]
struct TrainSettings {
    epochs: usize,
    batch: usize,
    lr: f32,
    seed: u64,
}

fn train_head(
    mut head: Head<AutoB>,
    data: TrainData<'_>,
    settings: TrainSettings,
    progress: &(dyn Fn(&Progress) + Sync),
    cancel: &(dyn Fn() -> bool + Sync),
) -> Result<Head<AutoB>, FinetuneError> {
    let TrainData {
        n_classes,
        feats,
        labels,
        train_idx,
        val_idx,
    } = data;
    let TrainSettings {
        epochs,
        batch,
        lr,
        seed,
    } = settings;
    let device_auto: burn::tensor::Device<AutoB> = Default::default();
    let ce = CrossEntropyLossConfig::new().init(&device_auto);
    let mut optim = SgdConfig::new().init();

    // `best_val` starts as NaN so the metric is well-defined when
    // `val_idx` is empty (`val_split == 0.0`); `evaluate` returns NaN
    // for empty splits and the update below ignores non-finite values.
    let mut best_val = f32::NAN;
    // Snapshot the head whenever a strictly better val_acc is
    // observed, return that snapshot at end of training (fall back to
    // the last-epoch head only when val_idx is empty and best_val
    // never updates).  Fixes the "last epoch is published, even when
    // earlier epochs were better" bug.
    //
    // Why round-trip through `.mpk` instead of an in-memory clone:
    // `Param::clone` preserves `ParamId` and reuses the same
    // `Tensor` value, whose backing buffer is Arc-shared with the
    // running head -- subsequent SGD steps can therefore corrupt the
    // snapshot.  The recorder path is the one round-trip that
    // produces an owning, immutable byte snapshot
    // (`crate::model::tests::head_save_load_round_trip` verifies bit
    // equivalence).  Head is ~272 KB; one save+load per best-update
    // is negligible relative to per-epoch training cost.
    let snapshot_dir =
        tempfile::tempdir().map_err(|e| finetune_io_err("<finetune snapshot tempdir>", e))?;
    let snapshot_path = snapshot_dir.path().join("best.mpk");
    let mut best_path: Option<PathBuf> = None;
    // Per-epoch shuffle target, allocated once and reused.  Equivalent
    // to `train_idx[pi(i)]` where pi is the Fisher-Yates permutation
    // produced by `shuffle_indices(n, seed)`, since refilling from
    // `train_idx` and applying the same swaps yields the same gather.
    let mut order: Vec<usize> = Vec::with_capacity(train_idx.len());
    for epoch in 0..epochs {
        check_cancel(cancel)?;
        order.clear();
        order.extend_from_slice(train_idx);
        shuffle_in_place(&mut order, seed.wrapping_add(epoch as u64));

        let mut running_loss = 0.0_f64;
        let mut running_correct = 0usize;
        let mut running_count = 0usize;

        for chunk in order.chunks(batch) {
            check_cancel(cancel)?;
            let mut flat = Vec::with_capacity(chunk.len() * FEATURE_DIM);
            for &i in chunk {
                flat.extend_from_slice(&feats[i]);
            }
            let x = Tensor::<AutoB, 2>::from_data(
                TensorData::new(flat, [chunk.len(), FEATURE_DIM]),
                &device_auto,
            );
            let targets: Vec<i32> = chunk.iter().map(|&i| labels[i] as i32).collect();
            let y = Tensor::<AutoB, 1, Int>::from_data(
                TensorData::new(targets, [chunk.len()]),
                &device_auto,
            );

            let logits = head.forward(x);
            let loss = ce.forward(logits.clone(), y.clone());

            // CrossEntropyLoss reduces with mean, so each scalar is the
            // per-example mean for its batch.  Weight by chunk size so the
            // epoch average is a true per-example mean even when the last
            // chunk is smaller than `batch`.
            running_loss += (loss.clone().into_scalar() as f64) * chunk.len() as f64;
            let pred: Vec<f32> = logits.detach().into_data().to_vec::<f32>().unwrap();
            for (row, &i) in chunk.iter().enumerate() {
                let start = row * n_classes;
                if argmax(&pred[start..start + n_classes]) == labels[i] {
                    running_correct += 1;
                }
            }
            running_count += chunk.len();

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &head);
            head = optim.step(lr as f64, head, grads);
        }

        let train_acc = running_correct as f32 / running_count as f32;
        let avg_loss = if running_count > 0 {
            running_loss / running_count as f64
        } else {
            0.0
        };
        let val_acc = evaluate(&head.valid(), n_classes, feats, labels, val_idx);
        if val_acc.is_finite() && (best_val.is_nan() || val_acc > best_val) {
            best_val = val_acc;
            // Internal best-epoch snapshot stays in raw Burn
            // recorder format: it never leaves the daemon and
            // round-trips through `Head::load_mpk` below.  The
            // `ACSTHEAD` schema is enforced only at the
            // published `cfg.out` boundary.  `save_mpk` consumes
            // the head, so clone first.
            head.clone().save_mpk(&snapshot_path)?;
            best_path = Some(snapshot_path.clone());
        }
        let metrics = EpochMetrics {
            epoch: epoch + 1,
            epochs,
            train_loss: avg_loss,
            train_acc,
            val_acc,
            best_val_acc: best_val,
        };
        progress(&Progress::with_metrics(
            format!(
                "epoch {:>3}/{}:  train_loss={:.4}  train_acc={:.4}  val_acc={:.4}  (best_val={:.4})",
                epoch + 1,
                epochs,
                avg_loss,
                train_acc,
                val_acc,
                best_val
            ),
            metrics,
        ));
    }
    match best_path {
        Some(p) => Ok(Head::<AutoB>::load_mpk(&p, &device_auto)?),
        None => Ok(head),
    }
}

fn panic_payload_to_string(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic payload".into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::Mutex;

    /// Serializes Burn's per-backend RNG slot for tests that
    /// re-seed it.  See `backend_seed_determines_head_init` for
    /// the rayon-RNG limitation.
    static RNG_LOCK: Mutex<()> = Mutex::new(());

    /// `Backend::seed` makes `Head::<AutoB>::new`'s weight
    /// initialisation deterministic.  Ignored by default: Burn's
    /// `burn-ndarray` `multi-threads` feature dispatches init work
    /// onto rayon workers whose RNG slot is not reachable from a
    /// `parking_lot::Mutex` here, so parallel `cargo test --lib`
    /// runs perturb the seeded state.  Run in isolation via
    /// `cargo test --lib backend_seed_determines_head_init` (single
    /// matched name => one process, one thread effectively) or
    /// `cargo test --lib -- --include-ignored --test-threads=1`.
    #[test]
    #[ignore = "requires process isolation; rayon-thread RNG races parallel runs"]
    fn backend_seed_determines_head_init() {
        let _rng_guard = RNG_LOCK.lock();
        let device: burn::tensor::Device<AutoB> = Default::default();
        let n_classes = 4;

        <AutoB as Backend>::seed(&device, 0xDEAD_BEEF);
        let h1 = Head::<AutoB>::new(n_classes, &device);
        let w1: Vec<f32> = h1.linear.weight.val().into_data().to_vec().expect("to_vec");

        <AutoB as Backend>::seed(&device, 0xDEAD_BEEF);
        let h2 = Head::<AutoB>::new(n_classes, &device);
        let w2: Vec<f32> = h2.linear.weight.val().into_data().to_vec().expect("to_vec");

        assert_eq!(w1.len(), w2.len(), "weight length drift");
        for (i, (a, b)) in w1.iter().zip(w2.iter()).enumerate() {
            assert!(
                (a - b).abs() < f32::EPSILON,
                "weight drift at idx {i}: a={a}, b={b} -- Backend::seed didn't produce determinism",
            );
        }

        // Different seed -> different weights (sanity check that the
        // determinism above wasn't accidentally a no-op via some
        // other source of state).
        <AutoB as Backend>::seed(&device, 0x1234_5678);
        let h3 = Head::<AutoB>::new(n_classes, &device);
        let w3: Vec<f32> = h3.linear.weight.val().into_data().to_vec().expect("to_vec");
        let any_diff = w1
            .iter()
            .zip(w3.iter())
            .any(|(a, b)| (a - b).abs() > f32::EPSILON);
        assert!(
            any_diff,
            "different seeds must produce different weights; the seed plumbing is a no-op",
        );
    }

    /// `check_feature_buffer_cap` accepts up to
    /// `MAX_EXAMPLES_FEATURE_BUFFER` examples and rejects anything
    /// strictly larger with `InvalidConfig`.  Two invariants are
    /// pinned:
    ///   1. `MAX_EXAMPLES_FEATURE_BUFFER` is the literal "256 MiB
    ///      / sizeof::<[f32; FEATURE_DIM]>" -- a future bump of the
    ///      cap surfaces as this test failing instead of silently
    ///      raising the OOM ceiling.
    ///   2. The diagnostic message includes the operator-actionable
    ///      "use a developer host" steer; testing the substring
    ///      rather than the exact text leaves room for wording
    ///      polish without test churn.
    #[test]
    fn feature_buffer_cap_rejects_overflow() {
        // The cap math: 256 MiB / 8 KB-per-row = 32 768 rows at
        // FEATURE_DIM = 2000.  Pin the constant explicitly so the
        // intent is visible in the test source, not just derived.
        let expected_cap = (256 * 1024 * 1024) / std::mem::size_of::<[f32; FEATURE_DIM]>();
        assert_eq!(
            MAX_EXAMPLES_FEATURE_BUFFER, expected_cap,
            "MAX_EXAMPLES_FEATURE_BUFFER drifted from the documented \
             256 MiB / FEATURE_DIM*4 derivation",
        );

        // At-cap is fine.
        check_feature_buffer_cap(MAX_EXAMPLES_FEATURE_BUFFER).expect("cap-equal must pass");
        // One past the cap rejects.
        let err = check_feature_buffer_cap(MAX_EXAMPLES_FEATURE_BUFFER + 1)
            .expect_err("cap+1 must reject");
        match err {
            FinetuneError::InvalidConfig(msg) => {
                assert!(
                    msg.contains("on-device cap"),
                    "diagnostic missing on-device cap context: {msg}",
                );
                assert!(
                    msg.contains("developer host"),
                    "diagnostic missing operator-actionable steer: {msg}",
                );
            }
            other => panic!("expected FinetuneError::InvalidConfig, got {other:?}"),
        }
        // Far-past the cap also rejects (sanity: the predicate is
        // `>` not `==`).
        assert!(
            matches!(
                check_feature_buffer_cap(MAX_EXAMPLES_FEATURE_BUFFER * 4),
                Err(FinetuneError::InvalidConfig(_)),
            ),
            "4x-cap must reject",
        );
    }

    /// Linearly separable 2-class features: class 0 has dim 0 = 1.0,
    /// class 1 has dim 1 = 1.0, all other dims zero.  SGD reaches
    /// val_acc = 1.0 within one epoch, so `best_val` is set at epoch 1
    /// and never strictly improves afterward; the fix must
    /// publish the epoch-1 snapshot, not the epoch-3 head.
    fn synthetic_separable_data() -> (Vec<[f32; FEATURE_DIM]>, Vec<usize>) {
        let mut feats = Vec::with_capacity(8);
        let mut labels = Vec::with_capacity(8);
        for _ in 0..4 {
            let mut a = [0.0f32; FEATURE_DIM];
            a[0] = 1.0;
            feats.push(a);
            labels.push(0);
            let mut b = [0.0f32; FEATURE_DIM];
            b[1] = 1.0;
            feats.push(b);
            labels.push(1);
        }
        (feats, labels)
    }

    fn weights_of(head: &Head<AutoB>) -> Vec<f32> {
        head.linear
            .weight
            .val()
            .into_data()
            .to_vec()
            .expect("to_vec")
    }

    /// Save a fresh head to a tempfile and return two independent
    /// loads.  The `Head::new` initializer pulls from the backend's
    /// RNG, whose state advances across calls, so two `Head::new`
    /// invocations would start from *different* random weights and
    /// post-SGD weights would diverge for reasons unrelated to .
    /// Round-tripping through `.mpk` materializes an immutable byte
    /// snapshot (`crate::model::tests::head_save_atomic_round_trip`
    /// proves bit-equivalence for the published format; the
    /// crate-private raw-Burn `save_mpk` used here is its
    /// header-less counterpart, used only for in-test fixtures /
    /// best-epoch snapshots that never leave the daemon).
    fn paired_initial_heads(
        n_classes: usize,
        device: &burn::tensor::Device<AutoB>,
    ) -> (tempfile::TempDir, Head<AutoB>, Head<AutoB>) {
        // RNG_LOCK serialises the `Head::new` call below
        // against the seed-determinism test.  Without it, that test's
        // intermediate state can race with this fixture's draw and
        // produce two different "initial heads" across parallel
        // tests, defeating the round-trip guarantee.  Acquired here
        // and released *after* `Head::new`; the loaded handles
        // `a`/`b` are immutable byte snapshots so they don't need
        // continued protection.
        let _rng_guard = RNG_LOCK.lock();
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("init.mpk");
        Head::<AutoB>::new(n_classes, device)
            .save_mpk(&path)
            .expect("save init head");
        let a = Head::<AutoB>::load_mpk(&path, device).expect("load a");
        let b = Head::<AutoB>::load_mpk(&path, device).expect("load b");
        (dir, a, b)
    }

    /// The head returned by `train_head` is the snapshot from
    /// the best-val_acc epoch, not the post-last-epoch head.  With a
    /// linearly separable synthetic dataset val_acc reaches 1.0 at
    /// epoch 1 and stays there, so `best_head` is the snapshot taken
    /// after epoch 1.  Compare against an independent run that starts
    /// from byte-identical weights and trains for exactly one epoch:
    /// the 3-epoch published head must byte-match.  Without the fix,
    /// the 3-epoch run would return weights moved by epoch-2 + epoch-3
    /// SGD steps and the comparison would fail.
    #[test]
    fn train_head_returns_best_not_last_epoch() {
        let device: burn::tensor::Device<AutoB> = Default::default();
        let (feats, labels) = synthetic_separable_data();
        let train_idx: Vec<usize> = (0..feats.len()).collect();
        let val_idx: Vec<usize> = train_idx.clone();
        let n_classes = 2;
        let (_init_dir, init_3, init_ref) = paired_initial_heads(n_classes, &device);

        let captured: Mutex<Vec<EpochMetrics>> = Mutex::new(Vec::new());
        let progress = |p: &Progress| {
            if let Some(m) = p.metrics.as_ref() {
                captured.lock().push(*m);
            }
        };
        let cancel = || false;

        let data = TrainData {
            n_classes,
            feats: &feats,
            labels: &labels,
            train_idx: &train_idx,
            val_idx: &val_idx,
        };
        let settings_3 = TrainSettings {
            epochs: 3,
            batch: 4,
            lr: 0.5,
            seed: 7,
        };
        let head_3 =
            train_head(init_3, data, settings_3, &progress, &cancel).expect("3-epoch train");

        let metrics = captured.into_inner();
        assert_eq!(metrics.len(), 3, "one EpochMetrics per epoch");
        // Mirror `train_head`'s strict `>` tracking: the *first* epoch
        // to reach the running maximum is the best snapshot.  (The
        // standard `max_by` returns the last tied value, which would
        // disagree with the tracker for plateau-shaped val_acc.)
        let mut best_epoch = 0;
        let mut best_val = f32::NAN;
        for (i, m) in metrics.iter().enumerate() {
            if m.val_acc.is_finite() && (best_val.is_nan() || m.val_acc > best_val) {
                best_val = m.val_acc;
                best_epoch = i + 1;
            }
        }
        assert!(
            best_epoch < 3,
            "test precondition: best epoch must be < last so the bug-affected branch is exercised; \
             observed metrics={metrics:?}",
        );

        let data_ref = TrainData {
            n_classes,
            feats: &feats,
            labels: &labels,
            train_idx: &train_idx,
            val_idx: &val_idx,
        };
        let settings_ref = TrainSettings {
            epochs: best_epoch,
            batch: 4,
            lr: 0.5,
            seed: 7,
        };
        let head_ref =
            train_head(init_ref, data_ref, settings_ref, &|_| {}, &cancel).expect("ref train");

        let w3 = weights_of(&head_3);
        let wref = weights_of(&head_ref);
        assert_eq!(w3.len(), wref.len(), "weight buffer length drift");
        for (i, (a, b)) in w3.iter().zip(wref.iter()).enumerate() {
            assert!(
                (a - b).abs() < f32::EPSILON,
                "weight drift at idx {i}: 3-epoch={a}, ref={b} -- published head is not the best snapshot",
            );
        }
    }

    /// `val_split == 0.0` => empty val set => `evaluate` yields NaN =>
    /// `best_val` never updates => `best_head` stays None => fallback
    /// to the last-epoch head.  Verify that fallback by running 2
    /// epochs with empty val and asserting the returned head differs
    /// from the head after 1 epoch (i.e. epoch-2 SGD steps actually
    /// took effect).
    #[test]
    fn train_head_falls_back_to_last_when_no_val() {
        let device: burn::tensor::Device<AutoB> = Default::default();
        let (feats, labels) = synthetic_separable_data();
        let train_idx: Vec<usize> = (0..feats.len()).collect();
        let val_idx: Vec<usize> = Vec::new();
        let n_classes = 2;
        let progress = |_: &Progress| {};
        let cancel = || false;
        let (_init_dir, init_2, init_1) = paired_initial_heads(n_classes, &device);

        let data2 = TrainData {
            n_classes,
            feats: &feats,
            labels: &labels,
            train_idx: &train_idx,
            val_idx: &val_idx,
        };
        let head_2 = train_head(
            init_2,
            data2,
            TrainSettings {
                epochs: 2,
                batch: 4,
                lr: 0.5,
                seed: 7,
            },
            &progress,
            &cancel,
        )
        .expect("2-epoch train (no val)");

        let data1 = TrainData {
            n_classes,
            feats: &feats,
            labels: &labels,
            train_idx: &train_idx,
            val_idx: &val_idx,
        };
        let head_1 = train_head(
            init_1,
            data1,
            TrainSettings {
                epochs: 1,
                batch: 4,
                lr: 0.5,
                seed: 7,
            },
            &progress,
            &cancel,
        )
        .expect("1-epoch train (no val)");

        let w2 = weights_of(&head_2);
        let w1 = weights_of(&head_1);
        let differs = w2
            .iter()
            .zip(w1.iter())
            .any(|(a, b)| (a - b).abs() > f32::EPSILON);
        assert!(
            differs,
            "no-val path must publish the last-epoch head; weights are identical to a 1-epoch run, \
             which means epoch-2 SGD steps were silently discarded",
        );
    }

    /// An empty class directory (created by an operator who
    /// staged the archive incorrectly, or by macOS's `__MACOSX`
    /// metadata extraction) fails closed at scan time with
    /// `BadDataset`.  Rejection fires BEFORE any feature
    /// extraction or optimizer step; the populated class is
    /// irrelevant because the scan rejects on the FIRST empty
    /// class it finds.
    #[test]
    fn run_rejects_empty_class_dir_before_training() {
        let dir = tempfile::tempdir().expect("tempdir");
        // Two classes; one populated, one empty.  Scan rejects
        // on `no` (alphabetically first) before walking `yes`.
        let yes = dir.path().join("yes");
        let no = dir.path().join("no");
        std::fs::create_dir(&yes).unwrap();
        std::fs::create_dir(&no).unwrap();
        #[allow(clippy::disallowed_methods)]
        std::fs::write(yes.join("a.wav"), b"placeholder").unwrap();

        let cfg = FinetuneConfig {
            data: dir.path().to_path_buf(),
            backbone: PathBuf::from("/nonexistent/backbone.mpk"),
            init_head: None,
            out: dir.path().join("out.mpk"),
            epochs: 1,
            batch: 1,
            lr: 0.01,
            val_split: 0.2,
            seed: 1,
        };
        let err = run(&cfg, &|_| {}, &|| false).expect_err("empty class must reject");
        match err {
            FinetuneError::BadDataset { path, reason } => {
                assert!(
                    path.ends_with("no"),
                    "BadDataset path must point at the offending class folder; got {path:?}"
                );
                assert!(
                    reason.contains("no non-hidden .wav sample files"),
                    "reason must explain the empty-class rejection; got {reason:?}"
                );
            }
            other => panic!("expected FinetuneError::BadDataset, got {other:?}"),
        }
    }

    /// `stratified_split_indices` guarantees every class with
    /// `class_n >= 2` lands at least one example in BOTH train
    /// and val partitions when `val_split > 0`.  Pre-fix, a
    /// global random shuffle could push an entire small class
    /// into one partition and leave it absent from the other.
    /// Imbalanced 3-class fixture (8 / 3 / 2) chosen so a naive
    /// `(n * val_split).round()` with `val_split = 0.25` would
    /// give the 2-example class zero val rows -- the new clamp
    /// pulls that to 1.
    #[test]
    fn stratified_split_represents_every_class_in_both_partitions() {
        // class 0: 8 examples (idx 0..8)
        // class 1: 3 examples (idx 8..11)
        // class 2: 2 examples (idx 11..13)
        let labels: Vec<usize> = (0..8)
            .map(|_| 0)
            .chain((0..3).map(|_| 1))
            .chain((0..2).map(|_| 2))
            .collect();
        let n_classes = 3;
        let val_split = 0.25_f32;
        let split = stratified_split_indices(&labels, n_classes, val_split, 12345, None)
            .expect("stratified split must succeed for class_n >= 2");

        // Every class must appear in both train and val.
        for class in 0..n_classes {
            let train_n = split.train.iter().filter(|&&i| labels[i] == class).count();
            let val_n = split.val.iter().filter(|&&i| labels[i] == class).count();
            assert!(
                train_n >= 1,
                "class {class} missing from train partition (train={train_n}, val={val_n}); \
                 stratification regressed",
            );
            assert!(
                val_n >= 1,
                "class {class} missing from val partition (train={train_n}, val={val_n}); \
                 stratification regressed",
            );
        }

        // Determinism: same seed must yield same partitions.
        let split2 = stratified_split_indices(&labels, n_classes, val_split, 12345, None).unwrap();
        assert_eq!(
            split.train, split2.train,
            "train indices must be deterministic for a given seed"
        );
        assert_eq!(
            split.val, split2.val,
            "val indices must be deterministic for a given seed"
        );

        // Distinct seeds shuffle differently (sanity).  Compare
        // via sorted union since the partition sizes are the
        // same; a different seed should at least permute the
        // membership of train vs val for the larger class.
        let split_other =
            stratified_split_indices(&labels, n_classes, val_split, 67890, None).unwrap();
        let train_set: std::collections::HashSet<_> = split.train.iter().collect();
        let train_other: std::collections::HashSet<_> = split_other.train.iter().collect();
        // It's possible (though unlikely) the two seeds produce
        // identical partitions for tiny classes; the assertion
        // is intentionally weak ("at least one different
        // permutation across both larger classes").  The
        // determinism check above is the load-bearing
        // assertion; this is a sanity guard.
        assert!(
            train_set != train_other,
            "two distinct seeds yielded byte-identical train partitions; \
             RNG plumbing may be a no-op",
        );
    }

    /// `stratified_split_indices` cannot satisfy "every class
    /// has at least 1 train AND 1 val example" when a class
    /// has only one example AND val is enabled.  The error
    /// names the singleton class so the operator can fix the
    /// dataset (or disable validation).
    #[test]
    fn stratified_split_rejects_singleton_class_when_val_enabled() {
        // class 0: 5 examples; class 1: 1 example.
        let labels: Vec<usize> = (0..5).map(|_| 0).chain(std::iter::once(1)).collect();
        let err = stratified_split_indices(&labels, 2, 0.2, 7, None)
            .expect_err("singleton class + val must reject");
        match err {
            FinetuneError::StratifiedSplitImpossible {
                class,
                per_class_kept,
                val_split,
            } => {
                assert_eq!(class, "class#1", "diagnostic must name the singleton class");
                assert_eq!(
                    per_class_kept,
                    vec![("class#0".into(), 5), ("class#1".into(), 1)]
                );
                assert!((val_split - 0.2).abs() < f32::EPSILON);
            }
            other => panic!("expected StratifiedSplitImpossible, got {other:?}"),
        }

        // val_split == 0.0 should accept the same shape (no val
        // partition implies no per-class val minimum).
        let ok = stratified_split_indices(&labels, 2, 0.0, 7, None)
            .expect("val_split=0 must accept singleton classes");
        assert_eq!(ok.train.len(), 6);
        assert_eq!(ok.val.len(), 0);
    }

    /// `validate_post_extract_quality` rejects when a class
    /// loses every example to preproc.  This is the
    /// "post-feature-extract" sibling of
    /// `EmptyClassAfterScan`: `scan_dataset` saw a wav, but
    /// every wav for that class failed to decode / resample /
    /// produced a non-finite spectrogram.  The error names the
    /// first offender and surfaces both kept and dropped
    /// per-class counts so the operator can see which class is
    /// degraded.
    #[test]
    fn validate_post_extract_rejects_class_with_zero_survivors() {
        // 2 classes; pre-scan saw 3 examples each (6 total).
        // Post-extract, class 0 kept all 3, class 1 lost
        // everything (0 surviving labels).
        let classes = vec!["yes".to_string(), "no".to_string()];
        let pre_scan = vec![3usize, 3];
        let labels = vec![0usize, 0, 0]; // class 1 entirely missing
        let err = validate_post_extract_quality(&classes, &pre_scan, 2, &labels, 6)
            .expect_err("class with zero survivors must reject");
        match err {
            FinetuneError::EmptyClassAfterExtract {
                class,
                per_class_kept,
                per_class_dropped,
            } => {
                assert_eq!(class, "no", "diagnostic must name the offending class");
                assert_eq!(
                    per_class_kept,
                    vec![("yes".into(), 3), ("no".into(), 0)],
                    "per-class kept counts must surface in the error",
                );
                assert_eq!(
                    per_class_dropped,
                    vec![("yes".into(), 0), ("no".into(), 3)],
                    "per-class dropped counts must surface in the error",
                );
            }
            other => panic!("expected EmptyClassAfterExtract, got {other:?}"),
        }
    }

    /// `validate_post_extract_quality` rejects when the
    /// aggregate drop ratio crosses [`MAX_DROP_RATIO`] (10 %).
    /// Pathological dataset with 100 inputs and 80 survivors
    /// (20 % dropped) MUST fail loudly so the operator sees a
    /// structured error rather than a head trained on the
    /// degraded survivor subset.
    #[test]
    fn validate_post_extract_rejects_high_drop_ratio() {
        // Two classes; both keep enough survivors to clear the
        // empty-class gate (so the drop-ratio gate is the
        // load-bearing rejection, not the empty-class gate).
        // 100 in, 80 out: 20 % drop -> exceeds 10 %.
        let classes = vec!["a".to_string(), "b".to_string()];
        let pre_scan = vec![50usize, 50];
        // 40 of each class survives.
        let labels: Vec<usize> = (0..40).map(|_| 0).chain((0..40).map(|_| 1)).collect();
        let err = validate_post_extract_quality(&classes, &pre_scan, 2, &labels, 100)
            .expect_err("20% drop ratio must reject");
        match err {
            FinetuneError::DropRatioExceeded {
                dropped,
                total,
                ratio,
                max_ratio,
                per_class_kept,
                per_class_dropped,
            } => {
                assert_eq!(dropped, 20);
                assert_eq!(total, 100);
                assert!(
                    (ratio - 0.20).abs() < 1e-4,
                    "ratio should be ~0.20: {ratio}"
                );
                assert!((max_ratio - MAX_DROP_RATIO).abs() < f32::EPSILON);
                assert_eq!(per_class_kept, vec![("a".into(), 40), ("b".into(), 40)]);
                assert_eq!(per_class_dropped, vec![("a".into(), 10), ("b".into(), 10)]);
            }
            other => panic!("expected DropRatioExceeded, got {other:?}"),
        }
    }

    /// `validate_post_extract_quality` accepts a clean dataset
    /// where every class kept all its examples.  Sanity check
    /// that the gate is `>` not `>=` (a 0% drop ratio must not
    /// be a rejection) and that no empty-class false-positive
    /// fires when class counts match exactly.
    #[test]
    fn validate_post_extract_accepts_clean_dataset() {
        let classes = vec!["yes".to_string(), "no".to_string()];
        let pre_scan = vec![5usize, 5];
        let labels: Vec<usize> = (0..5).map(|_| 0).chain((0..5).map(|_| 1)).collect();
        validate_post_extract_quality(&classes, &pre_scan, 2, &labels, 10)
            .expect("clean dataset must pass post-extract validation");
    }

    /// `validate_post_extract_quality` accepts a dataset right
    /// at the cap (10% drop). Confirms the predicate is `>` not
    /// `>=`; pathological 11 % is the test below.
    #[test]
    fn validate_post_extract_accepts_at_cap_drop_ratio() {
        // 100 in, 90 out: exactly 10 % drop.  Must accept.
        let classes = vec!["a".to_string(), "b".to_string()];
        let pre_scan = vec![50usize, 50];
        let labels: Vec<usize> = (0..45).map(|_| 0).chain((0..45).map(|_| 1)).collect();
        validate_post_extract_quality(&classes, &pre_scan, 2, &labels, 100)
            .expect("at-cap drop ratio must pass (predicate is strict >)");
    }

    /// Sanity: `per_class_counts_from_labels` correctly buckets
    /// a flat labels vec.  Used by both the empty-class-after-
    /// extract and drop-ratio gates; a bug here would silently
    /// misreport per-class diagnostics in the error messages.
    #[test]
    fn per_class_counts_from_labels_buckets_correctly() {
        let labels = vec![0usize, 1, 0, 2, 1, 0];
        let counts = per_class_counts_from_labels(3, &labels);
        assert_eq!(counts, vec![3, 2, 1]);
        // Empty input -> all zeros, length n_classes.
        assert_eq!(per_class_counts_from_labels(4, &[]), vec![0; 4]);
        // Out-of-range labels are silently dropped (defensive
        // against caller-side off-by-one); does not panic.
        let counts = per_class_counts_from_labels(2, &[0, 99, 1]);
        assert_eq!(counts, vec![1, 1]);
    }

    // MARK: scan_dataset contract pins

    /// An empty datasets root rejects with `BadDataset`.
    #[test]
    fn scan_rejects_no_class_folders() {
        let dir = tempfile::tempdir().expect("tempdir");
        let err = scan_dataset(dir.path()).expect_err("empty root rejects");
        match err {
            FinetuneError::BadDataset { path, reason } => {
                assert_eq!(path, dir.path().display().to_string());
                assert!(
                    reason.contains("no class folders"),
                    "reason must mention `no class folders`; got {reason:?}",
                );
            }
            other => panic!("expected BadDataset, got {other:?}"),
        }
    }

    /// Stray non-hidden non-dir root entries (file / symlink /
    /// device) reject with `BadDataset` rather than being silently
    /// skipped.
    #[test]
    fn scan_rejects_stray_root_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir(dir.path().join("cat")).unwrap();
        #[allow(clippy::disallowed_methods)]
        std::fs::write(dir.path().join("cat").join("a.wav"), b"x").unwrap();
        // Non-hidden stray file at the root.
        #[allow(clippy::disallowed_methods)]
        std::fs::write(dir.path().join("README.txt"), b"meta").unwrap();
        let err = scan_dataset(dir.path()).expect_err("stray root file rejects");
        match err {
            FinetuneError::BadDataset { path, reason } => {
                assert!(path.ends_with("README.txt"));
                assert!(
                    reason.contains("not a directory"),
                    "reason must mention non-directory rejection; got {reason:?}",
                );
            }
            other => panic!("expected BadDataset, got {other:?}"),
        }
    }

    /// Hidden root entries (leading `.`) are silently ignored,
    /// matching the upload-time `AssetPath` leading-dot rule.
    #[test]
    fn scan_skips_hidden_root_entries() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir(dir.path().join("cat")).unwrap();
        std::fs::create_dir(dir.path().join("dog")).unwrap();
        for cls in ["cat", "dog"] {
            #[allow(clippy::disallowed_methods)]
            std::fs::write(dir.path().join(cls).join("s.wav"), b"x").unwrap();
        }
        // Hidden directory + hidden file at the root: both ignored.
        std::fs::create_dir(dir.path().join(".cache")).unwrap();
        #[allow(clippy::disallowed_methods)]
        std::fs::write(dir.path().join(".DS_Store"), b"").unwrap();
        let (classes, examples) = scan_dataset(dir.path()).expect("hidden ignored");
        assert_eq!(classes, vec!["cat".to_string(), "dog".to_string()]);
        assert_eq!(examples.len(), 2);
    }

    /// Duplicate class labels under ASCII case-insensitive
    /// comparison reject with `BadDataset`.  Skipped on
    /// case-insensitive filesystems (default macOS APFS / HFS+)
    /// where the host cannot stage `Cat/` + `cat/` as siblings;
    /// Linux ext4 / xfs are case-sensitive so the test runs there.
    #[test]
    fn scan_rejects_case_insensitive_duplicate_labels() {
        let dir = tempfile::tempdir().expect("tempdir");
        // Probe case-sensitivity: create `case-probe-A`, then try
        // to create `case-probe-a`.  On a case-insensitive
        // filesystem the second create returns
        // `AlreadyExists`; on a case-sensitive filesystem both
        // succeed.
        let probe_a = dir.path().join("case-probe-A");
        let probe_b = dir.path().join("case-probe-a");
        std::fs::create_dir(&probe_a).expect("probe a");
        let case_sensitive = std::fs::create_dir(&probe_b).is_ok();
        std::fs::remove_dir_all(&probe_a).ok();
        std::fs::remove_dir_all(&probe_b).ok();
        if !case_sensitive {
            eprintln!(
                "skipping r2_me9_scan_rejects_case_insensitive_duplicate_labels: \
                 host filesystem is case-insensitive (cannot stage Cat/ + cat/ as siblings)"
            );
            return;
        }

        std::fs::create_dir(dir.path().join("Cat")).unwrap();
        std::fs::create_dir(dir.path().join("cat")).unwrap();
        for cls in ["Cat", "cat"] {
            #[allow(clippy::disallowed_methods)]
            std::fs::write(dir.path().join(cls).join("s.wav"), b"x").unwrap();
        }
        let err = scan_dataset(dir.path()).expect_err("case-insensitive duplicate rejects");
        match err {
            FinetuneError::BadDataset { reason, .. } => {
                assert!(
                    reason.contains("duplicate class label") && reason.contains("case-insensitive"),
                    "reason must mention case-insensitive duplicate; got {reason:?}",
                );
            }
            other => panic!("expected BadDataset, got {other:?}"),
        }
    }

    /// A class folder with no non-hidden `.wav` files anywhere
    /// under its subtree rejects with `BadDataset`.
    #[test]
    fn scan_rejects_empty_class_folder() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir(dir.path().join("cat")).unwrap();
        std::fs::create_dir(dir.path().join("dog")).unwrap();
        // Only `dog` has a wav; `cat` is empty.
        #[allow(clippy::disallowed_methods)]
        std::fs::write(dir.path().join("dog").join("s.wav"), b"x").unwrap();
        let err = scan_dataset(dir.path()).expect_err("empty class rejects");
        match err {
            FinetuneError::BadDataset { path, reason } => {
                assert!(path.ends_with("cat"));
                assert!(reason.contains("no non-hidden .wav"));
            }
            other => panic!("expected BadDataset, got {other:?}"),
        }
    }

    /// Classes are sorted by canonical byte order so the published
    /// head's label order is deterministic across hosts.
    #[test]
    fn scan_sorts_classes_by_byte_order() {
        let dir = tempfile::tempdir().expect("tempdir");
        for cls in ["zebra", "ant", "manatee"] {
            std::fs::create_dir(dir.path().join(cls)).unwrap();
            #[allow(clippy::disallowed_methods)]
            std::fs::write(dir.path().join(cls).join("s.wav"), b"x").unwrap();
        }
        let (classes, _) = scan_dataset(dir.path()).expect("scan");
        assert_eq!(
            classes,
            vec![
                "ant".to_string(),
                "manatee".to_string(),
                "zebra".to_string()
            ],
        );
    }

    /// Nested non-hidden `.wav` files under a class folder count
    /// as samples; hidden subdirs and non-`.wav` regular files are
    /// skipped.
    #[test]
    fn scan_recursive_discovery_picks_up_nested_wavs() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir(dir.path().join("cat")).unwrap();
        std::fs::create_dir(dir.path().join("cat").join("subdir")).unwrap();
        // Hidden subdir (skipped).
        std::fs::create_dir(dir.path().join("cat").join(".cache")).unwrap();
        std::fs::create_dir(dir.path().join("dog")).unwrap();
        for path in [
            "cat/a.wav",
            "cat/subdir/b.wav",
            "cat/subdir/c.txt",       // non-wav, skipped
            "cat/.cache/skipped.wav", // hidden, skipped
            "dog/d.wav",
        ] {
            #[allow(clippy::disallowed_methods)]
            std::fs::write(dir.path().join(path), b"x").unwrap();
        }
        let (classes, examples) = scan_dataset(dir.path()).expect("scan");
        assert_eq!(classes, vec!["cat".to_string(), "dog".to_string()]);
        // cat: a.wav + subdir/b.wav = 2; dog: d.wav = 1.
        assert_eq!(examples.len(), 3);
        let cat_count = examples.iter().filter(|(_, l)| *l == 0).count();
        assert_eq!(cat_count, 2);
    }

    /// Unsupported file types (FIFO, device, socket) inside a
    /// class folder reject with `BadDataset`.  Test uses a Unix
    /// domain socket because `std::fs` can create one without
    /// root.
    #[cfg(unix)]
    #[test]
    fn scan_rejects_unsupported_file_type_inside_class() {
        use std::os::unix::net::UnixListener;
        let dir = tempfile::tempdir().expect("tempdir");
        let cls = dir.path().join("cat");
        std::fs::create_dir(&cls).unwrap();
        #[allow(clippy::disallowed_methods)]
        std::fs::write(cls.join("a.wav"), b"x").unwrap();
        // Tempdir paths fit `sockaddr_un.sun_path` on macOS (104)
        // and Linux (108).
        let sock_path = cls.join("strange.sock");
        let _listener = UnixListener::bind(&sock_path).expect("bind unix socket");

        let err = scan_dataset(dir.path()).expect_err("unsupported file rejects");
        match err {
            FinetuneError::BadDataset { path, reason } => {
                assert!(path.ends_with("strange.sock"));
                assert!(reason.contains("unsupported file type"));
            }
            other => panic!("expected BadDataset, got {other:?}"),
        }
    }

    /// Root-level symlinks reject as non-directories;
    /// `symlink_metadata` does not follow the link, so a
    /// symlink-to-directory still rejects.
    #[cfg(unix)]
    #[test]
    fn scan_rejects_symlink_at_root_level() {
        use std::os::unix::fs::symlink;
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir(dir.path().join("cat")).unwrap();
        #[allow(clippy::disallowed_methods)]
        std::fs::write(dir.path().join("cat").join("a.wav"), b"x").unwrap();
        // Symlink the `cat` directory to `cat-link` (which would
        // otherwise look like a second class folder).
        symlink(dir.path().join("cat"), dir.path().join("cat-link")).unwrap();

        let err = scan_dataset(dir.path()).expect_err("symlink rejects");
        match err {
            FinetuneError::BadDataset { path, reason } => {
                assert!(path.ends_with("cat-link"));
                assert!(reason.contains("not a directory"));
            }
            other => panic!("expected BadDataset, got {other:?}"),
        }
    }

    /// `BadDataset` -> 400, `DatasetRead` -> 500; pins the
    /// operator-vs-daemon-internal split.
    #[test]
    fn finetune_error_kinds_classify_correctly() {
        use crate::common::error::{Categorized, ErrorKind};
        let bad = FinetuneError::BadDataset {
            path: "/x".into(),
            reason: "y".into(),
        };
        assert_eq!(bad.kind(), ErrorKind::UserInput);
        let read = FinetuneError::DatasetRead {
            path: "/x".into(),
            reason: "y".into(),
        };
        assert_eq!(read.kind(), ErrorKind::Internal);
    }
}
