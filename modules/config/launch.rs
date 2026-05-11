//! Launch-time deployment manifest (immutable at runtime) +
//! cross-validation helpers between user-pref `Config` and the
//! launch catalogues.
//!
//! Split out of `lib.rs` to bring the facade under the
//! 1,500-LoC layer-gate.  `LaunchConfig` and
//! `validate_policy_against_catalogue` are re-exported from
//! [`crate`]; existing import paths continue to work.

use crate::audio_io::mic_arbitrator::{
    CandidateSource, MicCandidate, MicCatalogue, MicPolicy, PolicyValidationError,
};
use crate::audio_io::mock::Waveform;
use crate::common::ids::MicId;
use crate::config::domain::{FileCfg, StreamCfg, TrainingDefaults};
use crate::config::error::{ConfigError, parse_err, read_err, write_err};
use crate::inference::BackboneCatalogue;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Launch-owned head settings.
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq, Default)]
#[serde(deny_unknown_fields)]
pub struct HeadLaunchConfig {
    /// Optional deployment-bundled default head.  When omitted,
    /// boot recovery and `POST /active { default: true }` cannot
    /// fall back to a bundled classifier.
    #[serde(default)]
    pub default: Option<DefaultHeadRef>,
}

impl HeadLaunchConfig {
    fn validate(&self) -> Result<(), String> {
        if let Some(default) = &self.default {
            default.validate()?;
        }
        Ok(())
    }
}

/// Explicit file pair for the deployment-bundled default head.
///
/// Keeping the `.mpk` and labels paths separate avoids daemon
/// assumptions about filenames or directory layouts; a launch TOML
/// can point at immutable assets wherever the deployment stores them.
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DefaultHeadRef {
    pub path: PathBuf,
    pub labels_path: PathBuf,
}

impl DefaultHeadRef {
    fn validate(&self) -> Result<(), String> {
        if self.path.as_os_str().is_empty() {
            return Err("head.default.path must be non-empty".into());
        }
        if self.labels_path.as_os_str().is_empty() {
            return Err("head.default.labels_path must be non-empty".into());
        }
        Ok(())
    }
}

/// Launch-time deployment manifest. **Read once at daemon boot;
/// never mutated by API, never hot-reloaded.** Operators edit the
/// file and restart the daemon to apply changes.
///
/// Holds catalogues / manifests that don't make sense to change
/// while the daemon is running:
///
/// * [`MicCatalogue`] -- which mics + which channels per mic are
///   available to the arbitrator.  The user-pref `Config::mic`
///   policy resolves against this catalogue at every mutation
///   point (boot, hot-reload, API).
/// * [`BackboneCatalogue`] -- ordered list of inference-backbone
///   candidates ([`crate::inference::BackboneKind`] + path +
///   optional sha256).
///   Loaded once at boot via
///   [`BackboneCatalogue::load_first_supported`]; kinds not
///   supported by the current build (e.g. `rknn` off cfg) are
///   silently skipped, so the same launch.toml can ship on both
///   host-dev macOS and aarch64 Rockchip devices.
/// * [`StreamCfg`] -- listener binds and stream admission policy.
///   Startup-only: not hot-reloaded and not API-mutable.
/// * [`HeadLaunchConfig`] -- optional explicit file pair for the
///   deployment-bundled default classifier head.
/// * [`TrainingDefaults`] -- default training hyperparameters
///   reported as starting values; never read on a hot path and no
///   API surface mutates them, so they live launch-side rather
///   than in the user-pref TOML.
/// * [`FileCfg`] -- file-service admission caps consumed once at
///   boot to construct `file_mgr::AdmissionCfg`.  Not hot-reloaded
///   (the constructed `FsServiceImpl` caches the caps), so it
///   belongs to the launch layer.
///
/// Future fields (host id, hardware-specific tuning constants, etc.)
/// can land here without disturbing the user-pref TOML's reload
/// semantics.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
#[serde(deny_unknown_fields)]
pub struct LaunchConfig {
    pub mic: MicCatalogue,
    /// Ordered list of inference-backbone candidates.  `#[serde(default)]`
    /// so older launch.toml files without a `[[backbone.candidates]]`
    /// table still load cleanly (the daemon then runs without
    /// inference, exactly as it does today when head files are
    /// missing).
    #[serde(default)]
    pub backbone: BackboneCatalogue,
    /// Stream listener binds + websocket admission policy.  Read
    /// once at boot so stream endpoints are not hot-configurable.
    pub stream: StreamCfg,
    /// Default-head launch settings.  `#[serde(default)]` keeps
    /// launch TOMLs without a bundled default valid; those daemons
    /// simply boot without bundled-default recovery.
    #[serde(default)]
    pub head: HeadLaunchConfig,
    /// Default training hyperparameters.  Loaded once at boot;
    /// per-job invocations override on a request-by-request basis,
    /// so the launch layer is the right home (the user-pref TOML
    /// never read this back at runtime).  `#[serde(default)]` so
    /// pre-migration launch TOMLs without a `[training_defaults]`
    /// block still load cleanly.
    #[serde(default)]
    pub training_defaults: TrainingDefaults,
    /// File-service admission caps.  Consumed once at boot to
    /// build the immutable `AdmissionCfg` handed to `FsServiceImpl`;
    /// no runtime mutator re-reads it.  `#[serde(default)]` keeps
    /// pre-migration launch TOMLs booting without an explicit
    /// `[file]` block.
    #[serde(default)]
    pub file: FileCfg,
}

impl LaunchConfig {
    /// First-boot defaults.  Includes a synthetic mock candidate so
    /// the daemon produces audio on a fresh macOS dev workstation
    /// without hand-editing the launch TOML -- mirrors the previous
    /// `Config::default_for` mock-audio fallback, just in the
    /// launch-immutable layer.  The backbone catalogue is empty by
    /// default: deployments must name concrete backbone artifacts in
    /// their launch TOML instead of inheriting daemon-hardcoded
    /// paths.
    pub fn default_for() -> Self {
        Self {
            mic: MicCatalogue {
                candidates: vec![MicCandidate {
                    id: MicId::from_static("default-mock"),
                    source: CandidateSource::Mock {
                        waveforms: vec![Waveform::Sine {
                            freq_hz: 1_000.0,
                            amplitude: 0.25,
                        }],
                        period_size: 512,
                        sample_rate: crate::common::dims::SampleRate::VALUE,
                    },
                    channels: vec![0],
                }],
            },
            backbone: BackboneCatalogue::default(),
            stream: StreamCfg::default(),
            head: HeadLaunchConfig::default(),
            training_defaults: TrainingDefaults::default(),
            file: FileCfg::default(),
        }
    }

    /// Load + validate from `path`.  If the file doesn't exist, an
    /// `Err(ConfigError::Read)` whose underlying `io::Error::kind()`
    /// is `NotFound` is returned -- the daemon's bootstrap path
    /// materializes [`Self::default_for`] in that case.
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let text = std::fs::read_to_string(path).map_err(|e| read_err(path.display(), e))?;
        let cfg: LaunchConfig = toml::from_str(&text).map_err(|e| parse_err(path.display(), e))?;
        if let Err((id, err)) = cfg.mic.validate() {
            return Err(ConfigError::Invalid {
                path: path.display().to_string(),
                msg: format!("launch mic catalogue: candidate {id}: {err}"),
            });
        }
        if let Err((idx, err)) = cfg.backbone.validate() {
            return Err(ConfigError::Invalid {
                path: path.display().to_string(),
                msg: format!("launch backbone catalogue: candidate[{idx}]: {err}"),
            });
        }
        cfg.stream.validate().map_err(|err| ConfigError::Invalid {
            path: path.display().to_string(),
            msg: format!("launch stream: {err}"),
        })?;
        cfg.head.validate().map_err(|err| ConfigError::Invalid {
            path: path.display().to_string(),
            msg: format!("launch head: {err}"),
        })?;
        // training_defaults / file validate at boot so a typo
        // (epochs = 0, max_upload_bytes = 0) surfaces in the
        // operator's systemd log rather than at first POST /train
        // / first upload.  Both used to live in the user-pref TOML
        // but were never read on a hot path or mutated by any API
        // route; the launch layer is the natural home.
        cfg.training_defaults
            .validate()
            .map_err(|err| ConfigError::Invalid {
                path: path.display().to_string(),
                msg: format!("launch training_defaults: {err}"),
            })?;
        cfg.file.validate().map_err(|err| ConfigError::Invalid {
            path: path.display().to_string(),
            msg: format!("launch file: {err}"),
        })?;
        Ok(cfg)
    }

    /// Persist to `path` via tempfile + atomic rename (same shape as
    /// [`crate::config::ConfigCell::persist`]).  Used by the daemon's
    /// bootstrap path when no launch TOML exists yet.
    pub fn persist(&self, path: &Path) -> Result<(), ConfigError> {
        write_launch_toml_atomically(path, self)
    }
}

/// Validate a [`MicPolicy`] (typically from a hot-reloaded
/// `Config`) against an immutable [`MicCatalogue`].  Surfaces a
/// `ConfigError::Invalid` so callers can propagate via the same
/// error channel as load-time issues.
pub fn validate_policy_against_catalogue(
    policy: &MicPolicy,
    catalogue: &MicCatalogue,
    path_for_diag: &Path,
) -> Result<(), ConfigError> {
    policy
        .validate_against(catalogue)
        .map_err(|e: PolicyValidationError| ConfigError::Invalid {
            path: path_for_diag.display().to_string(),
            msg: format!("mic policy: {e}"),
        })
}

fn write_launch_toml_atomically(path: &Path, cfg: &LaunchConfig) -> Result<(), ConfigError> {
    let text = toml::to_string_pretty(cfg)?;
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    if !dir.exists() {
        std::fs::create_dir_all(dir).map_err(|e| write_err(dir.display(), e))?;
    }
    crate::file_mgr::fs_atomic::put_atomic(path, text.as_bytes())
        .map_err(crate::config::watcher::file_to_config_err)
}
