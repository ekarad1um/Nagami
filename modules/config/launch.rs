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
use crate::config::error::{ConfigError, parse_err, read_err, write_err};
use crate::inference::{BackboneCatalogue, BackboneKind, BackboneRef};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

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
///   candidates ([`BackboneKind`] + path + optional sha256).
///   Loaded once at boot via
///   [`BackboneCatalogue::load_first_supported`]; kinds not
///   supported by the current build (e.g. `rknn` off cfg) are
///   silently skipped, so the same launch.toml can ship on both
///   host-dev macOS and aarch64 Rockchip devices.
///
/// Future fields (host id, hardware-specific tuning constants, etc.)
/// can land here without disturbing the user-pref TOML's reload
/// semantics.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct LaunchConfig {
    pub mic: MicCatalogue,
    /// Ordered list of inference-backbone candidates.  `#[serde(default)]`
    /// so older launch.toml files without a `[[backbone.candidates]]`
    /// table still load cleanly (the daemon then runs without
    /// inference, exactly as it does today when head files are
    /// missing).
    #[serde(default)]
    pub backbone: BackboneCatalogue,
}

impl LaunchConfig {
    /// First-boot defaults.  Includes a synthetic mock candidate so
    /// the daemon produces audio on a fresh macOS dev workstation
    /// without hand-editing the launch TOML -- mirrors the previous
    /// `Config::default_for` mock-audio fallback, just in the
    /// launch-immutable layer.  The backbone catalogue lists the
    /// canonical NPU + CPU candidates under `misc/` so a
    /// stock dev workflow keeps booting; operators on a real device
    /// override these to absolute paths in their own launch.toml.
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
            backbone: BackboneCatalogue {
                candidates: vec![
                    // Production NPU path (used on Linux + rknpu builds;
                    // silently skipped elsewhere by the loader).
                    BackboneRef {
                        kind: BackboneKind::Rknn,
                        path: PathBuf::from("misc/backbone.rknn"),
                        hash: None,
                    },
                    // CPU fallback (always supported).
                    BackboneRef {
                        kind: BackboneKind::Burn,
                        path: PathBuf::from("misc/backbone.mpk"),
                        hash: None,
                    },
                ],
            },
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
