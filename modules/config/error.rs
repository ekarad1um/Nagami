//! Config error type.
//!
//! Split out of `lib.rs` to bring the facade under the
//! 1,500-LoC layer-gate.  `ConfigError` is re-exported by [`crate`];
//! existing `config::ConfigError` import paths continue to work.

use thiserror::Error;

/// Per-section validation failure for [`crate::config::Config`].
///
/// Wraps the freeform `String` from each leaf validator
/// (`StreamCfg::validate`, `TrainingDefaults::validate`, etc.)
/// under a typed discriminator so the hot-reload callback (and
/// any future telemetry surface) can match on category instead
/// of regex-parsing log text.  `Display` renders as
/// `"<section>: <leaf>"`.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ConfigValidationError {
    /// `Config.inference` (`InferenceCfg::validate`) rejected.
    #[error("inference: {0}")]
    Inference(String),
    /// `Config.stream` (`StreamCfg::validate`) rejected.
    #[error("stream: {0}")]
    Stream(String),
    /// `Config.file` (`FileCfg::validate`) rejected.
    #[error("file: {0}")]
    File(String),
    /// `Config.training_defaults` (`TrainingDefaults::validate`)
    /// rejected.
    #[error("training_defaults: {0}")]
    TrainingDefaults(String),
    /// User-supplied [`crate::config::ConfigCell::watch_with`]
    /// callback rejected the reload (typically the daemon's
    /// cross-validation against the immutable launch catalogue).
    #[error("rejected by reload callback: {0}")]
    Callback(String),
}

/// Failure shapes from config load / mutate / persist.
/// Mapped to HTTP statuses via the
/// [`crate::common::error::Categorized`] impl below.
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("read {path}: {source}")]
    Read {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("write {path}: {source}")]
    Write {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("parse {path}: {source}")]
    Parse {
        path: String,
        #[source]
        source: toml::de::Error,
    },
    #[error("serialize: {0}")]
    Serialize(#[from] toml::ser::Error),
    #[error("watcher: {0}")]
    Watcher(#[from] notify::Error),
    #[error("persist: {0}")]
    Persist(#[from] tempfile::PersistError),
    /// Config parsed cleanly but a sub-section's `validate()`
    /// rejected its values.  Surfaced from `load()` so the daemon
    /// fails loudly at boot instead of clamping silently.
    #[error("invalid config {path}: {msg}")]
    Invalid { path: String, msg: String },
    /// A `mutate` / `mutate_then` callback re-entered `mutate_then`
    /// from the same thread (typically via the `after` hook
    /// invoking `config.mutate(...)` recursively).  Surfaced as a
    /// structured error rather than a silent `parking_lot::Mutex`
    /// deadlock (the underlying lock is non-reentrant).
    #[error("re-entrant config mutate detected; mutate callbacks must not re-enter mutate")]
    ReentrantMutate,
    /// Failed to spawn the debounce worker thread for `watch()`.
    /// In practice this only happens when the OS is out of thread
    /// resources entirely; surfaced explicitly so the failure mode
    /// isn't mis-attributed to a "read" error.
    #[error("spawn config-reload thread for {path}: {source}")]
    ThreadSpawn {
        path: String,
        #[source]
        source: std::io::Error,
    },
}

impl crate::common::error::Categorized for ConfigError {
    fn kind(&self) -> crate::common::error::ErrorKind {
        use crate::common::error::ErrorKind::*;
        match self {
            // Re-entrant mutate is a programmer error in
            // the daemon (a callback re-entered the lock).  Surface
            // it as `Internal` so it shows as 500 in API responses
            // -- operator can't fix without a code change.
            ConfigError::ReentrantMutate => Internal,
            // Operator-edited TOML failed parse or `validate()` --
            // they can fix the file and retry.
            ConfigError::Parse { .. } | ConfigError::Invalid { .. } => UserInput,
            // Anything else: daemon-internal IO / serializer /
            // notify-watcher / thread-spawn failure.
            ConfigError::Read { .. }
            | ConfigError::Write { .. }
            | ConfigError::Serialize(_)
            | ConfigError::Watcher(_)
            | ConfigError::Persist(_)
            | ConfigError::ThreadSpawn { .. } => Internal,
        }
    }
}

/// Shorthand for `ConfigError::Read { path: path.to_string(), source }`.
/// `path` is `impl Display` so call sites can pass `Path::display()`
/// adapters directly without per-site `to_string()` boilerplate.
pub(crate) fn read_err(path: impl std::fmt::Display, source: std::io::Error) -> ConfigError {
    ConfigError::Read {
        path: path.to_string(),
        source,
    }
}

/// Shorthand for `ConfigError::Write { path: path.to_string(), source }`.
/// Companion to [`read_err`].  Used by config + launch + watcher
/// writers; the `file_to_config_err` translator also routes a
/// `FileError::Io` destructure through here for consistency.
pub(crate) fn write_err(path: impl std::fmt::Display, source: std::io::Error) -> ConfigError {
    ConfigError::Write {
        path: path.to_string(),
        source,
    }
}

/// Shorthand for `ConfigError::Parse { path: path.to_string(), source }`.
/// Source is the TOML deserialiser error; the daemon and the
/// launch-config loader share the same parse-error shape.
pub(crate) fn parse_err(path: impl std::fmt::Display, source: toml::de::Error) -> ConfigError {
    ConfigError::Parse {
        path: path.to_string(),
        source,
    }
}
