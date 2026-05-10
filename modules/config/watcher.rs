//! Hot-reload worker: debounces filesystem events and runs the
//! validate-then-commit cycle for `Config`.
//!
//! Both helpers are `pub(crate)`; the only caller is
//! [`crate::config::ConfigCell::watch_with`].  The `ReloadCallback`
//! alias lives in `config.rs` since it is part of the public
//! `watch_with` signature.

use crate::config::Config;
use crate::config::ReloadCallback;
use crate::config::error::{ConfigError, ConfigValidationError, write_err};
use arc_swap::ArcSwap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub(crate) fn debounce_reload_worker(
    path: Arc<PathBuf>,
    inner: Arc<ArcSwap<Config>>,
    reload_count: Arc<std::sync::atomic::AtomicU64>,
    mutate_lock: Arc<parking_lot::Mutex<()>>,
    on_reload: Arc<ReloadCallback>,
    rx: std::sync::mpsc::Receiver<()>,
) {
    use std::sync::mpsc::RecvTimeoutError;
    use std::time::Duration;

    /// Quiet window after the most recent FS event before we
    /// consider the burst settled.  100 ms is short enough to feel
    /// instant to a human and long enough to absorb editor multi-
    /// write saves (vim tested at 3-5 events in <20 ms).
    const WATCHER_DEBOUNCE: Duration = Duration::from_millis(100);

    loop {
        // Block until the first kick.  `Err` means all senders have
        // dropped -- clean shutdown.
        if rx.recv().is_err() {
            return;
        }
        // Coalesce: keep extending the deadline as long as new
        // kicks arrive.  This is the "quiet window" semantics:
        // reload only after the file system goes quiet, no matter
        // how long the burst itself lasted.
        loop {
            match rx.recv_timeout(WATCHER_DEBOUNCE) {
                Ok(()) => continue,
                Err(RecvTimeoutError::Timeout) => break,
                Err(RecvTimeoutError::Disconnected) => return,
            }
        }
        // Burst settled -- perform exactly one reload.
        try_reload(
            &path,
            &inner,
            &reload_count,
            &mutate_lock,
            on_reload.as_ref(),
        );
    }
}

/// Read-parse-validate-swap cycle.  Errors are logged (not
/// returned) -- this runs in a background thread on operator
/// edits, so the only meaningful action is "log + keep previous".
///
/// # Concurrency
///
/// Acquires `mutate_lock` for the entire critical section, then
/// reads the file (lock-first, read-second).  The opposite order
/// would let us race an API mutation that has just taken the
/// lock and is about to write a new TOML body: we would read the
/// in-progress file mid-write and overwrite the API's value.
/// Holding the lock through file read, parse, validate, the
/// user-supplied `on_reload` callback, and the final
/// `inner.store` ensures concurrent API mutations serialize
/// behind the reload.
pub(crate) fn try_reload(
    path: &Path,
    inner: &ArcSwap<Config>,
    reload_count: &std::sync::atomic::AtomicU64,
    mutate_lock: &parking_lot::Mutex<()>,
    on_reload: &dyn Fn(&Config) -> Result<(), ConfigValidationError>,
) {
    // Lock first, THEN read.  The other order would let us read a
    // file mid-write by an API mutation that just acquired the
    // lock.
    let _guard = mutate_lock.lock();
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => {
            tracing::warn!(
                target: "config",
                path = %path.display(),
                err = %e,
                "config reload: read failed; keeping previous snapshot",
            );
            return;
        }
    };
    let cfg: Config = match toml::from_str(&text) {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(
                target: "config",
                path = %path.display(),
                err = %e,
                "config reload: parse failed; keeping previous snapshot",
            );
            return;
        }
    };
    if let Err(e) = cfg.validate() {
        tracing::warn!(
            target: "config",
            path = %path.display(),
            err = %e,
            "config reload: validation failed; keeping previous snapshot",
        );
        return;
    }
    // `MicPolicy` cross-validation against the launch catalogue
    // is delegated to the user-supplied `on_reload` callback --
    // the immutable launch catalogue isn't reachable from the
    // `config` crate alone.  The validate-then-commit flow below
    // gates `inner.store` on the callback returning `Ok`, so a
    // rejection cleanly preserves the prior snapshot (see the
    // match arms after the `cb_result` block).
    let prev = inner.load_full();
    // Compare BEFORE allocating the new Arc -- saves a heap
    // allocation on the spurious-reload case (e.g. editor wrote
    // identical content) which is a real and common scenario.
    if *prev == cfg {
        return;
    }
    drop(prev);

    // MARK: Validate-then-commit ordering
    // Run the user-supplied callback FIRST.  If it returns Err
    // (the daemon's cross-validation rejected the new value),
    // discard the reload entirely -- `inner` keeps its prior
    // value, no live ArcSwaps changed, log warn for the
    // operator.  Only on Ok do we commit to `inner` and
    // increment the reload counter.
    //
    // The callback may panic.  `catch_unwind` keeps the worker
    // alive; a panic is treated as Err-equivalent (reload
    // discarded).  `AssertUnwindSafe` is required because
    // `&dyn Fn(&Config) -> Result<(), ConfigValidationError>` is
    // not auto-UnwindSafe; the callback owns no shared mutable
    // state by contract (see `## Callback contract` on
    // `watch_with`).
    let cb_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| on_reload(&cfg)));
    match cb_result {
        Ok(Ok(())) => {
            // Validation passed and side effects (live ArcSwap
            // stores) have been applied by the callback.  Now
            // commit to `inner` so `config.snapshot()` agrees
            // with what's running.
            let arc = Arc::new(cfg);
            inner.store(arc);
            reload_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            tracing::info!(
                target: "config",
                path = %path.display(),
                "config reloaded from disk",
            );
        }
        Ok(Err(diag)) => {
            tracing::warn!(
                target: "config",
                path = %path.display(),
                err = %diag,
                "config reload rejected by callback; keeping prior snapshot",
            );
        }
        Err(payload) => {
            let msg = if let Some(s) = payload.downcast_ref::<&'static str>() {
                (*s).to_string()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "<non-string panic payload>".to_string()
            };
            tracing::error!(
                target: "config",
                panic = %msg,
                "config reload callback panicked; reload discarded -- fix the callback",
            );
        }
    }
}

/// Atomic write: serialize the TOML and delegate to the
/// canonical `file_mgr::fs_atomic::put_atomic` so the durability
/// protocol (tempfile + flush + sync_all + persist + parent
/// fsync) lives in exactly one place.
pub(crate) fn write_toml_atomically(path: &Path, cfg: &Config) -> Result<(), ConfigError> {
    let text = toml::to_string_pretty(cfg)?;
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    if !dir.exists() {
        std::fs::create_dir_all(dir).map_err(|e| write_err(dir.display(), e))?;
    }
    crate::file_mgr::fs_atomic::put_atomic(path, text.as_bytes()).map_err(file_to_config_err)
}

/// Map the two `FileError` variants `put_atomic` can return
/// into the parallel `ConfigError` shapes.  Other `FileError`
/// variants are unreachable from `put_atomic`; surface them as
/// `Write` with the formatted text so a future widening of the
/// primitive's failure set degrades gracefully instead of
/// panicking.  `pub(crate)` so `config::launch::write_launch_toml_atomically`
/// can share the mapping without duplicating it.
pub(crate) fn file_to_config_err(err: crate::file_mgr::FileError) -> ConfigError {
    match err {
        crate::file_mgr::FileError::Io { path, source } => write_err(path, source),
        crate::file_mgr::FileError::Persist(e) => ConfigError::Persist(e),
        other => write_err("", std::io::Error::other(format!("{other}"))),
    }
}
