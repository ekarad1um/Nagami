//! Runtime storage hygiene: periodic sweep of orphan `.tmp/`
//! entries (crashed uploads / staged-but-unfinished deletes) and
//! aged-out per-workspace job logs.
//!
//! # Scope
//!
//! Two distinct kinds of clutter accumulate under the workspace
//! tree over a daemon's lifetime:
//!
//! 1. **`.tmp/` orphans.** Three places stage temporaries:
//!    * `<root>/.tmp/` -- workspace-delete staging
//!      (`delete-workspace-<job>/` + tombstone JSON)
//!    * `<root>/active/.tmp/<activation_id>/` -- pre-publish
//!      active-head staging (atomic-renamed into
//!      `active/generations/<id>/` on success)
//!    * `<workspace>/.tmp/` -- per-workspace asset upload
//!      tempfiles, asset / converter / log delete tombstones +
//!      payloads
//!
//!    The boot-time recovery sweep ([`super::recovery::recover_all`])
//!    drains tombstones and unlinks stage orphans at every
//!    restart, but on a daemon that crashes hard and never
//!    restarts (or crashes between two restarts in a way that
//!    bypasses recovery -- e.g. `kill -9` during a window where
//!    the writer had already created the staging file but not yet
//!    completed the operation) the entries sit indefinitely.
//!    Power-loss orphans from `tempfile::NamedTempFile` (mkstemp)
//!    are the most common case: the file's `Drop` impl unlinks on
//!    clean shutdown but cannot run on hard crash.
//!
//! 2. **Per-workspace job logs.** `<workspace>/training_logs/`
//!    and `<workspace>/converter_logs/` accumulate one `*.jsonl`
//!    per job run.  There is no built-in retention -- they grow
//!    until the operator explicitly issues `DELETE
//!    /workspace/{id}/assets/{tree}` or until the workspace is
//!    removed.  On a busy operator a year of daily training jobs
//!    leaves 365 entries per workspace, each containing the
//!    full epoch / batch trace -- non-trivial storage on a
//!    bounded SBC.
//!
//! The daemon-root `<root>/logs/acousticsd.log.*` does **not**
//! flow through here: `tracing_appender::rolling::Rotation::DAILY`
//! plus `max_log_files(7)` already prunes it, and the rotation
//! lives inside the appender's own writer thread.
//!
//! # Safety -- aging vs. concurrent operations
//!
//! The reaper deletes entries whose mtime exceeds an age
//! threshold.  Concurrent writers race the sweep only if their
//! mtime is older than the threshold AND they are still active:
//!
//! * Upload tempfiles: an `NamedTempFile` in `<workspace>/.tmp/`
//!   has mtime ≈ upload duration (seconds-to-minutes on eMMC).
//!   The 24 h default threshold is three orders of magnitude
//!   above that.
//! * Activation staging dirs: rename-to-`generations/` is
//!   atomic; staging mtime ≤ activation duration (~milliseconds
//!   to a few seconds).  No realistic race.
//! * Delete payloads: drain is bounded to 256 entries / batch
//!   inside a single critical section; full-tree drains complete
//!   in seconds.  A 24 h-old payload means the daemon crashed
//!   mid-drain; the reaper finishes the operator's intent.
//! * Per-workspace log files: each `*.jsonl` is written by a
//!   producer whose mtime advances on every event.  An idle log
//!   older than the 30 d default is a stale job; pruning frees
//!   the inode.
//!
//! No coordination lock with `WorkspaceMgr` is required because
//! the threshold is far above any legitimate operation duration.
//! Racing the sweep at the inode level is benign: `NamedTempFile`'s
//! `persist()` returns `io::ErrorKind::NotFound` and the producer
//! treats it as a normal failed-upload (rollback + error to
//! caller); no torn write reaches the asset tree.
//!
//! # I/O shape
//!
//! Synchronous + blocking.  Callers (the daemon's periodic
//! `storage_reaper` background task) wrap in
//! `tokio::task::spawn_blocking` so the reader thread is free
//! during a sweep.  The walk is at most one
//! `read_dir` per workspace + a per-entry `metadata()` so the
//! cost is `O(workspaces * |.tmp entries|)` and dominated by
//! syscall round-trips.  On a 100-workspace device with empty
//! `.tmp/` dirs the sweep is microseconds; with a stuck
//! 1k-entry orphan it's milliseconds.  Either way well under the
//! 1 h period.

use crate::file_mgr::dataset::{CONVERTER_LOGS_DIR_NAME, TRAINING_LOGS_DIR_NAME};
use crate::file_mgr::error::{FileError, io_err};
use crate::file_mgr::schema::{
    ROOT_TMP_DIR_NAME, active_staging_dir, root_tmp_dir, workspaces_dir,
};
use std::path::Path;
use std::time::{Duration, SystemTime};

/// Age thresholds and dispatch knobs for a single sweep pass.
///
/// Both `tmp_age` and `log_age` are operator-tunable in spirit;
/// today they are constants in the daemon binary's reaper wiring
/// (`modules/daemon/main_body.rs`).  Promotion to the launch TOML
/// is a follow-up if operators ask for tighter / looser windows.
#[derive(Clone, Copy, Debug)]
pub struct SweepConfig {
    /// Reap `.tmp/` entries whose mtime is older than this.
    /// `Duration::from_secs(24 * 3600)` is the daemon default --
    /// orders of magnitude above any in-flight operation.
    pub tmp_age: Duration,
    /// Prune per-workspace `*_logs/*.jsonl` files older than
    /// this.  `Duration::from_secs(30 * 24 * 3600)` is the
    /// daemon default; keeps a month of job history per
    /// workspace.
    pub log_age: Duration,
}

/// Outcome of one [`sweep_once`] pass.  Numbers feed the
/// `WorkspaceMetrics` counters and the daemon's tracing log so
/// operators can see "did the reaper actually do anything" at a
/// glance.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SweepReport {
    /// `.tmp/` entries reaped (root + active + every workspace).
    pub tmp_orphans_reaped: u64,
    /// Stale `*.jsonl` files removed from `training_logs/` +
    /// `converter_logs/`.
    pub log_files_pruned: u64,
    /// Per-workspace directories scanned (count of distinct
    /// `<root>/workspaces/<id>/` walked).  Zero when the
    /// workspaces parent itself is absent (fresh daemon).
    pub workspaces_scanned: u64,
    /// Per-workspace sweep failures.  Each one is logged
    /// `tracing::warn!` with the path; the sweep continues with
    /// the next workspace so a single corrupt entry cannot
    /// disable the reaper.
    pub failures: u64,
}

impl SweepReport {
    /// True iff the sweep removed at least one entry.  Used by
    /// the daemon's periodic task to decide whether to log at
    /// `info` (work happened) or stay silent (cheap no-op).
    pub fn did_work(&self) -> bool {
        self.tmp_orphans_reaped > 0 || self.log_files_pruned > 0
    }
}

/// Run a single sweep pass over `<root>/.tmp/`,
/// `<root>/active/.tmp/`, and every `<workspace>/.tmp/` +
/// `*_logs/` underneath `<root>/workspaces/`.  Returns the
/// counts collected -- the caller wires them into
/// `WorkspaceMetrics`.
///
/// Per-workspace failures are isolated: a sweep failure on
/// workspace A does not skip workspace B.  The function only
/// returns `Err` when the workspaces-root walk itself fails
/// (e.g. read_dir on a corrupt FS) -- that case is rare enough
/// that propagating to the caller is the right shape, since the
/// daemon-level tracing call site logs the error and continues
/// at the next tick.
pub fn sweep_once(root: &Path, cfg: &SweepConfig) -> Result<SweepReport, FileError> {
    let now = SystemTime::now();
    let mut report = SweepReport::default();

    // 1. Root-level `.tmp/` (workspace-delete staging).
    sweep_dir_entries(&root_tmp_dir(root), now, cfg.tmp_age, &mut report);

    // 2. Active-head pre-publish staging.  `active_staging_dir`
    //    is the canonical helper for `<root>/active/.tmp/`; use
    //    it rather than inlining the join so a future refactor
    //    of the active-tree layout flows through one place.
    sweep_dir_entries(&active_staging_dir(root), now, cfg.tmp_age, &mut report);

    // 3. Per-workspace `.tmp/` + log dirs.
    let workspaces = workspaces_dir(root);
    let entries = match std::fs::read_dir(&workspaces) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(report),
        Err(e) => return Err(io_err(workspaces.display(), e)),
    };
    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!(
                    target: "file_mgr",
                    err = %e,
                    parent = %workspaces.display(),
                    "storage reaper: workspaces dir-iter entry failed",
                );
                report.failures += 1;
                continue;
            }
        };
        let file_type = match entry.file_type() {
            Ok(ft) => ft,
            Err(e) => {
                tracing::warn!(
                    target: "file_mgr",
                    err = %e,
                    path = %entry.path().display(),
                    "storage reaper: workspace file_type probe failed",
                );
                report.failures += 1;
                continue;
            }
        };
        if !file_type.is_dir() {
            continue;
        }
        report.workspaces_scanned += 1;
        let ws = entry.path();
        sweep_workspace(&ws, now, cfg, &mut report);
    }
    Ok(report)
}

/// Sweep one workspace's `.tmp/` + the two log dirs.
/// Internal helper kept lazy on the dir presence (any of the
/// three may legitimately be absent under the lazy-mkdir
/// contract introduced alongside this reaper).
fn sweep_workspace(ws: &Path, now: SystemTime, cfg: &SweepConfig, report: &mut SweepReport) {
    sweep_dir_entries(&ws.join(ROOT_TMP_DIR_NAME), now, cfg.tmp_age, report);
    prune_old_files(
        &ws.join(TRAINING_LOGS_DIR_NAME),
        now,
        cfg.log_age,
        &mut report.log_files_pruned,
        &mut report.failures,
    );
    prune_old_files(
        &ws.join(CONVERTER_LOGS_DIR_NAME),
        now,
        cfg.log_age,
        &mut report.log_files_pruned,
        &mut report.failures,
    );
}

/// Remove every direct entry in `dir` whose mtime is older than
/// `age`.  Missing `dir` is a no-op (lazy mkdir means many
/// workspaces never materialize their `.tmp/`).  Per-entry
/// failures (race against a concurrent writer, EACCES on a
/// readonly mount) increment `report.failures` and continue.
fn sweep_dir_entries(dir: &Path, now: SystemTime, age: Duration, report: &mut SweepReport) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return,
        Err(e) => {
            tracing::warn!(
                target: "file_mgr",
                err = %e,
                path = %dir.display(),
                "storage reaper: read_dir failed",
            );
            report.failures += 1;
            return;
        }
    };
    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!(
                    target: "file_mgr",
                    err = %e,
                    parent = %dir.display(),
                    "storage reaper: dir-iter entry failed",
                );
                report.failures += 1;
                continue;
            }
        };
        let path = entry.path();
        let metadata = match entry.metadata() {
            Ok(m) => m,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => {
                tracing::warn!(
                    target: "file_mgr",
                    err = %e,
                    path = %path.display(),
                    "storage reaper: metadata probe failed",
                );
                report.failures += 1;
                continue;
            }
        };
        let mtime = match metadata.modified() {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!(
                    target: "file_mgr",
                    err = %e,
                    path = %path.display(),
                    "storage reaper: mtime probe failed (platform does not expose mtime?)",
                );
                report.failures += 1;
                continue;
            }
        };
        // `duration_since` returns `Err` when the file's mtime is
        // *in the future* relative to `now` (clock skew on
        // NFS-style mounts, manual `touch -t`).  Treat as
        // not-yet-aged and skip.
        let aged_out = matches!(now.duration_since(mtime), Ok(d) if d > age);
        if !aged_out {
            continue;
        }
        let file_type = metadata.file_type();
        let res = if file_type.is_dir() {
            std::fs::remove_dir_all(&path)
        } else {
            std::fs::remove_file(&path)
        };
        match res {
            Ok(()) => {
                report.tmp_orphans_reaped += 1;
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // Raced with a concurrent producer that
                // finalized + cleaned up between our `stat` and
                // `remove`.  Treat as success-with-no-op.
            }
            Err(e) => {
                tracing::warn!(
                    target: "file_mgr",
                    err = %e,
                    path = %path.display(),
                    "storage reaper: remove failed",
                );
                report.failures += 1;
            }
        }
    }
}

/// Remove regular files in `dir` older than `age`.  Skips
/// subdirectories (defensive: production log dirs hold only flat
/// `*.jsonl` files, but an operator-pasted subdir should not
/// disappear silently).
fn prune_old_files(
    dir: &Path,
    now: SystemTime,
    age: Duration,
    pruned: &mut u64,
    failures: &mut u64,
) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return,
        Err(e) => {
            tracing::warn!(
                target: "file_mgr",
                err = %e,
                path = %dir.display(),
                "storage reaper: log read_dir failed",
            );
            *failures += 1;
            return;
        }
    };
    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!(
                    target: "file_mgr",
                    err = %e,
                    parent = %dir.display(),
                    "storage reaper: log dir-iter entry failed",
                );
                *failures += 1;
                continue;
            }
        };
        let path = entry.path();
        let metadata = match entry.metadata() {
            Ok(m) => m,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => {
                tracing::warn!(
                    target: "file_mgr",
                    err = %e,
                    path = %path.display(),
                    "storage reaper: log metadata probe failed",
                );
                *failures += 1;
                continue;
            }
        };
        if !metadata.file_type().is_file() {
            continue;
        }
        let mtime = match metadata.modified() {
            Ok(m) => m,
            Err(e) => {
                // Mirror `sweep_dir_entries`'s shape: log +
                // count rather than silently skipping.  A
                // platform that cannot expose mtime is a
                // configuration/build oddity an operator
                // should see in the logs, not a quiet drop.
                tracing::warn!(
                    target: "file_mgr",
                    err = %e,
                    path = %path.display(),
                    "storage reaper: log mtime probe failed",
                );
                *failures += 1;
                continue;
            }
        };
        let aged_out = matches!(now.duration_since(mtime), Ok(d) if d > age);
        if !aged_out {
            continue;
        }
        match std::fs::remove_file(&path) {
            Ok(()) => {
                *pruned += 1;
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                tracing::warn!(
                    target: "file_mgr",
                    err = %e,
                    path = %path.display(),
                    "storage reaper: log remove failed",
                );
                *failures += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    // Tests stage fixtures with `std::fs::*` + `filetime::set_file_mtime`;
    // the production constraint in `clippy.toml` (writes through
    // file_mgr) does not apply to test setup helpers.
    #![allow(clippy::disallowed_methods)]
    use super::*;
    use std::fs;
    use std::time::{Duration, UNIX_EPOCH};

    /// Helper: backdate a path's mtime to `now - age` so the
    /// reaper sees it as expired.  Uses `filetime` because
    /// `std::fs` does not expose mtime-setting on stable Rust.
    fn backdate(path: &Path, age: Duration) {
        let target = SystemTime::now()
            .checked_sub(age)
            .expect("backdate clock subtraction");
        set_mtime_at(path, target);
    }

    /// Helper: forward-date a path's mtime to `now + ahead`.
    /// Used by [`sweep_skips_future_mtime_entries`] to exercise
    /// the clock-skew safety branch (`duration_since` returns
    /// `Err` when mtime > now, which the reaper treats as
    /// not-yet-aged).
    fn forward_date(path: &Path, ahead: Duration) {
        let target = SystemTime::now()
            .checked_add(ahead)
            .expect("forward_date clock addition");
        set_mtime_at(path, target);
    }

    fn set_mtime_at(path: &Path, target: SystemTime) {
        let secs = target
            .duration_since(UNIX_EPOCH)
            .expect("post-epoch")
            .as_secs();
        let ft = filetime::FileTime::from_unix_time(secs as i64, 0);
        filetime::set_file_mtime(path, ft).expect("set mtime");
    }

    /// Helper: build an empty workspace tree skeleton with a
    /// per-workspace `.tmp/` and the two log dirs already
    /// materialized.  Mirrors the post-first-producer shape.
    fn build_workspace_skeleton(root: &Path, id: &str) -> std::path::PathBuf {
        let ws = root.join("workspaces").join(id);
        fs::create_dir_all(ws.join(".tmp")).unwrap();
        fs::create_dir_all(ws.join("training_logs")).unwrap();
        fs::create_dir_all(ws.join("converter_logs")).unwrap();
        ws
    }

    /// Daemon production-default thresholds: 24 h on `.tmp/`,
    /// 30 d on per-workspace log files.  Matches the constants
    /// wired in `daemon::main_body` so the tests exercise the
    /// same window operators see in deployment.  Tests that need
    /// a different threshold use struct-update syntax, e.g.
    /// `SweepConfig { tmp_age: Duration::from_nanos(1), ..default_cfg() }`.
    fn default_cfg() -> SweepConfig {
        SweepConfig {
            tmp_age: Duration::from_secs(24 * 3600),
            log_age: Duration::from_secs(30 * 24 * 3600),
        }
    }

    /// Sweep removes a file in `<root>/.tmp/` aged past the
    /// threshold AND leaves the dir itself in place (the dir's
    /// mtime is fresh enough to skip).
    #[test]
    fn sweep_reaps_aged_root_tmp_file() {
        let tmp = tempfile::tempdir().unwrap();
        let root_tmp = tmp.path().join(".tmp");
        fs::create_dir_all(&root_tmp).unwrap();
        let stale = root_tmp.join("delete-workspace-stale.json");
        fs::write(&stale, b"{}").unwrap();
        backdate(&stale, Duration::from_secs(48 * 3600));

        let report = sweep_once(tmp.path(), &default_cfg()).expect("sweep");
        assert_eq!(report.tmp_orphans_reaped, 1);
        assert_eq!(report.log_files_pruned, 0);
        assert_eq!(report.failures, 0);
        assert!(!stale.exists());
        assert!(root_tmp.is_dir(), "parent .tmp/ itself stays in place");
    }

    /// Sweep skips entries whose mtime is *in the future*
    /// relative to wall-clock `now`.  This is the clock-skew
    /// safety branch -- `SystemTime::duration_since` returns
    /// `Err(SystemTimeError)` for future stamps and the reaper
    /// pattern-matches only the `Ok(d)` arm.  An operator who
    /// touches a fixture file with a forward timestamp (or a
    /// FUSE/NFS mount with clock skew) must NOT see their data
    /// reaped just because the daemon's clock disagrees.
    #[test]
    fn sweep_skips_future_mtime_entries() {
        let tmp = tempfile::tempdir().unwrap();
        let root_tmp = tmp.path().join(".tmp");
        fs::create_dir_all(&root_tmp).unwrap();
        let future = root_tmp.join("forward-stamped.json");
        fs::write(&future, b"{}").unwrap();
        // 1 h ahead of wall clock -- well past the typical
        // skew an NTP-synced host would tolerate, but a
        // forensic / fixture scenario can produce it.
        forward_date(&future, Duration::from_secs(3600));

        // Use the tightest possible `tmp_age` (1 ns).  Any
        // sane "is the file older than X?" predicate would
        // reap a present-time file under this config; only the
        // future-mtime skip keeps `future` alive.
        let cfg = SweepConfig {
            tmp_age: Duration::from_nanos(1),
            ..default_cfg()
        };
        let report = sweep_once(tmp.path(), &cfg).expect("sweep");
        assert_eq!(
            report.tmp_orphans_reaped, 0,
            "future-mtime entry must not be reaped",
        );
        assert_eq!(
            report.failures, 0,
            "future-mtime skip must not surface as a failure",
        );
        assert!(future.exists(), "future-stamped fixture survived sweep");
    }

    /// Sweep leaves a fresh file alone (mtime within
    /// `tmp_age`).  Guards against the false-positive: a fast
    /// reaper would otherwise nuke in-flight uploads.
    #[test]
    fn sweep_skips_fresh_root_tmp_file() {
        let tmp = tempfile::tempdir().unwrap();
        let root_tmp = tmp.path().join(".tmp");
        fs::create_dir_all(&root_tmp).unwrap();
        let fresh = root_tmp.join("delete-workspace-fresh.json");
        fs::write(&fresh, b"{}").unwrap();
        // Default mtime: now.  No backdate.

        let report = sweep_once(tmp.path(), &default_cfg()).expect("sweep");
        assert_eq!(report.tmp_orphans_reaped, 0);
        assert!(fresh.exists());
    }

    /// Sweep handles a dir entry inside `.tmp/`: a stale
    /// `delete-workspace-<job>/payload/` subtree (the
    /// canonical workspace-delete staging shape) is removed
    /// recursively.
    #[test]
    fn sweep_reaps_aged_root_tmp_dir_tree() {
        let tmp = tempfile::tempdir().unwrap();
        let staging = tmp
            .path()
            .join(".tmp")
            .join("delete-workspace-stale")
            .join("payload");
        fs::create_dir_all(&staging).unwrap();
        fs::write(staging.join("a.bin"), b"junk").unwrap();
        let parent = staging.parent().unwrap();
        backdate(parent, Duration::from_secs(48 * 3600));

        let report = sweep_once(tmp.path(), &default_cfg()).expect("sweep");
        assert_eq!(report.tmp_orphans_reaped, 1);
        assert!(!parent.exists(), "whole subtree removed");
    }

    /// Per-workspace `.tmp/` entries are reaped on the same
    /// rule.  Confirms the recursion into `<workspaces>/<id>/`.
    #[test]
    fn sweep_reaps_aged_per_workspace_tmp() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = build_workspace_skeleton(tmp.path(), "ws-a");
        let stale = ws.join(".tmp").join("delete-assets-stale.json");
        fs::write(&stale, b"{}").unwrap();
        backdate(&stale, Duration::from_secs(48 * 3600));

        let report = sweep_once(tmp.path(), &default_cfg()).expect("sweep");
        assert_eq!(report.tmp_orphans_reaped, 1);
        assert_eq!(report.workspaces_scanned, 1);
        assert!(!stale.exists());
    }

    /// Log pruning removes a `.jsonl` older than `log_age`
    /// and keeps a fresh sibling.  Pins the per-workspace
    /// retention contract.
    #[test]
    fn sweep_prunes_aged_log_keeps_fresh() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = build_workspace_skeleton(tmp.path(), "ws-logs");
        let stale = ws.join("training_logs").join("stale.jsonl");
        let fresh = ws.join("training_logs").join("fresh.jsonl");
        fs::write(&stale, b"{}").unwrap();
        fs::write(&fresh, b"{}").unwrap();
        backdate(&stale, Duration::from_secs(60 * 24 * 3600));

        let report = sweep_once(tmp.path(), &default_cfg()).expect("sweep");
        assert_eq!(report.log_files_pruned, 1);
        assert!(!stale.exists());
        assert!(fresh.exists());
    }

    /// Log pruning skips subdirectories.  Production log dirs
    /// hold only `*.jsonl`, but an operator-pasted subdir
    /// should survive a sweep so the reaper does not silently
    /// nuke operator artifacts.
    #[test]
    fn sweep_skips_subdirs_inside_log_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = build_workspace_skeleton(tmp.path(), "ws-subdir");
        let sub = ws.join("training_logs").join("operator-stash");
        fs::create_dir_all(&sub).unwrap();
        backdate(&sub, Duration::from_secs(60 * 24 * 3600));

        let report = sweep_once(tmp.path(), &default_cfg()).expect("sweep");
        assert_eq!(report.log_files_pruned, 0);
        assert!(sub.is_dir());
    }

    /// Sweep tolerates a completely absent layout: a fresh
    /// daemon with no workspaces, no `.tmp/`, no `active/`
    /// returns an all-zero report rather than failing on
    /// ENOENT.  Mirrors the lazy-mkdir contract: subsystems
    /// materialize their own dirs.
    #[test]
    fn sweep_no_op_on_fresh_root() {
        let tmp = tempfile::tempdir().unwrap();
        let report = sweep_once(tmp.path(), &default_cfg()).expect("sweep no-op");
        assert_eq!(report, SweepReport::default());
    }

    /// Active-staging dirs (`<root>/active/.tmp/<uuid>/`)
    /// older than the threshold are reaped.  This is the
    /// crashed-activation cleanup path: an `activation_id`
    /// staging dir whose atomic-rename never completed.
    #[test]
    fn sweep_reaps_aged_active_staging() {
        let tmp = tempfile::tempdir().unwrap();
        let staging = tmp
            .path()
            .join("active")
            .join(".tmp")
            .join("00000000-0000-4000-8000-000000000001");
        fs::create_dir_all(&staging).unwrap();
        fs::write(staging.join("head.mpk"), b"junk").unwrap();
        backdate(&staging, Duration::from_secs(48 * 3600));

        let report = sweep_once(tmp.path(), &default_cfg()).expect("sweep");
        assert_eq!(report.tmp_orphans_reaped, 1);
        assert!(!staging.exists());
    }

    /// `SweepReport::did_work` returns true iff at least one
    /// entry was removed.  Used by the daemon's periodic task
    /// to decide whether to log at `info!` or stay silent.
    #[test]
    fn did_work_predicate_matches_counters() {
        let mut r = SweepReport::default();
        assert!(!r.did_work());
        r.failures = 5;
        assert!(!r.did_work(), "failures alone are not 'work'");
        r.tmp_orphans_reaped = 1;
        assert!(r.did_work());
        let r = SweepReport {
            log_files_pruned: 1,
            ..Default::default()
        };
        assert!(r.did_work());
    }
}
