//! Domain-specific config sub-sections: `StreamCfg` (launch),
//! `TrainingDefaults`, `FileCfg` (hot config).  Each
//! carries its own `validate()` predicate; `Config::validate`
//! aggregates them.
//!
//! Split out of `lib.rs` to bring the facade under the
//! 1,500-LoC layer-gate.  Public types are re-exported by [`crate`];
//! existing `config::StreamCfg` / `TrainingDefaults` import paths
//! continue to work.
//!
//! The active head is persisted under
//! `<workspace_root>/active/`; the daemon's first-boot path
//! materializes the bundled default generation if absent.  The
//! legacy `HeadRef` / `head_active` TOML contract was retired
//! during the workspace redesign.

use crate::stream_io::TransportPolicy;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Stream-server bind + permissions config.
///
/// The WS admission policy is per-listener: `tcp_policy` and
/// `uds_policy` each carry a [`TransportPolicy`] that the daemon
/// applies to the matching listener.  Defaults: `TransportPolicy::capped()`
/// for TCP, a relaxed policy for UDS (filesystem permissions are
/// already the auth boundary).  Both fields are `#[serde(default)]`
/// so existing TOML files load unchanged.
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct StreamCfg {
    /// Unix-domain-socket path for in-host clients.  Removed-and-rebound
    /// on daemon startup; permissions set per `uds_mode`.
    pub uds_path: PathBuf,
    /// Octal mode bits applied via `chmod` after bind.  0o660 is too
    /// tight for non-root tools; 0o666 lets any local user connect.
    /// Real deployments tune via group ownership instead.
    pub uds_mode: u32,
    /// `host:port` for the TCP listener.  `0.0.0.0:0` to bind ephemeral.
    pub tcp_bind: String,
    /// Broadcast channel capacity (per stream).  Lagging clients beyond
    /// this trigger a WS close (status 1011); see `stream_io`.
    pub broadcast_capacity: usize,
    /// Per-listener WS admission policy for the TCP listener.
    /// Capped 32-conn policy with strict subprotocol gate;
    /// `#[serde(default)]` so an existing TOML config without a
    /// `[stream.tcp_policy]` table loads cleanly.
    #[serde(default = "default_tcp_policy")]
    pub tcp_policy: TransportPolicy,
    /// Per-listener WS admission policy for the UDS
    /// listener.  Default is `TransportPolicy::capped()` with the
    /// subprotocol gate relaxed: UDS access is already gated by
    /// filesystem permissions (see `uds_mode`), and forcing local
    /// CLI tools to hand-roll `Sec-WebSocket-Protocol` is just
    /// friction.  Operators who want the strict gate can override
    /// in TOML.
    #[serde(default = "default_uds_policy")]
    pub uds_policy: TransportPolicy,
}

impl Default for StreamCfg {
    fn default() -> Self {
        Self {
            // Launch-time default for first boot.  Bundled dev and
            // production deployments should pin their actual socket
            // location in launch.toml; the daemon ensures the parent
            // before validation when it auto-materializes defaults.
            uds_path: PathBuf::from("var/run/acoustics_lab.sock"),
            uds_mode: 0o666,
            tcp_bind: "127.0.0.1:8787".into(),
            broadcast_capacity: 64,
            tcp_policy: default_tcp_policy(),
            uds_policy: default_uds_policy(),
        }
    }
}

/// Default `TransportPolicy` for the TCP listener: cap of 32
/// connections per stream, strict `Sec-WebSocket-Protocol` gate.
fn default_tcp_policy() -> TransportPolicy {
    TransportPolicy::capped()
}

/// Default `TransportPolicy` for the UDS listener.
/// Same connection cap as TCP, but `require_subprotocol = false`:
/// UDS access is already gated by filesystem permissions
/// (`uds_mode`), and local CLI tools (curl-like dev scripts)
/// rarely bother with the subprotocol header.  Operators who
/// want the strict gate can override `[stream.uds_policy]` in
/// TOML.
fn default_uds_policy() -> TransportPolicy {
    TransportPolicy {
        require_subprotocol: false,
        ..TransportPolicy::capped()
    }
}

impl StreamCfg {
    /// Predicates over the listener-side fields.  The plan's
    /// originally-listed bitrate/fec/complexity belong to the opus
    /// encoder, not this struct; `StreamCfg` is the network-listener
    /// surface (bind addresses + broadcast fan-out depth), so the
    /// validator covers exactly those fields.
    ///
    /// MARK: Defence in depth
    ///
    /// `validate` is the **config-time** gate; `stream_io::bind_*`
    /// re-checks at bind time (race-safe deletion, etc.).  Both layers
    /// fail closed.  The split exists because some checks (host
    /// resolution, fd ownership) are inherently bind-time, while
    /// others (path shape, parent dir hygiene) are static and
    /// surface in operator-edited TOML far earlier than first
    /// connect.  See the comment on the private
    /// `Self::validate_uds_path` helper for the precise UDS
    /// contract.
    pub fn validate(&self) -> Result<(), String> {
        self.validate_tcp_bind()?;
        self.validate_uds_path()?;
        if self.broadcast_capacity == 0 {
            return Err(
                "stream.broadcast_capacity must be > 0 (zero would fail every send)".into(),
            );
        }
        // POSIX file modes are 12 bits (setuid/setgid/sticky + 9
        // permission bits).  Anything above 0o7777 is a typo.
        if self.uds_mode > 0o7777 {
            return Err(format!(
                "stream.uds_mode {:#o} exceeds the 12-bit POSIX mode range (max 0o7777)",
                self.uds_mode
            ));
        }
        Ok(())
    }

    /// Validate `tcp_bind` shape only.
    ///
    /// The auth/exposure posture is the operator's responsibility:
    /// production deployments front the daemon with an Nginx (or
    /// equivalent) reverse proxy that handles TLS termination and
    /// auth, so the daemon trusts every request that reaches it.
    /// Any host (`127.0.0.1`, `0.0.0.0`, `[::]`, a specific NIC
    /// address, `localhost`) is accepted; we only reject shapes
    /// that cannot bind at all (missing colon, non-u16 port, empty
    /// host).
    fn validate_tcp_bind(&self) -> Result<(), String> {
        if self.tcp_bind.is_empty() {
            return Err("stream.tcp_bind must be non-empty".into());
        }
        // Split on the LAST colon so IPv6 bracketed forms
        // (`[::1]:8787`) parse without a full RFC-3986 parser.
        let (host_raw, port) = self.tcp_bind.rsplit_once(':').ok_or_else(|| {
            format!(
                "stream.tcp_bind {:?} must be host:port (missing colon)",
                self.tcp_bind
            )
        })?;
        port.parse::<u16>().map_err(|_| {
            format!(
                "stream.tcp_bind {:?} port component must parse as u16; got {port:?}",
                self.tcp_bind
            )
        })?;
        // Strip the IPv6 brackets, if any, so `[]:8787` and
        // `:8787` both fall into the empty-host check.
        let host = host_raw
            .strip_prefix('[')
            .and_then(|s| s.strip_suffix(']'))
            .unwrap_or(host_raw);
        if host.is_empty() {
            return Err(format!(
                "stream.tcp_bind {:?} has empty host (use e.g. 127.0.0.1, 0.0.0.0, or [::])",
                self.tcp_bind
            ));
        }
        Ok(())
    }

    /// Static UDS-path hygiene.  This is the **config-time** gate;
    /// `stream_io::bind_uds` owns the race-safe bind-time deletion
    /// and fd-handover.  We reject the worst footguns before they
    /// reach the bind path:
    ///
    /// 1. Path must have a parent directory.  A bare filename
    ///    (`acoustics_lab.sock`) without a parent points at CWD,
    ///    which the daemon-supervised process model treats as
    ///    undefined.  We require an explicit parent so operators
    ///    don't accidentally bind in `/`.
    /// 2. The parent directory must currently exist.  We do not
    ///    auto-create it -- the absence is almost certainly a typo
    ///    in the TOML; creating it would mask the typo and quietly
    ///    install a socket in an unexpected location.
    /// 3. If the path already resolves to *something*, that thing
    ///    MUST be a Unix socket file.  `symlink_metadata` (not
    ///    `metadata`) is used so a symlink at the path is a hard
    ///    reject -- following the symlink at bind time is a
    ///    well-known race (TOCTOU between check and unlink-then-bind).
    ///    Regular files, directories, FIFOs, devices: all rejected.
    /// 4. On Linux, the parent directory must not be world-writable
    ///    unless its sticky bit is set (the `/tmp` shape).  Best-effort:
    ///    on macOS we skip this check because dev hosts typically
    ///    bind under `~/...` (mode 0o755) and the LAN posture is
    ///    different; the documentation comment surfaces the gap.
    fn validate_uds_path(&self) -> Result<(), String> {
        if self.uds_path.as_os_str().is_empty() {
            return Err("stream.uds_path must be non-empty".into());
        }
        let parent = self.uds_path.parent().ok_or_else(|| {
            format!(
                "stream.uds_path {:?} has no parent directory; specify a full path \
                 (e.g. /run/acoustics_lab.sock)",
                self.uds_path.display()
            )
        })?;
        // A path of `foo.sock` parses to parent `""`, which serde +
        // PathBuf treat as "current dir" and we still want to
        // reject -- explicit empty-parent is the same operator
        // mistake as no-parent.
        if parent.as_os_str().is_empty() {
            return Err(format!(
                "stream.uds_path {:?} has no parent directory; specify a full path \
                 (e.g. /run/acoustics_lab.sock)",
                self.uds_path.display()
            ));
        }
        // `symlink_metadata` (not `Path::exists()`) so we can
        // distinguish "absent" from "present but inaccessible
        // (EACCES)".  `Path::exists()` collapses both to
        // `false`, producing a misleading "does not exist"
        // diagnostic for a perms-denied parent that the operator
        // can actually see.  Also catches a symlinked parent
        // (TOCTOU surface for the bind-time unlink).
        match std::fs::symlink_metadata(parent) {
            Ok(md) => {
                if md.file_type().is_symlink() {
                    return Err(format!(
                        "stream.uds_path {:?}: parent {} is a symlink; refuse \
                         (parent symlinks defeat the bind-time TOCTOU protections)",
                        self.uds_path.display(),
                        parent.display(),
                    ));
                }
                if !md.file_type().is_dir() {
                    return Err(format!(
                        "stream.uds_path {:?}: parent {} exists but is not a directory",
                        self.uds_path.display(),
                        parent.display(),
                    ));
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(format!(
                    "stream.uds_path {:?}: parent directory {} does not exist; \
                     create it (e.g. systemd-tmpfiles) or pick an existing path",
                    self.uds_path.display(),
                    parent.display(),
                ));
            }
            Err(e) => {
                return Err(format!(
                    "stream.uds_path {:?}: stat parent {} failed: {e}",
                    self.uds_path.display(),
                    parent.display(),
                ));
            }
        }
        // Use `symlink_metadata` so a symlink at the configured
        // path is a hard reject rather than silently followed.
        // `metadata` would `stat()` (follow symlinks) and let an
        // attacker who controls the parent dir replace the socket
        // with a symlink to e.g. `/etc/passwd` -- the bind-time
        // unlink in `stream_io` would then delete the target.
        match std::fs::symlink_metadata(&self.uds_path) {
            Ok(md) => {
                let ft = md.file_type();
                if ft.is_symlink() {
                    return Err(format!(
                        "stream.uds_path {:?} is a symlink; refuse to bind through \
                         a symlink (would expose stream_io's unlink to a TOCTOU on \
                         the symlink target)",
                        self.uds_path.display()
                    ));
                }
                if ft.is_file() {
                    return Err(format!(
                        "stream.uds_path {:?} is a regular file; refuse to bind \
                         (the bind-time unlink would delete operator data)",
                        self.uds_path.display()
                    ));
                }
                if ft.is_dir() {
                    return Err(format!(
                        "stream.uds_path {:?} is a directory; refuse to bind",
                        self.uds_path.display()
                    ));
                }
                // What remains: socket / FIFO / block / char device.
                // `is_socket` lives behind `unix::fs::FileTypeExt`;
                // we accept "socket" silently here and let
                // `stream_io::bind_uds` re-check race-safely.  FIFO /
                // device: also rejected so the bind-time unlink
                // doesn't touch them.
                #[cfg(unix)]
                {
                    use std::os::unix::fs::FileTypeExt;
                    if !ft.is_socket() {
                        return Err(format!(
                            "stream.uds_path {:?} is a {} (not a unix socket); \
                             refuse to bind",
                            self.uds_path.display(),
                            describe_unix_file_type(&ft),
                        ));
                    }
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // Fresh path -- bind will create it.  Normal case.
            }
            Err(e) => {
                return Err(format!(
                    "stream.uds_path {:?}: stat failed: {e}",
                    self.uds_path.display()
                ));
            }
        }
        // Best-effort parent-dir mode check.  Linux-only: the
        // macOS dev-host pattern (binding under `$HOME` mode
        // 0o755) does not fit the world-writable-needs-sticky
        // shape and the security model differs.
        #[cfg(target_os = "linux")]
        {
            use std::os::unix::fs::PermissionsExt;
            if let Ok(md) = std::fs::metadata(parent) {
                let mode = md.permissions().mode();
                // World-writable bit (`0o002`) without sticky bit
                // (`0o1000`) lets any local user replace the socket
                // path between bind attempts.  `/tmp` has both
                // (`drwxrwxrwt`); a misconfigured runtime dir
                // typically has one and not the other.
                let world_writable = (mode & 0o002) != 0;
                let sticky = (mode & 0o1000) != 0;
                if world_writable && !sticky {
                    return Err(format!(
                        "stream.uds_path {:?}: parent directory {} is world-writable \
                         without the sticky bit (mode {:#o}); refuse to bind into \
                         a directory any local user can hijack",
                        self.uds_path.display(),
                        parent.display(),
                        mode,
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Best-effort human-readable name for a non-socket Unix file
/// type.  Used only inside the `validate_uds_path` diagnostic so an
/// operator who points `uds_path` at e.g. a FIFO sees "fifo" rather
/// than the opaque `FileType { ... }` debug form.
#[cfg(unix)]
fn describe_unix_file_type(ft: &std::fs::FileType) -> &'static str {
    use std::os::unix::fs::FileTypeExt;
    if ft.is_fifo() {
        "fifo"
    } else if ft.is_block_device() {
        "block device"
    } else if ft.is_char_device() {
        "char device"
    } else {
        // Shouldn't reach here: caller already handled file/dir/
        // symlink/socket; remaining types are platform-exotic.
        "non-socket file"
    }
}

/// Default training hyperparameters; per-job invocations can override.
/// Kept in the daemon config so the API exposes "what would happen if
/// you started training right now" without re-reading workspace metadata.
///
/// The `extract_caps` override field was removed during the
/// workspace redesign (archive extraction is no longer a
/// daemon-side operation; datasets are now uploaded as single
/// files via `PUT /workspace/{id}/assets/{*path}`).  Legacy
/// TOMLs carrying `[training_defaults.extract_caps]` will fail
/// validation with serde's "unknown field" error -- the upgrade
/// note in `docs/BUILD.md` documents the removal.
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TrainingDefaults {
    pub epochs: u32,
    pub batch_size: u32,
    pub learning_rate_e6: u32, // micro-lr (e.g. 100 = 1e-4); ints round-trip in TOML cleanly
}

impl Default for TrainingDefaults {
    fn default() -> Self {
        Self {
            epochs: 32,
            batch_size: 16,
            learning_rate_e6: 100, // 1e-4
        }
    }
}

impl TrainingDefaults {
    /// Reject TOML-supplied training defaults that would either
    /// fail every job at admission (`epochs == 0`) or push the
    /// daemon into OOM territory before the operator notices
    /// (epochs/batch in the millions).  `TrainingConfig::validate`
    /// catches the zero cases at job-spawn time, but that is too
    /// late: an operator who edits `training_defaults.epochs = 0`
    /// expects the daemon to reject at boot, not to look healthy
    /// until they POST `/train`.
    ///
    /// Upper bounds are conservative -- not the highest the
    /// hardware can sustain, but tight enough to refuse a typo
    /// (`epochs = 100000` instead of `100`).  Operators with
    /// legitimately bigger workloads override per-job via
    /// `POST /workspace/{id}/training` (`TrainingConfig::validate`).
    ///
    /// `learning_rate_e6` is a micro-lr (`1` = 1e-6, `1_000_000`
    /// = 1.0).  Zero is rejected here for the same boot-vs-job-
    /// spawn argument.  The `1_000_000` cap keeps lr below 1.0;
    /// lr = 1.0 is essentially always a typo for finetuning.
    pub fn validate(&self) -> Result<(), String> {
        const MAX_EPOCHS: u32 = 10_000;
        const MAX_BATCH: u32 = 4_096;
        const MAX_LR_E6: u32 = 1_000_000;

        if self.epochs == 0 {
            return Err("training_defaults.epochs must be >= 1".into());
        }
        if self.epochs > MAX_EPOCHS {
            return Err(format!(
                "training_defaults.epochs {} exceeds {} (almost \
                 certainly a typo; override per-job if you really \
                 need this many epochs)",
                self.epochs, MAX_EPOCHS
            ));
        }
        if self.batch_size == 0 {
            return Err("training_defaults.batch_size must be >= 1".into());
        }
        if self.batch_size > MAX_BATCH {
            return Err(format!(
                "training_defaults.batch_size {} exceeds {} (almost \
                 certainly a typo; override per-job if you really \
                 need this big a batch)",
                self.batch_size, MAX_BATCH
            ));
        }
        if self.learning_rate_e6 == 0 {
            return Err(
                "training_defaults.learning_rate_e6 must be >= 1 (zero lr never converges)".into(),
            );
        }
        if self.learning_rate_e6 > MAX_LR_E6 {
            return Err(format!(
                "training_defaults.learning_rate_e6 {} exceeds {} \
                 (~1.0 lr; almost certainly a typo)",
                self.learning_rate_e6, MAX_LR_E6
            ));
        }
        Ok(())
    }
}

/// Operator-tunable file-service admission caps.
///
/// Mirrors `file_mgr::AdmissionCfg` (the run-time enforcement
/// surface) but lives in `config` so operators can override via TOML
/// without recompiling.  The daemon reads this block at boot and
/// constructs the `AdmissionCfg` it hands to `FsServiceImpl`.
///
/// On-device defaults: 256 MiB per upload, 4 concurrent uploads
/// per the workspace-redesign §9 storage table.  Operators on a
/// developer host with abundant RAM can lift the per-upload cap
/// explicitly:
///
/// ```toml
/// [file]
/// max_upload_bytes = 1073741824 # 1 GiB
/// max_concurrent_uploads = 8
/// ```
///
/// `max_concurrent_uploads` is `usize` here for ergonomics
/// inside the `[file]` table; it converts to `AdmissionCfg`'s
/// `u32` via a saturating cast at the boundary in `daemon::main`
/// (`try_into().unwrap_or(u32::MAX)` -- harmless in practice).
#[derive(Clone, Copy, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct FileCfg {
    /// Per-request hard ceiling on uncompressed upload bytes.
    /// Default 256 MiB.
    pub max_upload_bytes: u64,
    /// Maximum number of in-flight uploads across the whole
    /// `WorkspaceMgr`.  Default 4 per the workspace-redesign §9
    /// storage table (raised from 2 for better bulk-load
    /// throughput on bounded SBC deployments; see §8 "Tunable
    /// upload concurrency").  Held as `usize` because the
    /// underlying `tokio::sync::Semaphore` takes `usize`; the
    /// daemon bridges to `file_mgr::AdmissionCfg`'s `u32` shape.
    pub max_concurrent_uploads: usize,
}

impl Default for FileCfg {
    fn default() -> Self {
        Self {
            max_upload_bytes: 256 * 1024 * 1024, // 256 MiB
            max_concurrent_uploads: 4,
        }
    }
}

impl FileCfg {
    /// Operator-supplied caps must keep the daemon
    /// functional.  Zero on either field would refuse every
    /// upload (no permits / nothing fits) and is almost certainly
    /// a typo; reject at boot rather than silently hanging the
    /// upload path.
    pub fn validate(&self) -> Result<(), String> {
        if self.max_upload_bytes == 0 {
            return Err("file.max_upload_bytes must be > 0".into());
        }
        if self.max_concurrent_uploads == 0 {
            return Err("file.max_concurrent_uploads must be > 0".into());
        }
        Ok(())
    }
}
