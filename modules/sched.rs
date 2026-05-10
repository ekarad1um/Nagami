//! Per-thread CPU affinity + SCHED_FIFO realtime priority
//! helpers.
//!
//! # Why a separate module (not under `crate::common`)
//!
//! [`crate::common`] carries `#![forbid(unsafe_code)]` as a
//! deliberate guardrail.  The two syscall wrappers below need
//! `unsafe` FFI; `forbid` cannot be overridden by sub-modules
//! (unlike `deny`), so the helpers ship in this standalone
//! module instead.
//!
//! # What the helpers do
//!
//! - [`pin_to_core`] -- sets the calling thread's CPU
//!   affinity to a single core via `sched_setaffinity(0, ...)`
//!   (the `pid=0` shorthand for "calling thread").
//! - [`set_realtime`] -- switches the calling thread to
//!   `SCHED_FIFO` with the given priority via
//!   `sched_setscheduler(0, SCHED_FIFO, ...)`.
//!
//! Both helpers operate on the calling thread.  Call them
//! from inside the thread you want to configure (typically as
//! the first line after `thread::Builder::spawn(move || {
//! ... })` for std threads, or inside
//! `tokio::Builder::on_thread_start` for tokio workers).
//!
//! # Failure modes
//!
//! Both helpers return `io::Result<()>`.  Validation errors
//! (`InvalidInput`) surface uniformly on every platform; the
//! Linux backend additionally surfaces kernel errnos.  The
//! dominant production failure on Linux is `EPERM` from
//! [`set_realtime`] without `CAP_SYS_NICE`; current-thread
//! affinity changes via [`pin_to_core`] do not normally need
//! `CAP_SYS_NICE` but can still fail on cpuset/cgroup
//! restrictions, an out-of-range core ID, or single-core
//! systems requesting `core > 0`.  The daemon MUST stay up
//! if any of these fail; call sites use `let _ =
//! sched::pin_to_core(...)` and `tracing::warn!` on failure
//! rather than `?`-propagating.
//!
//! # FFI source of truth
//!
//! `cpu_set_t`, `sched_param`, `SCHED_FIFO`, `CPU_ZERO`,
//! `CPU_SET`, `sched_setaffinity`, and `sched_setscheduler`
//! come from the workspace `libc` dependency, so this module
//! does not own a private mirror of the Linux kernel ABI.
//!
//! # Deploy note
//!
//! The daemon binary needs `CAP_SYS_NICE` for [`set_realtime`]
//! to succeed without running as root.  Grant via:
//!
//! ```bash
//! sudo setcap cap_sys_nice+ep /usr/local/bin/acousticsd
//! ```
//!
//! Without the cap the daemon starts and runs at
//! `SCHED_OTHER` (kernel default) -- the only observable
//! effect is occasional ALSA underruns under load.
//!
//! # Non-Linux platforms
//!
//! macOS dev hosts and Windows targets get no-op shims for the
//! syscall side, but [`set_realtime`] still validates priority
//! consistently with Linux so dev-host config tests catch
//! invalid settings.  Production runs on Linux; macOS is
//! dev-only; the helpers keep call sites ifdef-free.

#![warn(missing_debug_implementations)]

use std::io;

/// Validate that `priority` is in the documented Linux
/// `SCHED_FIFO` range (`1..=99`).  Runs before the cfg
/// dispatch so the no-op shim and the real syscall path
/// produce identical `InvalidInput` errors for invalid input.
fn validate_realtime_priority(priority: i32) -> io::Result<()> {
    if (1..=99).contains(&priority) {
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("realtime priority {priority} out of SCHED_FIFO range 1..=99"),
        ))
    }
}

/// Pin the calling thread to a single CPU core.
///
/// On Linux: `sched_setaffinity(0, sizeof(cpu_set_t), mask)`
/// with `mask` having only bit `core` set.  On other
/// platforms: no-op.
///
/// # Errors
///
/// - `InvalidInput` when `core` is out of `cpu_set_t` range
///   (`>= libc::CPU_SETSIZE`, typically 1024).
/// - The underlying syscall errno on any other failure
///   (e.g. `EINVAL` on a single-core system requesting
///   `core > 0`, or cgroup/cpuset rejection).
pub fn pin_to_core(core: usize) -> io::Result<()> {
    #[cfg(target_os = "linux")]
    {
        linux::pin_to_core(core)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = core;
        Ok(())
    }
}

/// Switch the calling thread to `SCHED_FIFO` with `priority`.
///
/// Valid priority range is `1..=99` inclusive (per
/// `sched_get_priority_min(SCHED_FIFO)` and
/// `..._max(SCHED_FIFO)` on Linux); callers above 50 risk
/// preempting kernel housekeeping.  Daemon wiring uses 50 for
/// the audio source thread and 30 for inference.
///
/// On Linux: `sched_setscheduler(0, SCHED_FIFO, &{
/// sched_priority: priority })`.  On other platforms: no-op
/// (after validation).
///
/// # Errors
///
/// - `InvalidInput` when `priority` is outside `1..=99` (on
///   every platform, before any syscall).
/// - On Linux: `EPERM` without `CAP_SYS_NICE`, or any other
///   `sched_setscheduler` errno.
pub fn set_realtime(priority: i32) -> io::Result<()> {
    validate_realtime_priority(priority)?;
    #[cfg(target_os = "linux")]
    {
        linux::set_realtime(priority)
    }
    #[cfg(not(target_os = "linux"))]
    {
        Ok(())
    }
}

#[cfg(target_os = "linux")]
mod linux {
    use std::io;
    use std::mem::MaybeUninit;

    pub(super) fn pin_to_core(core: usize) -> io::Result<()> {
        // libc's `CPU_SET` has no bounds check, so reject
        // out-of-range cores deterministically here rather
        // than racing the syscall's `EINVAL`.
        if core >= libc::CPU_SETSIZE as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "core {core} out of cpu_set_t range (max {})",
                    libc::CPU_SETSIZE - 1
                ),
            ));
        }
        // Stack-only: no allocation in the pin path.  The
        // `MaybeUninit` lets us defer initialisation to libc's
        // `CPU_ZERO` without depending on the layout of
        // libc's private `bits` field.
        //
        // SAFETY: `CPU_ZERO` writes to every byte of `set`
        // before any read; the `&mut` reference is exclusive
        // and the storage outlives the call.
        let mut set = MaybeUninit::<libc::cpu_set_t>::uninit();
        unsafe {
            libc::CPU_ZERO(&mut *set.as_mut_ptr());
            libc::CPU_SET(core, &mut *set.as_mut_ptr());
        }
        // SAFETY: `set` is fully initialised by `CPU_ZERO`
        // above; `pid=0` is the calling thread per
        // `sched_setaffinity(2)`; the kernel only reads
        // through the pointer for `cpusetsize` bytes, which
        // matches `size_of::<cpu_set_t>()`.
        let rc = unsafe {
            libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), set.as_ptr())
        };
        if rc < 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }

    pub(super) fn set_realtime(priority: i32) -> io::Result<()> {
        // `sched_param` on Linux is a single-field struct
        // (`sched_priority`), but libc owns the layout to stay
        // ABI-correct across kernel/libc revisions.
        //
        // SAFETY: zero-initialised `sched_param` is valid;
        // `sched_priority` is the only meaningful field for
        // `SCHED_FIFO`.
        let mut param: libc::sched_param = unsafe { std::mem::zeroed() };
        param.sched_priority = priority;
        // SAFETY: `param` lives on the stack for the duration
        // of the syscall; `pid=0` is the calling thread per
        // `sched_setscheduler(2)`; the kernel only reads
        // `sched_priority`.
        let rc = unsafe { libc::sched_setscheduler(0, libc::SCHED_FIFO, &param) };
        if rc < 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Out-of-range core selection surfaces as `InvalidInput`
    /// at the helper's own input gate, ahead of any syscall.
    #[cfg(target_os = "linux")]
    #[test]
    fn pin_to_core_out_of_range_rejects_locally() {
        let err = pin_to_core(libc::CPU_SETSIZE as usize).expect_err("must reject");
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        let err = pin_to_core(usize::MAX).expect_err("must reject");
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    /// Non-Linux `pin_to_core` is a pure no-op; callers can
    /// `let _ = sched::pin_to_core(...)` without ifdef'ing.
    #[cfg(not(target_os = "linux"))]
    #[test]
    fn non_linux_pin_is_noop() {
        assert!(pin_to_core(0).is_ok());
        assert!(pin_to_core(usize::MAX).is_ok());
    }

    /// Calling [`pin_to_core(0)`] from a test thread on Linux
    /// either succeeds (the test runner has the cap) or
    /// returns a recognisable errno.  Both are valid; what
    /// we guard against is panic / hang / undefined behavior.
    #[cfg(target_os = "linux")]
    #[test]
    fn pin_to_core_zero_returns_a_result() {
        match pin_to_core(0) {
            Ok(()) => {}
            Err(e) => {
                let raw = e.raw_os_error();
                assert!(
                    matches!(raw, Some(errno) if (1..=255).contains(&errno)),
                    "unexpected pin_to_core(0) error: {e:?}",
                );
            }
        }
    }

    /// Validation runs before any cfg dispatch, so every
    /// platform rejects out-of-range realtime priorities the
    /// same way (`InvalidInput`) without touching the kernel.
    #[test]
    fn set_realtime_invalid_priority_rejects_locally() {
        for &bad in &[0_i32, -1, 100, 255, i32::MIN, i32::MAX] {
            let err = set_realtime(bad).expect_err("must reject out-of-range priority");
            assert_eq!(
                err.kind(),
                io::ErrorKind::InvalidInput,
                "priority {bad} should be InvalidInput",
            );
        }
    }

    /// Valid priorities pass validation; the syscall result
    /// itself is platform-dependent (Ok on the no-op shim;
    /// possibly `EPERM` on Linux without `CAP_SYS_NICE`).
    /// What we guard against is the validation gate
    /// rejecting a documented-valid value.
    #[test]
    fn set_realtime_valid_priority_returns_a_result() {
        for &good in &[1_i32, 30, 50, 99] {
            match set_realtime(good) {
                Ok(()) => {}
                Err(e) => {
                    // Validation must not be the failure source for in-range values.
                    assert_ne!(
                        e.kind(),
                        io::ErrorKind::InvalidInput,
                        "valid priority {good} was rejected as InvalidInput: {e:?}",
                    );
                }
            }
        }
    }
}
