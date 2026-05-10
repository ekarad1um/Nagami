//! `StatusReporter` trait + impl on
//! [`crate::status::StatusMonitor`].  Held by the api crate as
//! `Arc<dyn StatusReporter>` so `/api/v1/status` doesn't
//! need the concrete monitor type -- tests substitute mocks
//! returning canned snapshots.
//!
//! The trait surface is intentionally just `snapshot()` --
//! `register()` is daemon-only (boot-time wiring) and stays
//! on the concrete [`crate::status::StatusMonitor`].
//!
//! Named `StatusReporter` (not `StatusSnapshot` per the
//! original sketch) to avoid collision with the
//! [`crate::status::StatusSnapshot`] data type the trait method
//! returns.

use crate::status::{BroadcastLagSnapshot, StatusMonitor, StatusSnapshot};

/// Read-only handle for assembling the daemon's status
/// snapshot.  Production impl: [`StatusMonitor`] (which also
/// owns the registration side that the daemon uses at boot).
pub trait StatusReporter: Send + Sync + std::fmt::Debug {
    /// Assemble a fresh [`StatusSnapshot`] capturing every
    /// registered subsystem's most recent heartbeat plus the
    /// most recent process-wide metrics sample plus the
    /// supplied broadcast-lag counters.
    ///
    /// Wait-free.  Process-wide metrics come
    /// from the `ArcSwap<MetricsSnapshot>` published by
    /// `StatusMonitor`'s background sampler task; the request
    /// path no longer touches sysinfo.  Callers no longer need to
    /// wrap in `spawn_blocking` (the prior `Mutex<System>` +
    /// per-request `refresh_specifics` syscall is gone).
    fn snapshot(&self, broadcast_lags: BroadcastLagSnapshot) -> StatusSnapshot;
}

impl StatusReporter for StatusMonitor {
    fn snapshot(&self, broadcast_lags: BroadcastLagSnapshot) -> StatusSnapshot {
        StatusMonitor::snapshot(self, broadcast_lags)
    }
}

// Object-safety smoke.
#[cfg(test)]
const _: fn() = || {
    fn assert_obj_safe<T: ?Sized>() {}
    assert_obj_safe::<dyn StatusReporter>();
};
