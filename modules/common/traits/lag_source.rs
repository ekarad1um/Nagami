//! Broadcast-lag observability trait.
//!
//! `/api/v1/status` surfaces cumulative WS-receiver lag counts so
//! operators can see when subscribers can't keep up with the
//! audio + inference broadcast channels.  The counters live in
//! `stream_io::BroadcastLagCounters` (per-WS-receiver `fetch_add`
//! on `Lagged`); the API needs to read them at status-poll time
//! without depending on `stream_io` directly.
//!
//! [`LagSource`] is the abstraction: an `Arc<dyn LagSource>`
//! flows through `api::AppState`.  Production wires
//! `stream_io::BroadcastLagCounters`; tests substitute mocks
//! that return canned snapshots.
//!
//! [`BroadcastLagSnapshot`] lives here rather than in `status` so
//! the trait surface is self-contained -- `common` owns no
//! workspace deps, so `status` cannot host the type.  `status`
//! re-exports it for the existing `status::BroadcastLagSnapshot`
//! import path.

/// Cumulative broadcast-lag counters captured at snapshot time.
///
/// Fields are u64 monotonic counters reset only on daemon
/// restart; the API exposes them as-is so operators can chart
/// drift over time.  `Default` (all zeros) is the right value
/// for pre-wiring (unit tests, or before `stream_io` has emitted
/// any lag event).
///
/// Counter semantics are dropped MESSAGES, not lag EVENTS.
/// The underlying `tokio::sync::broadcast::Receiver::recv`
/// returns `RecvError::Lagged(n)` where `n` is the count of
/// messages the receiver missed; the producer increments by
/// `n` (not by 1) so a receiver that lagged by 100 packets in
/// a single event contributes 100 to the counter.  Field
/// names spell the semantic out so a UI dashboard does not
/// mis-label this as an event count.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BroadcastLagSnapshot {
    /// Cumulative WS audio-broadcast messages dropped due to a
    /// receiver lagging the broadcast channel.  Each lag event
    /// adds the missed-message count (`RecvError::Lagged(n)`)
    /// rather than 1, so the counter measures total messages
    /// dropped, not lag-event count.
    pub audio_messages_dropped: u64,
    /// Cumulative WS inference-broadcast messages dropped due
    /// to a receiver lagging the broadcast channel.  Same
    /// dropped-messages-not-events semantics as `audio_messages_dropped`.
    pub inference_messages_dropped: u64,
}

/// Read-side view of per-stream broadcast-lag counters.
///
/// Implemented by `stream_io::BroadcastLagCounters` in
/// production.  [`Self::snapshot`] is wait-free (atomic loads);
/// call it once per `/api/v1/status` request.
///
/// `Send + Sync + 'static` so an `Arc<dyn LagSource>` can be
/// stored in `api::AppState` and shared across handler tasks.
pub trait LagSource: Send + Sync + 'static {
    fn snapshot(&self) -> BroadcastLagSnapshot;
}
