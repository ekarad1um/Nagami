//! Object-safe traits shared across the daemon.
//!
//! One trait per file; each pairs with a production impl in the
//! owning module (traits land alongside their first impl).  The
//! API + daemon consume these as `Arc<dyn TraitName>` so test
//! mocks can substitute without rebuilding the world.
//!
//! - [`lag_source`] -- read-side view of `stream_io`'s per-stream
//!   broadcast-lag counters.
//! - [`head_store`] -- read + atomic-swap surface for the active
//!   classifier head; backed in production by
//!   `inference::HotHead`'s [`crate::common::version::VersionedSwap`].

pub mod head_store;
pub mod lag_source;
