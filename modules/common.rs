//! Acoustics Lab contract module.
//!
//! `common` is the vocabulary every other module uses to talk
//! about itself: dimension newtypes, validated identifiers,
//! `ErrorKind` / `Categorized`, time + version primitives,
//! object-safe traits.
//!
//! # Discipline
//!
//! - **No heavy deps.** The dependency surface is intentionally
//!   small: `thiserror`, `serde` (with `rc` for `MicId`'s
//!   `Arc<str>`), `uuid`, `time`, `arc-swap`, `parking_lot`.  No
//!   tokio, axum, prost, alsa, opus, burn, rknn -- those belong
//!   in the modules that use them.
//! - **Public surface is a contract.** Anything `pub` here is a
//!   cross-module concern; treat changes accordingly.
//! - **No unsafe.** The inner `#![forbid(unsafe_code)]` is the
//!   guardrail; if a future need arises it goes in a separate
//!   module (see [`crate::sched`] for the precedent).

#![forbid(unsafe_code)]

pub mod asset_path;
pub mod dims;
pub mod error;
// `.mpk` head-artifact persistence header; pure-bytes module so
// both the converter (writes) and inference (reads) share the
// spec.
pub mod head_header;
// Lowercase-hex byte encoder; the layer guard forbids
// `inference -> file_mgr`, so the digest helper lives here in
// `common` where every digest-stamping layer (file_mgr, converter,
// inference, api) is allowed to import it.
pub mod hex;
pub mod ids;
pub mod time;
pub mod traits;
pub mod version;
// Workspace / head / job foundation types -- on-disk schema
// contracts consumed by `file_mgr`, the in-memory `JobRegistry`,
// and the API response layer.
pub mod workspace;
