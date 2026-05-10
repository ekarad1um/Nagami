//! Acoustics Lab daemon -- boot sequence, `drain_registry`, subsystem wiring.
//!
//! The daemon does NOT self-restart failed subsystems; an
//! external process supervisor (systemd `Type=notify` or
//! equivalent) owns restart.  The bounded-shutdown contract +
//! failure-model rationale lives in the private
//! `drain_registry` submodule.
//!
//! [`run`] is the only public entry point; the `acousticsd`
//! binary's `fn main()` is a 2-line wrapper.

#![warn(missing_debug_implementations)]

pub(crate) mod drain_registry;

mod main_body;

pub use main_body::run;
