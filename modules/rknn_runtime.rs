//! Minimal safe wrapper around `librknnrt.so` /
//! `librknnmrt.so` (Rockchip NPU runtime library).
//!
//! Scope: single-input / single-output inference with
//! zero-copy borrowed buffers at the FFI boundary.
//!
//! # Failure model
//!
//! The librknnrt FFI surface is fundamentally unsafe Rust:
//! a SIGSEGV inside `rknn_run` (or any other `extern "C"`
//! call) is **not** catchable by Rust's
//! [`std::panic::catch_unwind`].  POSIX signal delivery
//! happens at the kernel boundary, before Rust's unwind
//! machinery sees any control flow.  The
//! `load_nonexistent_library_errors_cleanly` test only
//! proves library-LOAD failure surfaces cleanly via
//! [`Error::LibraryLoad`]; runtime SIGSEGV inside an FFI
//! call propagates to a SIGABRT and exits the process.
//!
//! **This is the deliberate failure model.**  The daemon's
//! external supervisor (`scripts/run_acousticsd.sh`)
//! restarts unconditionally on non-zero exit including
//! SIGSEGV (139), SIGABRT (134), and SIGTERM (143).
//! Operator observability comes from the rolling log
//! (`${workspace_root}/logs/acousticsd.log.*`); the
//! structured panic-hook record captures location +
//! payload + backtrace before the abort.
//!
//! For the "wedged but not crashed" case (e.g. NPU
//! firmware deadlock that leaves `rknn_run` blocked
//! indefinitely with no signal), the inference engine's
//! heartbeat-pump watchdog (search the daemon body for
//! `STALE_ABORT_AFTER`) detects >5 s of `frames_emitted`
//! non-advancement and calls [`std::process::abort`] from
//! the watchdog thread.  That guard sits at the engine
//! level rather than per-FFI-call because the engine runs
//! inside [`tokio::task::spawn_blocking`] (no
//! [`tokio::time::timeout`] reachable) and a per-call
//! timeout would add hot-path overhead.

#![warn(missing_debug_implementations)]

// Frozen FFI bindings -- checked into
// `rknn_runtime/bindings.rs`.  Small and audited (7
// functions, ~12 types); no build-time bindgen
// dependency.  Everything in `sys` is `unsafe` by
// definition; the safe wrapper lives in submodules below.
//
// Visibility is `pub(crate)` (NOT `pub`) so the FFI blast
// radius stays inside this module tree.  Downstream callers
// must go through [`Session`] / [`InputSlice`] /
// [`OutputSlice`], which check every return code and enforce
// buffer aliasing rules.  Exposing the raw C-shaped structs
// and `unsafe extern "C"` declarations crate-wide would let
// other modules bypass those checks (the rknpu2 swallowed-
// error pattern this wrapper exists to prevent) and would
// also couple them to bindgen output we may want to refresh.
#[allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    dead_code,
    clippy::all
)]
pub(crate) mod sys {
    include!("rknn_runtime/bindings.rs");
}

mod error;
mod ffi;
mod infer;
mod session;
// `utils` exposes the librknnrt path-discovery helper, which is only
// reachable from the rknpu-gated code path in `inference::backbone`.
// Mirror that cfg here so the helper is not flagged dead on hosts.
#[cfg(all(target_os = "linux", feature = "rknpu"))]
pub(crate) mod utils;

pub use error::{Error, Result};
pub use infer::{InputSlice, OutputSlice};
pub use session::{DataType, IoCount, QntType, SdkVersion, Session, TensorAttr, TensorFormat};

// Sanity tests -- run on host (no NPU).  Verify struct layouts match C, each
// DataType raw value is distinct, each TensorFormat maps to the expected
// constant, the error-code decoder covers every declared RKNN_ERR_*, and
// library-load failure surfaces as Error::LibraryLoad.

#[cfg(test)]
mod tests {
    use super::*;

    /// The C ABI layout of `rknn_input_output_num` is `{u32, u32}`.  On every
    /// target we care about (aarch64, x86_64) that's 8 bytes with no
    /// padding.
    #[test]
    fn layout_rknn_input_output_num() {
        assert_eq!(std::mem::size_of::<sys::rknn_input_output_num>(), 8);
        assert_eq!(std::mem::align_of::<sys::rknn_input_output_num>(), 4);
    }

    /// `rknn_input` has an 8-byte pointer field, so with alignment
    /// padding around it the struct is 32 bytes on 64-bit targets.
    /// A mismatch here means the FFI ABI no longer matches the
    /// vendored library and would corrupt every call.
    #[test]
    fn layout_rknn_input_is_32_bytes_on_64bit() {
        assert_eq!(std::mem::size_of::<sys::rknn_input>(), 32);
        assert_eq!(std::mem::align_of::<sys::rknn_input>(), 8);
    }

    /// `rknn_output` has an 8-byte pointer plus u8/u8/u32/u32; alignment
    /// rounds it to 24 bytes on 64-bit.
    #[test]
    fn layout_rknn_output_is_24_bytes_on_64bit() {
        assert_eq!(std::mem::size_of::<sys::rknn_output>(), 24);
        assert_eq!(std::mem::align_of::<sys::rknn_output>(), 8);
    }

    /// `rknn_context` is `uint64_t` per the header.
    #[test]
    fn rknn_context_is_u64() {
        assert_eq!(std::mem::size_of::<sys::rknn_context>(), 8);
    }

    /// `rknn_tensor_attr` is the largest FFI struct we touch: each
    /// `input_attr` / `output_attr` query passes
    /// `size_of::<rknn_tensor_attr>()` as a u32 to `rknn_query`, and
    /// the runtime rejects any mismatch with `RKNN_ERR_PARAM_INVALID`.
    /// Catching the drift here on the host (no NPU needed) is cheaper
    /// than discovering it as a runtime query failure.
    #[test]
    fn layout_rknn_tensor_attr_is_376_bytes() {
        // 4 (index) + 4 (n_dims) + 64 ([u32;16] dims) + 256 ([c_char;256] name)
        // + 4 (n_elems) + 4 (size) + 4 (fmt) + 4 (type_) + 4 (qnt_type)
        // + 1 (fl) + 3 pad + 4 (zp) + 4 (scale) + 4 (w_stride)
        // + 4 (size_with_stride) + 1 (pass_through) + 3 pad + 4 (h_stride) = 376.
        assert_eq!(std::mem::size_of::<sys::rknn_tensor_attr>(), 376);
        assert_eq!(std::mem::align_of::<sys::rknn_tensor_attr>(), 4);
    }

    /// `rknn_sdk_version` is two `char[256]` strings packed end-to-end.
    /// No padding (alignment 1).
    #[test]
    fn layout_rknn_sdk_version_is_512_bytes() {
        assert_eq!(std::mem::size_of::<sys::rknn_sdk_version>(), 512);
        assert_eq!(std::mem::align_of::<sys::rknn_sdk_version>(), 1);
    }

    /// `rknn_init_extend` includes a 112-byte `reserved` field; if
    /// Rockchip ever changes its size the struct shrinks/grows and we
    /// need to know.  Today we always pass `null_mut` so layout doesn't
    /// affect runtime behaviour, but sanity-checking it now means a
    /// future caller passing a real one is caught at the host build.
    #[test]
    fn layout_rknn_init_extend_is_136_bytes() {
        assert_eq!(std::mem::size_of::<sys::rknn_init_extend>(), 136);
        assert_eq!(std::mem::align_of::<sys::rknn_init_extend>(), 8);
    }

    /// `rknn_run_extend` and `rknn_output_extend` are passed as
    /// `null_mut` from `Session::infer`; layout asserts here protect
    /// against silent ABI drift if a future caller starts populating
    /// them.
    #[test]
    fn layout_rknn_run_extend_is_24_bytes() {
        assert_eq!(std::mem::size_of::<sys::rknn_run_extend>(), 24);
        assert_eq!(std::mem::align_of::<sys::rknn_run_extend>(), 8);
    }

    #[test]
    fn layout_rknn_output_extend_is_8_bytes() {
        assert_eq!(std::mem::size_of::<sys::rknn_output_extend>(), 8);
        assert_eq!(std::mem::align_of::<sys::rknn_output_extend>(), 8);
    }

    /// Known DataType variants must each map to a distinct raw value.  The
    /// reverse direction (`from_raw`) is private and covered indirectly
    /// through on-device `rknn_query` paths; here we just guard against
    /// accidental collisions in `to_raw`.
    #[test]
    fn datatype_to_raw_all_distinct() {
        let raws: Vec<_> = [
            DataType::Float32,
            DataType::Float16,
            DataType::Int8,
            DataType::Uint8,
            DataType::Int16,
            DataType::Uint16,
            DataType::Int32,
            DataType::Uint32,
            DataType::Int64,
            DataType::Bool,
            DataType::Int4,
            DataType::Bfloat16,
        ]
        .iter()
        .map(|d| d.to_raw())
        .collect();
        let mut uniq = raws.clone();
        uniq.sort();
        uniq.dedup();
        assert_eq!(
            raws.len(),
            uniq.len(),
            "DataType variants must map to distinct raw values"
        );
    }

    /// Exhaustive coverage: every `RKNN_TENSOR_*` data-type constant in
    /// `bindings.rs` (excluding the sentinel `RKNN_TENSOR_TYPE_MAX`)
    /// must round-trip through `to_raw` without falling into the
    /// `Unknown(_)` arm.  Catches the historical "header gained
    /// `RKNN_TENSOR_UINT32` but the safe wrapper still mapped it to
    /// `Unknown(7)`" regression -- the wrapper would silently lose
    /// dtype information and skip host-buffer size validation for that
    /// dtype.
    #[test]
    fn datatype_round_trip_covers_all_known_constants() {
        let constants: &[(sys::rknn_tensor_type, &str)] = &[
            (sys::_rknn_tensor_type::RKNN_TENSOR_FLOAT32, "FLOAT32"),
            (sys::_rknn_tensor_type::RKNN_TENSOR_FLOAT16, "FLOAT16"),
            (sys::_rknn_tensor_type::RKNN_TENSOR_INT8, "INT8"),
            (sys::_rknn_tensor_type::RKNN_TENSOR_UINT8, "UINT8"),
            (sys::_rknn_tensor_type::RKNN_TENSOR_INT16, "INT16"),
            (sys::_rknn_tensor_type::RKNN_TENSOR_UINT16, "UINT16"),
            (sys::_rknn_tensor_type::RKNN_TENSOR_INT32, "INT32"),
            (sys::_rknn_tensor_type::RKNN_TENSOR_UINT32, "UINT32"),
            (sys::_rknn_tensor_type::RKNN_TENSOR_INT64, "INT64"),
            (sys::_rknn_tensor_type::RKNN_TENSOR_BOOL, "BOOL"),
            (sys::_rknn_tensor_type::RKNN_TENSOR_INT4, "INT4"),
            (sys::_rknn_tensor_type::RKNN_TENSOR_BFLOAT16, "BFLOAT16"),
        ];
        // `from_raw` is private; round-trip via `to_raw` after building
        // each variant from the raw constant by name.  The intent is
        // that no constant in this list serializes back to itself via
        // an `Unknown(_)` arm -- if a future bindings refresh adds a
        // new constant we expect to extend the variant list above and
        // this test together.
        for (raw, name) in constants {
            // Build the matching DataType variant explicitly so the
            // compiler enforces that we keep this list in sync with
            // the enum.
            let variant = match *raw {
                x if x == sys::_rknn_tensor_type::RKNN_TENSOR_FLOAT32 => DataType::Float32,
                x if x == sys::_rknn_tensor_type::RKNN_TENSOR_FLOAT16 => DataType::Float16,
                x if x == sys::_rknn_tensor_type::RKNN_TENSOR_INT8 => DataType::Int8,
                x if x == sys::_rknn_tensor_type::RKNN_TENSOR_UINT8 => DataType::Uint8,
                x if x == sys::_rknn_tensor_type::RKNN_TENSOR_INT16 => DataType::Int16,
                x if x == sys::_rknn_tensor_type::RKNN_TENSOR_UINT16 => DataType::Uint16,
                x if x == sys::_rknn_tensor_type::RKNN_TENSOR_INT32 => DataType::Int32,
                x if x == sys::_rknn_tensor_type::RKNN_TENSOR_UINT32 => DataType::Uint32,
                x if x == sys::_rknn_tensor_type::RKNN_TENSOR_INT64 => DataType::Int64,
                x if x == sys::_rknn_tensor_type::RKNN_TENSOR_BOOL => DataType::Bool,
                x if x == sys::_rknn_tensor_type::RKNN_TENSOR_INT4 => DataType::Int4,
                x if x == sys::_rknn_tensor_type::RKNN_TENSOR_BFLOAT16 => DataType::Bfloat16,
                x => panic!("RKNN_TENSOR_{name} (raw={x}) has no matching DataType variant"),
            };
            assert_eq!(
                variant.to_raw(),
                *raw,
                "round-trip mismatch for RKNN_TENSOR_{name}",
            );
        }
        // Sentinel: `RKNN_TENSOR_TYPE_MAX` is intentionally NOT a
        // mappable variant; if a future bindings refresh promotes it
        // (or any value below it grows past the current max), the
        // assertion here will hint at the gap.
        assert!(
            sys::_rknn_tensor_type::RKNN_TENSOR_TYPE_MAX as usize >= constants.len(),
            "RKNN_TENSOR_TYPE_MAX shrank below the known constant list -- bindings drift?",
        );
    }

    /// TensorFormat ctor-mapping produces the expected RKNN constants.
    #[test]
    fn tensor_format_to_raw_is_correct() {
        assert_eq!(
            TensorFormat::Nchw.to_raw(),
            sys::_rknn_tensor_format::RKNN_TENSOR_NCHW
        );
        assert_eq!(
            TensorFormat::Nhwc.to_raw(),
            sys::_rknn_tensor_format::RKNN_TENSOR_NHWC
        );
        assert_eq!(
            TensorFormat::Nc1Hwc2.to_raw(),
            sys::_rknn_tensor_format::RKNN_TENSOR_NC1HWC2
        );
        assert_eq!(
            TensorFormat::Undefined.to_raw(),
            sys::_rknn_tensor_format::RKNN_TENSOR_UNDEFINED
        );
    }

    /// Every negative error code declared in the header must be decoded to
    /// a known name by `rknn_error_name`, not fall through to "UNKNOWN".
    /// This prevents "we added a variant to the enum but forgot to extend
    /// the decoder" drift.
    #[test]
    fn error_name_covers_every_known_code() {
        for (code, expected) in [
            (sys::RKNN_ERR_FAIL, "RKNN_ERR_FAIL"),
            (sys::RKNN_ERR_TIMEOUT, "RKNN_ERR_TIMEOUT"),
            (
                sys::RKNN_ERR_DEVICE_UNAVAILABLE,
                "RKNN_ERR_DEVICE_UNAVAILABLE",
            ),
            (sys::RKNN_ERR_MALLOC_FAIL, "RKNN_ERR_MALLOC_FAIL"),
            (sys::RKNN_ERR_PARAM_INVALID, "RKNN_ERR_PARAM_INVALID"),
            (sys::RKNN_ERR_MODEL_INVALID, "RKNN_ERR_MODEL_INVALID"),
            (sys::RKNN_ERR_CTX_INVALID, "RKNN_ERR_CTX_INVALID"),
            (sys::RKNN_ERR_INPUT_INVALID, "RKNN_ERR_INPUT_INVALID"),
            (sys::RKNN_ERR_OUTPUT_INVALID, "RKNN_ERR_OUTPUT_INVALID"),
            (sys::RKNN_ERR_DEVICE_UNMATCH, "RKNN_ERR_DEVICE_UNMATCH"),
            (
                sys::RKNN_ERR_INCOMPATILE_PRE_COMPILE_MODEL,
                "RKNN_ERR_INCOMPATILE_PRE_COMPILE_MODEL",
            ),
            (
                sys::RKNN_ERR_INCOMPATILE_OPTIMIZATION_LEVEL_VERSION,
                "RKNN_ERR_INCOMPATILE_OPTIMIZATION_LEVEL_VERSION",
            ),
            (
                sys::RKNN_ERR_TARGET_PLATFORM_UNMATCH,
                "RKNN_ERR_TARGET_PLATFORM_UNMATCH",
            ),
        ] {
            assert_eq!(error::rknn_error_name(code), expected, "code = {code}");
        }
        // RKNN_SUCC = 0 -- decoder returns the name so diagnostic prints are
        // still informative if someone mis-passes success as error.
        assert_eq!(error::rknn_error_name(sys::RKNN_SUCC as _), "RKNN_SUCC");
        // Unknown code.
        assert_eq!(error::rknn_error_name(-9999), "UNKNOWN");
    }

    /// Library load with a nonsense path must surface cleanly as
    /// `Error::LibraryLoad`, not panic and not silently succeed.
    #[test]
    fn load_nonexistent_library_errors_cleanly() {
        let mut fake_model = vec![0u8; 128];
        // SAFETY: the path is guaranteed not to exist, so library
        // loading fails before any FFI symbol is resolved or invoked.
        let err =
            unsafe { Session::load(std::path::Path::new("/no/such/library.so"), &mut fake_model) }
                .expect_err("should fail");
        match err {
            Error::LibraryLoad { .. } => {} // expected
            other => panic!("expected LibraryLoad, got {other:?}"),
        }
    }
}
