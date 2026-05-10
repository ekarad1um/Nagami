//! Zero-copy inference API.
//!
//! Design goals:
//!   * No allocation inside `Session::infer` -- caller owns all buffers.
//!   * Buffers cross the FFI as raw pointers with the correct length /
//!     dtype descriptors; librknnrt handles any internal type/layout
//!     conversion per our `pass_through` flag.
//!   * Every C call's return code is checked (the lesson from the
//!     rknpu2-swallowed-silent-failure bug).
//!
//! Ownership:
//!   * `InputSlice<'a>` borrows `&[T]` -- data must outlive the `infer` call.
//!   * `OutputSlice<'a>` borrows `&mut [T]` -- librknnrt writes in place;
//!     we use `is_prealloc=1` so there is nothing to `rknn_outputs_release`
//!     afterwards in terms of freeing user data, but we still call it as a
//!     safety no-op (the C API documents it as safe for is_prealloc=1).

use crate::rknn_runtime::{
    error::{Error, Result, check},
    session::{DataType, Session, TensorFormat},
    sys,
};
use std::os::raw::c_void;

/// The rknn C ABI reports buffer sizes as `u32`.  Buffers above this limit
/// cannot be described to the runtime.  Surfacing as a typed error keeps
/// with the crate's "no silent truncation" discipline.
fn n_bytes_to_u32(what: &'static str, n_bytes: usize) -> Result<u32> {
    u32::try_from(n_bytes).map_err(|_| Error::ShapeMismatch {
        what,
        expected: u32::MAX as usize,
        got: n_bytes,
    })
}

/// Borrowed input buffer + metadata for one input tensor.
///
/// Construct via `InputSlice::f32(...)`, `::f16(...)`, `::i8(...)`,
/// `::u8(...)`.  Each ctor records the element count and dtype; the
/// byte size reported to rknn_inputs_set is `size_of_val(data)`.
///
/// **Why `&mut [T]` not `&[T]`?** The C ABI of `rknn_inputs_set` declares
/// `void *buf` (non-const). librknnrt's documented behavior is to *read*
/// from `buf` and copy/normalize bytes into NPU memory -- it does not
/// write to the user buffer.  But the C type says it could; if a future
/// runtime ever did mutate, a `*const -> *mut` cast from a `&[T]` would be
/// undefined behavior.  Requiring `&mut [T]` makes the call sound under
/// the strict reading of the ABI without relying on Rockchip's docs.
///
/// `n_bytes` is stored as `usize` so construction is infallible even for
/// hypothetical buffers near the host word size; the conversion to the C
/// ABI's `u32` happens in `Session::infer`, which returns
/// `Error::ShapeMismatch` if the value won't fit (the practical limit is
/// ~4 GiB per tensor).
#[derive(Debug)]
pub struct InputSlice<'a> {
    pub(crate) index: u32,
    pub(crate) ptr: *mut c_void,
    pub(crate) n_bytes: usize,
    pub(crate) dtype: DataType,
    pub(crate) fmt: TensorFormat,
    pub(crate) pass_through: bool,
    // Lifetime + variance binding for an exclusive borrow of the input
    // buffer; ensures no other Rust reference can read or write the
    // backing memory while librknnrt has the pointer.
    _marker: std::marker::PhantomData<&'a mut [u8]>,
}

impl<'a> InputSlice<'a> {
    /// Fp32 host buffer. librknnrt will convert to the model's native dtype
    /// (typically fp16) when `pass_through=false`.  With `pass_through=true`,
    /// the buffer is copied raw -- only valid if the model's input is also
    /// fp32 (byte-matched).
    pub fn f32(index: u32, data: &'a mut [f32]) -> Self {
        InputSlice {
            index,
            n_bytes: std::mem::size_of_val(data),
            ptr: data.as_mut_ptr() as *mut c_void,
            dtype: DataType::Float32,
            fmt: TensorFormat::Undefined,
            pass_through: false,
            _marker: std::marker::PhantomData,
        }
    }

    /// Fp16 host buffer (`&mut [u16]` = bit-pattern of fp16).  Typically
    /// `pass_through=true` -- bytes match the NPU's native dtype exactly.
    pub fn f16(index: u32, data: &'a mut [u16]) -> Self {
        InputSlice {
            index,
            n_bytes: std::mem::size_of_val(data),
            ptr: data.as_mut_ptr() as *mut c_void,
            dtype: DataType::Float16,
            fmt: TensorFormat::Undefined,
            pass_through: true,
            _marker: std::marker::PhantomData,
        }
    }

    /// Int8 host buffer (for quantized int8 models).
    pub fn i8(index: u32, data: &'a mut [i8]) -> Self {
        InputSlice {
            index,
            n_bytes: std::mem::size_of_val(data),
            ptr: data.as_mut_ptr() as *mut c_void,
            dtype: DataType::Int8,
            fmt: TensorFormat::Undefined,
            pass_through: true,
            _marker: std::marker::PhantomData,
        }
    }

    /// Uint8 host buffer (image inputs typically).
    pub fn u8(index: u32, data: &'a mut [u8]) -> Self {
        InputSlice {
            index,
            n_bytes: std::mem::size_of_val(data),
            ptr: data.as_mut_ptr() as *mut c_void,
            dtype: DataType::Uint8,
            fmt: TensorFormat::Undefined,
            pass_through: true,
            _marker: std::marker::PhantomData,
        }
    }

    /// Builder: set the source-layout hint.  Required with
    /// `pass_through=false` on rv1126b -- the librknnrt normalize pipeline
    /// rejects `Undefined` with "only support NHWC src layout!".
    pub fn with_format(mut self, fmt: TensorFormat) -> Self {
        self.fmt = fmt;
        self
    }

    /// Builder: toggle `pass_through`.  Defaults are dtype-specific (see
    /// ctor docs); override when you know better.
    pub fn with_pass_through(mut self, pass_through: bool) -> Self {
        self.pass_through = pass_through;
        self
    }
}

/// Borrowed output buffer + metadata. librknnrt writes into `ptr` in place.
///
/// Use `OutputSlice::f32_preallocated(...)` for the standard "dequantize
/// to fp32" path (`want_float=true`).  For native-dtype output use
/// `f16_preallocated` (bit-pattern via `&mut [u16]`).
///
/// `n_bytes` is stored as `usize`; see `InputSlice` for the same rationale.
#[derive(Debug)]
pub struct OutputSlice<'a> {
    pub(crate) index: u32,
    pub(crate) ptr: *mut c_void,
    pub(crate) n_bytes: usize,
    pub(crate) want_float: bool,
    _marker: std::marker::PhantomData<&'a mut [u8]>,
}

impl<'a> OutputSlice<'a> {
    /// Pre-allocated fp32 output buffer, with `want_float=true` so
    /// librknnrt dequantizes from the model's native fp16 into our fp32.
    /// `buf.len()` must equal the model's `num_elements` for this tensor.
    pub fn f32_preallocated(index: u32, buf: &'a mut [f32]) -> Self {
        OutputSlice {
            index,
            n_bytes: std::mem::size_of_val(buf),
            ptr: buf.as_mut_ptr() as *mut c_void,
            want_float: true,
            _marker: std::marker::PhantomData,
        }
    }

    /// Pre-allocated fp16 output buffer (raw bit-pattern via `&mut [u16]`),
    /// with `want_float=false` so librknnrt writes the model's native
    /// fp16 directly.
    pub fn f16_preallocated(index: u32, buf: &'a mut [u16]) -> Self {
        OutputSlice {
            index,
            n_bytes: std::mem::size_of_val(buf),
            ptr: buf.as_mut_ptr() as *mut c_void,
            want_float: false,
            _marker: std::marker::PhantomData,
        }
    }
}

impl Session {
    /// Run one inference cycle: `rknn_inputs_set` ->
    /// `rknn_run` -> `rknn_outputs_get` ->
    /// `rknn_outputs_release`.  Zero allocations after
    /// construction of `input` / `output` descriptors.
    ///
    /// Caller provides pre-sized buffers.  On return, the
    /// output buffer contains the dequantized (or raw)
    /// result.
    pub fn infer(&mut self, input: InputSlice<'_>, output: OutputSlice<'_>) -> Result<()> {
        // Bound-check byte counts against the C ABI's u32
        // field width before we touch FFI: above
        // [`u32::MAX`] would truncate silently.  All
        // `ShapeMismatch` `what` strings use the "bytes"
        // suffix uniformly so operators can grep the logs.
        let in_size = n_bytes_to_u32("input buffer bytes", input.n_bytes)?;
        let out_size = n_bytes_to_u32("output buffer bytes", output.n_bytes)?;
        // Defence-in-depth: compare the caller's byte count
        // against the EXPECTED host-buffer size derived from
        // the cached tensor attr (zero FFI overhead;
        // populated at [`Session::load`]).
        //
        // **Two distinct expected-byte computations**
        // depending on `pass_through`:
        //
        // - `pass_through=false` (default for
        //   [`InputSlice::f32`]): the host buffer is in the
        //   CALLER's dtype (typically fp32); librknnrt
        //   converts to the model's native dtype
        //   internally.  Expected host bytes = `attr.n_elems
        //   * sizeof(input.dtype)`, NOT `attr.size`.  Using
        //   `attr.size` directly would break the standard
        //   `fp32 -> fp16` NPU path.
        //
        // - `pass_through=true` (typical for
        //   `InputSlice::f16/i8/u8`): the host buffer is
        //   byte-identical to the model's native dtype;
        //   librknnrt copies it raw.  Expected host bytes
        //   == `attr.size` exactly (which equals `n_elems
        //   * sizeof(model_dtype)`).
        //
        // For dtypes with no byte-per-element answer (`Int4` sub-byte
        // packing, `Unknown(_)` from a future SDK), we skip the
        // n_elems-based check and fall through to the upstream
        // `RknnBackbone::load`-style guard at the caller layer.
        if let Some(attr) = self.input_attr_cached(input.index) {
            let expected = if input.pass_through {
                attr.size as usize
            } else {
                match input.dtype.bytes_per_elem() {
                    Some(b) => attr.n_elems as usize * b,
                    None => input.n_bytes, // skip check for sub-byte / unknown
                }
            };
            if input.n_bytes != expected {
                return Err(Error::ShapeMismatch {
                    what: "input buffer bytes",
                    expected,
                    got: input.n_bytes,
                });
            }
        }
        if let Some(attr) = self.output_attr_cached(output.index) {
            // OutputSlice doesn't carry a `dtype` field -- its host
            // dtype is implicit in `want_float`:
            // - `want_float=true`  -> fp32 host buffer; librknnrt
            //   dequantizes the model's native dtype (typically fp16)
            //   into the caller's fp32 scratch.  Expected host bytes
            //   = `n_elems * sizeof(f32)` = `n_elems * 4`.
            // - `want_float=false` -> host buffer is byte-identical to
            //   the model's native dtype (raw write).  Expected host
            //   bytes = `attr.size` (model native).
            let expected = if output.want_float {
                attr.n_elems as usize * 4
            } else {
                attr.size as usize
            };
            if output.n_bytes != expected {
                return Err(Error::ShapeMismatch {
                    what: "output buffer bytes",
                    expected,
                    got: output.n_bytes,
                });
            }
        }

        // 1. rknn_inputs_set
        let mut rk_in = sys::rknn_input {
            index: input.index,
            buf: input.ptr,
            size: in_size,
            pass_through: if input.pass_through { 1 } else { 0 },
            type_: input.dtype.to_raw(),
            fmt: input.fmt.to_raw(),
        };
        // SAFETY: `input.ptr` originates from `&mut [T]` (see InputSlice
        // ctors) and is exclusively borrowed for the lifetime of this
        // function -- no other Rust reference can read or write the
        // backing memory.  `rk_in` is a stack-local; the C ABI signature
        // `rknn_input *` is mutable but Rockchip documents this call as
        // synchronous and read-only on `buf`.  Even if a future runtime
        // wrote to `buf`, the &mut bound on InputSlice keeps Rust's
        // aliasing invariants intact.
        let code = unsafe {
            (self.table().inputs_set)(self.context(), 1, &mut rk_in as *mut sys::rknn_input)
        };
        check("rknn_inputs_set", "set input 0", code)?;

        // 2. rknn_run (blocking)
        let code = unsafe { (self.table().run)(self.context(), std::ptr::null_mut()) };
        check("rknn_run", "run inference", code)?;

        // 3. rknn_outputs_get (pre-allocated)
        let mut rk_out = sys::rknn_output {
            want_float: if output.want_float { 1 } else { 0 },
            is_prealloc: 1, // our buffer, librknnrt writes into it
            index: output.index,
            buf: output.ptr,
            size: out_size,
        };
        let code = unsafe {
            (self.table().outputs_get)(
                self.context(),
                1,
                &mut rk_out as *mut sys::rknn_output,
                std::ptr::null_mut(),
            )
        };
        check("rknn_outputs_get", "get output 0", code)?;

        // 4. rknn_outputs_release
        // Per rknn_api.h: with is_prealloc=TRUE, the user's buf is NOT freed
        // (only FALSE-path bufs allocated by librknnrt are).  We still call
        // this to release any internal accounting state the runtime tracks
        // per-`rknn_outputs_get`.
        let code = unsafe {
            (self.table().outputs_release)(self.context(), 1, &mut rk_out as *mut sys::rknn_output)
        };
        check("rknn_outputs_release", "release output 0", code)?;
        Ok(())
    }
}
