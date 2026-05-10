//! Session -- owns a loaded RKNN context and its runtime library, exposes
//! safe query + inference methods.  Zero-copy at the inference boundary
//! (see `infer.rs`).

use crate::rknn_runtime::{
    error::{Error, Result, check},
    ffi::SymbolTable,
    sys,
};
use std::{marker::PhantomData, mem::MaybeUninit, os::raw::c_char, path::Path};

/// Number of input + output tensors on a loaded model.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct IoCount {
    pub n_input: u32,
    pub n_output: u32,
}

/// SDK version strings reported by `rknn_query(RKNN_QUERY_SDK_VERSION)`.
#[derive(Debug, Clone)]
pub struct SdkVersion {
    pub api: String,
    pub driver: String,
}

/// Tensor attribute reported by `rknn_query(RKNN_QUERY_{INPUT,OUTPUT}_ATTR)`.
/// Fields mirror `rknn_tensor_attr` from the header but with decoded
/// strings/enums.  `dims` is exposed only up to the valid `n_dims`.
#[derive(Debug, Clone)]
pub struct TensorAttr {
    pub index: u32,
    pub name: String,
    pub dims: Vec<u32>,
    /// Total number of elements (logical -- doesn't include NPU-internal padding).
    pub n_elems: u32,
    /// Bytes in the logical form: `n_elems * size_of::<type>()`.
    pub size: u32,
    pub dtype: DataType,
    pub format: TensorFormat,
    pub qnt_type: QntType,
}

/// Tensor element type.  Unknown values from the header are mapped to
/// `Unknown(raw)` without panicking -- matches Rockchip's future-extensibility
/// convention (they add new tensor types across SDK versions).
///
/// The set of known variants tracks every `RKNN_TENSOR_*` constant exposed
/// by `bindings.rs`; new SDK additions surface as `Unknown(raw)` until
/// `from_raw` / `to_raw` are extended.  See the
/// `datatype_round_trip_covers_all_known_constants` test below.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum DataType {
    Float32,
    Float16,
    Int8,
    Uint8,
    Int16,
    Uint16,
    Int32,
    Uint32,
    Int64,
    Bool,
    Int4,
    Bfloat16,
    Unknown(u32),
}

impl DataType {
    pub(crate) fn to_raw(self) -> sys::rknn_tensor_type {
        match self {
            DataType::Float32 => sys::_rknn_tensor_type::RKNN_TENSOR_FLOAT32,
            DataType::Float16 => sys::_rknn_tensor_type::RKNN_TENSOR_FLOAT16,
            DataType::Int8 => sys::_rknn_tensor_type::RKNN_TENSOR_INT8,
            DataType::Uint8 => sys::_rknn_tensor_type::RKNN_TENSOR_UINT8,
            DataType::Int16 => sys::_rknn_tensor_type::RKNN_TENSOR_INT16,
            DataType::Uint16 => sys::_rknn_tensor_type::RKNN_TENSOR_UINT16,
            DataType::Int32 => sys::_rknn_tensor_type::RKNN_TENSOR_INT32,
            DataType::Uint32 => sys::_rknn_tensor_type::RKNN_TENSOR_UINT32,
            DataType::Int64 => sys::_rknn_tensor_type::RKNN_TENSOR_INT64,
            DataType::Bool => sys::_rknn_tensor_type::RKNN_TENSOR_BOOL,
            DataType::Int4 => sys::_rknn_tensor_type::RKNN_TENSOR_INT4,
            DataType::Bfloat16 => sys::_rknn_tensor_type::RKNN_TENSOR_BFLOAT16,
            DataType::Unknown(x) => x,
        }
    }

    fn from_raw(raw: sys::rknn_tensor_type) -> Self {
        match raw {
            sys::_rknn_tensor_type::RKNN_TENSOR_FLOAT32 => DataType::Float32,
            sys::_rknn_tensor_type::RKNN_TENSOR_FLOAT16 => DataType::Float16,
            sys::_rknn_tensor_type::RKNN_TENSOR_INT8 => DataType::Int8,
            sys::_rknn_tensor_type::RKNN_TENSOR_UINT8 => DataType::Uint8,
            sys::_rknn_tensor_type::RKNN_TENSOR_INT16 => DataType::Int16,
            sys::_rknn_tensor_type::RKNN_TENSOR_UINT16 => DataType::Uint16,
            sys::_rknn_tensor_type::RKNN_TENSOR_INT32 => DataType::Int32,
            sys::_rknn_tensor_type::RKNN_TENSOR_UINT32 => DataType::Uint32,
            sys::_rknn_tensor_type::RKNN_TENSOR_INT64 => DataType::Int64,
            sys::_rknn_tensor_type::RKNN_TENSOR_BOOL => DataType::Bool,
            sys::_rknn_tensor_type::RKNN_TENSOR_INT4 => DataType::Int4,
            sys::_rknn_tensor_type::RKNN_TENSOR_BFLOAT16 => DataType::Bfloat16,
            x => DataType::Unknown(x),
        }
    }

    /// Bytes per element for the host buffer that carries this dtype.
    /// Used by the `Session::infer` validation to compute the EXPECTED
    /// host-buffer size from the model's `n_elems` (when
    /// `pass_through=false`, librknnrt converts the host buffer to the
    /// model's native dtype internally -- so the host bytes are
    /// `n_elems * sizeof(host_dtype)`, NOT `n_elems * sizeof(model_dtype)`).
    ///
    /// Returns `None` for `Int4` (sub-byte; element count is not the
    /// natural unit) and `Unknown(_)` (caller's responsibility).
    pub fn bytes_per_elem(self) -> Option<usize> {
        match self {
            DataType::Bool | DataType::Int8 | DataType::Uint8 => Some(1),
            DataType::Float16 | DataType::Bfloat16 | DataType::Int16 | DataType::Uint16 => Some(2),
            DataType::Float32 | DataType::Int32 | DataType::Uint32 => Some(4),
            DataType::Int64 => Some(8),
            DataType::Int4 | DataType::Unknown(_) => None,
        }
    }
}

/// Tensor layout hint.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum TensorFormat {
    Nchw,
    Nhwc,
    Nc1Hwc2,
    Undefined,
    Unknown(u32),
}

impl TensorFormat {
    pub(crate) fn to_raw(self) -> sys::rknn_tensor_format {
        match self {
            TensorFormat::Nchw => sys::_rknn_tensor_format::RKNN_TENSOR_NCHW,
            TensorFormat::Nhwc => sys::_rknn_tensor_format::RKNN_TENSOR_NHWC,
            TensorFormat::Nc1Hwc2 => sys::_rknn_tensor_format::RKNN_TENSOR_NC1HWC2,
            TensorFormat::Undefined => sys::_rknn_tensor_format::RKNN_TENSOR_UNDEFINED,
            TensorFormat::Unknown(x) => x,
        }
    }

    fn from_raw(raw: sys::rknn_tensor_format) -> Self {
        match raw {
            sys::_rknn_tensor_format::RKNN_TENSOR_NCHW => TensorFormat::Nchw,
            sys::_rknn_tensor_format::RKNN_TENSOR_NHWC => TensorFormat::Nhwc,
            sys::_rknn_tensor_format::RKNN_TENSOR_NC1HWC2 => TensorFormat::Nc1Hwc2,
            sys::_rknn_tensor_format::RKNN_TENSOR_UNDEFINED => TensorFormat::Undefined,
            x => TensorFormat::Unknown(x),
        }
    }
}

/// Quantization type.  Surfaced for diagnostics; not consumed by this crate.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum QntType {
    None,
    Dfp,
    AffineAsymmetric,
    Unknown(u32),
}

impl QntType {
    fn from_raw(raw: sys::rknn_tensor_qnt_type) -> Self {
        match raw {
            sys::_rknn_tensor_qnt_type::RKNN_TENSOR_QNT_NONE => QntType::None,
            sys::_rknn_tensor_qnt_type::RKNN_TENSOR_QNT_DFP => QntType::Dfp,
            sys::_rknn_tensor_qnt_type::RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC => {
                QntType::AffineAsymmetric
            }
            x => QntType::Unknown(x),
        }
    }
}

/// Loaded RKNN model context.  Single-threaded per Rockchip's docs -- we
/// mark it `Send` (ownership can move between threads) but not `Sync`
/// (concurrent API calls from multiple threads on the same context are UB).
///
/// The `!Sync` property is enforced by `PhantomData<Cell<()>>`, which is
/// `Send` but not `Sync` -- stable Rust equivalent of a negative impl.
pub struct Session {
    /// Cached input / output tensor attrs for zero-cost
    /// defence-in-depth validation in [`Session::infer`];
    /// without it the only protection against an `InputSlice`
    /// byte-count mismatch (e.g. fp32 buffer to a fp16-input
    /// model) is upstream caller validation.
    ///
    /// **Field-declaration order is conventional, NOT
    /// load-bearing.**  Rust runs `Drop::drop` to completion
    /// before any field is dropped, so the
    /// `(self.table.destroy)(self.context)` call in
    /// `Session::Drop` always sees both fields alive.  We keep
    /// `caches -> context -> table` as "release in reverse
    /// acquisition order" for the mental model only — zero
    /// runtime effect.
    input_attrs: Vec<TensorAttr>,
    output_attrs: Vec<TensorAttr>,
    context: sys::rknn_context,
    table: SymbolTable,
    _not_sync: PhantomData<std::cell::Cell<()>>,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("context", &format_args!("{:#x}", self.context))
            .field("table", &self.table)
            .finish()
    }
}

impl Session {
    /// Load the runtime library from `lib_path` and initialize an RKNN
    /// context from `model` bytes.
    ///
    /// `model` is `&mut` because `rknn_init`'s C signature is
    /// `void* model` (non-const), so the runtime is permitted to mutate
    /// the buffer.  Callers that need the original bytes preserved should
    /// pass a freshly read copy.
    ///
    /// # Safety
    ///
    /// Callers must satisfy ALL of:
    ///
    /// 1. **Trusted lib.**  `lib_path` is a trusted Rockchip
    ///    librknnrt build (vendor distribution).
    /// 2. **ABI match.**  Its ABI matches the vendored
    ///    `rknn_runtime/bindings.rs` (host `layout_*` tests
    ///    verify this against a representative `.so`; bumping
    ///    the lib needs a re-gen).
    /// 3. **No path tampering.**  Deployment chain pins the
    ///    resolved path between boot and `Session::load` (file
    ///    permissions, immutable rootfs, restricted `RKNN_LIB`).
    ///
    /// Mismatched ABI is UB even though the per-call `unsafe`
    /// block looks innocent.  Once `Session::load` returns, the
    /// `&mut self` methods are safe -- the trust obligation is
    /// fully discharged at construction.
    ///
    /// # Model buffer lifetime
    ///
    /// We pass `flag = 0`, so `rknn_init` internalises the
    /// model bytes before returning; dropping `model` after the
    /// call is safe.  A future flip to
    /// `RKNN_FLAG_MODEL_BUFFER_ZERO_COPY` would require storing
    /// `model` in `Session` to outlive the context.
    pub unsafe fn load(lib_path: &Path, model: &mut [u8]) -> Result<Self> {
        let table = SymbolTable::load(lib_path)?;
        // rknn_init's `size` parameter is u32.  Protect against silent
        // truncation on hypothetical >4 GiB model buffers (real .rknn files
        // are MB-sized, but a wrapper promising "no silent failures" should
        // not open that door).
        let model_size = u32::try_from(model.len()).map_err(|_| Error::ShapeMismatch {
            what: "model bytes",
            expected: u32::MAX as usize,
            got: model.len(),
        })?;
        let mut ctx: sys::rknn_context = 0;
        let code = unsafe {
            (table.init)(
                &mut ctx,
                model.as_mut_ptr() as *mut _,
                model_size,
                0,                    // flag: no special flags
                std::ptr::null_mut(), // extend: unused
            )
        };
        check("rknn_init", "load model", code)?;
        // Populate the attr cache.  Build a
        // partially-constructed [`Session`] with empty
        // Vecs first so
        // we can use `&self`-bound `io_count` / `input_attr` /
        // `output_attr` helpers to populate them; then store the
        // populated Vecs in the final Session value.  On any query
        // failure the partial Session drops, releasing the rknn
        // context via Session::Drop -- no leak.
        let mut session = Session {
            table,
            context: ctx,
            input_attrs: Vec::new(),
            output_attrs: Vec::new(),
            _not_sync: PhantomData,
        };
        let io = session.io_count()?;
        session.input_attrs.reserve_exact(io.n_input as usize);
        session.output_attrs.reserve_exact(io.n_output as usize);
        for i in 0..io.n_input {
            session.input_attrs.push(session.input_attr(i)?);
        }
        for i in 0..io.n_output {
            session.output_attrs.push(session.output_attr(i)?);
        }
        Ok(session)
    }

    /// Cached input tensor attr from load time.  O(1), no FFI roundtrip.
    /// Returns None if `index` is out of range.  Use this instead of
    /// `input_attr(index)` on the hot path.
    pub fn input_attr_cached(&self, index: u32) -> Option<&TensorAttr> {
        self.input_attrs.get(index as usize)
    }

    /// Cached output tensor attr from load time.  Same semantics as
    /// [`Self::input_attr_cached`].
    pub fn output_attr_cached(&self, index: u32) -> Option<&TensorAttr> {
        self.output_attrs.get(index as usize)
    }

    /// Query SDK + driver version strings.
    pub fn sdk_version(&self) -> Result<SdkVersion> {
        let mut ver = MaybeUninit::<sys::rknn_sdk_version>::zeroed();
        let code = unsafe {
            (self.table.query)(
                self.context,
                sys::_rknn_query_cmd::RKNN_QUERY_SDK_VERSION,
                ver.as_mut_ptr() as *mut _,
                std::mem::size_of::<sys::rknn_sdk_version>() as u32,
            )
        };
        check("rknn_query", "SDK version", code)?;
        let ver = unsafe { ver.assume_init() };
        Ok(SdkVersion {
            api: c_fixed_str_to_string(&ver.api_version)?,
            driver: c_fixed_str_to_string(&ver.drv_version)?,
        })
    }

    /// Query input + output tensor counts.
    pub fn io_count(&self) -> Result<IoCount> {
        let mut io = MaybeUninit::<sys::rknn_input_output_num>::zeroed();
        let code = unsafe {
            (self.table.query)(
                self.context,
                sys::_rknn_query_cmd::RKNN_QUERY_IN_OUT_NUM,
                io.as_mut_ptr() as *mut _,
                std::mem::size_of::<sys::rknn_input_output_num>() as u32,
            )
        };
        check("rknn_query", "input/output count", code)?;
        let io = unsafe { io.assume_init() };
        Ok(IoCount {
            n_input: io.n_input,
            n_output: io.n_output,
        })
    }

    /// Query the attribute of input tensor `index`.
    pub fn input_attr(&self, index: u32) -> Result<TensorAttr> {
        self.query_tensor_attr(
            index,
            sys::_rknn_query_cmd::RKNN_QUERY_INPUT_ATTR,
            "input attr",
        )
    }

    /// Query the attribute of output tensor `index`.
    pub fn output_attr(&self, index: u32) -> Result<TensorAttr> {
        self.query_tensor_attr(
            index,
            sys::_rknn_query_cmd::RKNN_QUERY_OUTPUT_ATTR,
            "output attr",
        )
    }

    fn query_tensor_attr(
        &self,
        index: u32,
        cmd: sys::rknn_query_cmd,
        context: &'static str,
    ) -> Result<TensorAttr> {
        // Per the C API, the caller writes `info.index` before calling
        // rknn_query; the library fills the rest of the struct.
        let mut attr = MaybeUninit::<sys::rknn_tensor_attr>::zeroed();
        unsafe {
            (*attr.as_mut_ptr()).index = index;
        }
        let code = unsafe {
            (self.table.query)(
                self.context,
                cmd,
                attr.as_mut_ptr() as *mut _,
                std::mem::size_of::<sys::rknn_tensor_attr>() as u32,
            )
        };
        check("rknn_query", context, code)?;
        let attr = unsafe { attr.assume_init() };

        let n_dims = attr.n_dims as usize;
        // Defensive: an n_dims beyond the fixed-capacity `dims[RKNN_MAX_DIMS]`
        // would indicate a C-side write-past-end.  Report as param-invalid
        // rather than truncating silently.
        if n_dims > sys::RKNN_MAX_DIMS as usize {
            return Err(Error::Rknn {
                name: "rknn_query",
                code: sys::RKNN_ERR_PARAM_INVALID,
                context: "n_dims exceeds RKNN_MAX_DIMS",
            });
        }

        Ok(TensorAttr {
            index: attr.index,
            name: c_fixed_str_to_string(&attr.name)?,
            dims: attr.dims[..n_dims].to_vec(),
            n_elems: attr.n_elems,
            size: attr.size,
            dtype: DataType::from_raw(attr.type_),
            format: TensorFormat::from_raw(attr.fmt),
            qnt_type: QntType::from_raw(attr.qnt_type),
        })
    }

    /// Crate-internal: symbol table accessor for the `infer` module.
    pub(crate) fn table(&self) -> &SymbolTable {
        &self.table
    }

    /// Crate-internal: opaque context handle.
    pub(crate) fn context(&self) -> sys::rknn_context {
        self.context
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // Best-effort cleanup.  Drop must never panic, so just ignore error codes.
        let _ = unsafe { (self.table.destroy)(self.context) };
    }
}

/// Decode a fixed-capacity null-terminated C string into `String`.
/// `rknn_api.h` uses `char name[256]` / `char api_version[256]` style
/// arrays; the useful bytes end at the first `\0`.
fn c_fixed_str_to_string<const N: usize>(buf: &[c_char; N]) -> Result<String> {
    // Find the NUL terminator within the fixed buffer.  If none present,
    // the entire buffer is valid text (malformed but handleable).
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, N) };
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(N);
    let s = std::str::from_utf8(&bytes[..end])?;
    Ok(s.to_string())
}
