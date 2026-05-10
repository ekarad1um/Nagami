//! Runtime-loaded FFI dispatch table.  Holds the `libloading::Library` +
//! 7 resolved symbols.  All symbols resolve eagerly at construction time, so
//! a missing symbol is a hard error at `SymbolTable::load` -- not a deferred
//! surprise at inference time.

use crate::rknn_runtime::sys::{
    rknn_context, rknn_init_extend, rknn_input, rknn_output, rknn_output_extend, rknn_query_cmd,
    rknn_run_extend,
};
use libloading::Library;
use std::ffi::c_int;
use std::os::raw::c_void;
use std::path::Path;
use std::sync::Arc;

type FnInit =
    unsafe extern "C" fn(*mut rknn_context, *mut c_void, u32, u32, *mut rknn_init_extend) -> c_int;
type FnDestroy = unsafe extern "C" fn(rknn_context) -> c_int;
type FnQuery = unsafe extern "C" fn(rknn_context, rknn_query_cmd, *mut c_void, u32) -> c_int;
type FnInputsSet = unsafe extern "C" fn(rknn_context, u32, *mut rknn_input) -> c_int;
type FnRun = unsafe extern "C" fn(rknn_context, *mut rknn_run_extend) -> c_int;
type FnOutputsGet =
    unsafe extern "C" fn(rknn_context, u32, *mut rknn_output, *mut rknn_output_extend) -> c_int;
type FnOutputsRelease = unsafe extern "C" fn(rknn_context, u32, *mut rknn_output) -> c_int;

/// Resolved FFI table.  Every call in the safe wrapper goes through here.
/// Debug-printable but symbols are opaque fn pointers -- we just show that
/// the table is populated.
pub(crate) struct SymbolTable {
    // The loaded library is kept here for the lifetime of the resolved
    // function pointers.  Dropping it would unmap the code pages the
    // pointers point at.
    //
    // Wrapped in `Arc<Library>` so a future
    // SessionPool that wants N concurrent sessions sharing one
    // `dlopen` (~5 MB mapped) can `Arc::clone` rather than re-
    // `dlopen`'ing per session.  Cost today is one `Arc::clone` per
    // session-create on the cold path (init only); the hot path
    // never touches `_lib`.  Field name kept `_lib` (leading
    // underscore) so the existing "carried for lifetime, not
    // dispatched through" semantics stays explicit at the
    // declaration site.
    _lib: Arc<Library>,
    pub init: FnInit,
    pub destroy: FnDestroy,
    pub query: FnQuery,
    pub inputs_set: FnInputsSet,
    pub run: FnRun,
    pub outputs_get: FnOutputsGet,
    pub outputs_release: FnOutputsRelease,
}

impl std::fmt::Debug for SymbolTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SymbolTable")
            .field("loaded", &true)
            .field("symbols", &7u32)
            .finish()
    }
}

impl SymbolTable {
    /// Load the shared library at `path` and resolve all 7 symbols eagerly.
    ///
    /// Safety: the loaded library is trusted to export the RKNN C ABI.
    /// We error on missing symbols but cannot verify their signatures at
    /// runtime -- mismatched ABI would be UB.  In practice the target
    /// libraries (`librknnrt.so` 2.x, `librknnmrt.so`) are ABI-stable.
    pub(crate) fn load(path: &Path) -> Result<Self, LoadError> {
        // libloading::Library::new is unsafe because any .so can have
        // arbitrary init-code side effects.  In our case we trust Rockchip's
        // vendored library.
        let lib = unsafe { Library::new(path) }.map_err(|source| LoadError::Library {
            path: path.to_path_buf(),
            source,
        })?;

        // Resolve each symbol.  `Symbol<'_>` borrows the library; deref
        // returns `&T`, and `T: Copy` (fn pointer) lets us copy the pointer
        // out.  The copied pointer stays valid as long as `_lib` is held by
        // the returned `SymbolTable`.
        unsafe fn resolve<T: Copy>(
            lib: &Library,
            name_bytes: &[u8],
            name_str: &'static str,
        ) -> Result<T, LoadError> {
            let sym: libloading::Symbol<'_, T> =
                unsafe { lib.get(name_bytes) }.map_err(|source| LoadError::Symbol {
                    name: name_str,
                    source,
                })?;
            Ok(*sym)
        }

        // Names inlined at each call so reorders stay local.  Null-terminated
        // for libloading's C-string lookup; display-name is the error-report
        // form.
        let init: FnInit = unsafe { resolve(&lib, b"rknn_init\0", "rknn_init") }?;
        let destroy: FnDestroy = unsafe { resolve(&lib, b"rknn_destroy\0", "rknn_destroy") }?;
        let query: FnQuery = unsafe { resolve(&lib, b"rknn_query\0", "rknn_query") }?;
        let inputs_set: FnInputsSet =
            unsafe { resolve(&lib, b"rknn_inputs_set\0", "rknn_inputs_set") }?;
        let run: FnRun = unsafe { resolve(&lib, b"rknn_run\0", "rknn_run") }?;
        let outputs_get: FnOutputsGet =
            unsafe { resolve(&lib, b"rknn_outputs_get\0", "rknn_outputs_get") }?;
        let outputs_release: FnOutputsRelease =
            unsafe { resolve(&lib, b"rknn_outputs_release\0", "rknn_outputs_release") }?;

        Ok(SymbolTable {
            _lib: Arc::new(lib),
            init,
            destroy,
            query,
            inputs_set,
            run,
            outputs_get,
            outputs_release,
        })
    }
}

/// Errors from the library-loading phase.  Kept separate from the main
/// `Error` enum so the wrapper can convert them to a common variant without
/// losing detail.
#[derive(Debug)]
pub(crate) enum LoadError {
    Library {
        path: std::path::PathBuf,
        source: libloading::Error,
    },
    Symbol {
        name: &'static str,
        source: libloading::Error,
    },
}
