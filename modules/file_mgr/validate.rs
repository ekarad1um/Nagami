//! Asset-name + extension validation, plus the small
//! filesystem helpers (sha256 streaming, directory fsync).

use crate::common::ids::AssetId;
use crate::file_mgr::error::{FileError, io_err};
use sha2::{Digest, Sha256};
use std::path::Path;

/// Validate a single asset filename (the trailing component
/// of an asset path).
///
/// Thin pre-rename wrapper around [`AssetId::parse`]: every
/// rule (ASCII allowlist `[A-Za-z0-9._-]`, leading-`.`
/// reject, length cap, non-empty) lives there.  Keeping the
/// `&str -> Result<(), FileError>` shape so call sites that
/// only need to validate (api routes, training cfg) don't
/// pay an [`AssetId`] allocation; the upload commit path
/// constructs the [`AssetId`] once it owns the name string.
///
/// Errors flatten to [`FileError::InvalidName`] (lossy, but
/// the diagnostic mirrors what the operator sees) so existing
/// callers' `match FileError::InvalidName(_)` patterns keep
/// working.  Use [`AssetId::parse`] directly when the
/// structured [`crate::common::ids::IdError`] is needed.
pub fn validate_asset_name(name: &str) -> Result<(), FileError> {
    AssetId::parse(name)
        .map(|_| ())
        .map_err(|e| FileError::InvalidName(e.to_string()))
}

pub(crate) fn validate_extension(name: &str, allowed: &[&'static str]) -> Result<(), FileError> {
    // We do NOT use [`std::path::Path::extension`] because it
    // only reports the LAST component (`.gz` of `.tar.gz`); we
    // want to accept both `.tar.gz` and `.tgz` specifically.
    //
    // No allocation in the happy path: compare the tail of
    // `name` against `.<ext>` byte-for-byte,
    // case-insensitively.  We work on `&[u8]` rather than
    // `&str` slices because [`validate_asset_name`] permits
    // multi-byte UTF-8 (Linux allows it), and `name.len() -
    // need` could otherwise land mid-codepoint and panic the
    // str slice.  The expected list is `&[&'static str]` and
    // is always ASCII, so a byte-level case-insensitive
    // compare matches the prior str compare on inputs we
    // accept and never panics on inputs we reject.
    let name_bytes = name.as_bytes();
    for ext in allowed {
        let ext_bytes = ext.as_bytes();
        let need = ext_bytes.len() + 1; // `.` + ext
        if name_bytes.len() < need {
            continue;
        }
        let tail = &name_bytes[name_bytes.len() - need..];
        if tail[0] == b'.' && tail[1..].eq_ignore_ascii_case(ext_bytes) {
            return Ok(());
        }
    }
    // For the diagnostic, emit everything after the first dot
    // -- for `foo.tar.gz` that's `tar.gz`, which lines up with
    // the expected list.  Falls back to the bare `name` when
    // there's no dot.  Safe to slice `&str` here:
    // `find('.')` returns a byte index that's always a char
    // boundary (`.` is ASCII).
    let got = name
        .find('.')
        .map(|i| &name[i + 1..])
        .unwrap_or(name)
        .to_string();
    Err(FileError::InvalidExtension {
        got,
        expected: allowed.to_vec(),
    })
}

// `hex_lowercase` lives at `crate::common::hex` so the inference
// backbone (forbidden from depending on `file_mgr` per the layer
// guard at `tests/dependency_edge_guard.rs`) can share the same
// encoder.  Re-exported here for `file_mgr`'s submodule consumers
// (`uploader`, `recovery`, `active_head_writer`) and for the
// `pub(crate)` re-export at the `file_mgr.rs` root that the
// api/routes layer reaches through.
pub(crate) use crate::common::hex::hex_lowercase;

/// SHA-256 a file in 64 KiB chunks, returning the
/// lowercase-hex digest.  Used by
/// [`crate::file_mgr::WorkspaceMgr::validate`] so multi-GB
/// dataset uploads don't have to be slurped into RAM.
///
/// We read directly from the [`std::fs::File`] (no
/// `BufReader`) because the 64 KiB chunk size is itself the
/// I/O unit; wrapping in a same-sized BufReader would just
/// add a bounce buffer and an extra 64 KiB heap allocation
/// per call.
pub(crate) fn sha256_file_streaming(path: &Path) -> Result<String, FileError> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| io_err(path.display(), e))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let n = f.read(&mut buf).map_err(|e| io_err(path.display(), e))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex_lowercase(&hasher.finalize()))
}

/// Fsync the parent directory of `path` so the rename that
/// just put `path` in place is itself durable.  POSIX
/// `rename(2)` is atomic w.r.t. concurrent observers, but
/// the directory entry update only reaches stable storage
/// when the directory inode is fsynced.  Without this, a hard
/// power loss after a successful `persist()` can undo the
/// rename: the file content is durable (the tempfile was
/// fsynced pre-rename), but the directory entry still names
/// the temp.
///
/// Best-effort on platforms that don't support fsync on a
/// directory (Windows): `File::open` of a directory may
/// itself fail; we surface that as an error since callers
/// expect Linux/macOS where this works.
pub(crate) fn fsync_dir(path: &std::path::Path) -> std::io::Result<()> {
    std::fs::File::open(path)?.sync_all()
}
