//! Canonical "write bytes to a file atomically" primitive.
//!
//! Backs every workspace metadata write and the converter +
//! training on-disk artifact paths so the durability discipline
//! is consistent across the daemon.
//!
//! # Protocol
//!
//! 1. Create a [`tempfile::NamedTempFile`] in `final_path`'s
//!    parent directory (so the rename below is intra-filesystem
//!    and POSIX-atomic).
//! 2. Write `bytes` and call [`std::fs::File::sync_all`] on the
//!    tempfile so its data reaches stable storage before the
//!    rename publishes the new name.
//! 3. [`tempfile::NamedTempFile::persist`] (atomic rename) into
//!    `final_path`.
//! 4. fsync the parent directory so the directory-entry update
//!    also reaches stable storage; without this, a power loss
//!    after step 3 returns can revert the rename.
//!
//! Steps 2 + 4 close the durability gap that pure `tempfile +
//! persist` leaves: data + name both reach disk before the call
//! returns.
//!
//! # Failure modes
//!
//! - Tempfile creation / write / sync I/O failure ->
//!   [`FileError::Io`] (the partial tempfile is auto-dropped).
//! - `persist` failure -> [`FileError::Persist`] (the rename
//!   itself failed -- typically `EXDEV` when `final_path`'s
//!   parent is on a different filesystem than the source).
//! - Parent fsync failure -> [`FileError::Io`] (rare; reported
//!   so callers can treat the write as un-durable).

use crate::file_mgr::error::{FileError, io_err};
use crate::file_mgr::validate::fsync_dir;
use std::path::Path;

/// Atomically write `bytes` to `final_path`.
///
/// The path's parent directory must exist; the temporary
/// staging file is created inside it (intra-FS guarantee for
/// the rename).  On success, both the file's data AND the
/// directory entry have reached stable storage.  On any
/// failure, `final_path` is unchanged: the tempfile is dropped
/// without persisting and no partial bytes appear under the
/// final name.
pub fn put_atomic(final_path: &Path, bytes: &[u8]) -> Result<(), FileError> {
    use std::io::Write;
    let parent = final_path.parent().ok_or_else(|| {
        io_err(
            final_path.display(),
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "put_atomic: path has no parent directory",
            ),
        )
    })?;
    let mut tmp =
        tempfile::NamedTempFile::new_in(parent).map_err(|e| io_err(parent.display(), e))?;
    tmp.write_all(bytes)
        .map_err(|e| io_err(final_path.display(), e))?;
    tmp.flush().map_err(|e| io_err(final_path.display(), e))?;
    // Durability barrier: `flush` only drains Rust-side
    // buffering.  Without `sync_all` the rename below can
    // become visible BEFORE the file's data reaches stable
    // storage.
    tmp.as_file()
        .sync_all()
        .map_err(|e| io_err(final_path.display(), e))?;
    tmp.persist(final_path)?;
    // fsync the parent dir so the rename's directory-entry
    // update also reaches stable storage.  Without this, a
    // power loss after `persist` returns can revert the
    // rename.
    fsync_dir(parent).map_err(|e| io_err(parent.display(), e))?;
    Ok(())
}
