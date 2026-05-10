//! Runtime path helpers.

use std::path::PathBuf;

/// Search standard Linux locations for `librknnrt.so` / `librknnmrt.so`.
/// Returns paths in discovery order -- first hit is typically what the
/// operator wants.  The CLI layer of the consuming binary should still
/// accept an explicit `--library` override.
pub fn find_library_candidates() -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = vec![
        PathBuf::from("/usr/lib"),
        PathBuf::from("/usr/local/lib"),
        PathBuf::from("/usr/lib/aarch64-linux-gnu"),
    ];
    if let Ok(ld) = std::env::var("LD_LIBRARY_PATH") {
        for part in ld.split(':').filter(|s| !s.is_empty()) {
            dirs.push(PathBuf::from(part));
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        dirs.push(PathBuf::from(home).join(".local/lib"));
    }

    let mut hits = Vec::new();
    for d in &dirs {
        for name in ["librknnrt.so", "librknnmrt.so"] {
            let p = d.join(name);
            if p.exists() {
                hits.push(p);
            }
        }
    }
    hits
}
