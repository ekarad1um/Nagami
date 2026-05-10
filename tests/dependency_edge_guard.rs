//! Lightweight machine-checked layer-graph guard.
//!
//! Walks every production Rust file under `modules/<top>/`, extracts
//! `crate::<other>::` references, and asserts the resulting (source,
//! target) edges all appear in an allowlist matching
//! `docs/ARCH_BOUNDARIES.md`.  Catches a future cross-layer import
//! before it lands without resorting to a workspace split.
//!
//! Heuristics:
//!
//!   * `#[cfg(test)]` modules + the trailing in-file `mod tests { ... }`
//!     pattern are excluded by truncating the file at the first
//!     `#[cfg(test)]` attribute occurrence.  Production code that
//!     places a `#[cfg(test)]` block in the middle of a module would
//!     lose visibility of the trailing production code, but the
//!     existing convention is "tests at the end" and a violation
//!     would surface as a false negative (missed allowed edge), not a
//!     false positive (spurious failure).
//!   * Single-line comments (`//`, `///`, `//!`) are stripped per
//!     line so a `crate::X::` reference inside a doc comment does
//!     not count as a real edge.
//!   * Block comments (`/* ... */`) are NOT stripped.  Production
//!     code in this crate uses line comments exclusively; a block
//!     comment containing `crate::X::` would surface as a false
//!     positive that the allowlist must absorb.
//!
//! Source of truth: this test's `ALLOWED_EDGES` table.  When the
//! layer graph in `docs/ARCH_BOUNDARIES.md` changes, update the
//! table in lockstep.

#![allow(clippy::disallowed_methods)]

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

/// Top-level modules in `modules/lib.rs`.  A `crate::X::` reference
/// where `X` is in this list counts as a layer edge.  References to
/// any other identifier (typically locals or sub-modules of the
/// current top module) do not.
const TOP_MODULES: &[&str] = &[
    "audio_buffer",
    "audio_io",
    "api",
    "common",
    "config",
    "converter",
    "daemon",
    "dsp",
    "file_mgr",
    "inference",
    "model",
    "opus_stream",
    "preproc",
    "proto",
    "rknn_runtime",
    "sched",
    "status",
    "stream_io",
    "training",
];

/// Allowed direct dependencies per `docs/ARCH_BOUNDARIES.md` edge
/// map.  An edge `(source, target)` is permitted iff `target` is in
/// `ALLOWED_EDGES[source]`.
///
/// Update this table in lockstep with `docs/ARCH_BOUNDARIES.md` when
/// adding or removing a layer-graph edge.  A new forbidden edge
/// surfaces here as a test failure naming the source file, not as a
/// merge-time review surprise.
const ALLOWED_EDGES: &[(&str, &[&str])] = &[
    ("audio_buffer", &[]),
    ("audio_io", &["audio_buffer", "common", "dsp", "sched"]),
    // `api`'s set is the broadest non-daemon module: it adapts every
    // application domain except producer-only modules (audio_buffer,
    // proto, opus_stream, stream_io).
    (
        "api",
        &[
            "audio_io",
            "common",
            "config",
            "converter",
            "file_mgr",
            "inference",
            "model",
            "status",
            "training",
        ],
    ),
    ("common", &[]),
    // `config` -> `file_mgr` is the canonical durability edge (the
    // config writers delegate to `file_mgr::fs_atomic::put_atomic`);
    // the second-round `lib_2nd` review documented this as an
    // intentional lateral edge to a dependency-light primitive.
    (
        "config",
        &["audio_io", "common", "file_mgr", "inference", "stream_io"],
    ),
    ("converter", &["common", "file_mgr", "model"]),
    // `daemon` is the composition root; every L2/L3/L4 module is
    // permitted.  Listed explicitly so a future module addition
    // surfaces as an allowlist edit rather than an implicit
    // wildcard.
    (
        "daemon",
        &[
            "api",
            "audio_buffer",
            "audio_io",
            "common",
            "config",
            "converter",
            "dsp",
            "file_mgr",
            "inference",
            "model",
            "opus_stream",
            "preproc",
            "proto",
            "rknn_runtime",
            "sched",
            "status",
            "stream_io",
            "training",
        ],
    ),
    // `dsp -> common` is the `Categorized` trait impl on
    // `dsp::resample::StreamingResampleError`; `common::error` is
    // the canonical trait home, so any module surfacing a typed
    // error has this edge.
    ("dsp", &["common"]),
    ("file_mgr", &["common"]),
    (
        "inference",
        &[
            "audio_buffer",
            "common",
            "model",
            "preproc",
            "proto",
            "rknn_runtime",
        ],
    ),
    ("model", &["common"]),
    (
        "opus_stream",
        &["audio_buffer", "audio_io", "common", "dsp", "proto"],
    ),
    // `preproc -> audio_io` consumes the capture-side caps
    // (`MAX_CHANNELS`, `MIN_SAMPLE_RATE`, `MAX_SAMPLE_RATE`) for
    // WAV ingest validation.  Sharing the caps avoids divergent
    // capture-vs-WAV admission policy; the trade-off is recorded
    // in `docs/ARCH_BOUNDARIES.md`.
    ("preproc", &["audio_io", "common", "dsp"]),
    ("proto", &["common"]),
    ("rknn_runtime", &[]),
    ("sched", &[]),
    ("status", &["common"]),
    ("stream_io", &["common", "proto"]),
    ("training", &["common", "file_mgr", "model", "preproc"]),
];

/// Walk `modules/` and return every Rust file's path paired with
/// the top-level module it belongs to.
fn discover_production_files() -> Vec<(String, PathBuf)> {
    let modules_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("modules");
    let mut out = Vec::new();
    for top in TOP_MODULES {
        // Each top-level module has both a `<top>.rs` parent file
        // and (optionally) a `<top>/` directory of children.  Scan
        // both.
        let parent = modules_root.join(format!("{top}.rs"));
        if parent.exists() {
            out.push((top.to_string(), parent));
        }
        let dir = modules_root.join(top);
        if dir.is_dir() {
            walk_dir(&dir, top, &mut out);
        }
    }
    out
}

fn walk_dir(dir: &Path, top: &str, acc: &mut Vec<(String, PathBuf)>) {
    let entries = fs::read_dir(dir).expect("read modules dir");
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk_dir(&path, top, acc);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            // Skip child `tests.rs` files (the in-tree integration
            // test fixtures); they live alongside production code
            // but don't carry layer-graph contracts.
            if path.file_name().and_then(|n| n.to_str()) == Some("tests.rs") {
                continue;
            }
            acc.push((top.to_string(), path));
        }
    }
}

/// Strip the trailing `#[cfg(test)]` block (if any) and per-line
/// comments.  Returns the production-only source text.
fn production_segment(src: &str) -> String {
    let cut = src.find("#[cfg(test)]").unwrap_or(src.len());
    let head = &src[..cut];
    let mut out = String::with_capacity(head.len());
    for line in head.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("//") {
            // Drop the entire line (covers `//`, `///`, `//!`).
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// Extract every `crate::<ident>::` reference's identifier from
/// `src`, returning the unique set.  Identifiers not matching one of
/// `TOP_MODULES` are filtered out (they would be intra-module sub-
/// paths that don't represent a layer edge).
fn extract_edges(src: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let needle = "crate::";
    let mut i = 0;
    while let Some(pos) = src[i..].find(needle) {
        let start = i + pos + needle.len();
        // Read identifier characters.
        let bytes = src.as_bytes();
        let mut end = start;
        while end < bytes.len() {
            let c = bytes[end];
            if c.is_ascii_alphanumeric() || c == b'_' {
                end += 1;
            } else {
                break;
            }
        }
        if end > start {
            let ident = &src[start..end];
            if TOP_MODULES.contains(&ident) {
                out.insert(ident.to_string());
            }
        }
        i = end;
    }
    out
}

/// Production-grade dependency-edge guard.  Any new `crate::<top>::`
/// reference outside the allowlist surfaces here as a clear failure.
///
/// To debug a failure:
///
///   1. Read the failure message; it names the source file + the
///      forbidden edge.
///   2. If the edge is intentional, update both
///      `docs/ARCH_BOUNDARIES.md` AND `ALLOWED_EDGES` above (in
///      lockstep) and re-run.
///   3. If the edge is unintentional, fix the import in the source
///      file.
#[test]
fn no_forbidden_layer_edges() {
    let allowed: BTreeMap<&str, BTreeSet<&str>> = ALLOWED_EDGES
        .iter()
        .map(|(src, targets)| (*src, targets.iter().copied().collect()))
        .collect();

    let files = discover_production_files();
    assert!(
        files.len() >= 30,
        "discover_production_files returned only {} files; expected >= 30 across {} top modules",
        files.len(),
        TOP_MODULES.len(),
    );

    let mut violations: Vec<String> = Vec::new();
    for (top, path) in &files {
        let src = match fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => panic!("read {}: {e}", path.display()),
        };
        let prod = production_segment(&src);
        let edges = extract_edges(&prod);
        let allow_for_src = allowed
            .get(top.as_str())
            .unwrap_or_else(|| panic!("no allowlist entry for top module `{top}`"));
        for tgt in &edges {
            if tgt == top {
                // Self-references (`crate::audio_io::source::...`
                // inside `audio_io`) are not layer edges.
                continue;
            }
            if !allow_for_src.contains(tgt.as_str()) {
                violations.push(format!(
                    "FORBIDDEN EDGE: {} -> {} in {}\n  (allowed targets for `{}`: {:?})",
                    top,
                    tgt,
                    path.strip_prefix(env!("CARGO_MANIFEST_DIR"))
                        .unwrap_or(path)
                        .display(),
                    top,
                    allow_for_src,
                ));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "{} forbidden layer-graph edge(s) found; update either the source \
         file (fix the import) OR `docs/ARCH_BOUNDARIES.md` + the \
         `ALLOWED_EDGES` table in this test (when introducing a deliberate \
         new edge):\n\n{}",
        violations.len(),
        violations.join("\n\n"),
    );
}

/// Sanity check on the test's own infrastructure: every top-level
/// module has an allowlist entry, and every entry references a
/// known top-level module.  Surfaces typos in the table.
#[test]
fn allowlist_table_is_well_formed() {
    let allowed_keys: BTreeSet<&str> = ALLOWED_EDGES.iter().map(|(src, _)| *src).collect();
    let top_set: BTreeSet<&str> = TOP_MODULES.iter().copied().collect();

    let missing: Vec<&str> = top_set.difference(&allowed_keys).copied().collect();
    assert!(
        missing.is_empty(),
        "ALLOWED_EDGES is missing entries for top modules: {missing:?}",
    );

    for (src, targets) in ALLOWED_EDGES {
        for tgt in *targets {
            assert!(
                top_set.contains(tgt),
                "ALLOWED_EDGES[{src}] references unknown top module `{tgt}`",
            );
        }
    }
}
