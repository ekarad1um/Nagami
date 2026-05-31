//! Nagami CLI binary.  Thin wrapper that translates command-line
//! arguments into a [`nagami::config::Config`], drives [`nagami::run`],
//! and reports results through stdout, stderr, or an in-place rewrite.
//!
//! Exit codes:
//!
//! * `0` - success (or `--check` passed with no proposed changes).
//! * `1` - `--check` detected that minification would modify the input.
//! * `2` - any other failure (I/O, parse, validation, emit).

#![cfg(feature = "cli")]

use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, ValueEnum};

/// CLI-facing mirror of [`nagami::config::Profile`].  Kept separate so
/// the `clap` derives do not leak into the library's public surface.
#[derive(Debug, Clone, Copy, ValueEnum)]
enum CLIProfile {
    Baseline,
    Aggressive,
    Max,
}

impl From<CLIProfile> for nagami::config::Profile {
    fn from(value: CLIProfile) -> Self {
        match value {
            CLIProfile::Baseline => nagami::config::Profile::Baseline,
            CLIProfile::Aggressive => nagami::config::Profile::Aggressive,
            CLIProfile::Max => nagami::config::Profile::Max,
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "nagami",
    about = "Shrinks WGSL shaders via Naga IR optimization passes",
    long_about = "Nagami[n] - Naga + Minify. Shrinks WGSL shaders by lowering to Naga IR, running optimization passes, and emitting minimal valid WGSL."
)]
struct Args {
    #[arg(value_name = "INPUT", help = "Input path. Use '-' to read from stdin.")]
    input: PathBuf,

    #[arg(
        short = 'o',
        value_name = "OUTPUT",
        conflicts_with = "in_place",
        help = "Output path. Use '-' to write to stdout."
    )]
    output: Option<PathBuf>,

    #[arg(long, conflicts_with = "output", help = "Overwrite INPUT in place.")]
    in_place: bool,

    #[arg(short = 'p', long, value_enum, default_value_t = CLIProfile::Max)]
    profile: CLIProfile,

    #[arg(
        long = "preserve-symbol",
        value_name = "NAME",
        help = "Keep this symbol name unchanged (globals, functions, constants, overrides, struct types, struct members). Can be used multiple times."
    )]
    preserve_symbols: Vec<String>,

    #[arg(
        long,
        conflicts_with_all = ["output", "in_place", "trace", "trace_dir", "validate_each_pass"],
        help = "Exit with status 1 if minification would change the input. \
                Read-only - no output, trace, or validation side effects."
    )]
    check: bool,

    #[arg(long, help = "Print minification stats to stderr.")]
    stats: bool,

    #[arg(
        short = 'q',
        long,
        conflicts_with = "stats",
        help = "Suppress non-error CLI output (including stats)."
    )]
    quiet: bool,

    #[arg(long, help = "Enable per-pass trace output.")]
    trace: bool,

    #[arg(
        long,
        value_name = "DIR",
        requires = "trace",
        help = "Directory for trace dumps (default: trace)."
    )]
    trace_dir: Option<PathBuf>,

    #[arg(
        long,
        requires = "trace",
        help = "Re-validate emitted WGSL text after every pass and roll back on failure."
    )]
    validate_each_pass: bool,

    #[arg(
        long,
        value_name = "FILE",
        help = "WGSL file with external declarations to prepend (excluded from output)."
    )]
    preamble: Option<PathBuf>,

    #[arg(
        long,
        value_name = "N",
        help = "Max expression node count for inlining a function (default: 48, baseline/aggressive: 24)."
    )]
    max_inline_node_count: Option<usize>,

    #[arg(
        long,
        value_name = "N",
        help = "Max call sites for inlining a function (default: 6, baseline/aggressive: 3)."
    )]
    max_inline_call_sites: Option<usize>,

    #[arg(
        long,
        overrides_with = "no_mangle",
        help = "Mangle struct types, struct members, and constant names (on by default)."
    )]
    mangle: bool,

    #[arg(
        long = "no-mangle",
        overrides_with = "mangle",
        help = "Disable mangling even when profile implies it."
    )]
    no_mangle: bool,

    #[arg(long, help = "Beautify the output with indentation and newlines.")]
    beautify: bool,

    #[arg(
        long,
        value_name = "N",
        default_value_t = 2,
        help = "Number of spaces per indentation level (default: 2)."
    )]
    indent: u8,

    #[arg(
        long,
        value_name = "N",
        conflicts_with = "sig_figs",
        help = "Round every float literal to at most N decimal places (lossy)."
    )]
    decimal_places: Option<u8>,

    #[arg(
        long,
        value_name = "N",
        help = "Round every float literal to at most N significant figures (lossy)."
    )]
    sig_figs: Option<u8>,
}

/// Translate the two CLI precision flags into a [`FloatPrecision`].
/// Clap rejects `--decimal-places` and `--sig-figs` together at parse
/// time (`conflicts_with = "sig_figs"` on `decimal_places`), so this
/// helper only needs to map the at-most-one-set case.  When neither
/// flag is given, all float kinds default to
/// [`nagami::config::PrecisionMode::Full`].
fn precision_from_cli(
    decimal_places: Option<u8>,
    sig_figs: Option<u8>,
) -> nagami::config::FloatPrecision {
    use nagami::config::{FloatPrecision, PrecisionMode};
    if let Some(p) = decimal_places {
        FloatPrecision::all(PrecisionMode::DecimalPlaces(p))
    } else if let Some(s) = sig_figs {
        FloatPrecision::all(PrecisionMode::SignificantFigures(s))
    } else {
        FloatPrecision::default()
    }
}

fn main() -> ExitCode {
    match run_cli() {
        Ok(should_fail_check) => {
            if should_fail_check {
                ExitCode::from(1)
            } else {
                ExitCode::SUCCESS
            }
        }
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(2)
        }
    }
}

/// Parse arguments, load input, run the pipeline, and emit output.
///
/// Returns `Ok(true)` when `--check` was requested and the pipeline
/// would modify the input (the `main` wrapper translates this into
/// exit code 1).  Every other success path returns `Ok(false)`.
fn run_cli() -> Result<bool, Box<dyn std::error::Error>> {
    let args = Args::parse();

    if args.in_place && is_dash_path(&args.input) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "--in-place cannot be used with stdin input ('-')",
        )
        .into());
    }

    // Refuse to write the minified output over the `--preamble` file: the
    // run strips the preamble's declarations from the output, so writing
    // that result back would silently and permanently destroy them.  The
    // destructive destination is reachable via `--in-place` (writes the
    // input) and `-o <path>` (writes that path); `-o -` (stdout) has no
    // file.  `same_file` matches by inode where possible, so symlinks and
    // hard links to the preamble are caught too, and a not-yet-created
    // output (or a missing preamble, which errors at the read below) never
    // false-matches.
    if let Some(preamble_path) = args.preamble.as_ref() {
        let dest: Option<&Path> = if args.in_place {
            Some(args.input.as_path())
        } else {
            args.output.as_deref().filter(|p| !is_dash_path(p))
        };
        if let Some(dest) = dest
            && same_file(dest, preamble_path)
        {
            let flag = if args.in_place {
                "--in-place"
            } else {
                "-o <output>"
            };
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "{flag} cannot write to the same file as --preamble \
                     (the rewrite would delete the preamble's declarations)"
                ),
            )
            .into());
        }
    }

    let input = read_input(&args.input)
        .map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", args.input.display())))?;

    let preamble =
        if let Some(preamble_path) = &args.preamble {
            Some(fs::read_to_string(preamble_path).map_err(|e| {
                io::Error::new(e.kind(), format!("{}: {e}", preamble_path.display()))
            })?)
        } else {
            None
        };

    let config = nagami::config::Config {
        profile: args.profile.into(),
        preserve_symbols: args.preserve_symbols,
        mangle: match (args.mangle, args.no_mangle) {
            (true, _) => Some(true),
            (_, true) => Some(false),
            _ => None,
        },
        beautify: args.beautify,
        indent: args.indent,
        float_precision: precision_from_cli(args.decimal_places, args.sig_figs),
        max_inline_node_count: args.max_inline_node_count,
        max_inline_call_sites: args.max_inline_call_sites,
        trace: nagami::config::TraceConfig {
            enabled: args.trace,
            dump_dir: args.trace_dir,
            validate_each_pass: args.validate_each_pass,
            ..Default::default()
        },
        preamble,
    };

    let output = nagami::run(&input, &config)?;
    let changed = output.source != input;

    if args.check {
        if args.stats {
            print_summary(&output.report);
        }
        return Ok(changed);
    }

    if args.in_place {
        // Write through a sibling temp file plus an atomic rename so a
        // crash mid-write cannot corrupt the user's input.  A direct
        // `fs::write` would leave the input truncated on power loss or
        // an interrupted process.
        write_atomic(&args.input, &output.source)
            .map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", args.input.display())))?;
    } else if let Some(path) = args.output.as_deref() {
        // Prefix the offending path on failure, matching read_input / the
        // preamble read / the in-place write above - EXCEPT for explicit
        // `-o -` (stdout), where `-` is not a meaningful path to report, so
        // the error stays bare (consistent with the no-`-o` stdout branch).
        let r = write_output(path, &output.source);
        if is_dash_path(path) {
            r?;
        } else {
            r.map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))?;
        }
    } else {
        write_output(Path::new("-"), &output.source)?;
    }

    let show_summary = args.stats || (!args.quiet && (args.in_place || args.output.is_some()));
    if show_summary {
        print_summary(&output.report);
    }

    Ok(false)
}

/// `true` when `path` is the single dash convention for stdin/stdout.
fn is_dash_path(path: &Path) -> bool {
    path == Path::new("-")
}

/// `true` when both paths resolve to the same on-disk file.  On Unix this
/// compares device + inode, which also equates hard links (distinct names
/// sharing one inode that path canonicalization cannot match); elsewhere it
/// falls back to canonical-path comparison.  Returns `false` when either
/// path is missing or cannot be stat-ed, so a not-yet-created output is
/// never mistaken for an existing file.
fn same_file(a: &Path, b: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        match (fs::metadata(a), fs::metadata(b)) {
            (Ok(ma), Ok(mb)) => ma.dev() == mb.dev() && ma.ino() == mb.ino(),
            _ => false,
        }
    }
    #[cfg(not(unix))]
    {
        match (fs::canonicalize(a), fs::canonicalize(b)) {
            (Ok(ca), Ok(cb)) => ca == cb,
            _ => false,
        }
    }
}

/// Read input from `path` or from stdin when `path` is the `-` sentinel.
fn read_input(path: &Path) -> Result<String, io::Error> {
    if is_dash_path(path) {
        let mut buffer = String::new();
        io::stdin().read_to_string(&mut buffer)?;
        Ok(buffer)
    } else {
        fs::read_to_string(path)
    }
}

/// Write `content` to `path`, redirecting to stdout when `path` is the
/// `-` sentinel.  Flushes stdout on completion so short runs are not
/// truncated by a buffered writer left open at process exit.
fn write_output(path: &Path, content: &str) -> Result<(), io::Error> {
    if is_dash_path(path) {
        let mut stdout = io::stdout().lock();
        stdout.write_all(content.as_bytes())?;
        stdout.flush()?;
        Ok(())
    } else {
        fs::write(path, content)
    }
}

/// Atomic file replace: write `content` to a sibling temp path then
/// `rename` over `path`.  The rename is atomic on POSIX (and acts as
/// `MoveFileEx(MOVEFILE_REPLACE_EXISTING)` on Windows via
/// `std::fs::rename`), so the destination contains either the old or
/// the new contents at every observable moment - never a truncated or
/// half-written intermediate.
///
/// Errors during temp-file creation, write, or rename are surfaced
/// to the caller verbatim - there is no non-atomic direct-write
/// fallback.  A fallback would non-atomically overwrite the user's
/// input on exactly the failures (ENOSPC mid-write, EACCES on the
/// parent directory, ENOTSUP on the FS) the atomic write defends
/// against, defeating the invariant.  Surfacing the error up front
/// keeps the user's existing file intact.
///
/// Temp filenames are salted with pid + nanos and opened with
/// `create_new`, so two concurrent invocations cannot share or
/// silently clobber each other's staged content even if their
/// timestamps collide.
fn write_atomic(path: &Path, content: &str) -> Result<(), io::Error> {
    let dir = path.parent().filter(|p| !p.as_os_str().is_empty());
    let file_name = path
        .file_name()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "path has no file name"))?;
    let pid = std::process::id();
    // Salt with pid + nanos to keep concurrent invocations from
    // colliding on the same temp name without pulling in a tempfile
    // dependency.  `create_new` below guarantees we never overwrite
    // an existing temp; on collision we surface `AlreadyExists` so
    // the caller can retry or report.
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let mut tmp_name = file_name.to_os_string();
    tmp_name.push(format!(".nagami-tmp-{pid}-{stamp}"));
    let tmp_path: PathBuf = match dir {
        Some(d) => d.join(&tmp_name),
        None => PathBuf::from(&tmp_name),
    };

    // Open with create_new so the open fails with `AlreadyExists` on
    // collision rather than silently truncating another process's
    // staged file.  Drop the file handle before rename so Windows
    // can unlink the source.  The stage-and-rename block runs inside
    // a closure so every error path (open / write_all / sync_all /
    // rename) routes through the same best-effort cleanup of the
    // staged temp file; otherwise a mid-write failure would leave a
    // `.nagami-tmp-*` file alongside the input.
    let result: io::Result<()> = (|| {
        {
            let mut file = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&tmp_path)?;
            file.write_all(content.as_bytes())?;
            // Propagate `sync_all` errors so ENOSPC-at-flush is
            // surfaced to the caller.  Page-cache writes succeed even
            // when the underlying device is out of space; ENOSPC then
            // shows up on `sync_all`, and if we swallow it the
            // subsequent `rename` succeeds on a file whose contents
            // never reached the disk - a power loss immediately after
            // would leave the user with a renamed-but-torn file and
            // no error signal.  On filesystems where `sync_all` is a
            // no-op (most modern filesystems return success), this
            // costs nothing; on filesystems where it spuriously
            // fails, the user gets a clear durability error rather
            // than silent data loss.
            file.sync_all()?;
        }
        fs::rename(&tmp_path, path)
    })();
    if result.is_err() {
        // Best-effort cleanup; ignored if the temp was never
        // created (open failed) or already moved (rename succeeded).
        let _ = fs::remove_file(&tmp_path);
    }
    result
}

/// Print a human-readable byte-delta summary to stderr.  Handles the
/// degenerate `input == 0` case without dividing by zero and reports
/// both shrink ("saved") and growth ("grew by") deltas so an upstream
/// tool can still parse the line after a regression.
fn print_summary(report: &nagami::pipeline::Report) {
    let input = report.input_bytes;
    let output = report.output_bytes;
    let passes = report.pass_reports.len();

    if output <= input {
        let saved = input - output;
        let pct = if input == 0 {
            0.0
        } else {
            (saved as f64 / input as f64) * 100.0
        };
        eprintln!(
            "Minified: {} -> {} bytes (saved {} bytes, {:.2}% smaller, {} passes)",
            input, output, saved, pct, passes
        );
    } else {
        let growth = output - input;
        let pct = if input == 0 {
            0.0
        } else {
            (growth as f64 / input as f64) * 100.0
        };
        eprintln!(
            "Minified: {} -> {} bytes (grew by {} bytes, +{:.2}%, {} passes)",
            input, output, growth, pct, passes
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    /// Run clap's internal `debug_assert` on the derived `Args`
    /// command so arg-name drift in `conflicts_with` / `requires`
    /// lists is caught at test time instead of at user-invocation
    /// time.  Without this guard, renaming a field (e.g. `trace_dir`
    /// -> `trace_dump_dir`) without updating every conflict list that
    /// mentions it would compile fine and only panic when a user ran
    /// `nagami --check --trace-dir /x`.
    #[test]
    fn args_command_definition_is_internally_consistent() {
        Args::command().debug_assert();
    }

    /// Pin the flag -> [`FloatPrecision`] translation.  Clap rejects the
    /// both-set case at parse time, so only the at-most-one-set arms are
    /// reachable; this covers each of them plus the neither-set default.
    #[test]
    fn precision_from_cli_maps_flags_to_modes() {
        use nagami::config::{FloatPrecision, PrecisionMode};
        assert_eq!(
            precision_from_cli(Some(3), None),
            FloatPrecision::all(PrecisionMode::DecimalPlaces(3))
        );
        assert_eq!(
            precision_from_cli(None, Some(2)),
            FloatPrecision::all(PrecisionMode::SignificantFigures(2))
        );
        assert_eq!(precision_from_cli(None, None), FloatPrecision::default());
    }
}
