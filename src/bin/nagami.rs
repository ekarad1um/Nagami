//! Nagami CLI: arguments -> [`nagami::config::Config`] -> [`nagami::run`],
//! output to stdout, a file, or in place.
//!
//! Exit codes:
//!
//! * `0` - success (or `--check` passed with no proposed changes).
//! * `1` - `--check` detected that minification would modify the input.
//! * `2` - any other failure (I/O, parse, validation, emit), including a
//!   text-only bailout under `--strict-fallback`.

#![cfg(feature = "cli")]

use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, ValueEnum};

/// `json`: one document on stdout with the source as a field, so agent
/// callers parse a single stream; diagnostics stay on stderr either way.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
enum OutputFormat {
    #[default]
    Text,
    Json,
}

/// Mirror of [`nagami::config::Profile`] so the `clap` derives stay out of
/// the library's public surface.
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
    version,
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
        conflicts_with_all = ["output", "in_place", "trace", "trace_dir", "validate_each_pass", "name_map"],
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
        help = "Re-validate emitted WGSL text after every pass and escalate any failure to a hard error (instead of the default silent per-pass rollback)."
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

    #[arg(long, help = "Fail instead of shipping a text-only bailout.")]
    strict_fallback: bool,

    #[arg(
        long,
        value_name = "PATH",
        help = "Write the original -> final identifier map as JSON."
    )]
    name_map: Option<PathBuf>,

    #[arg(
        long,
        value_enum,
        default_value_t,
        help = "Stdout format: raw WGSL, or one JSON document with source, report, and name map."
    )]
    format: OutputFormat,
}

/// Clap rejects both flags together, so only the at-most-one-set case
/// needs mapping; neither set means `Full` for every float kind.
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
    // Forward-substitution builds expression trees as deep as the input's
    // statement count and several IR walks recurse per level: the 8 MiB
    // main stack overflows (SIGABRT under `panic = "abort"`) near ~9k
    // chained reassignments; a lazily-committed 256 MiB worker stack moves
    // that cliff far past any real shader.
    const WORKER_STACK_BYTES: usize = 256 * 1024 * 1024;
    match std::thread::Builder::new()
        .name("nagami".into())
        .stack_size(WORKER_STACK_BYTES)
        .spawn(cli_main)
    {
        Ok(worker) => match worker.join() {
            Ok(code) => code,
            // Unreachable under `panic = "abort"`; a defined exit beats an
            // unwrap if that ever changes.
            Err(_) => ExitCode::from(2),
        },
        // Stack reservation failed: run on the default stack.
        Err(_) => cli_main(),
    }
}

fn cli_main() -> ExitCode {
    match run_cli() {
        Ok(code) => ExitCode::from(code),
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(2)
        }
    }
}

/// Exit code per the module header; hard failures are the `Err` arm.
fn run_cli() -> Result<u8, Box<dyn std::error::Error>> {
    let args = Args::parse();

    if args.in_place && is_dash_path(&args.input) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "--in-place cannot be used with stdin input ('-')",
        )
        .into());
    }

    // Writing the output over the `--preamble` file would destroy the
    // declarations the run strips.
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

    // Preflight --name-map destinations so a refusal changes nothing on
    // disk (a late guard would fire after --in-place rewrote the input);
    // a not-yet-created -o falls back to path equality.
    if let Some(map_path) = args.name_map.as_deref() {
        if is_dash_path(map_path) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "--name-map cannot write to stdout ('-'); use --format json, \
                 whose document embeds the map",
            )
            .into());
        }
        let hits = |other: &Path| same_file(map_path, other) || map_path == other;
        let clobbered = if hits(&args.input) && !is_dash_path(&args.input) {
            Some("the input")
        } else if args.preamble.as_deref().is_some_and(hits) {
            Some("the preamble")
        } else if args
            .output
            .as_deref()
            .filter(|p| !is_dash_path(p))
            .is_some_and(hits)
        {
            Some("the -o output")
        } else {
            None
        };
        if let Some(what) = clobbered {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("--name-map {} would overwrite {what}", map_path.display()),
            )
            .into());
        }
    }

    // stdout carries raw WGSL or the JSON document, never both.
    if args.format == OutputFormat::Json && args.output.as_deref().is_some_and(is_dash_path) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "-o - conflicts with --format json (the JSON document on stdout \
             embeds the source; drop -o - or use -o <file>)",
        )
        .into());
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
        },
        preamble,
    };

    let output = nagami::run(&input, &config)?;
    let changed = output.source != input;

    // Not gated on --quiet, which silences only the success summary.
    if let Some(reason) = &output.report.bailout {
        if args.strict_fallback {
            return Err(format!("--strict-fallback: {reason}").into());
        }
        eprintln!(
            "warning: output is lexically compacted only, the IR pipeline did not apply:\n{reason}"
        );
    }

    if args.check {
        if args.format == OutputFormat::Json {
            println!("{}", nagami::json::render_output(&output));
        } else if args.stats {
            print_summary(&output.report);
        }
        return Ok(u8::from(changed));
    }

    // Before the shader, so an auxiliary-write failure leaves the user's
    // files untouched.
    if let Some(path) = args.name_map.as_deref() {
        fs::write(
            path,
            nagami::json::render_name_map(output.name_map.as_ref()),
        )
        .map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))?;
    }

    if args.in_place {
        write_atomic(&args.input, &output.source)
            .map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", args.input.display())))?;
    } else if let Some(path) = args.output.as_deref() {
        // `-o -` gets no path prefix (`-` is no path); `-o` onto the input
        // itself writes atomically like --in-place.
        let r = if !is_dash_path(path) && same_file(path, &args.input) {
            write_atomic(path, &output.source)
        } else {
            write_output(path, &output.source)
        };
        if is_dash_path(path) {
            r?;
        } else {
            r.map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))?;
        }
    } else if args.format == OutputFormat::Text {
        write_output(Path::new("-"), &output.source)?;
    }

    if args.format == OutputFormat::Json {
        println!("{}", nagami::json::render_output(&output));
    } else {
        let show_summary = args.stats || (!args.quiet && (args.in_place || args.output.is_some()));
        if show_summary {
            print_summary(&output.report);
        }
    }

    Ok(0)
}

fn is_dash_path(path: &Path) -> bool {
    path == Path::new("-")
}

/// Same on-disk file: device + inode on Unix (hard links included),
/// canonical paths elsewhere.  `false` when either path cannot be
/// stat-ed, so a not-yet-created output never matches.
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

fn read_input(path: &Path) -> Result<String, io::Error> {
    if is_dash_path(path) {
        let mut buffer = String::new();
        io::stdin().read_to_string(&mut buffer)?;
        Ok(buffer)
    } else {
        fs::read_to_string(path)
    }
}

/// Flushes stdout so short runs are not truncated at process exit.
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

/// Sibling temp file, then `rename` over `path`: atomic on POSIX and
/// `MoveFileEx(REPLACE_EXISTING)` on Windows, so the destination always
/// holds the old or the new contents.  No non-atomic fallback: it would
/// overwrite the input on exactly the failures (ENOSPC, EACCES, ENOTSUP)
/// the atomic write defends against.  Temp names are salted with pid +
/// nanos and opened `create_new`, so concurrent invocations cannot
/// clobber each other.
fn write_atomic(path: &Path, content: &str) -> Result<(), io::Error> {
    let dir = path.parent().filter(|p| !p.as_os_str().is_empty());
    let file_name = path
        .file_name()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "path has no file name"))?;
    let pid = std::process::id();
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

    // The handle is dropped before rename (Windows cannot unlink an open
    // source); the closure routes every error path through one temp-file
    // cleanup.
    let result: io::Result<()> = (|| {
        {
            let mut file = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&tmp_path)?;
            file.write_all(content.as_bytes())?;
            // Page-cache writes succeed on a full device; ENOSPC surfaces at
            // `sync_all`, and swallowing it would rename a file whose
            // contents never reached the disk.
            file.sync_all()?;
        }
        fs::rename(&tmp_path, path)
    })();
    if result.is_err() {
        // Ignored when the temp was never created or already renamed.
        let _ = fs::remove_file(&tmp_path);
    }
    result
}

/// Byte-delta summary on stderr; growth is reported too so a parser
/// survives a regression.
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

    /// Clap's `debug_assert` catches arg-name drift in `conflicts_with` /
    /// `requires` lists, which otherwise panics only at user invocation.
    #[test]
    fn args_command_definition_is_internally_consistent() {
        Args::command().debug_assert();
    }

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
