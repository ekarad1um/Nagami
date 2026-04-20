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
        conflicts_with_all = ["output", "in_place"],
        help = "Exit with status 1 if minification would change the input."
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
        help = "Maximum decimal places for float literals (lossy; omit for full precision)."
    )]
    max_precision: Option<u8>,
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
        max_precision: args.max_precision,
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
        fs::write(&args.input, &output.source)
            .map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", args.input.display())))?;
    } else if let Some(path) = args.output.as_deref() {
        write_output(path, &output.source)?;
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
