//! Pipeline driver: the [`Pass`] trait, the convergence loop, and the
//! per-pass bookkeeping (validation, rollback, trace dumps, [`PassReport`]).

use std::path::PathBuf;
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::config::Config;
use crate::error::Error;
use crate::io;

mod context;
mod report;

pub use context::PassContext;
pub use report::{PassReport, Report};

/// Hard cap on sweeps; real pipelines converge well under it.
const MAX_PIPELINE_SWEEPS: usize = 16;

/// An IR optimization pass that mutates a naga module in place.
pub trait Pass {
    /// Short, unique identifier for reports and trace filenames.
    fn name(&self) -> &'static str;
    /// Run against `module`; `Ok(true)` iff it was modified.  The driver
    /// validates only after a declared change, skips a pass whose input is
    /// unchanged since it last contributed none, and re-runs accepted
    /// passes to rebuild the pre-failure state after a rollback: a pass
    /// must be a deterministic function of the module and must not touch
    /// it (or the name log) while returning `Ok(false)`.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] on an unrecoverable failure.
    fn run(&mut self, module: &mut naga::Module, ctx: &PassContext<'_>) -> Result<bool, Error>;
}

/// naga's WGSL text for `module` under an already computed `info`, so the
/// trace / `validate_each_pass` emissions never pay the validator twice
/// per pass.  A stale `info` panics inside the backend through
/// out-of-bounds arena indexing; debug builds re-validate as a drift guard.
fn emit_wgsl_with_info(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
) -> Result<String, Error> {
    // The backend hits `unreachable!()` (an abort under `panic = "abort"`) on
    // override-sized arrays and non-const global initializers; the
    // comment-only placeholder re-parses as an empty module, which is why the
    // text round-trip check skips these modules.
    if crate::module_needs_naga_baseline_skip(module) {
        return Ok("// (naga WGSL backend skipped: unsupported emit)\n".to_string());
    }
    // Catches a module left invalid, not an `info` whose arena view is stale.
    debug_assert!(
        io::validate_module(module).is_ok(),
        "emit_wgsl_with_info called with a stale ModuleInfo: module no longer validates"
    );
    naga::back::wgsl::write_string(module, info, naga::back::wgsl::WriterFlags::empty())
        .map_err(|e| Error::Emit(e.to_string()))
}

// MARK: Driver

/// Run the IR pipeline to a fixed point: [`crate::passes::build_ir_passes`]
/// in order, repeated until no pass reports a change, capped at
/// `MAX_PIPELINE_SWEEPS`.  Every declared change is validated; a failure
/// rolls back, or escalates under
/// [`crate::config::TraceConfig::validate_each_pass`].
///
/// # Errors
///
/// Returns [`Error`] on an unrecoverable pass failure, or on a validation
/// failure when rollback is disabled.
pub fn run_ir_passes(
    module: &mut naga::Module,
    config: &Config,
    report: &mut Report,
) -> Result<crate::name_map::NameLog, Error> {
    let passes = crate::passes::build_ir_passes(config);
    run_ir_passes_with(module, config, report, passes)
}

/// Driver parameterised on the pass list so tests can inject synthetic passes.
fn run_ir_passes_with(
    module: &mut naga::Module,
    config: &Config,
    report: &mut Report,
    mut passes: Vec<Box<dyn Pass>>,
) -> Result<crate::name_map::NameLog, Error> {
    let trace_run_dir = prepare_trace_dir(config)?;
    let mut sweeps = 0usize;
    // Module-scope renames across sweeps, for the name map.
    let name_log = std::cell::RefCell::new(crate::name_map::NameLog::default());
    let trace_enabled = config.trace.enabled;
    let needs_text_validation = config.trace.validate_each_pass;
    // Trace / CI runs report every pass; the plain path skips idle ones.
    let full_fidelity = trace_enabled || needs_text_validation;

    // Validator info reused by the trace / CI text emissions; plain
    // minification emits no intermediate text and leaves it unseeded.
    let mut current_info: Option<naga::valid::ModuleInfo> = if full_fidelity {
        Some(io::validate_module(module)?)
    } else {
        None
    };

    // `version` counts accepted changes; `clean_at[i]` is the version at
    // which pass `i` last contributed none (no change, or rejected).  Passes
    // are deterministic in the module, so a pass clean at the current
    // version is skipped: it would contribute none again.  Output and
    // convergence are unaffected; only the report omits the idle runs.
    let mut version = 0u64;
    let mut clean_at: Vec<Option<u64>> = vec![None; passes.len()];

    loop {
        let mut any_changed = false;

        // Rollback state: module + name log (a stale log would report renames
        // the shipped names lack) as of the sweep start, and the passes whose
        // accepted changes it lacks.  A validation failure restores it and
        // replays exactly those passes: determinism reproduces the pre-pass
        // state, and a rejected pass never joins the list, so a second
        // failure cannot resurrect the first one's output.  Taken before the
        // first pass that runs (an all-idle sweep clones nothing); never
        // under `validate_each_pass`, where every failure is an `Err`.
        let mut backup: Option<(naga::Module, crate::name_map::NameLog)> = None;
        let mut accepted: Vec<usize> = Vec::new();

        for i in 0..passes.len() {
            if !full_fidelity && clean_at[i] == Some(version) {
                continue;
            }
            if backup.is_none() && !needs_text_validation {
                backup = Some((module.clone(), name_log.borrow().clone()));
            }

            let before_text = if trace_enabled {
                let info = current_info
                    .as_ref()
                    .expect("current_info is seeded whenever trace/validate_each_pass is on");
                Some(emit_wgsl_with_info(module, info)?)
            } else {
                None
            };
            let before_bytes = before_text.as_ref().map(|text| text.len());

            #[cfg(not(target_arch = "wasm32"))]
            let start = Instant::now();
            let ctx = PassContext {
                config,
                trace_run_dir: trace_run_dir.as_deref(),
                name_log: Some(&name_log),
            };

            let declared_changed = passes[i].run(module, &ctx)?;
            #[cfg(not(target_arch = "wasm32"))]
            let duration_us = start.elapsed().as_micros() as u64;
            #[cfg(target_arch = "wasm32")]
            let duration_us = 0u64;
            let mut validation_ok = true;
            let mut rolled_back = false;
            let mut text_validation_ok = None;

            // No declared change -> the module is still the state the last
            // validation blessed, so only CI mode re-validates; an
            // under-reporting pass trips the traced debug_assert below.
            if declared_changed || needs_text_validation {
                match io::validate_module(module) {
                    Ok(info) => {
                        current_info = Some(info);
                        if declared_changed {
                            version += 1;
                            accepted.push(i);
                        }
                    }
                    Err(e) => {
                        if needs_text_validation {
                            // CI mode: a regressing pass is an error, not a warning.
                            return Err(Error::Validation(format!(
                                "pass '{}' produced invalid IR: {}",
                                passes[i].name(),
                                e
                            )));
                        }
                        eprintln!(
                            "warning: validation failed after pass '{}', rolling back: {}",
                            passes[i].name(),
                            e
                        );
                        // Pre-pass state again, which `current_info` (if
                        // seeded) already describes.
                        let (saved_module, saved_log) = backup
                            .as_ref()
                            .expect("backup is taken whenever validate_each_pass is off");
                        *module = saved_module.clone();
                        *name_log.borrow_mut() = saved_log.clone();
                        for &earlier in &accepted {
                            passes[earlier].run(module, &ctx)?;
                        }
                        validation_ok = false;
                        rolled_back = true;
                    }
                }
            }
            if !declared_changed || rolled_back {
                clean_at[i] = Some(version);
            }

            let after_text = if rolled_back {
                // Pre-pass state again: the before text is the after text.
                if trace_enabled {
                    before_text.clone()
                } else {
                    None
                }
            } else if full_fidelity {
                let info = current_info
                    .as_ref()
                    .expect("current_info is refreshed after every successful pass");
                Some(emit_wgsl_with_info(module, info)?)
            } else {
                None
            };
            let after_bytes = after_text.as_ref().map(|text| text.len());

            // A baseline-skipped module's placeholder text re-parses as an
            // empty module: no verdict rather than a spurious clean round-trip.
            if !rolled_back
                && needs_text_validation
                && !crate::module_needs_naga_baseline_skip(module)
            {
                let ok = io::validate_wgsl_text(
                    after_text
                        .as_deref()
                        .expect("after text must be available for text validation"),
                )
                .is_ok();
                text_validation_ok = Some(ok);
                if !ok {
                    return Err(Error::Validation(format!(
                        "pass '{}' produced IR that round-trips to invalid WGSL text",
                        passes[i].name()
                    )));
                }
            }

            let changed_by_text = match (before_text.as_ref(), after_text.as_ref()) {
                (Some(before), Some(after)) => before != after,
                _ => false,
            };
            // Convergence follows the declarations alone: `changed_by_text`
            // exists only under `--trace`, and letting it drive the loop
            // would make convergence depth depend on a debug flag.  It only
            // enriches the report and, in debug builds, exposes an
            // under-reporting pass.
            any_changed |= !rolled_back && declared_changed;
            #[cfg(debug_assertions)]
            if trace_enabled && !rolled_back && !declared_changed {
                debug_assert!(
                    !changed_by_text,
                    "pass '{}' under-reports: emitted text changed but it returned Ok(false)",
                    passes[i].name()
                );
            }
            let changed = !rolled_back && (declared_changed || changed_by_text);

            let pass_report = PassReport {
                pass_name: passes[i].name().to_string(),
                before_bytes,
                after_bytes,
                changed,
                duration_us,
                validation_ok,
                text_validation_ok,
                rolled_back,
            };

            if trace_enabled {
                dump_trace_step(
                    &ctx,
                    report.pass_reports.len(),
                    before_text
                        .as_deref()
                        .expect("before text must be available when tracing"),
                    after_text
                        .as_deref()
                        .expect("after text must be available when tracing"),
                    &pass_report,
                )?;
            }

            report.pass_reports.push(pass_report);
        }

        sweeps += 1;
        if !any_changed || sweeps >= MAX_PIPELINE_SWEEPS {
            if sweeps >= MAX_PIPELINE_SWEEPS && any_changed {
                report.converged = false;
                // The report is observable on the `Err` path too, so the
                // sweep count lands before the CI-mode escalation (a
                // warning otherwise).
                report.sweeps = sweeps;
                if config.trace.validate_each_pass {
                    return Err(Error::Validation(format!(
                        "pipeline did not converge after {MAX_PIPELINE_SWEEPS} sweeps; \
                         a pass is producing oscillating IR"
                    )));
                }
                eprintln!("warning: pipeline did not converge after {MAX_PIPELINE_SWEEPS} sweeps");
            }
            break;
        }
    }

    report.sweeps = sweeps;
    Ok(name_log.into_inner())
}

// MARK: Trace directory allocation

/// Prepare the trace output directory for the current run.  Returns
/// `None` when tracing is disabled or when running on wasm (which has
/// no filesystem); otherwise creates `trace/<base>/run-{stamp}` with
/// collision-safe suffix handling.
fn prepare_trace_dir(config: &Config) -> Result<Option<PathBuf>, Error> {
    if !config.trace.enabled {
        return Ok(None);
    }

    #[cfg(target_arch = "wasm32")]
    {
        let _ = config;
        Ok(None)
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let base = config
            .trace
            .dump_dir
            .clone()
            .unwrap_or_else(|| PathBuf::from("trace"));
        std::fs::create_dir_all(&base)?;
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| Error::Io(e.to_string()))?
            .as_millis();
        Ok(Some(allocate_trace_run_dir(&base, stamp)?))
    }
}

/// Atomically claim a fresh `run-{stamp}[-suffix]` directory under `base`.
///
/// Repeated calls within the same millisecond, or concurrent processes
/// that happen to agree on a stamp, would otherwise collide on the
/// plain `run-{stamp}` name.  Using `std::fs::create_dir` (not
/// `create_dir_all`) as the claim primitive means the OS returns
/// `AlreadyExists` on collision, at which point the suffix increments
/// and the caller retries.  First available wins with no check-then-act
/// race window.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn allocate_trace_run_dir(
    base: &std::path::Path,
    stamp: u128,
) -> Result<PathBuf, Error> {
    const MAX_SUFFIX_TRIES: u32 = 10_000;
    for suffix in 0..MAX_SUFFIX_TRIES {
        let candidate = if suffix == 0 {
            base.join(format!("run-{stamp}"))
        } else {
            base.join(format!("run-{stamp}-{suffix}"))
        };
        match std::fs::create_dir(&candidate) {
            Ok(()) => return Ok(candidate),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e.into()),
        }
    }
    Err(Error::Io(format!(
        "failed to allocate trace run directory under {base:?} after {MAX_SUFFIX_TRIES} tries",
    )))
}

/// Write one `step-NNN-<pass>/` directory (`before.wgsl`, `after.wgsl`,
/// `meta.txt`) under the run's trace folder; a no-op on wasm or with
/// tracing off.
fn dump_trace_step(
    ctx: &PassContext<'_>,
    step_index: usize,
    before_text: &str,
    after_text: &str,
    report: &PassReport,
) -> Result<(), Error> {
    if !ctx.config.trace.enabled {
        return Ok(());
    }

    #[cfg(target_arch = "wasm32")]
    {
        let _ = (ctx, step_index, before_text, after_text, report);
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let Some(run_dir) = ctx.trace_run_dir else {
            return Ok(());
        };
        let step_dir = run_dir.join(format!("step-{step_index:03}-{}", report.pass_name));
        std::fs::create_dir_all(&step_dir)?;
        std::fs::write(step_dir.join("before.wgsl"), before_text)?;
        std::fs::write(step_dir.join("after.wgsl"), after_text)?;
        std::fs::write(
            step_dir.join("meta.txt"),
            format!(
                "pass_name={}\nbefore_bytes={}\nafter_bytes={}\nchanged={}\nduration_us={}\nvalidation_ok={}\ntext_validation_ok={}\nrolled_back={}\n",
                report.pass_name,
                report
                    .before_bytes
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "n/a".to_string()),
                report
                    .after_bytes
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "n/a".to_string()),
                report.changed,
                report.duration_us,
                report.validation_ok,
                report
                    .text_validation_ok
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "n/a".to_string()),
                report.rolled_back,
            ),
        )?;
        Ok(())
    }
}

// MARK: Trace directory tests

#[cfg(all(test, not(target_arch = "wasm32")))]
mod trace_dir_tests {
    use super::allocate_trace_run_dir;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Unique-per-test directory under the OS temp dir, cleaned up on
    /// `Drop`.  Hand-rolled to avoid pulling `tempfile` in purely for
    /// this one test group.
    struct TestTempDir(PathBuf);

    impl TestTempDir {
        fn new() -> Self {
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
            let pid = std::process::id();
            let path = std::env::temp_dir().join(format!("nagami-trace-test-{pid}-{counter}"));
            std::fs::create_dir_all(&path).expect("create test tempdir");
            Self(path)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestTempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn allocates_plain_run_dir_when_no_collision() {
        let tmp = TestTempDir::new();
        let path = allocate_trace_run_dir(tmp.path(), 12345).expect("allocate");
        assert_eq!(path, tmp.path().join("run-12345"));
        assert!(path.exists());
    }

    #[test]
    fn adds_suffix_when_base_name_collides() {
        let tmp = TestTempDir::new();
        std::fs::create_dir(tmp.path().join("run-42")).expect("pre-create");
        let path = allocate_trace_run_dir(tmp.path(), 42).expect("allocate");
        assert_eq!(path, tmp.path().join("run-42-1"));
        assert!(path.exists());
    }

    #[test]
    fn escalates_suffix_until_available() {
        let tmp = TestTempDir::new();
        std::fs::create_dir(tmp.path().join("run-7")).expect("pre-create");
        std::fs::create_dir(tmp.path().join("run-7-1")).expect("pre-create");
        std::fs::create_dir(tmp.path().join("run-7-2")).expect("pre-create");
        let path = allocate_trace_run_dir(tmp.path(), 7).expect("allocate");
        assert_eq!(path, tmp.path().join("run-7-3"));
    }

    #[test]
    fn concurrent_claims_do_not_share_a_directory() {
        // Simulate two near-simultaneous callers agreeing on a stamp
        // by invoking the allocator twice back-to-back.  They must
        // land in distinct directories.  This is the exact failure
        // the suffix retry targets: with the original `create_dir_all`
        // primitive both callers would share `run-{stamp}` because
        // `create_dir_all` is idempotent.
        let tmp = TestTempDir::new();
        let first = allocate_trace_run_dir(tmp.path(), 99).expect("first");
        let second = allocate_trace_run_dir(tmp.path(), 99).expect("second");
        assert_ne!(first, second);
        assert!(first.exists());
        assert!(second.exists());
    }
}

// MARK: Driver tests

#[cfg(test)]
mod driver_tests {
    //! Rollback, hard-fail escalation, and idle-skip coverage on synthetic
    //! passes; each is a deterministic function of the module, as
    //! `Pass::run` demands.
    use std::cell::Cell;
    use std::rc::Rc;

    use super::{Pass, PassContext, run_ir_passes_with};
    use crate::config::Config;
    use crate::error::Error;
    use crate::io;
    use crate::pipeline::report::Report;

    fn first_function(module: &mut naga::Module) -> &mut naga::Function {
        match module.entry_points.first_mut() {
            Some(ep) => &mut ep.function,
            None => {
                let (_, f) = module
                    .functions
                    .iter_mut()
                    .next()
                    .expect("test module must have a function or entry point");
                f
            }
        }
    }

    /// Corrupts on EVERY run: the driver's bookkeeping, not pass-side
    /// memory, must keep a rejected pass out of the replayed state.
    struct CorruptingPass;

    impl Pass for CorruptingPass {
        fn name(&self) -> &'static str {
            "synthetic_corrupt"
        }
        fn run(
            &mut self,
            module: &mut naga::Module,
            _ctx: &PassContext<'_>,
        ) -> Result<bool, Error> {
            let function = first_function(module);
            let lit = function.expressions.append(
                naga::Expression::Literal(naga::Literal::I32(0)),
                naga::Span::UNDEFINED,
            );
            // Store through a scalar literal: structurally invalid.
            function.body.push(
                naga::Statement::Store {
                    pointer: lit,
                    value: lit,
                },
                naga::Span::UNDEFINED,
            );
            Ok(true)
        }
    }

    /// Adds `var t: u32;` once: a valid, idempotent change the rollback
    /// tests can watch survive.
    struct AddLocalPass;

    impl Pass for AddLocalPass {
        fn name(&self) -> &'static str {
            "synthetic_add_local"
        }
        fn run(
            &mut self,
            module: &mut naga::Module,
            _ctx: &PassContext<'_>,
        ) -> Result<bool, Error> {
            let ty = module.types.insert(
                naga::Type {
                    name: None,
                    inner: naga::TypeInner::Scalar(naga::Scalar::U32),
                },
                naga::Span::UNDEFINED,
            );
            let function = first_function(module);
            if !function.local_variables.is_empty() {
                return Ok(false);
            }
            function.local_variables.append(
                naga::LocalVariable {
                    name: Some("t".to_owned()),
                    ty,
                    init: None,
                },
                naga::Span::UNDEFINED,
            );
            Ok(true)
        }
    }

    /// Counts the driver's invocations; never changes anything.
    struct CountingPass(Rc<Cell<usize>>);

    impl Pass for CountingPass {
        fn name(&self) -> &'static str {
            "synthetic_count"
        }
        fn run(&mut self, _: &mut naga::Module, _: &PassContext<'_>) -> Result<bool, Error> {
            self.0.set(self.0.get() + 1);
            Ok(false)
        }
    }

    /// Substrate for every synthetic pass.
    const TINY_WGSL: &str = "@compute @workgroup_size(1) fn main() { }\n";

    fn parsed_module() -> naga::Module {
        io::parse_wgsl(TINY_WGSL).expect("tiny wgsl parses")
    }

    fn baseline_config(validate_each_pass: bool) -> Config {
        let mut cfg = Config::default();
        cfg.trace.validate_each_pass = validate_each_pass;
        cfg
    }

    #[test]
    fn validate_each_pass_escalates_ir_corruption_to_err() {
        let mut module = parsed_module();
        let cfg = baseline_config(/*validate_each_pass=*/ true);
        let mut report = Report::new(0);
        let passes: Vec<Box<dyn Pass>> = vec![Box::new(CorruptingPass)];
        let result = run_ir_passes_with(&mut module, &cfg, &mut report, passes);
        match result {
            Err(Error::Validation(msg)) => {
                assert!(
                    msg.contains("synthetic_corrupt"),
                    "validation error should name the offending pass; got: {msg}"
                );
                assert!(
                    io::validate_module(&module).is_err(),
                    "escalation must not roll back"
                );
            }
            other => panic!("expected Err(Error::Validation(..)), got {other:?}"),
        }
    }

    #[test]
    fn without_validate_each_pass_ir_corruption_is_rolled_back() {
        let mut module = parsed_module();
        let cfg = baseline_config(/*validate_each_pass=*/ false);
        let mut report = Report::new(0);
        let passes: Vec<Box<dyn Pass>> = vec![Box::new(CorruptingPass)];
        let result = run_ir_passes_with(&mut module, &cfg, &mut report, passes);
        assert!(
            result.is_ok(),
            "without the flag, rollback must keep the pipeline on the happy path; got {result:?}"
        );
        io::validate_module(&module)
            .expect("module must be restored to a valid state after rollback");
        let step = report
            .pass_reports
            .iter()
            .find(|p| p.pass_name == "synthetic_corrupt")
            .expect("synthetic_corrupt step should appear in the report");
        assert!(step.rolled_back, "step must be flagged as rolled_back");
        assert!(
            !step.validation_ok,
            "step must be flagged as validation_ok=false"
        );
    }

    /// Two failures in one sweep: the second rollback must rebuild from the
    /// accepted pass only; replaying every earlier pass would ship the first
    /// corruption unvalidated.
    #[test]
    fn rollback_replays_only_accepted_passes() {
        let mut module = parsed_module();
        let cfg = baseline_config(/*validate_each_pass=*/ false);
        let mut report = Report::new(0);
        let passes: Vec<Box<dyn Pass>> = vec![
            Box::new(CorruptingPass),
            Box::new(AddLocalPass),
            Box::new(CorruptingPass),
        ];
        run_ir_passes_with(&mut module, &cfg, &mut report, passes)
            .expect("rollbacks keep the pipeline on the happy path");
        io::validate_module(&module).expect("both corruptions must be rolled back");
        let function = first_function(&mut module);
        assert_eq!(
            function.local_variables.len(),
            1,
            "the accepted pass's change must survive the later rollback"
        );
        assert!(
            function
                .body
                .iter()
                .all(|s| !matches!(s, naga::Statement::Store { .. })),
            "no corrupting store may survive"
        );
        assert!(
            report
                .pass_reports
                .iter()
                .filter(|p| p.pass_name == "synthetic_corrupt")
                .all(|p| p.rolled_back && !p.validation_ok)
        );
        assert_eq!(
            report
                .pass_reports
                .iter()
                .filter(|p| p.pass_name == "synthetic_add_local" && p.changed)
                .count(),
            1
        );
    }

    /// An idle pass is not re-run until the module changes; trace / CI
    /// modes keep every run.
    #[test]
    fn idle_passes_are_skipped_until_the_module_changes() {
        for (validate_each_pass, expected_runs, expected_reports) in [(false, 1, 3), (true, 2, 4)] {
            let runs = Rc::new(Cell::new(0usize));
            let mut module = parsed_module();
            let cfg = baseline_config(validate_each_pass);
            let mut report = Report::new(0);
            let passes: Vec<Box<dyn Pass>> = vec![
                Box::new(AddLocalPass),
                Box::new(CountingPass(Rc::clone(&runs))),
            ];
            run_ir_passes_with(&mut module, &cfg, &mut report, passes).expect("converges");
            // Sweep 2 re-runs only the pass that changed (now idle) and
            // skips the one already clean at that version.
            assert_eq!(report.sweeps, 2, "validate_each_pass={validate_each_pass}");
            assert_eq!(
                runs.get(),
                expected_runs,
                "validate_each_pass={validate_each_pass}"
            );
            assert_eq!(
                report.pass_reports.len(),
                expected_reports,
                "validate_each_pass={validate_each_pass}"
            );
        }
    }
}
