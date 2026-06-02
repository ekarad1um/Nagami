//! Pipeline driver: owns the [`Pass`] trait, the convergence loop, and
//! all per-pass bookkeeping (validation, rollback, trace dumps).
//!
//! Passes are pure IR-to-IR transforms; the driver wraps them with
//! validation, optional text round-tripping, optional rollback, and a
//! per-pass [`PassReport`] that downstream consumers (CLI, wasm
//! bindings, tests) surface verbatim.

use std::path::PathBuf;
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::config::Config;
#[cfg(not(target_arch = "wasm32"))]
use crate::config::TraceDumpFormat;
use crate::error::Error;
use crate::io;

mod context;
mod report;

pub use context::PassContext;
pub use report::{PassReport, Report};

/// Hard cap on driver iterations.  Prevents pathological pass
/// interactions from looping forever; real pipelines converge in
/// well under this many sweeps.
const MAX_PIPELINE_SWEEPS: usize = 16;

/// An IR optimization pass that mutates a naga module in place.
pub trait Pass {
    /// Short, unique identifier for reports and trace filenames.
    fn name(&self) -> &'static str;
    /// Execute the pass against `module`.
    ///
    /// Returns `Ok(true)` when the module was modified and `Ok(false)`
    /// when the pass made no change.  The driver validates after every
    /// run, so passes may produce intermediate invalid IR only if they
    /// restore validity before returning.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if the pass encounters an unrecoverable failure.
    fn run(&mut self, module: &mut naga::Module, ctx: &PassContext<'_>) -> Result<bool, Error>;
}

/// Emit WGSL text for `module` using naga's backend and the supplied
/// `ModuleInfo`.  Skips validation, so callers MUST ensure `info`
/// matches the current IR state; a stale `ModuleInfo` can panic deep
/// inside the backend through out-of-bounds type or expression indexing.
///
/// The pipeline threads a single `ModuleInfo` across pass boundaries
/// precisely so this helper avoids re-validating for the before/after
/// text emission paths; without it, trace and `validate_each_pass`
/// builds would pay the validator cost twice per pass.
///
/// In `cfg(debug_assertions)` builds (which include `cargo test`),
/// the helper re-validates `module` anyway as a drift guard - so
/// debug-build emission costs the same as the redundant path that
/// release intentionally avoids.  Release builds (where the perf
/// gain matters) skip the check entirely.
fn emit_wgsl_with_info(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
) -> Result<String, Error> {
    // naga's WGSL backend hits `unreachable!()` emitting an override-sized
    // array size, which aborts the process under release `panic = "abort"`.
    // `--trace` and `--validate-each-pass` run this on every intermediate IR,
    // so return a placeholder rather than the panicking backend.  The
    // placeholder is comment-only WGSL that re-parses as an empty module, so
    // the `validate_each_pass` text check below MUST skip these modules or it
    // would record a spurious clean round-trip.
    if crate::module_has_override_sized_array(module) {
        return Ok("// (override-sized array: naga WGSL backend skipped)\n".to_string());
    }
    // Debug-only drift guard: re-validate so a future caller that
    // forgets to refresh after mutating `module` sees a clean panic
    // here instead of a cryptic crash inside the backend.  Release
    // builds skip the check entirely; see the doc-comment above for
    // the debug-vs-release cost trade-off.
    #[cfg(debug_assertions)]
    {
        debug_assert!(
            io::validate_module(module).is_ok(),
            "emit_wgsl_with_info called with a stale ModuleInfo: module no longer validates"
        );
    }
    naga::back::wgsl::write_string(module, info, naga::back::wgsl::WriterFlags::empty())
        .map_err(|e| Error::Emit(e.to_string()))
}

// MARK: Driver

/// Run the IR optimization pipeline to a fixed point.
///
/// Passes from [`crate::passes::build_ir_passes`] execute in order and
/// the whole sequence repeats until no pass reports a change, capped
/// at `MAX_PIPELINE_SWEEPS` sweeps.  Every pass is validated; failures
/// either roll back or escalate to [`Error::Validation`] depending on
/// [`crate::config::TraceConfig::validate_each_pass`].
///
/// # Errors
///
/// Returns [`Error`] if any pass fails with an unrecoverable error or
/// if the module fails validation and rollback is disabled.
pub fn run_ir_passes(
    module: &mut naga::Module,
    config: &Config,
    report: &mut Report,
) -> Result<(), Error> {
    let passes = crate::passes::build_ir_passes(config);
    run_ir_passes_with(module, config, report, passes)
}

/// Internal driver parameterised on the pass list so tests can inject
/// synthetic passes (e.g. an IR-corrupting pass for rollback / hard-fail
/// coverage) without rebuilding the full pipeline.
fn run_ir_passes_with(
    module: &mut naga::Module,
    config: &Config,
    report: &mut Report,
    mut passes: Vec<Box<dyn Pass>>,
) -> Result<(), Error> {
    let trace_run_dir = prepare_trace_dir(config)?;
    let mut sweeps = 0usize;

    // Thread a `ModuleInfo` across pass boundaries so neither text
    // emission nor per-pass validation ever re-runs the validator
    // redundantly.  Seeding is lazy: without tracing or
    // `validate_each_pass`, emission helpers are never called and the
    // common hot path (plain minification) avoids paying for an upfront
    // `validate_module`.  Every successful pass refreshes `current_info`
    // from its post-validation result, so the initial seed matters only
    // for the very first `before_text`.
    let needs_info_upfront = config.trace.enabled || config.trace.validate_each_pass;
    let mut current_info: Option<naga::valid::ModuleInfo> = if needs_info_upfront {
        Some(io::validate_module(module)?)
    } else {
        None
    };

    loop {
        let mut any_changed = false;

        for pass in passes.iter_mut() {
            let trace_enabled = config.trace.enabled;
            let needs_text_validation = config.trace.validate_each_pass;

            let before_text = if trace_enabled {
                let info = current_info
                    .as_ref()
                    .expect("current_info is seeded whenever trace/validate_each_pass is on");
                Some(emit_wgsl_with_info(module, info)?)
            } else {
                None
            };
            let before_bytes = before_text.as_ref().map(|text| text.len());

            // Pre-pass backup used to roll back invalid IR.
            //
            // naga's arenas derive `Clone` without a custom `clone_from`,
            // so the default impl allocates a fresh copy anyway; reusing
            // a long-lived backup via `clone_from + mem::swap` would
            // bring no capacity reuse until naga ships its own impl.
            //
            // Under `validate_each_pass` every failure path escalates
            // to `Err` without ever consuming the backup, so we skip
            // the clone entirely on that CI-oriented path where
            // `module.clone()` is otherwise the hottest call.  Taking
            // the backup below in that mode would indicate a logic bug.
            let mut backup: Option<naga::Module> = if needs_text_validation {
                None
            } else {
                Some(module.clone())
            };

            #[cfg(not(target_arch = "wasm32"))]
            let start = Instant::now();
            let ctx = PassContext {
                config,
                trace_run_dir: trace_run_dir.as_deref(),
            };

            let declared_changed = pass.run(module, &ctx)?;
            #[cfg(not(target_arch = "wasm32"))]
            let duration_us = start.elapsed().as_micros() as u64;
            #[cfg(target_arch = "wasm32")]
            let duration_us = 0u64;
            let mut validation_ok = true;
            let mut rolled_back = false;
            let mut text_validation_ok = None;

            match io::validate_module(module) {
                Ok(info) => {
                    // Adopt the fresh info unconditionally.  Even when
                    // this iteration did not consume `current_info`
                    // (only the trace / validate_each_pass emission
                    // paths do), caching it here means the next pass
                    // whose `before_text` is needed already has a
                    // current snapshot and avoids a redundant validation.
                    current_info = Some(info);
                }
                Err(e) => {
                    if needs_text_validation {
                        // `validate_each_pass` escalates IR failures to
                        // a hard `Err` so CI surfaces a regressing pass
                        // instead of burying a warning in stderr.  The
                        // text-validation branch further down applies
                        // the same policy for symmetry.
                        return Err(Error::Validation(format!(
                            "pass '{}' produced invalid IR: {}",
                            pass.name(),
                            e
                        )));
                    }
                    // We took the !needs_text_validation branch above
                    // when seeding `backup`, so it is always `Some` on
                    // this branch.  The `expect` documents the
                    // invariant rather than introducing a runtime
                    // fallback that would also have to return an Err.
                    let b = backup.take().expect(
                        "backup must be Some when validate_each_pass is off; \
                         see pre-pass backup gate above",
                    );
                    eprintln!(
                        "warning: validation failed after pass '{}', rolling back: {}",
                        pass.name(),
                        e
                    );
                    *module = b;
                    // `current_info` (if seeded) still matches the
                    // restored backup since it was last refreshed
                    // against the pre-pass state, so re-validation
                    // is unnecessary.
                    validation_ok = false;
                    rolled_back = true;
                }
            }

            let after_text = if rolled_back {
                // Module is back to its pre-pass state, so reuse
                // `before_text` when tracing rather than emitting the
                // identical string a second time.
                if trace_enabled {
                    before_text.clone()
                } else {
                    None
                }
            } else if trace_enabled || needs_text_validation {
                let info = current_info
                    .as_ref()
                    .expect("current_info is refreshed after every successful pass");
                Some(emit_wgsl_with_info(module, info)?)
            } else {
                None
            };
            let after_bytes = after_text.as_ref().map(|text| text.len());

            // Override-sized-array modules emit a comment-only placeholder
            // (the backend would abort), which re-parses as a valid empty
            // module; skip the check so a spurious clean round-trip is not
            // recorded, leaving `text_validation_ok` as `None`.
            if !rolled_back
                && needs_text_validation
                && !crate::module_has_override_sized_array(module)
            {
                let ok = io::validate_wgsl_text(
                    after_text
                        .as_deref()
                        .expect("after text must be available for text validation"),
                )
                .is_ok();
                text_validation_ok = Some(ok);
                if !ok {
                    // Mirror the IR-validation policy above: under
                    // `validate_each_pass`, a failed text round-trip
                    // escalates to a structured error.
                    return Err(Error::Validation(format!(
                        "pass '{}' produced IR that round-trips to invalid WGSL text",
                        pass.name()
                    )));
                }
            }

            let changed_by_text = match (before_text.as_ref(), after_text.as_ref()) {
                (Some(before), Some(after)) => before != after,
                _ => false,
            };
            // Convergence is driven SOLELY by each pass's own `declared_changed`
            // return.  `changed_by_text` requires `before_text`, which exists
            // only under `--trace`, so folding it into the loop signal would
            // make a deterministic minifier's convergence depth depend on a
            // debug flag (an under-reporting pass would sweep one extra time
            // only when traced).  It is kept solely to enrich the per-pass
            // report below.
            any_changed |= !rolled_back && declared_changed;
            // Debug guard: a pass that mutated the IR yet returned `Ok(false)`
            // is an under-reporter - its emitted text differs here while
            // `declared_changed` is false.  Catch it loudly in tests rather
            // than letting traced and production convergence silently fork.
            #[cfg(debug_assertions)]
            if trace_enabled && !rolled_back && !declared_changed {
                debug_assert!(
                    !changed_by_text,
                    "pass '{}' under-reports: emitted text changed but it returned Ok(false)",
                    pass.name()
                );
            }
            let changed = !rolled_back && (declared_changed || changed_by_text);

            let pass_report = PassReport {
                pass_name: pass.name().to_string(),
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
                // Persist the sweep count BEFORE the early-Err return
                // so callers inspecting the report after the error
                // (we take `&mut report`, the report state is
                // observable on the Err path) see the actual
                // sweep-count we ran for, not the `Report::new`
                // default of 0.
                report.sweeps = sweeps;
                // Under `validate_each_pass` (CI mode) a non-converged
                // pipeline is a regression to surface, not a warning to
                // bury in stderr.  Escalate to a structured error so
                // CI assertions catch it.  In the default (non-CI)
                // mode keep the existing warning + partial-output
                // behaviour - callers can inspect `report.converged`
                // explicitly.
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
    Ok(())
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

/// Write one `step-NNN-<pass>/` directory under the run's trace folder.
/// Silently returns `Ok(())` on wasm, when tracing is disabled, or when
/// the configured dump format is anything other than
/// [`TraceDumpFormat::WGSL`].
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
        if !matches!(ctx.config.trace.dump_format, TraceDumpFormat::WGSL) {
            return Ok(());
        }
        let Some(run_dir) = ctx.trace_run_dir else {
            return Ok(());
        };
        let step_dir = run_dir.join(format!("step-{step_index:03}-{}", report.pass_name));
        std::fs::create_dir_all(&step_dir)?;

        if ctx.config.trace.dump_before_after {
            std::fs::write(step_dir.join("before.wgsl"), before_text)?;
        }
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

// MARK: Hard-fail escalation tests

#[cfg(test)]
mod hard_fail_tests {
    //! Coverage for the `validate_each_pass` hard-fail escalation paths.
    //!
    //! Two symmetric behaviours are locked in:
    //!
    //! * With `trace.validate_each_pass = true`, any pass that produces
    //!   invalid IR must return `Err(Error::Validation)` instead of
    //!   silently rolling back.  Rollback-and-warn is reserved for the
    //!   default non-CI configuration.
    //! * Without the flag, the same corruption must be rolled back and
    //!   the pipeline must finish successfully with the pre-pass IR
    //!   state fully restored.
    //!
    //! The synthetic `CorruptingPass` below appends a structurally
    //! invalid `Store` (pointer operand is a scalar literal) to the
    //! first function or entry point it sees.  The validator rejects
    //! such a store because the pointer operand matches neither the
    //! value-pointer nor module-pointer shape.
    use super::{Pass, PassContext, run_ir_passes_with};
    use crate::config::Config;
    use crate::error::Error;
    use crate::io;
    use crate::pipeline::report::Report;

    struct CorruptingPass {
        ran: bool,
    }

    impl Pass for CorruptingPass {
        fn name(&self) -> &'static str {
            "synthetic_corrupt"
        }
        fn run(
            &mut self,
            module: &mut naga::Module,
            _ctx: &PassContext<'_>,
        ) -> Result<bool, Error> {
            // Idempotent: corrupt only on the first invocation so the
            // convergence loop terminates in the no-escalate scenario
            // where rollback restores the valid state.
            if self.ran {
                return Ok(false);
            }
            self.ran = true;
            let function = match module.entry_points.first_mut() {
                Some(ep) => &mut ep.function,
                None => {
                    let (_, f) = module
                        .functions
                        .iter_mut()
                        .next()
                        .expect("test module must have a function or entry point");
                    f
                }
            };
            let lit = function.expressions.append(
                naga::Expression::Literal(naga::Literal::I32(0)),
                naga::Span::UNDEFINED,
            );
            // A Store whose pointer is a scalar literal is structurally
            // invalid; the pointer operand must resolve to a pointer or
            // value-pointer type.  Validator rejects this before ever
            // inspecting the value side.
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

    /// Tiny compute shader used as the corruption substrate.  Chosen
    /// so the built-in pass pipeline (which these tests bypass) is
    /// irrelevant to the outcome.
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
        let passes: Vec<Box<dyn Pass>> = vec![Box::new(CorruptingPass { ran: false })];
        let result = run_ir_passes_with(&mut module, &cfg, &mut report, passes);
        match result {
            Err(Error::Validation(msg)) => {
                assert!(
                    msg.contains("synthetic_corrupt"),
                    "validation error should name the offending pass; got: {msg}"
                );
                // Module must still reflect the corrupting pass output:
                // the escalation path intentionally does NOT roll back,
                // and a future refactor that starts rolling back on
                // escalation would silently break this contract.
                assert!(
                    io::validate_module(&module).is_err(),
                    "expected module to still reflect the corrupting pass output \
                     (escalation path intentionally does not roll back)"
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
        let passes: Vec<Box<dyn Pass>> = vec![Box::new(CorruptingPass { ran: false })];
        let result = run_ir_passes_with(&mut module, &cfg, &mut report, passes);
        assert!(
            result.is_ok(),
            "without the flag, rollback must keep the pipeline on the happy path; got {result:?}"
        );
        // The corruption must have been reverted: the restored module
        // validates again and the step report flags the rollback.
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
}
