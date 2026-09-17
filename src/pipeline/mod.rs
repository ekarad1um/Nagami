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

pub use context::{AnalysisCache, ModuleInfoCell, PassContext, Rendered, TailRender};
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
    fn run(&mut self, module: &mut naga::Module, ctx: &PassContext<'_>) -> Result<bool, Error>;
}

/// naga's WGSL text for `module` under an already computed `info`, so the
/// trace / `validate_each_pass` emissions never pay the validator twice
/// per pass; a stale `info` panics inside the backend through
/// out-of-bounds arena indexing.
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

/// How many sweeps a pass list gets.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Sweeps {
    /// Until no pass reports a change, capped at `MAX_PIPELINE_SWEEPS`; a
    /// cap hit is reported as non-convergence.
    ToFixedPoint,
    /// Exactly one, for a list whose passes reach their fixed point in one
    /// application and price against the converged module.
    Once,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Validation {
    /// After every declared change, before the next pass runs; a failure
    /// rolls that pass back on the spot.  Trace and CI runs, whose
    /// per-pass text needs the info, and the re-run that locates the
    /// failure a lazy run met.
    PerPass,
    /// When a pass reads the info and once at the end: most passes never
    /// read it.  A failure cannot say which pass is at fault, so the
    /// pipeline is re-run per pass from a backup.
    Lazy,
}

#[derive(Clone, Copy)]
struct Shared<'a> {
    cell: &'a ModuleInfoCell,
    name_log: &'a std::cell::RefCell<crate::name_map::NameLog>,
    tail: &'a TailRender,
    analyses: &'a AnalysisCache,
}

/// Run the IR pipeline: [`crate::passes::build_ir_passes`] to a fixed
/// point, then [`crate::passes::build_tail_passes`] once.  `info` must
/// describe `module` as passed in; the returned info describes it as
/// returned, so a caller never validates the same state twice.
pub fn run_ir_passes(
    module: &mut naga::Module,
    info: naga::valid::ModuleInfo,
    config: &Config,
    report: &mut Report,
    preamble_names: std::collections::HashSet<String>,
) -> Result<Converged, Error> {
    let tail = TailRender {
        preamble_names,
        ..TailRender::default()
    };
    let mut passes = crate::passes::build_ir_passes(config);
    let mut tail_passes = crate::passes::build_tail_passes(config);
    run_lists(
        module,
        info,
        config,
        report,
        &tail,
        &mut passes,
        &mut tail_passes,
    )
}

/// [`run_ir_passes`] over explicit pass lists.  Every declared change is
/// validated: per pass where a mode needs the per-pass info, lazily
/// otherwise.  A lazy run that meets an invalid module is discarded and
/// redone per pass from a backup, which reports and rolls back the pass
/// at fault as the per-pass run always did, so the lazy run has no
/// observable outcome of its own.
fn run_lists(
    module: &mut naga::Module,
    info: naga::valid::ModuleInfo,
    config: &Config,
    report: &mut Report,
    tail: &TailRender,
    passes: &mut [Box<dyn Pass>],
    tail_passes: &mut [Box<dyn Pass>],
) -> Result<Converged, Error> {
    let name_log = std::cell::RefCell::new(crate::name_map::NameLog::default());
    let analyses = AnalysisCache::default();
    let per_pass = config.trace.enabled || config.trace.validate_each_pass;
    let mut cell = ModuleInfoCell::new(info);
    if !per_pass {
        let backup = module.clone();
        let reports = report.pass_reports.len();
        let converged = report.converged;
        let shared = Shared {
            cell: &cell,
            name_log: &name_log,
            tail,
            analyses: &analyses,
        };
        match run_both(
            module,
            config,
            report,
            shared,
            passes,
            tail_passes,
            Validation::Lazy,
        ) {
            Err(_) if cell.failed() => {}
            outcome => {
                let info = outcome?;
                return Ok(Converged {
                    name_log: name_log.into_inner(),
                    info,
                    render: tail.render.take(),
                });
            }
        }
        // Back to the pre-run state: the report entries, the renames and
        // the tail's render were the discarded run's.
        *module = backup;
        *name_log.borrow_mut() = crate::name_map::NameLog::default();
        report.pass_reports.truncate(reports);
        report.converged = converged;
        *tail.render.borrow_mut() = None;
        analyses.clear();
        cell = ModuleInfoCell::new(io::validate_module(module)?);
    }
    let shared = Shared {
        cell: &cell,
        name_log: &name_log,
        tail,
        analyses: &analyses,
    };
    let info = run_both(
        module,
        config,
        report,
        shared,
        passes,
        tail_passes,
        Validation::PerPass,
    )?;
    Ok(Converged {
        name_log: name_log.into_inner(),
        info,
        render: tail.render.take(),
    })
}

/// The IR list to a fixed point, the tail once, then the info of the
/// result.
fn run_both(
    module: &mut naga::Module,
    config: &Config,
    report: &mut Report,
    shared: Shared<'_>,
    passes: &mut [Box<dyn Pass>],
    tail_passes: &mut [Box<dyn Pass>],
    validation: Validation,
) -> Result<naga::valid::ModuleInfo, Error> {
    run_ir_passes_with(
        module,
        config,
        report,
        passes,
        shared,
        Sweeps::ToFixedPoint,
        validation,
    )?;
    if !tail_passes.is_empty() {
        run_ir_passes_with(
            module,
            config,
            report,
            tail_passes,
            shared,
            Sweeps::Once,
            validation,
        )?;
    }
    shared.cell.take_or_validate(module)
}

/// What [`run_ir_passes`] leaves beside the module: the renames it made,
/// naga's analysis of the result, and the tail's render of it (`None`
/// without that render).
pub struct Converged {
    /// The module-scope renames, for the name map.
    pub name_log: crate::name_map::NameLog,
    /// Validation info describing the module as returned.
    pub info: naga::valid::ModuleInfo,
    /// The tail's render, for the emission that ships.
    pub render: Option<Rendered>,
}

/// A hash of the module's `Debug` rendering, the whole IR included: the
/// under-reporting check's oracle.  Debug builds only; the rendering links
/// every naga `Debug` impl, which the release binary must not carry.
#[cfg(debug_assertions)]
fn module_fingerprint(module: &naga::Module) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    format!("{module:?}").hash(&mut hasher);
    hasher.finish()
}

/// Driver parameterised on the pass list so tests can inject synthetic passes.
fn run_ir_passes_with(
    module: &mut naga::Module,
    config: &Config,
    report: &mut Report,
    passes: &mut [Box<dyn Pass>],
    shared: Shared<'_>,
    sweep_policy: Sweeps,
    validation: Validation,
) -> Result<(), Error> {
    let Shared {
        cell,
        name_log,
        tail,
        analyses,
    } = shared;
    let trace_run_dir = prepare_trace_dir(config)?;
    let mut sweeps = 0usize;
    let trace_enabled = config.trace.enabled;
    let needs_text_validation = config.trace.validate_each_pass;
    // Trace / CI runs report every pass; the plain path skips idle ones.
    let full_fidelity = trace_enabled || needs_text_validation;
    let lazy = validation == Validation::Lazy;
    debug_assert!(
        !(lazy && full_fidelity),
        "per-pass text needs per-pass info"
    );

    // `version` counts accepted changes; `clean_at[i]` is the version at
    // which pass `i` last contributed none (no change, or rejected).  Passes
    // are deterministic in the module, so a pass clean at the current
    // version is skipped; only the report omits the idle runs.
    let mut version = 0u64;
    let mut clean_at: Vec<Option<u64>> = vec![None; passes.len()];

    // Per-pass rollback state: module + name log (a stale log would report
    // renames the shipped names lack) as of the run start, plus every
    // accepted pass run since, in order.  A failure restores it and replays
    // exactly those: determinism rebuilds the pre-pass state, and a rejected
    // pass never joins the list, so a second failure cannot resurrect the
    // first one's output.  Cloned once per run; not under
    // `validate_each_pass`, where every failure is an `Err`, nor under lazy
    // validation, whose failure redoes the pipeline from its own backup.
    let mut backup: Option<(naga::Module, crate::name_map::NameLog)> = None;
    let mut accepted: Vec<usize> = Vec::new();

    'sweeps: loop {
        let mut any_changed = false;

        for i in 0..passes.len() {
            // `opt_bisect_limit` reached: the sweep it cuts short still
            // counts in the report, and the cut is not a fixed point.
            if config
                .trace
                .opt_bisect_limit
                .is_some_and(|limit| version >= limit)
            {
                sweeps += 1;
                report.converged = false;
                break 'sweeps;
            }
            if !full_fidelity && clean_at[i] == Some(version) {
                continue;
            }
            if !lazy && backup.is_none() && !needs_text_validation {
                backup = Some((module.clone(), name_log.borrow().clone()));
            }

            let before_text = if trace_enabled {
                Some(emit_wgsl_with_info(module, &*cell.get(module)?)?)
            } else {
                None
            };
            let before_bytes = before_text.as_ref().map(|text| text.len());

            #[cfg(not(target_arch = "wasm32"))]
            let start = Instant::now();

            // The rollback replay and the `clean_at` skip both rest on "no
            // declared change, no change": a pass that rewrites the arena
            // behind a `false` leaves the replayed state short of the
            // timeline's, which `compact` then reports every sweep.
            // `changed_by_text` cannot see an orphaned expression, so debug
            // builds hash the IR itself.
            #[cfg(debug_assertions)]
            let fingerprint_before = module_fingerprint(module);
            let declared_changed = passes[i].run(
                module,
                &PassContext {
                    config,
                    info: cell,
                    name_log: Some(name_log),
                    tail: Some(tail),
                    analyses,
                },
            )?;
            // A pass may swallow the error a failed lazy validation returned
            // it; the cell remembers, and the run stops here rather than
            // pile more passes onto an invalid module.
            if cell.failed() {
                return Err(Error::Validation(format!(
                    "pass '{}' read an invalid module",
                    passes[i].name()
                )));
            }
            #[cfg(debug_assertions)]
            debug_assert!(
                declared_changed || module_fingerprint(module) == fingerprint_before,
                "pass '{}' under-reports: the IR changed but it returned Ok(false)",
                passes[i].name()
            );
            #[cfg(not(target_arch = "wasm32"))]
            let duration_us = start.elapsed().as_micros() as u64;
            #[cfg(target_arch = "wasm32")]
            let duration_us = 0u64;
            let mut validation_ok = true;
            let mut rolled_back = false;
            let mut text_validation_ok = None;

            if lazy {
                if declared_changed {
                    cell.invalidate();
                    analyses.clear();
                    version += 1;
                }
            } else if declared_changed || needs_text_validation {
                // No declared change -> the module is still the state the
                // last validation blessed, so only CI mode re-validates; an
                // under-reporting pass trips the fingerprint debug_assert.
                match io::validate_module(module) {
                    Ok(info) => {
                        cell.set(info);
                        if declared_changed {
                            analyses.clear();
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
                        // Pre-pass state again.  Each replayed pass reads
                        // the info of the state it first ran on, so the
                        // replay re-validates as the timeline did, on a
                        // path few shaders take; replay is deterministic
                        // by the `Pass` contract, and re-deriving the final
                        // info turns a contract violation into an error
                        // instead of a stale `info` the backend indexes
                        // out of bounds.
                        let (saved_module, saved_log) = backup
                            .as_ref()
                            .expect("backup is taken whenever validate_each_pass is off");
                        *module = saved_module.clone();
                        *name_log.borrow_mut() = saved_log.clone();
                        analyses.clear();
                        cell.set(io::validate_module(module)?);
                        for &earlier in &accepted {
                            passes[earlier].run(
                                module,
                                &PassContext {
                                    config,
                                    info: cell,
                                    name_log: Some(name_log),
                                    tail: Some(tail),
                                    analyses,
                                },
                            )?;
                            analyses.clear();
                            cell.set(io::validate_module(module)?);
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
                Some(emit_wgsl_with_info(module, &*cell.get(module)?)?)
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
                let verdict = io::validate_wgsl_text(
                    after_text
                        .as_deref()
                        .expect("after text must be available for text validation"),
                );
                text_validation_ok = Some(verdict.is_ok());
                if let Err(e) = verdict {
                    // The diagnostic is the whole point of this mode; without
                    // it the pass name alone leaves nothing to act on.
                    return Err(Error::Validation(format!(
                        "pass '{}' produced IR that round-trips to invalid WGSL text: {e}",
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
            // would make convergence depth depend on a debug flag; it only
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

            if let Some(run_dir) = trace_run_dir.as_deref() {
                dump_trace_step(
                    run_dir,
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
        if sweep_policy == Sweeps::Once {
            return Ok(());
        }
        if !any_changed || sweeps >= MAX_PIPELINE_SWEEPS {
            if sweeps >= MAX_PIPELINE_SWEEPS && any_changed {
                // An invalid module can keep a lazy run's passes busy to
                // the cap; the per-pass re-run judges that, not this
                // warning.
                if lazy {
                    drop(cell.get(module)?);
                }
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
    Ok(())
}

// MARK: Trace directory allocation

/// `None` when tracing is off or on wasm (no filesystem); otherwise a
/// fresh `<base>/run-{stamp}[-suffix]`.
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

/// Claims `run-{stamp}[-suffix]` under `base` with `create_dir` (not
/// `create_dir_all`) as the primitive: same-millisecond or concurrent
/// callers agreeing on a stamp get `AlreadyExists`, the suffix increments,
/// and first available wins with no check-then-act race window.
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
/// `meta.txt`) under `run_dir`; a no-op on wasm.
fn dump_trace_step(
    run_dir: &std::path::Path,
    step_index: usize,
    before_text: &str,
    after_text: &str,
    report: &PassReport,
) -> Result<(), Error> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (run_dir, step_index, before_text, after_text, report);
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
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

    /// Hand-rolled so `tempfile` is not a dev-dependency for one test group.
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
        // Two callers agreeing on a stamp: an idempotent `create_dir_all`
        // primitive would hand both `run-{stamp}`.
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

    use super::{
        AnalysisCache, ModuleInfoCell, Pass, PassContext, Shared, TailRender, Validation, run_both,
        run_lists,
    };

    fn run_to_fixed_point(
        module: &mut naga::Module,
        info: naga::valid::ModuleInfo,
        config: &Config,
        report: &mut Report,
        mut passes: Vec<Box<dyn Pass>>,
    ) -> Result<naga::valid::ModuleInfo, Error> {
        let tail = TailRender::default();
        run_lists(module, info, config, report, &tail, &mut passes, &mut [])
            .map(|converged| converged.info)
    }
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

    /// Adds `var t: u32;` when exactly `.0` locals exist: a valid change the
    /// rollback tests can watch survive, stageable across sweeps.
    struct AddLocalPass(usize);

    impl Pass for AddLocalPass {
        fn name(&self) -> &'static str {
            "synthetic_add_local"
        }
        fn run(
            &mut self,
            module: &mut naga::Module,
            _ctx: &PassContext<'_>,
        ) -> Result<bool, Error> {
            if first_function(module).local_variables.len() != self.0 {
                return Ok(false);
            }
            let ty = module.types.insert(
                naga::Type {
                    name: None,
                    inner: naga::TypeInner::Scalar(naga::Scalar::U32),
                },
                naga::Span::UNDEFINED,
            );
            let function = first_function(module);
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

    /// Reads the module's info, counting the reads that succeed; never
    /// changes anything.
    struct ReadInfoPass(Rc<Cell<usize>>);

    impl Pass for ReadInfoPass {
        fn name(&self) -> &'static str {
            "synthetic_read_info"
        }
        fn run(&mut self, module: &mut naga::Module, ctx: &PassContext<'_>) -> Result<bool, Error> {
            ctx.info(module)?;
            self.0.set(self.0.get() + 1);
            Ok(false)
        }
    }

    /// Reads the shared analysis of the first body and keeps every answer,
    /// so a recomputed one cannot land at a freed address; never changes
    /// anything.
    struct ReadLensPass(Rc<std::cell::RefCell<Vec<super::context::AccessLens>>>);

    impl Pass for ReadLensPass {
        fn name(&self) -> &'static str {
            "synthetic_read_lens"
        }
        fn run(&mut self, module: &mut naga::Module, ctx: &PassContext<'_>) -> Result<bool, Error> {
            // The tiny module's one body, the entry point, is body 0.
            let function = &module.entry_points[0].function;
            let lens = ctx.access_lens(0, function, module);
            self.0.borrow_mut().push(lens);
            Ok(false)
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
        let info = io::validate_module(&module).expect("valid input");
        let result = run_to_fixed_point(&mut module, info, &cfg, &mut report, passes);
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
        let info = io::validate_module(&module).expect("valid input");
        let result = run_to_fixed_point(&mut module, info, &cfg, &mut report, passes);
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
            Box::new(AddLocalPass(0)),
            Box::new(CorruptingPass),
        ];
        let info = io::validate_module(&module).expect("valid input");
        run_to_fixed_point(&mut module, info, &cfg, &mut report, passes)
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

    /// The backup is taken once per run, so a rejection in sweep two must
    /// replay sweep one's accepted work as well as its own.
    #[test]
    fn rollback_replays_accepted_passes_from_earlier_sweeps() {
        let mut module = parsed_module();
        let cfg = baseline_config(/*validate_each_pass=*/ false);
        let mut report = Report::new(0);
        // Sweep 1: the second pass adds the first local; sweep 2: the first
        // pass adds the second.  The corrupting pass is rejected in both.
        let passes: Vec<Box<dyn Pass>> = vec![
            Box::new(AddLocalPass(1)),
            Box::new(AddLocalPass(0)),
            Box::new(CorruptingPass),
        ];
        let info = io::validate_module(&module).expect("valid input");
        run_to_fixed_point(&mut module, info, &cfg, &mut report, passes).expect("converges");
        io::validate_module(&module).expect("both corruptions must be rolled back");
        assert_eq!(first_function(&mut module).local_variables.len(), 2);
        let adds = report
            .pass_reports
            .iter()
            .filter(|r| r.pass_name == "synthetic_add_local" && r.changed)
            .count();
        let rollbacks = report.pass_reports.iter().filter(|r| r.rolled_back).count();
        assert_eq!((adds, rollbacks, report.sweeps), (2, 2, 3));
    }

    /// Trace / CI modes keep every run.
    #[test]
    fn idle_passes_are_skipped_until_the_module_changes() {
        for (validate_each_pass, expected_runs, expected_reports) in [(false, 1, 3), (true, 2, 4)] {
            let runs = Rc::new(Cell::new(0usize));
            let mut module = parsed_module();
            let cfg = baseline_config(validate_each_pass);
            let mut report = Report::new(0);
            let passes: Vec<Box<dyn Pass>> = vec![
                Box::new(AddLocalPass(0)),
                Box::new(CountingPass(Rc::clone(&runs))),
            ];
            let info = io::validate_module(&module).expect("valid input");
            run_to_fixed_point(&mut module, info, &cfg, &mut report, passes).expect("converges");
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

    /// `opt_bisect_limit = N` ships exactly the first N accepted changes.
    #[test]
    fn opt_bisect_limit_keeps_exactly_n_accepted_changes() {
        for (limit, expected_locals) in [(0u64, 0usize), (1, 1), (2, 2), (5, 2)] {
            let mut module = parsed_module();
            let mut cfg = baseline_config(/*validate_each_pass=*/ false);
            cfg.trace.opt_bisect_limit = Some(limit);
            let mut report = Report::new(0);
            let passes: Vec<Box<dyn Pass>> =
                vec![Box::new(AddLocalPass(0)), Box::new(AddLocalPass(1))];
            let info = io::validate_module(&module).expect("valid input");
            run_to_fixed_point(&mut module, info, &cfg, &mut report, passes).expect("runs");
            assert_eq!(
                first_function(&mut module).local_variables.len(),
                expected_locals,
                "limit={limit}"
            );
            assert_eq!(
                report.pass_reports.iter().filter(|p| p.changed).count(),
                expected_locals,
                "limit={limit}"
            );
        }
    }

    /// The report as the tests compare it: every field but the timing.
    fn report_key(report: &Report) -> impl PartialEq + std::fmt::Debug {
        (
            report.converged,
            report.sweeps,
            report
                .pass_reports
                .iter()
                .map(|p| {
                    (
                        p.pass_name.clone(),
                        p.before_bytes,
                        p.after_bytes,
                        p.changed,
                        p.validation_ok,
                        p.text_validation_ok,
                        p.rolled_back,
                    )
                })
                .collect::<Vec<_>>(),
        )
    }

    /// The lazy driver is an optimisation of the per-pass one: module and
    /// report agree whether the run is clean, meets an invalid module a
    /// pass reads, or one nobody reads.
    #[test]
    fn lazy_and_per_pass_validation_agree() {
        type Passes = Vec<Box<dyn Pass>>;
        let lists: Vec<fn() -> Passes> = vec![
            || vec![Box::new(AddLocalPass(0)), Box::new(AddLocalPass(1))],
            || {
                vec![
                    Box::new(CorruptingPass),
                    Box::new(ReadInfoPass(Rc::new(Cell::new(0)))),
                ]
            },
            || vec![Box::new(AddLocalPass(0)), Box::new(CorruptingPass)],
            || {
                vec![
                    Box::new(AddLocalPass(1)),
                    Box::new(AddLocalPass(0)),
                    Box::new(CorruptingPass),
                    Box::new(ReadInfoPass(Rc::new(Cell::new(0)))),
                ]
            },
        ];
        let cfg = baseline_config(/*validate_each_pass=*/ false);
        for (i, list) in lists.iter().enumerate() {
            let mut lazy_module = parsed_module();
            let mut lazy_report = Report::new(0);
            let info = io::validate_module(&lazy_module).expect("valid input");
            run_to_fixed_point(&mut lazy_module, info, &cfg, &mut lazy_report, list())
                .expect("lazy run converges");

            let mut module = parsed_module();
            let mut report = Report::new(0);
            let name_log = std::cell::RefCell::new(crate::name_map::NameLog::default());
            let tail = TailRender::default();
            let cell = ModuleInfoCell::new(io::validate_module(&module).expect("valid input"));
            let analyses = AnalysisCache::default();
            let shared = Shared {
                cell: &cell,
                name_log: &name_log,
                tail: &tail,
                analyses: &analyses,
            };
            run_both(
                &mut module,
                &cfg,
                &mut report,
                shared,
                &mut list(),
                &mut [],
                Validation::PerPass,
            )
            .expect("per-pass run converges");

            assert_eq!(report_key(&lazy_report), report_key(&report), "list {i}");
            assert_eq!(
                format!("{lazy_module:?}"),
                format!("{module:?}"),
                "list {i}"
            );
        }
    }

    /// Lazy validation pays once per read of a changed module, plus the
    /// final one; per-pass validation pays once per change.
    #[test]
    fn lazy_validation_pays_per_read_not_per_change() {
        let cfg = baseline_config(/*validate_each_pass=*/ false);
        for (reads, expected) in [(false, 1), (true, 2)] {
            let mut module = parsed_module();
            let mut report = Report::new(0);
            let name_log = std::cell::RefCell::new(crate::name_map::NameLog::default());
            let tail = TailRender::default();
            let cell = ModuleInfoCell::new(io::validate_module(&module).expect("valid input"));
            let analyses = AnalysisCache::default();
            let shared = Shared {
                cell: &cell,
                name_log: &name_log,
                tail: &tail,
                analyses: &analyses,
            };
            let mut passes: Vec<Box<dyn Pass>> = vec![Box::new(AddLocalPass(0))];
            if reads {
                passes.push(Box::new(ReadInfoPass(Rc::new(Cell::new(0)))));
            }
            passes.push(Box::new(AddLocalPass(1)));
            if reads {
                passes.push(Box::new(ReadInfoPass(Rc::new(Cell::new(0)))));
            }
            run_both(
                &mut module,
                &cfg,
                &mut report,
                shared,
                &mut passes,
                &mut [],
                Validation::Lazy,
            )
            .expect("converges");
            assert_eq!(cell.validations.get(), expected, "reads={reads}");
        }
    }

    /// Passes that leave the module unchanged share an analysis; an
    /// accepted change drops it.
    #[test]
    fn shared_analyses_survive_idle_passes_and_not_a_change() {
        let seen = Rc::new(std::cell::RefCell::new(Vec::new()));
        let mut module = parsed_module();
        let cfg = baseline_config(/*validate_each_pass=*/ false);
        let mut report = Report::new(0);
        let passes: Vec<Box<dyn Pass>> = vec![
            Box::new(ReadLensPass(Rc::clone(&seen))),
            Box::new(ReadLensPass(Rc::clone(&seen))),
            Box::new(AddLocalPass(0)),
            Box::new(ReadLensPass(Rc::clone(&seen))),
        ];
        let info = io::validate_module(&module).expect("valid input");
        run_to_fixed_point(&mut module, info, &cfg, &mut report, passes).expect("converges");
        let seen = seen.borrow();
        assert!(seen.len() >= 3, "{}", seen.len());
        assert!(
            Rc::ptr_eq(&seen[0], &seen[1]),
            "idle passes share the entry"
        );
        assert!(!Rc::ptr_eq(&seen[1], &seen[2]), "a change drops it");
    }

    /// A pass reading an invalid module fails its read; the re-run per
    /// pass names and rolls back the pass at fault, and the reader then
    /// reads the restored module.
    #[test]
    fn a_read_of_an_invalid_module_names_the_pass_at_fault() {
        let reads = Rc::new(Cell::new(0usize));
        let mut module = parsed_module();
        let cfg = baseline_config(/*validate_each_pass=*/ false);
        let mut report = Report::new(0);
        let passes: Vec<Box<dyn Pass>> = vec![
            Box::new(CorruptingPass),
            Box::new(ReadInfoPass(Rc::clone(&reads))),
        ];
        let info = io::validate_module(&module).expect("valid input");
        run_to_fixed_point(&mut module, info, &cfg, &mut report, passes)
            .expect("the rollback keeps the pipeline on the happy path");
        io::validate_module(&module).expect("the corruption is rolled back");
        assert_eq!(
            reads.get(),
            1,
            "the reader succeeds once, after the rollback"
        );
        let names: Vec<_> = report
            .pass_reports
            .iter()
            .map(|p| (p.pass_name.as_str(), p.rolled_back, p.validation_ok))
            .collect();
        assert_eq!(
            names,
            vec![
                ("synthetic_corrupt", true, false),
                ("synthetic_read_info", false, true)
            ]
        );
    }
}
