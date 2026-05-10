//! Drain registry + bounded shutdown orchestration.
//!
//! ## Failure model: external supervision
//!
//! This module does NOT perform runtime supervision: it does not
//! observe task completion while the daemon is running, does not
//! restart failed tasks, and does not classify subsystems as
//! `unhealthy` on background failure.  Those responsibilities
//! live with an EXTERNAL process supervisor (systemd
//! `Type=notify`, runit, s6, or equivalent).  The daemon's
//! contract is:
//!
//! * Boot deterministically.
//! * Surface health via `/api/v1/status` and `Heartbeat::*`
//!   sender so the external supervisor can act on degraded /
//!   unhealthy signals.
//! * On SIGTERM, drain in bounded time.  If draining cannot
//!   complete within the budget, return `false` from
//!   [`DrainRegistry::shutdown_and_drain`] so the caller can flush
//!   non-supervised tail work (mic arbitrator stop, log appender
//!   flush) and exit non-zero, prompting the external supervisor
//!   to restart a fresh process.
//!
//! ## What this module IS responsible for
//!
//! 1. **Registration.**  Every task the daemon spawns whose
//!    shutdown ordering matters is registered here so the drain
//!    sequence has a single, audited list to walk.
//! 2. **Cancellation propagation.**  Tasks that cooperate with
//!    cancellation register a [`tokio_util::sync::CancellationToken`]
//!    via [`DrainRegistry::register_major_with_token`].  On
//!    [`DrainRegistry::cancel_all`] every registered token is
//!    cancelled before any handle is awaited.  This is the only
//!    way to make `tokio::task::spawn_blocking` workers (e.g. the
//!    inference engine, the training jobs' epoch loops) observe
//!    shutdown -- `AbortHandle::abort()` does NOT stop an already-
//!    running blocking closure.
//! 3. **Bounded drain.**  [`DrainRegistry::shutdown_and_drain`]
//!    cancels every registered token, then awaits every registered
//!    handle under per-task budgets capped by an outer deadline.
//!    Returns `true` on clean drain, `false` on outer-deadline
//!    expiry so the caller can complete tail work before exiting
//!    non-zero.
//!
//! ## Drain ordering
//!
//! The producer (mic arbitrator) is silenced BEFORE
//! [`DrainRegistry::shutdown_and_drain`] runs, so consumers
//! (inference, opus, ws) drain into a quiet pipeline.  The
//! arbitrator stays outside this registry: it is not a
//! `JoinHandle`, and its pre-drain `signal_stop()` is the
//! load-bearing ordering constraint.

use std::time::Duration;
use tokio::task::{AbortHandle, JoinHandle};
use tokio_util::sync::CancellationToken;

/// Task tier -- controls per-task drain budget.
#[derive(Debug, Clone, Copy)]
enum Tier {
    /// Long-running consumers (inference engine, opus encoder,
    /// HTTP listeners).  Drain budget defaults to 5 s -- enough
    /// for an `InferenceEngine::run_blocking` to observe the
    /// shutdown token between iterations (~250 ms) plus
    /// in-flight request settling.
    Major,
    /// Fire-and-keep refreshers (heartbeat pumps, the training
    /// reaper).  Drain budget is 1 s -- one interval tick is
    /// enough for the `select! { _ = shutdown.cancelled() }`
    /// arm to fire.
    Background,
}

/// One supervised task.  `Inner` is type-erased: the registry
/// only needs to await + abort + classify the outcome, all of
/// which work through the trait-object surface in `Inner`.
struct Slot {
    name: &'static str,
    tier: Tier,
    inner: Box<dyn DrainableHandle + Send + 'static>,
}

/// Trait-object facade over `JoinHandle<Result<T, E>>` for
/// arbitrary T/E.  Lets the registry hold a heterogeneous
/// vec of handles without naming each `(T, E)` pair.  The
/// `drain` returns the structured outcome; the registry
/// formats it.
trait DrainableHandle {
    /// Get an abort handle (cheap clone) for the timeout-then-
    /// abort path.  Stable across `drain` calls though we only
    /// drain once.
    fn abort_handle(&self) -> AbortHandle;

    /// Await the underlying `JoinHandle` and reduce the
    /// result to a `TaskOutcome` for logging.  Consumes self.
    fn drain(self: Box<Self>) -> futures_util::future::BoxFuture<'static, TaskOutcome>;
}

/// Outcome of awaiting a task handle.  Used by the drain logger
/// so each subsystem produces one structured log line at
/// shutdown.
enum TaskOutcome {
    /// Task ran to completion with `Ok(_)` -- the cleanest
    /// shutdown path; the cancellation token was observed and
    /// the task exited its run loop cleanly.
    Clean,
    /// Task ran to completion but returned `Err(_)`.  The
    /// formatted error message is held inline so the
    /// registry can log it without needing to re-poll the
    /// handle.
    Error(String),
    /// Tokio cancelled the join handle (typically because we
    /// called `.abort()` after the per-task timeout fired).
    Cancelled,
    /// Task panicked.  The panic-payload doesn't survive the
    /// `JoinError` boundary in a useful form; we surface the
    /// `JoinError`'s `Display` (which carries the panic
    /// location) so post-mortems have signal.
    Panicked(String),
}

/// Concrete `DrainableHandle` impl for a `JoinHandle<Result<T, E>>`.
struct ResultHandle<T, E>
where
    T: Send + 'static,
    E: Send + std::fmt::Display + 'static,
{
    handle: JoinHandle<Result<T, E>>,
}

impl<T, E> DrainableHandle for ResultHandle<T, E>
where
    T: Send + 'static,
    E: Send + std::fmt::Display + 'static,
{
    fn abort_handle(&self) -> AbortHandle {
        self.handle.abort_handle()
    }

    fn drain(self: Box<Self>) -> futures_util::future::BoxFuture<'static, TaskOutcome> {
        Box::pin(async move {
            match self.handle.await {
                Ok(Ok(_)) => TaskOutcome::Clean,
                Ok(Err(e)) => TaskOutcome::Error(e.to_string()),
                Err(je) if je.is_cancelled() => TaskOutcome::Cancelled,
                Err(je) => TaskOutcome::Panicked(je.to_string()),
            }
        })
    }
}

/// Concrete `DrainableHandle` impl for a `JoinHandle<()>`.
/// Background tasks (heartbeat refreshers, the reaper) don't
/// return a `Result`; they exit via the shutdown token.
struct UnitHandle {
    handle: JoinHandle<()>,
}

impl DrainableHandle for UnitHandle {
    fn abort_handle(&self) -> AbortHandle {
        self.handle.abort_handle()
    }

    fn drain(self: Box<Self>) -> futures_util::future::BoxFuture<'static, TaskOutcome> {
        Box::pin(async move {
            match self.handle.await {
                Ok(()) => TaskOutcome::Clean,
                Err(je) if je.is_cancelled() => TaskOutcome::Cancelled,
                Err(je) => TaskOutcome::Panicked(je.to_string()),
            }
        })
    }
}

/// Drain registry.  Owns the registered handles and, optionally,
/// per-task cancellation tokens; on
/// [`Self::shutdown_and_drain`], cancels every token and joins
/// every handle concurrently under per-task budgets capped by an
/// outer total budget.
///
/// Construction is cheap (two empty vecs).  Holds no spawned
/// tasks of its own -- purely a registry.
///
/// See the module-level docs for the failure model: drains, does
/// not restart.  External supervisor handles process restart.
pub struct DrainRegistry {
    slots: Vec<Slot>,
    /// Cancellation tokens registered alongside their handles.
    /// Cancelled in bulk by [`Self::cancel_all`] before any
    /// per-handle drain begins -- this is the only way blocking
    /// closures (e.g. `tokio::task::spawn_blocking`) observe
    /// shutdown, since `AbortHandle::abort()` does not stop an
    /// already-running blocking closure.
    cancel_tokens: Vec<CancellationToken>,
    /// Hooks to call before draining; used by the daemon to set
    /// the cancel flag on every active training job (and any
    /// other registry-shaped subsystem) so blocking work
    /// observes shutdown immediately.  Each hook returns the
    /// number of subsystems it cancelled, used only for logging.
    pre_drain_hooks: Vec<Box<dyn FnOnce() -> usize + Send + 'static>>,
}

impl std::fmt::Debug for DrainRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DrainRegistry")
            .field("registered", &self.slots.len())
            .field("cancel_tokens", &self.cancel_tokens.len())
            .field("pre_drain_hooks", &self.pre_drain_hooks.len())
            .finish_non_exhaustive()
    }
}

impl Default for DrainRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl DrainRegistry {
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            cancel_tokens: Vec::new(),
            pre_drain_hooks: Vec::new(),
        }
    }

    /// Register a major-tier task -- long-running consumers
    /// whose `Result<T, E>` outcome is interesting on
    /// post-mortem.  Drain budget defaults to 5 s.
    ///
    /// For tasks that wrap `tokio::task::spawn_blocking`,
    /// prefer [`Self::register_major_with_token`] so the
    /// blocking closure observes shutdown via the token; an
    /// `AbortHandle::abort()` cannot stop a running blocking
    /// closure.
    pub fn register_major<T, E>(&mut self, name: &'static str, handle: JoinHandle<Result<T, E>>)
    where
        T: Send + 'static,
        E: Send + std::fmt::Display + 'static,
    {
        self.slots.push(Slot {
            name,
            tier: Tier::Major,
            inner: Box::new(ResultHandle { handle }),
        });
    }

    /// Register a major-tier task AND its cancellation token.
    /// On [`Self::cancel_all`] / [`Self::shutdown_and_drain`]
    /// the token is cancelled before any handle is awaited,
    /// giving the task (and any blocking closures it spawned
    /// that polled the token) a chance to exit cleanly within
    /// its drain budget.
    pub fn register_major_with_token<T, E>(
        &mut self,
        name: &'static str,
        handle: JoinHandle<Result<T, E>>,
        token: CancellationToken,
    ) where
        T: Send + 'static,
        E: Send + std::fmt::Display + 'static,
    {
        self.cancel_tokens.push(token);
        self.register_major(name, handle);
    }

    /// Register a background-tier task -- fire-and-keep
    /// refreshers (heartbeat pumps, the training reaper) whose
    /// only shutdown signal is the cancellation token.  Drain
    /// budget defaults to 1 s.
    pub fn register_bg(&mut self, name: &'static str, handle: JoinHandle<()>) {
        self.slots.push(Slot {
            name,
            tier: Tier::Background,
            inner: Box::new(UnitHandle { handle }),
        });
    }

    /// Register a pre-drain hook.  Called by
    /// [`Self::shutdown_and_drain`] BEFORE per-task drain begins
    /// (and AFTER `cancel_all`).  Used by the daemon to set the
    /// cancel flag on every active training job so its blocking
    /// epoch loop observes shutdown.  The hook returns the count
    /// of subsystems it cancelled (logged by the drain sequence).
    pub fn register_pre_drain_hook<F>(&mut self, hook: F)
    where
        F: FnOnce() -> usize + Send + 'static,
    {
        self.pre_drain_hooks.push(Box::new(hook));
    }

    /// Cancel every registered cancellation token.  Idempotent
    /// (cancelling an already-cancelled token is a no-op).
    /// Does NOT await any handles -- pair with
    /// [`Self::shutdown_and_drain`] for the full sequence.
    ///
    /// Returns the number of tokens cancelled.
    pub fn cancel_all(&self) -> usize {
        for token in &self.cancel_tokens {
            token.cancel();
        }
        self.cancel_tokens.len()
    }

    /// Bounded drain: cancel every registered token, run pre-
    /// drain hooks, then await every registered handle under
    /// per-task budgets capped by `outer_budget`.  Per-task
    /// timeouts log a warn and abort the handle.
    ///
    /// Returns `true` if the drain completed within the outer
    /// budget, `false` if the budget expired.  On `false` the
    /// caller MUST hard-exit the process (after flushing any
    /// non-supervised tail work such as the log appender guard
    /// and the mic-arbitrator stop), because
    /// `tokio::task::spawn_blocking` workers (inference engine,
    /// training jobs) cannot be aborted from async code: the
    /// worker thread runs to completion or until the process
    /// exits.  Dropping the runtime would block the drop until
    /// those workers finish, defeating the bounded shutdown
    /// guarantee.  Hard-exiting and letting systemd restart is
    /// cleaner than hanging.
    ///
    /// This function deliberately does NOT call
    /// `std::process::exit` itself: doing so would skip the
    /// non-supervised cleanup the caller still owns (mic
    /// arbitrator join, tracing-appender flush via
    /// `WorkerGuard` drop).  Returning the overrun signal lets
    /// the caller perform that cleanup before exiting.
    ///
    /// Caller MUST have already silenced any non-supervised
    /// producers (the mic arbitrator) BEFORE invoking this.  The
    /// registry doesn't enforce producer-first ordering -- the
    /// arbitrator isn't a `JoinHandle`.
    #[must_use = "outer-budget overrun requires the caller to flush \
                  non-supervised tail work and then std::process::exit(1)"]
    pub async fn shutdown_and_drain(mut self, outer_budget: Duration) -> bool {
        const MAJOR_BUDGET: Duration = Duration::from_secs(5);
        const BG_BUDGET: Duration = Duration::from_secs(1);

        let cancelled = self.cancel_all();
        if cancelled > 0 {
            tracing::debug!(
                target: "acoustics",
                tokens = cancelled,
                "drain: cancelled supervised cancellation tokens",
            );
        }
        let mut hook_total = 0usize;
        for hook in self.pre_drain_hooks.drain(..) {
            hook_total = hook_total.saturating_add(hook());
        }
        if hook_total > 0 {
            tracing::info!(
                target: "acoustics",
                cancelled = hook_total,
                "drain: pre-drain hooks cancelled subsystems",
            );
        }

        // Collect abort handles BEFORE moving slots into the
        // per-task drain futures.  When the outer-budget
        // `timeout` below fires it cancels the whole
        // `drain_all_inner` future -- which drops every
        // pending `drain_one` *before* its inner per-task
        // `timeout` arm can run `abort()`.  Without this
        // upfront collection, a budget overrun would leave
        // every still-pending task running while the caller
        // hard-exits, defeating the bounded-shutdown contract.
        // `spawn_blocking` workers still cannot be aborted (the
        // module doc explains why), but regular async tasks
        // CAN be, and they MUST be.
        let aborts: Vec<AbortHandle> = self.slots.iter().map(|s| s.inner.abort_handle()).collect();

        let drain_one = |slot: Slot| async move {
            let Slot { name, tier, inner } = slot;
            let budget = match tier {
                Tier::Major => MAJOR_BUDGET,
                Tier::Background => BG_BUDGET,
            };
            let abort = inner.abort_handle();
            match tokio::time::timeout(budget, inner.drain()).await {
                Ok(outcome) => log_outcome(name, outcome),
                Err(_) => {
                    tracing::warn!(
                        target: "acoustics",
                        task = name,
                        budget_secs = budget.as_secs(),
                        "task did not exit within shutdown budget; aborting",
                    );
                    abort.abort();
                }
            }
        };

        let drain_all_inner = futures_util::future::join_all(self.slots.into_iter().map(drain_one));
        if tokio::time::timeout(outer_budget, drain_all_inner)
            .await
            .is_err()
        {
            // Outer budget expired with at least one task still
            // pending.  Abort every collected handle so async
            // tasks stop holding their state across the caller's
            // hard-exit.  Cheap (`AbortHandle::abort` is one
            // atomic store); idempotent for handles whose drain
            // already completed.
            for abort in &aborts {
                abort.abort();
            }
            tracing::warn!(
                target: "acoustics",
                outer_budget_secs = outer_budget.as_secs(),
                aborted = aborts.len(),
                "drain did not complete within outer budget; \
                 aborted pending async tasks. caller will exit \
                 non-zero so external supervisor restarts \
                 (spawn_blocking workers cannot be aborted in-process)",
            );
            return false;
        }
        true
    }
}

fn log_outcome(name: &'static str, outcome: TaskOutcome) {
    match outcome {
        TaskOutcome::Clean => {
            tracing::debug!(target: "acoustics", task = name, "task ended cleanly");
        }
        TaskOutcome::Error(e) => {
            tracing::warn!(target: "acoustics", task = name, err = %e, "task returned an error");
        }
        TaskOutcome::Cancelled => {
            tracing::debug!(target: "acoustics", task = name, "task was cancelled");
        }
        TaskOutcome::Panicked(je) => {
            tracing::error!(target: "acoustics", task = name, err = %je, "task panicked");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::Notify;

    /// `cancel_all` cancels every registered token.  Token
    /// cancellation is what blocking closures rely on to observe
    /// shutdown.
    #[tokio::test]
    async fn cancel_all_cancels_registered_tokens() {
        let mut reg = DrainRegistry::new();
        let token_a = CancellationToken::new();
        let token_b = CancellationToken::new();
        let handle_a: JoinHandle<Result<(), &'static str>> = {
            let token = token_a.clone();
            tokio::spawn(async move {
                token.cancelled().await;
                Ok(())
            })
        };
        let handle_b: JoinHandle<Result<(), &'static str>> = {
            let token = token_b.clone();
            tokio::spawn(async move {
                token.cancelled().await;
                Ok(())
            })
        };
        reg.register_major_with_token("a", handle_a, token_a.clone());
        reg.register_major_with_token("b", handle_b, token_b.clone());

        let cancelled = reg.cancel_all();
        assert_eq!(cancelled, 2);
        assert!(token_a.is_cancelled());
        assert!(token_b.is_cancelled());
    }

    /// `register_pre_drain_hook` runs each hook exactly once
    /// during `shutdown_and_drain`.  This is the lever the
    /// daemon uses to cancel active training jobs before
    /// awaiting any handle.
    #[tokio::test]
    async fn pre_drain_hooks_run_during_shutdown() {
        let mut reg = DrainRegistry::new();
        let counter = Arc::new(AtomicUsize::new(0));
        {
            let c = counter.clone();
            reg.register_pre_drain_hook(move || {
                c.fetch_add(7, Ordering::SeqCst);
                7
            });
        }
        {
            let c = counter.clone();
            reg.register_pre_drain_hook(move || {
                c.fetch_add(3, Ordering::SeqCst);
                3
            });
        }
        // Empty drain (no slots) -- still fires hooks.
        // Generous outer budget: drain returns `false` only on
        // budget overrun, but with no slots that cannot happen.
        let drained_clean = reg.shutdown_and_drain(Duration::from_secs(5)).await;
        assert!(drained_clean, "empty drain must complete within budget");
        assert_eq!(counter.load(Ordering::SeqCst), 10);
    }

    /// A task that observes its registered token within the
    /// budget exits cleanly (TaskOutcome::Clean is logged at
    /// debug level; we observe the side effect via the
    /// completion of `shutdown_and_drain`).
    #[tokio::test]
    async fn token_cancelled_task_drains_cleanly_within_budget() {
        let mut reg = DrainRegistry::new();
        let token = CancellationToken::new();
        let started = Arc::new(Notify::new());
        let started_for_task = started.clone();
        let handle: JoinHandle<Result<(), &'static str>> = {
            let token = token.clone();
            tokio::spawn(async move {
                started_for_task.notify_one();
                token.cancelled().await;
                Ok(())
            })
        };
        reg.register_major_with_token("cooperative", handle, token);
        // Wait until the task is parked on the token so the
        // drain race is deterministic.
        started.notified().await;

        let start = std::time::Instant::now();
        let drained_clean = reg.shutdown_and_drain(Duration::from_secs(2)).await;
        let elapsed = start.elapsed();
        assert!(
            drained_clean,
            "cooperative drain must complete within budget"
        );
        // Cooperative task; should drain in well under the
        // outer budget (typically <50 ms) -- assert the bound
        // is wide enough to be CI-stable.
        assert!(
            elapsed < Duration::from_secs(2),
            "cooperative drain took {elapsed:?}; expected <<2 s outer budget",
        );
    }
}
