#!/usr/bin/env bash
# Acoustics Lab daemon (`acousticsd`) -- minimal supervisor wrapper.
#
# The daemon is intentionally not run under systemd; fatal-thread
# panics (mic-arbitrator, supervisor, config-watcher) call
# `process::abort` so the failure is visible.  This wrapper provides
# the matching respawn loop with exponential backoff so a recurring
# panic doesn't fork-bomb the process table.
#
# Usage:
#   scripts/run_acousticsd.sh [--config <path>] [--launch-config <path>] [other acousticsd args...]
#
# Environment overrides:
#   ACOUSTICSD_BIN                 path to the binary (default: /usr/local/bin/acousticsd)
#   RUN_ACOUSTICSD_BACKOFF_MAX     cap on backoff seconds (default: 60)
#   RUN_ACOUSTICSD_BACKOFF_INITIAL initial backoff seconds after first crash (default: 2)
#
# ## CAP_SYS_NICE for SCHED_FIFO
#
# `acousticsd`'s audio + inference threads call `sched_setscheduler(0,
# SCHED_FIFO, ...)` to switch to realtime priority (50 for audio, 30
# for inference) and `sched_setaffinity(0, ...)` to pin to specific
# cores (1 = audio, 2 = inference, 3 = tokio workers on a 4-core SBC,
# leaving core 0 for kernel / IRQ).
#
# Both syscalls require `CAP_SYS_NICE`; without it the daemon starts
# and runs at SCHED_OTHER (kernel default).  Each failed attempt logs
# a one-line `tracing::warn!`; the only observable effect of running
# unprivileged is occasional ALSA underruns under load.  To grant the
# cap (one-time, on each install of the binary):
#
#   sudo setcap cap_sys_nice+ep "$ACOUSTICSD_BIN"
#
# Verify after granting:
#
#   getcap "$ACOUSTICSD_BIN"   # -> ".../acousticsd = cap_sys_nice+ep"
#   cat /proc/$(pidof acousticsd)/sched | grep policy
#                              # -> "policy 1" (= SCHED_FIFO) for the
#                              #    audio + inference threads
#
# The cap is preserved across binary upgrades only if the upgrade
# uses `cp` + re-`setcap`; many package managers strip caps on
# install -- re-grant after every deploy.
set -u

ACOUSTICSD_BIN="${ACOUSTICSD_BIN:-/usr/local/bin/acousticsd}"
backoff="${RUN_ACOUSTICSD_BACKOFF_INITIAL:-2}"
backoff_max="${RUN_ACOUSTICSD_BACKOFF_MAX:-60}"

if [[ ! -x "$ACOUSTICSD_BIN" ]]; then
    echo "run_acousticsd: $ACOUSTICSD_BIN is not executable" >&2
    exit 127
fi

# Forward SIGTERM / SIGINT to the child so the daemon's drain
# completes before we exit. Without the trap the child keeps running
# after the wrapper dies (`kill -9 $$` orphans it).
child_pid=
shutdown=0
forward_signal() {
    shutdown=1
    if [[ -n "$child_pid" ]]; then
        kill -TERM "$child_pid" 2>/dev/null || true
    fi
}
trap forward_signal TERM INT HUP

while :; do
    "$ACOUSTICSD_BIN" "$@" &
    child_pid=$!
    wait "$child_pid"
    rc=$?
    child_pid=

    if [[ "$shutdown" -eq 1 ]]; then
        echo "run_acousticsd: forwarded shutdown signal; exiting rc=$rc" >&2
        exit "$rc"
    fi

    # Clean exit (e.g. --check finished healthy): propagate it.
    if [[ "$rc" -eq 0 ]]; then
        echo "run_acousticsd: acousticsd exited rc=0; not respawning" >&2
        exit 0
    fi

    echo "run_acousticsd: acousticsd exited rc=$rc; restarting in ${backoff}s" >&2
    sleep "$backoff"

    # Exponential backoff up to the cap so a tight crash loop doesn't
    # eat 100% CPU. Reset on every clean exit (the loop restarts from
    # initial when a future crash follows a long uptime -- there's no
    # "uptime threshold" tracking; restarting from initial after a
    # crash is fine for this use case).
    backoff=$(( backoff * 2 ))
    if [[ "$backoff" -gt "$backoff_max" ]]; then
        backoff="$backoff_max"
    fi
done
