#!/usr/bin/env bash
# capture_baseline.sh -- run the three hot-path benches and emit
# `tests/benches/baseline_<git-sha>.json` aggregating their median,
# mean, and stderr from criterion's per-bench `estimates.json`.
#
# Usage:
#   bash scripts/capture_baseline.sh
#
# Re-baseline policy lives in `docs/BENCHS.md`; this script is the
# tool referenced there.

set -euo pipefail

# Run from the workspace root (the directory containing this script's
# parent). All cargo + git commands are relative to that location.
cd "$(dirname "$0")/.."
WORKSPACE_ROOT="$PWD"

if ! command -v cargo >/dev/null; then
    echo "error: cargo not on PATH" >&2
    exit 1
fi

if ! command -v python3 >/dev/null; then
    echo "error: python3 not on PATH (used to aggregate criterion JSON)" >&2
    exit 1
fi

if [[ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ]]; then
    echo "warning: working tree is dirty -- captured baseline will not"     >&2
    echo "         reproducibly correspond to the named SHA. Stash or"      >&2
    echo "         commit before re-baselining."                            >&2
fi

GIT_SHA="$(git rev-parse --short=12 HEAD)"
echo "Capturing baseline at SHA $GIT_SHA"

# Run the three benches sequentially. Each takes ~10-20 s in a warm
# build; allow ~3 minutes wall-clock total on a laptop, longer on
# the Pi.
echo "[1/3] cargo bench -p acoustics-lab --bench audio_buffer_ring_throughput"
cargo bench -p acoustics-lab --bench audio_buffer_ring_throughput -q

echo "[2/3] cargo bench -p acoustics-lab --bench opus_stream_encode_frame"
cargo bench -p acoustics-lab --bench opus_stream_encode_frame -q

echo "[3/3] cargo bench -p acoustics-lab --bench inference_engine_run_window"
cargo bench -p acoustics-lab --bench inference_engine_run_window -q

OUTPUT="tests/benches/baseline_${GIT_SHA}.json"
echo "Writing $OUTPUT"

# Aggregate criterion's per-bench JSON into a single workspace-level
# baseline file. Schema kept stable -- see docs/BENCHS.md for
# the field-by-field contract that future regression-checking tools
# rely on.
python3 - "$GIT_SHA" "$OUTPUT" <<'PY'
import json, pathlib, datetime, platform, subprocess, sys

git_sha, output = sys.argv[1], sys.argv[2]
base = pathlib.Path("target/criterion")
benches = {
    "audio_buffer/push_period":  "audio_buffer_push_period/1024_samples",
    "audio_buffer/peek_window":  "audio_buffer_peek_window/44032_samples",
    "opus_stream/encode_frame":  "opus_stream_encode_frame/1024_samples_in_per_call",
    "inference/run_window_burn": "inference_run_window_burn/preproc+backbone+head",
}

def load(rel):
    p = base / rel / "new" / "estimates.json"
    if not p.exists():
        sys.exit(f"missing criterion output: {p} -- bench did not run?")
    with p.open() as f:
        d = json.load(f)
    return {
        "median_ns":          d["median"]["point_estimate"],
        "median_ci_low_ns":   d["median"]["confidence_interval"]["lower_bound"],
        "median_ci_high_ns":  d["median"]["confidence_interval"]["upper_bound"],
        "mean_ns":            d["mean"]["point_estimate"],
        "stderr_ns":          d["mean"]["standard_error"],
    }

doc = {
    "captured_at_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    "git_sha":         git_sha,
    "host": {
        "kernel": platform.system() + " " + platform.release(),
        "arch":   platform.machine(),
        "rustc":  subprocess.run(
            ["rustc", "--version"], capture_output=True, text=True
        ).stdout.strip(),
    },
    "regression_threshold_pct": 5.0,
    "comment": "Hot-path baseline. Subsequent changes must not regress median by >5% without a waiver. Re-capture via scripts/capture_baseline.sh.",
    "benches": {name: load(rel) for name, rel in benches.items()},
}

pathlib.Path(output).parent.mkdir(parents=True, exist_ok=True)
with open(output, "w") as f:
    json.dump(doc, f, indent=2)
    f.write("\n")
print(f"wrote {output}")
PY

echo "Done. Diff against the prior baseline_*.json with:"
echo "  diff <(jq . tests/benches/baseline_<old>.json) <(jq . $OUTPUT)"
