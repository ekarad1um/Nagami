# Hot-path benches and the baseline contract

Three criterion benches pin the hot-path budget for the daemon's
audio + inference pipeline; the `audio_buffer` bench measures two
points (push + peek), giving four named measurements total.

## Benches

| Bench (`[[bench]]` name) | Measures |
|---|---|
| `audio_buffer_ring_throughput` | `Writer::push` of one capture period (1024 samples), `Reader::peek_into` of one inference window (44032 samples) |
| `opus_stream_encode_frame` | One `OpusEngine::process_pcm`: 1024-sample chunk in, Opus packet out (resample + encode) |
| `inference_engine_run_window` | One inference cycle: spectrogram preproc + Burn backbone + head matmul |

Source under [`tests/benches/`](../tests/benches/).  Run individually:

```sh
cargo bench -p acoustics-lab --bench audio_buffer_ring_throughput
cargo bench -p acoustics-lab --bench opus_stream_encode_frame
cargo bench -p acoustics-lab --bench inference_engine_run_window
```

## Re-baseline policy

Recorded numbers live in `tests/benches/baseline_<git-sha>.json`.
Re-baseline only when the measurement target has moved -- a backend
change, a deliberate algorithmic rewrite, or a hardware refresh.
Routine refactors must not regress median runtime by more than
`regression_threshold_pct` (currently 5%) at any of the four
measurements.

```sh
bash scripts/capture_baseline.sh
```

The script runs all three benches in release, aggregates
median + mean + stderr from criterion's per-bench
`estimates.json`, and writes a new
`tests/benches/baseline_<git-sha>.json`.  Commit alongside the
change that justified it; keep prior baselines for regression
investigation.

## Diffing two baselines

```sh
diff <(jq . tests/benches/baseline_<old>.json) \
     <(jq . tests/benches/baseline_<new>.json)
```

Median deltas within `regression_threshold_pct` are noise; anything
larger needs a commit-message waiver or a fix.
