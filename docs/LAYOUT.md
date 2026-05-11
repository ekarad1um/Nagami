# Bundled fixture and runtime layout

Two trees: the operator-supplied / repo-bundled `misc/` tree
(fixtures + dev TOML + bundled-default head) and the runtime
workspace tree the daemon writes into at `<workspace_root>/`.
Sibling docs:
[ARCHITECTURE.md](ARCHITECTURE.md),
[API.md](API.md),
[BUILD.md](BUILD.md),
[PROTO.md](PROTO.md).

## Repo / bundled tree

```
acoustics_lab/
|-- misc/
|   |-- backbones/                 -- backbone weights (.mpk + .rknn)
|   |-- datasets/                  -- training datasets (operator-supplied)
|   |-- etc/config.toml            -- hot user-pref template (mirrors `<workspace>/config.toml`)
|   |-- heads/
|   |   |-- default/               -- bundled default head, copied into active/ on first boot
|   |       |-- head.mpk
|   |       |-- labels.txt
|   |-- models/                    -- TFJS source models for the converter
|   |   |-- (sc_preproc_model/, model.json, group1-shard{1,2}of2, ...) -- upstream Google Speech-Commands bundle (multi-shard, `words`; gitignored; see get_tfjs_sc_model.sh)
|   |-- npys/                      -- spectrogram-parity NPY fixtures
|   |-- share/                     -- runtime UDS socket dir
|   |   |-- acousticsd.sock        -- daemon's stream listener (gitignored)
|   |-- workspace/                 -- daemon's per-experiment workspace root (gitignored)
```

### `misc/npys/` (spectrogram-parity fixtures)

| File | Size | Consumed by |
|---|---|---|
| `waveform_{0..4}.npy` | 5 x 176 KiB | the preproc parity tests + the inference parity tests |
| `spectrogram_{0..4}.npy` | 5 x 40 KiB | the preproc parity tests |
| `window_blackman_2048.npy` | 8 KiB | `preproc_window_source` |
| `logits_{0..4}.npy` | 5 x 208 B | the inference parity tests |
| `probs_{0..4}.npy` | 5 x 208 B | the inference parity tests |

Total ~1.1 MiB.  Deterministic captures from the canonical
pipeline; the parity tests assert byte-stable reproduction
within fp32 rounding.

### `misc/backbones/` (backbone weights)

| File | Size | Consumed by |
|---|---|---|
| `backbone.mpk` | 5.5 MiB | host-dev Burn fp32 path + training + Burn parity tests |
| `backbone.rknn` | 2.8 MiB | NPU production path (`linux + rknpu` feature only) |

### `misc/heads/` (classifier heads)

| File | Size | Notes |
|---|---|---|
| `default/{head.mpk, labels.txt}` | 24 KiB + 113 B | 20-class Speech-Commands bundled-default; the daemon copies these into `active/generations/<activation_id>/` on first boot or when `POST /active {default: true}` runs |

The fixture directory name `default/` is operator-facing only; the
bundled default's runtime identity is the valid UUID-v4
`00000000-0000-4000-8000-000000000000`.  The directory name is never
parsed as a `HeadId`.

Regenerable via the
[`regen_fixtures`](../examples/regen_fixtures.rs) example —
subcommand `default-head`.

### `misc/models/` (TFJS source models for the converter)

Consumed via [`extract_head_from_tfjs_dir`](../modules/converter.rs):

- `{model.json, metadata.json, group1-shard{1,2}of2, sc_preproc_model/}`
  — upstream Google Speech-Commands TFJS bundle (multi-shard,
  `words`-schema metadata, head matched by shape against
  `BACKBONE_FEATURE_DIM`).  Fetched by
  [`get_tfjs_sc_model.sh`](../misc/models/get_tfjs_sc_model.sh);
  gitignored (anchored to the top of `misc/models/`).

### `misc/etc/config.toml`

Bundled user-pref template.  Mirrors the shape the daemon
auto-creates at `<workspace>/config.toml` on first boot: mic
policy, inference cadence, training defaults, file caps.  No
stream listener binds and no persisted active-head pointer here —
`<workspace>/active/` is the single source for active head state.
Operators copying the dev workflow into a real deployment can use
this file as a starting point for `<workspace>/config.toml`; the
daemon itself does not read `misc/etc/config.toml` directly (it
reads `<workspace>/config.toml`).

### `misc/etc/launch.toml`

Local-dev launch catalogue, what `--config` points at.  Distinct
schema from the user-pref TOML: lists mic candidates, backbone
candidates, stream listener binds, and `[head.default]`
(`path` + `labels_path`) read once at daemon boot.  Passing the
same TOML to `--config` and as `<workspace>/config.toml` is
rejected at boot.

### Runtime-only directories (gitignored)

`misc/share/` (UDS socket), `misc/workspace/` (per-experiment
data), and `misc/datasets/` (operator-supplied training data)
are populated at runtime.

### How tests + the regen CLI resolve the fixture root

A `crate_root()` helper resolves `CARGO_MANIFEST_DIR` and joins
the relative `misc/...` path — no `..` ascent, no
`canonicalize()`.  Used by the preproc parity tests, the
inference + converter test mods, and the [`regen_fixtures`](../examples/regen_fixtures.rs)
example.

## System libraries (NOT relocatable)

The Rockchip NPU runtime ships with the rv1126b Linux image
and cannot be bundled:

| File | Loaded by | Notes |
|---|---|---|
| `librknnrt.so` | [`rknn_runtime::utils`](../modules/rknn_runtime/utils.rs) | NPU runtime |
| `librknnmrt.so` | same | Multi-core variant |

Search order: `$RKNN_LIB`, then `/usr/lib/`, `/usr/local/lib/`,
`/usr/lib/aarch64-linux-gnu/`, every directory in
`$LD_LIBRARY_PATH`, and `~/.local/lib/`.

## Runtime tree (`<workspace_root>/`)

The daemon owns the entire workspace tree under
`<workspace_root>/`.  External tampering is unsupported (no
scan / reconcile API); operators either let the daemon mutate
through its API or wipe and recreate.

Every subdirectory inside the tree is **created lazily by the
writer that first touches it** -- only `<workspace_root>/` and
`<workspace_root>/logs/` are materialized at boot.  A fresh
workspace is just `workspace.json` + `heads.json`; the leaf
subdirs (`datasets/`, `converters/`, `heads/`,
`training_logs/`, `converter_logs/`, `.tmp/`) appear when their
respective producer first runs.  The same lazy-mkdir rule
applies at the root level: `.tmp/` and `active/` only appear
once needed.

```
<workspace_root>/
    config.toml                                   -- mutable user-pref config; auto-created if missing
    workspaces/<workspace_id>/                    -- created on workspace create
        workspace.json                            -- hot core metadata (WorkspaceCore)
        heads.json                                -- compact 2-slot head index (HeadIndex)
        datasets/                                 -- lazy: created on first dataset upload
            <path>/<file>
        converters/                               -- lazy: created on first converter run
            <path>/<file>
        heads/                                    -- lazy: created on first head publish (max 2)
            <head_id>.mpk                         -- raw weights (Burn .mpk)
            <head_id>.json                        -- per-head manifest (HeadManifest)
        training_logs/<job_id>.jsonl              -- lazy: created on first train job
        converter_logs/<job_id>.jsonl             -- lazy: created on first convert job
        .tmp/                                     -- lazy: created on first upload / delete staging
    .tmp/                                         -- lazy: created on first workspace delete
    active/                                       -- lazy: created on first head activation
        current.json                              -- {activation_id}; atomic active generation pointer
        generations/                              -- retain current + previous generation only
            <activation_id>/
                head.mpk                          -- materialized active-head weights
                labels.txt                        -- materialized from manifest `labels[]`
                manifest.json                     -- ActiveHeadManifest body
        .tmp/                                     -- activation staging
    var/run/
        acoustics_lab.sock                        -- default UDS listener (configurable in launch TOML)
    logs/                                         -- eager: created at boot for tracing-appender
        acousticsd.log                            -- daily rotated; 7-file retention
```

### Background storage hygiene

A `storage_reaper` background task (every 1 h; see
[`modules/file_mgr/storage_reaper.rs`](../modules/file_mgr/storage_reaper.rs))
keeps the lazy tree bounded:

- Reaps `.tmp/` entries (files or dir subtrees) whose mtime is
  older than 24 h across `<root>/.tmp/`, `<root>/active/.tmp/`,
  and every `<workspace>/.tmp/`.  Covers the "daemon kept
  running after a hard crash" gap that boot recovery cannot
  reach.
- Prunes per-workspace `training_logs/*.jsonl` and
  `converter_logs/*.jsonl` older than 30 d so a long-lived
  workspace does not grow unbounded.
- Leaves `<root>/logs/acousticsd.log.*` untouched -- the
  `tracing-appender::rolling::Rotation::DAILY` policy with
  `max_log_files(7)` already prunes it.

Counters surface via `GET /api/v1/status` under
`workspace_metrics.tmp_orphans_reaped_total`,
`log_files_pruned_total`, `storage_reaper_failures_total`.

`config.toml` is the hot-reloadable user-preference TOML
(mic policy + inference cadence -- the only fields actually
mutated at runtime).  Boot-time constants (training defaults,
file admission caps, stream binds, the mic catalogue, etc.)
live in the launch TOML.  The daemon writes a default body on
first boot and watches the file for operator edits at runtime;
it lives inside the workspace tree so a single
`--workspace <PATH>` argument is sufficient to locate everything
mutable the daemon owns.

**Backbone artefact**: a single Burn `.mpk` file lives outside
the workspace tree, configured via `[[backbone.candidates]]`
in the launch TOML.  Both the inference engine ([`load_first_supported`](../modules/inference/backbone.rs))
and the trainer (`POST /workspace/{id}/train`) read it.  The
trainer picks the first candidate with `kind = "burn"` at boot
and refuses the request with a structured 404 if none is
configured.  The daemon does not auto-create the path or accept
uploads — operators (or deployment scripts) place the file
there once.

The trainer treats this file as **frozen, read-only**: it
loads the weights into RAM at job start (no long-lived fd, no
mmap) and writes only to `<workspace>/heads/<head_id>.{mpk,json}`.
Combined with the daemon-wide `max_train_jobs = 1` single-flight
admission (1-permit `tokio::sync::Semaphore` in
[`training::JobRegistry`](../modules/training.rs)), this means the
file is safe to read concurrently with operator hot-swap via
`cp`: an in-flight job continues against its cached bytes, and
the next job picks up the new file at its own job-start read.
The storage reaper never touches this path either ([sweep scope
is `.tmp/` + per-workspace job logs only](../modules/file_mgr/storage_reaper.rs)).

The launch-time TOML (mic catalogue, backbone catalogue, stream
binds, `[head.default]`) is operator-supplied via `--config <PATH>`
and lives OUTSIDE the workspace tree so it can be deployment-
managed independently.

### File and directory roles

| Path | Role | Schema | Atomicity invariants |
|---|---|---|---|
| `config.toml` | Hot-reloadable user-preference TOML | [`Config`](../modules/config.rs) | Atomic rewrite (tempfile + fsync + rename); auto-created on first boot if missing |
| `workspaces/<id>/workspace.json` | Hot core metadata; eagerly loaded into `ArcSwap<WorkspaceCore>` | [`WorkspaceCore`](#workspacecore) | Atomic rewrite via `put_atomic` (tempfile + fsync + rename + parent fsync); revision-bump precedes dataset byte mutation |
| `workspaces/<id>/heads.json` | Compact head index (<=2 entries); eagerly loaded into `ArcSwap<HeadIndex>` | [`HeadIndex`](#headindex) | The publish point for trained heads — committed AFTER `<head_id>.{mpk,json}` are renamed |
| `workspaces/<id>/datasets/<path>/<file>` | Daemon-owned dataset tree | raw bytes | Each accepted mutation advances `workspace.json.workspace_revision` BEFORE bytes change |
| `workspaces/<id>/converters/<path>/<file>` | Daemon-owned converter input tree | raw bytes | Same revision-before-bytes invariant as the dataset tree |
| `workspaces/<id>/heads/<head_id>.mpk` | Raw trained-head weights | Burn `.mpk` | Index-atomic publish: staged in `.tmp/`, fsynced, renamed BEFORE `heads.json` references it |
| `workspaces/<id>/heads/<head_id>.json` | Per-head manifest with inline labels | [`HeadManifest`](#headmanifest) | Same publish ordering as the matching `.mpk` |
| `workspaces/<id>/training_logs/<job_id>.jsonl` | Append-only train job events | one [`JobEvent`](#jobevent) per line | Writers flush at least once per second and on terminal state; no per-line fsync |
| `workspaces/<id>/converter_logs/<job_id>.jsonl` | Append-only convert job events | one `JobEvent` per line | Same as `training_logs/` |
| `workspaces/<id>/.tmp/` | Per-workspace staging for atomic rename | (opaque) | Every mutating writer fsyncs the tempfile before rename |
| `.tmp/delete-workspace-<job_id>/payload/...` | Staged payload of an in-flight async workspace delete | (opaque) | Boot recovery resumes the drain |
| `active/current.json` | Atomic active-generation pointer | [`ActiveCurrentPointer`](#activecurrentpointer) | Atomic rewrite via `put_atomic`; only published AFTER the pointed generation is fsynced |
| `active/generations/<activation_id>/head.mpk` | Materialized active-head weights (independent of any workspace) | Burn `.mpk` | Owned by the generation directory; deleting source workspace does not affect this |
| `active/generations/<activation_id>/labels.txt` | Materialized from manifest `labels[]` | `\n`-joined UTF-8 | Boot recovery regenerates from `manifest.labels[]` if missing or hash-mismatched |
| `active/generations/<activation_id>/manifest.json` | Active-head provenance | [`ActiveHeadManifest`](#activeheadmanifest) | Validated at every read (`validate()` enforces discriminator <-> runtime_head_id consistency) |
| `active/.tmp/<activation_id>/` | Activation staging directory | (mirror of generation) | Renamed into `generations/` before `current.json` rewrites |
| `var/run/acoustics_lab.sock` | Default UDS listener | (opaque) | Daemon creates parent dir at `0o700` if missing; warns on world-writable parents |
| `logs/acousticsd.log` | Daemon stderr/stdout (daily rotation) | text | 7-file retention |

### Crash consistency invariants

- **Dataset mutations**: `workspace.json.workspace_revision` is
  advanced BEFORE dataset bytes change.  A crash mid-write may
  leave a head conservatively `Stale` (its
  `workspace_revision.id` no longer matches the
  workspace's current `workspace_revision.id`) but never leaves
  a head `Current` after a dataset mutation.
- **Trained heads**: `heads.json` is the publish point.  The
  `<head_id>.{mpk,json}` files are renamed FIRST; only then
  does `heads.json` rewrite.  A crash before the index commit
  leaves only daemon-owned orphan residue, which boot recovery
  sweeps.  `heads.json` referencing a missing head file is a
  corruption error (cannot happen on the nominal path).
- **Active head**: the staging directory is renamed under
  `generations/` and fsynced BEFORE `current.json` rewrites.
  A crash before `current.json` commit leaves an orphan
  generation under `generations/` (swept on next activation /
  boot).  The previous generation is retained until the new
  pointer is durable.
- **Async deletes**: a tombstone (`.tmp/delete-*-<job_id>.json`)
  is written + fsynced BEFORE any rename / unlink.  Boot
  recovery completes interrupted drains by the tombstone alone.

## Schemas (full)

The shapes below are the source-of-truth on-disk JSON.  All
parse with `#[serde(deny_unknown_fields)]` (with the documented
exception of `ActiveHeadManifest`, where the `serde(flatten)`
+ internally-tagged enum combination cannot compose with
`deny_unknown_fields`; structural validation runs via
[`ActiveHeadManifest::validate`](../modules/common/workspace.rs)).

### `WorkspaceCore`

`<workspace>/workspace.json` — hot core metadata.

```json
{
  "id":                 "<UUIDv4>",
  "name":               "main",
  "tags":               ["pilot", "speech-commands"],
  "created_at":         "2026-05-07T12:34:56Z",
  "workspace_revision": { "id": 5, "at": "2026-05-07T13:00:00Z" },
  "head_count":         2
}
```

`head_count` is derived from `heads.json` and boot-repairable.
`workspace_revision.id` increments by one per accepted upload
or delete under EITHER tree (`datasets/` or `converters/`);
`workspace_revision.at` records the RFC3339 wall-clock.  `tags`
is mutable user metadata (max 32 entries; <= 64 UTF-8 bytes each;
Unicode case-insensitive uniqueness via `str::to_lowercase`).
Tag edits do NOT bump `workspace_revision`.

Hard size cap: `MAX_WORKSPACE_CORE_BYTES = 64 KiB`.

### `HeadIndex`

`<workspace>/heads.json` — compact head index.  Hard cap at 2
entries (sliding window, most-recent-first).  Each record is 6
fields; trainer/converter input metadata (dataset path,
training-cfg payload) lives in the per-head manifest.

```json
{
  "heads": [
    {
      "head_id":           "<UUIDv4>",
      "workspace_revision": { "id": 5, "at": "2026-05-07T13:00:00Z" },
      "sha256":            "<hex>",
      "n_classes":         12,
      "size_bytes":        26112,
      "created_at":        "2026-05-07T13:01:00Z"
    }
  ]
}
```

Failed jobs never appear here — only successful publishes.

### `HeadManifest`

`<workspace>/heads/<head_id>.json` — per-head manifest (8
fields).  The producer's input paths + numeric training cfg
live in the matching `{training,converter}_logs/<job_id>.jsonl`
durable record, not here.

```json
{
  "head_id":           "<UUIDv4>",
  "workspace_id":      "<UUIDv4>",
  "workspace_revision": { "id": 5, "at": "2026-05-07T13:00:00Z" },
  "sha256":            "<hex>",
  "n_classes":         12,
  "size_bytes":        26112,
  "created_at":        "2026-05-07T13:01:00Z",
  "labels":            ["cat", "dog", "..."]
}
```

`labels` is the canonical recovery source for
`active/generations/<id>/labels.txt`.  All three structs
(`WorkspaceCore`, `HeadIndex`, `HeadManifest`) carry
`#[serde(deny_unknown_fields)]`, so any unknown-field body on
disk parse-fails through `read_workspace_core` /
`read_head_index` / `read_head_manifest`.

### `ActiveCurrentPointer`

`<workspace_root>/active/current.json`.

```json
{ "activation_id": "<directory-name>" }
```

`activation_id` is the directory name only; not parsed as a
`HeadId` (the operator-facing fixture path
`misc/heads/default/` is a directory name, never an
identifier).  `deny_unknown_fields` fails closed on a
hand-edited or future-shape file.

### `ActiveHeadManifest`

`<workspace_root>/active/generations/<activation_id>/manifest.json`
— internally tagged on `origin`.

`origin = "head"`:
```json
{
  "origin":                  "head",
  "source_workspace_id":     "<UUIDv4>",
  "workspace_revision": { "id": 5, "at": "2026-05-07T13:00:00Z" },
  "source_head_id":          "<UUIDv4>",
  "runtime_head_id":         "<UUIDv4>",
  "sha256":                  "<hex>",
  "labels_sha256":           "<hex>",
  "n_classes":               12,
  "labels":                  ["cat", "dog", "..."],
  "activated_at":            "2026-05-07T13:05:00Z"
}
```

`origin = "default"` (no `source_*` fields):
```json
{
  "origin":          "default",
  "runtime_head_id": "00000000-0000-4000-8000-000000000000",
  "sha256":          "<hex>",
  "labels_sha256":   "<hex>",
  "n_classes":       20,
  "labels":          ["...", "..."],
  "activated_at":    "2026-05-07T13:05:00Z"
}
```

Structural invariants (enforced by [`ActiveHeadManifest::validate`](../modules/common/workspace.rs)):
- For `origin: head`, `runtime_head_id` MUST equal
  `source_head_id` so emitted `InferenceFrame.head_id` carries
  the trained head's recorded identity.
- For `origin: default`, `runtime_head_id` MUST equal the
  bundled-default UUID `00000000-0000-4000-8000-000000000000`.

`labels[]` is canonical recovery data for `labels.txt`; boot
regenerates the file if it is missing or its sha256 differs.

### `JobEvent`

One JSONL line in `<workspace>/{training,converter}_logs/<job_id>.jsonl`.

```json
{
  "seq":      42,
  "at":       "2026-05-07T13:01:23Z",
  "state":    "running",
  "progress": { "done": 3, "total": 10 },
  "message":  "starting epoch 4"
}
```

`seq` is monotonic per job.  `state` is one of `queued` |
`running` | `succeeded` | `failed` | `cancelled`; omitted when
the event is a progress / log-only update.  `progress.total`
is optional (clients render a percent only when present).
`message` is line-capped by `max_log_line_bytes` (default
8 KiB).
