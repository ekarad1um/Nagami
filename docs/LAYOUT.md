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
|   |-- etc/dev.toml               -- daemon config (local dev)
|   |-- heads/
|   |   |-- 00000000-default/      -- bundled default head, copied into active/ on first boot
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
| `00000000-default/{head.mpk, labels.txt}` | 24 KiB + 113 B | 20-class Speech-Commands bundled-default; the daemon copies these into `active/generations/<activation_id>/` on first boot or when `POST /active {default: true}` runs |

The fixture directory name `00000000-default/` is operator-facing
only; the bundled default's runtime identity is the valid
UUID-v4 `00000000-0000-4000-8000-000000000000`.  The directory
name is never parsed as a `HeadId`.

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

### `misc/etc/dev.toml`

Local-dev daemon config.  Points at `misc/workspace/` for the
workspace root and `misc/share/acousticsd.sock` for the UDS
listener.  Paths resolve relative to the daemon's CWD.  No
persisted active-head pointer here — `<workspace_root>/active/`
is the single source.

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

```
<workspace_root>/
    backbone/
        backbone.mpk                              -- shared default; deployment-managed
        backbone.meta.json                        -- {sha256, n_features}
    workspaces/<workspace_id>/
        workspace.json                            -- hot core metadata (WorkspaceCore)
        heads.json                                -- compact 2-slot head index (HeadIndex)
        datasets/                                 -- daemon-owned dataset tree (nested folders allowed)
            <path>/<file>
        converters/                               -- daemon-owned converter input tree (nested folders allowed)
            <path>/<file>
        heads/                                    -- max 2 trained heads
            <head_id>.mpk                         -- raw weights (Burn .mpk)
            <head_id>.json                        -- per-head manifest (HeadManifest)
        training_logs/<job_id>.jsonl              -- append-only train job events
        converter_logs/<job_id>.jsonl             -- append-only convert job events
        .tmp/                                     -- per-request staging for atomic rename + delete-assets-* / delete-converters-* tombstones
    .tmp/                                         -- root staging for async workspace deletes
    active/
        current.json                              -- {activation_id}; atomic active generation pointer
        generations/                              -- retain current + previous generation only
            <activation_id>/
                head.mpk                          -- materialized active-head weights
                labels.txt                        -- materialized from manifest `labels[]`
                manifest.json                     -- ActiveHeadManifest body
        .tmp/                                     -- activation staging
    var/run/
        acoustics_lab.sock                        -- default UDS listener (configurable)
    logs/
        acousticsd.log                            -- daily rotated; 7-file retention
```

`backbone/` is loaded at boot from a deployment-bundled file.
Operators do not upload backbones via the API.

### File and directory roles

| Path | Role | Schema | Atomicity invariants |
|---|---|---|---|
| `backbone/backbone.mpk` | Frozen feature extractor (shared by every workspace and the runtime engine) | Burn `.mpk` | Deployment-supplied; daemon does not mutate |
| `backbone/backbone.meta.json` | `{sha256, n_features}` companion | JSON | Deployment-supplied |
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
is mutable user metadata (max 32 entries; <= 64 bytes each;
ASCII case-insensitive uniqueness).  Tag edits do NOT bump
`workspace_revision`.

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
`misc/heads/00000000-default/` is a directory name, never an
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
