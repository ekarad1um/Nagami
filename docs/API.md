# Web API

The `acousticsd` daemon exposes an HTTP/JSON control plane under
`/api/v1/*` and two binary-protobuf streaming endpoints under
`/stream/*`.  Both surfaces are mounted on a TCP listener (default
`127.0.0.1:8787`) and a Unix-domain socket listener (default
`<workspace_root>/var/run/acoustics_lab.sock`); per-listener
admission policies live in `[stream.tcp_policy]` /
`[stream.uds_policy]` of the dev TOML.

Workspaces own a daemon-only `datasets/` tree (path-validated
through [`AssetPath`](#assetpath)); trained heads are bounded
to a 2-slot per workspace; the active classifier head lives
under `<root>/active/` as an independent generation tree,
decoupled from any source workspace.  See
[ARCHITECTURE.md](ARCHITECTURE.md) for the lifecycle model and
[LAYOUT.md](LAYOUT.md) for the on-disk shape.

## Conventions

- Requests and responses are JSON unless noted (multipart for
  `/upload`; SSE for `/jobs/{id}/events`).
- Successful responses use HTTP 200 OK except where noted.
- Every error response is wrapped in:
  ```json
  { "error": "<message>", "code": "<machine_readable_code>" }
  ```
- Identifiers (`{id}`, `{job_id}`, `{head_id}`) are UUIDv4,
  lowercase, no braces.
- DTOs deny unknown fields; a typo'd key surfaces as 400 rather
  than silently using a default value.
- Producer endpoints (`/train`, `/convert`, `/upload` with
  caveats, `/active`) return immediately with the daemon-allocated
  identifiers; observe progress via `/jobs/...` and the durable
  JSONL log endpoints.

## Error envelope and codes

Every non-2xx response carries the same envelope:

```json
{ "error": "<human-readable message>", "code": "<short token>" }
```

The `code` field is one of the canonical taxonomy tokens (mapped
1-to-1 from [`ErrorKind`](../modules/common/error.rs)) plus a
small set of discriminator overrides for the cases dashboards
need to distinguish at a glance.

| Status | Code | When |
|---|---|---|
| 400 | `bad_request` | Malformed body, validation failed, [`AssetPath`](#assetpath) traversal, multipart field-order violation |
| 404 | `not_found` | Workspace, job, asset, or head missing |
| 405 | `method_not_allowed` | Path matched, verb did not |
| 409 | `conflict` | Generic conflict (sibling-name collision, terminal-state cancel) |
| 409 | `another_train_running` | A train job is already unfinished daemon-wide (`max_train_jobs = 1`) |
| 409 | `job_conflict` | An overlapping running job references this workspace or an ancestor/descendant dataset path |
| 409 | `event_gap` | SSE replay cursor is older than the in-memory ring; client backfills via the JSONL log endpoint |
| 425 | `too_early` | `?min_version=N` exceeds the live snapshot version (read-your-writes retry) |
| 429 | `too_many_requests` | Streaming connection cap reached |
| 500 | `internal` | Logic bug, unexpected IO, `spawn_blocking` join error |
| 501 | `not_implemented` | Recognised endpoint, currently stubbed |
| 503 | `unavailable` | Downstream service / device temporarily missing |

`event_gap` 409 bodies carry two extra fields so clients can
position the JSONL backfill exactly:

```json
{
  "error": "event ring overflow; backfill via /{training,converter}_logs",
  "code": "event_gap",
  "oldest_seq": 7,
  "latest_seq": 1024
}
```

## AssetPath

Every operator-supplied path resolved under a workspace's
`datasets/` tree (upload target, dataset GET, dataset DELETE,
train / convert input) is validated through [`AssetPath`](../modules/common/asset_path.rs):

- `/`-joined sequence of components.
- Each component matches the byte allowlist `[A-Za-z0-9._-]`,
  is non-empty, and does NOT start with `.` (rules out `.`,
  `..`, and `.hidden` with one rule).
- Total path length <= 256 bytes.
- Per-component length <= 255 bytes (filesystem `NAME_MAX` floor).
- Depth <= 8 components.
- Empty path rejected.
- Backslash, NUL, control bytes, non-ASCII bytes (`%`, `\n`,
  CJK, em-dash, ...) all fall outside the allowlist and reject.

Defence in depth: the route layer URL-decodes wildcard captures
before constructing an `AssetPath`, so smuggled `%2E%2E%2F` is
decoded to `../` and rejected via the leading-dot rule; an
undecoded `%` falls outside the allowlist and rejects too.

## Health

### `GET /api/v1/health`

Liveness ping for process supervisors.

Response: `{ "status": "ok" }`

## Workspace lifecycle

A workspace is a per-experiment directory under
`<workspace_root>/workspaces/<workspace_id>/`.  Hot metadata
lives in two small files (`workspace.json`,  `heads.json`)
that are eagerly loaded into per-workspace `ArcSwap` cells; the
list / summary paths NEVER walk `datasets/`.

### `POST /api/v1/workspace`

Create a new workspace.

Request:
```json
{
  "name": "pilot-1",
  "tags": ["smoke-test"]
}
```

Constraints: non-empty UTF-8, <=128 bytes (byte count, not
char count), no controls, no NUL, no path separators, no
leading/trailing whitespace.  Names are unique under Unicode
case-insensitive comparison via `str::to_lowercase` (simple case
folding: handles `Café`/`café` and Cyrillic/Greek case pairs;
does not normalize NFC vs NFD or apply locale-specific folding
like Turkish dotted/dotless I).  `tags` is an optional list of
strings under the same charset rules with a 64-byte per-tag cap
and 32-entry total cap.

Response:
```json
{
  "id": "<UUIDv4>",
  "name": "pilot-1",
  "tags": ["smoke-test"],
  "created_at": "2026-05-07T12:34:56Z",
  "workspace_revision": { "id": 0, "at": "2026-05-07T12:34:56Z" }
}
```

### `GET /api/v1/workspace`

List workspace identities.  Reads only the cached `workspace.json`
core per workspace; never walks `datasets/`.

Response:
```json
{
  "workspaces": [
    { "id": "<UUIDv4>", "name": "pilot-1", "created_at": "2026-05-07T12:34:56Z" }
  ]
}
```

### `GET /api/v1/workspace/{id}`

Workspace summary.  Returns the cached `workspace.json` core
plus the cached `heads.json` index (each entry derived against
the workspace's current `workspace_revision`).  Does NOT walk or
return dataset files; use [`GET /workspace/{id}/assets`](#get-apiv1workspaceidassets)
for that.

Response:
```json
{
  "id": "<UUIDv4>",
  "name": "pilot-1",
  "tags": ["smoke-test"],
  "created_at": "2026-05-07T12:34:56Z",
  "workspace_revision": { "id": 5, "at": "2026-05-07T13:00:00Z" },
  "heads": [
    {
      "head_id": "<UUIDv4>",
      "workspace_revision": { "id": 5, "at": "2026-05-07T13:00:00Z" },
      "sha256": "<hex>",
      "n_classes": 12,
      "size_bytes": 26112,
      "created_at": "2026-05-07T13:01:00Z",
      "status": "current"
    }
  ]
}
```

`status` is one of `current` (head's `workspace_revision.id`
matches the workspace's current `workspace_revision.id`) or
`stale`.

Per-head provenance (input paths + numeric training cfg) lives
only in `<workspace>/heads/<head_id>.json`
([`HeadManifest`](#headmanifest)) and in the matching
`converter_logs/<job_id>.jsonl` /
`training_logs/<job_id>.jsonl` files; this summary surface
intentionally omits it.

Errors: 404 `not_found` if the id does not exist.

### `PATCH /api/v1/workspace/{id}`

Atomic update of operator metadata. Name and tag edits do NOT advance `workspace_revision` or head freshness, as they are not workspace mutations.

Request:
```json
{
  "name": "pilot-2",
  "tags": ["prod"]
}
```

Requires at least one of `name` or `tags`. Returns the same response shape as `POST /api/v1/workspace`.

Errors: 400 `bad_request` if no fields are provided; 404 `not_found` if the workspace does not exist.

### `DELETE /api/v1/workspace/{id}`

Async workspace delete.  Stages the entire workspace tree under
`<workspace_root>/.tmp/delete-workspace-<job_id>/payload`,
returns the owning `job_id`, then drains the staged payload off
the request hot path in bounded batches.  Active inference is
not affected: the active generation owns independent bytes.

Response: `{ "job_id": "<UUIDv4>" }`

Errors: 404 `not_found` if the workspace does not exist;
409 `job_conflict` if any running job references the workspace
or any dataset path under it.

## Workspace assets

Both `datasets/` and `converters/` trees are daemon-owned:
file mutations only flow through `/upload` and
`/assets/{*path}` DELETE, with the top-level component
selecting which tree.  Operator-side rsync / scp into either
tree is unsupported.  Every accepted mutation under either
tree advances `workspace.json`'s `WorkspaceRevision` (one
counter spans both trees).

### `POST /api/v1/workspace/{id}/upload`

Single-file upload.  Multipart/form-data with two fields in
this exact order:

| Field | Type | Required | Notes |
|---|---|---|---|
| `path` | text, <= 320 B | yes | Workspace-rooted [`AssetPath`](#assetpath).  Must start with `datasets/` or `converters/`.  `converters/` requires at least one child component (`converters/<name>`).  `datasets/` requires at least two child components (`datasets/<class>/<file>`) because the trainer keys class labels off the first subdirectory of `datasets/`; deeper subtrees under a class are fine.  A leading `/` is accepted and stripped. |
| `file` | binary stream | yes | File body; capped at `max_upload_bytes` (default 256 MiB) |

`path` MUST precede `file`; the `file` field arriving first is
rejected with 400 (the streaming write needs the validated
destination before allocating a tempfile).  Legacy `kind` field
is rejected explicitly.

Response (`DatasetUploadReceipt`):
```json
{
  "path": "datasets/audio_dataset/cat/sample.wav",
  "sha256": "<hex>",
  "size_bytes": 24364,
  "workspace_revision_id": 6
}
```

The receipt's `path` field echoes the workspace-rooted form;
both `datasets/<...>` and `converters/<...>` produce the same
shape.

Pipeline: stream body to `<workspace>/.tmp/<random>` while
hashing, fsync, lock the workspace, conflict-check (only an
in-flight `WorkspaceDelete` for this workspace conflicts;
uploads, file-deletes, train, and convert all coexist),
atomically rewrite `workspace.json` with the next
`WorkspaceRevision`, rename tempfile into `<tree>/<path>`,
fsync the parent dir, publish the new core cache, unlock.

Errors: 400 `bad_request` if `path` is not under
`datasets/<...>` or `converters/<...>`, fails the dataset
class-folder depth gate (`datasets/<file>` is rejected; the
trainer needs `datasets/<class>/<file>`), or fails
[`AssetPath`](#assetpath) validation; 404 `not_found`
for a missing workspace; 409 `job_conflict` if a
`WorkspaceDelete` is in flight for this workspace; 413
`bad_request` if the body exceeds `max_upload_bytes`.

### `GET /api/v1/workspace/{id}/assets`

Paginated direct-child listing rooted at the workspace dir
(returns top-level entries `datasets`, `converters`, `heads`,
`training_logs`, `converter_logs`; internal `.tmp/` is filtered
out).  When `?path=` is supplied,
lists the named sub-tree (e.g. `?path=datasets/audio` lists
direct children of `<ws>/datasets/audio/`).  Reads filesystem
entries; does not take the workspace mutation mutex.

Query:
- `path` (optional) — workspace-rooted [`AssetPath`](#assetpath).
  Defaults to the workspace dir's top level.
- `offset` (optional u64) — default 0.
- `limit` (optional u64) — default 100, max 1000.

Response (`DatasetListing`):
```json
{
  "entries": [
    { "name": "cat",        "kind": "dir",  "size_bytes": null, "mtime": "2026-05-07T12:00:00Z" },
    { "name": "labels.txt", "kind": "file", "size_bytes": 113,  "mtime": "2026-05-07T12:01:23Z" }
  ],
  "total": 2,
  "offset": 0,
  "limit": 100
}
```

`mtime` is the filesystem's last-modification timestamp formatted
as RFC3339 UTC, present on every entry (file or directory).  A
birth-time field (`ctime` in some Unix vocabularies) is **not**
exposed: POSIX status-change time is not the intended semantic,
and Linux btime requires `statx(STATX_BTIME)` plumbing that is
not yet wired through.  Adding a future birth-time field is
non-breaking.

### `GET /api/v1/workspace/{id}/assets/{*path}`

Stream a regular file or list a directory's direct children.
Path is validated as [`AssetPath`](#assetpath); the route
layer URL-decodes wildcard captures before validation.

Streaming I/O does NOT hold the workspace mutation mutex
(`mutex-release-before-stream`); concurrent uploads to the
same workspace are not blocked by an in-flight read.
`Content-Type` is selected by extension only (no MIME sniffing
of bytes).

Response on a regular file: raw bytes; `Content-Type` from the
[MIME table](#mime-type-table-for-asset-get); `Content-Length`
populated.

Response on a directory: same `DatasetListing` shape as
`GET /assets?path=`.

#### Optional JSONL paging

For files whose name ends in `.jsonl` (the durable per-job event
backstop the daemon writes under `converter_logs/`, and — once
the training producer lands — `training_logs/`), the route
accepts two optional query parameters:

- `after_seq` (optional `u64`) — cursor; yields only events with
  `seq > after_seq`.  Default `0`.
- `limit` (optional `usize`) — page ceiling; default `200`,
  hard-clamped to `1000` server-side.

When either query parameter is set, the response shape becomes:

```json
{
  "events": [ /* JobEvent objects, one per line */ ],
  "next_after_seq": 42
}
```

`next_after_seq` is the cursor to pass back on the next call;
when the page is empty it echoes the caller's `after_seq` so a
poll that catches up reads `next_after_seq == after_seq`.

The JSONL paging branch only activates on `.jsonl` files and only
when at least one of the two query parameters is set; otherwise
the route stays in byte-stream mode (no `Range:` header support
yet — query-parameter and byte-range namespaces are reserved
orthogonally for a future implementation).

Errors: 404 `not_found` for a missing asset or workspace;
400 `bad_request` for an invalid path, if the resolved target is
neither a regular file nor a directory, if `?after_seq=` /
`?limit=` is set on a non-`.jsonl` file, or if either is set
when the resolved target is a directory.

### `DELETE /api/v1/workspace/{id}/assets/{*path}`

Workspace-asset delete dispatcher.  Two semantics share the
endpoint, picked by the path's top-level component.

#### Async tombstone+stage+drain — `datasets/...` and `converters/...`

Validates the path, asks the JobRegistry for the per-tree lease
(`DatasetDelete` for `datasets/...`, `ConverterDelete` for
`converters/...`), records a tombstone (filename prefix
`delete-assets-<job_id>` / `delete-converters-<job_id>`),
atomically rewrites `workspace.json` with the next
`WorkspaceRevision`, renames the target into the matching staging
dir, then drains the staged payload in bounded batches off the
lock.

The bare tree name (`DELETE /assets/datasets`,
`DELETE /assets/converters`) wipes the whole tree: the entire
`<workspace>/<tree>/` directory is renamed into staging and the
empty tree dir is recreated so the workspace's structural shape
survives.

Response: **202 Accepted** + `{ "job_id": "<UUIDv4>" }`.  The
202 status reflects the async semantic: the rename + tombstone
landed durably under the per-workspace mutex, but the staged
drain runs in the background.  Clients can poll
`GET /jobs/{job_id}` or stream `GET /jobs/{job_id}/events` for
terminal state.

#### Sync wipe — `training_logs[/...]` and `converter_logs[/...]`

JSONL backstop for per-job training / convert events.  Refuses
with `409 conflict` while a producer (Train for `training_logs`,
Convert for `converter_logs`) is currently active in the same
workspace; otherwise unlinks the matching `.jsonl` file
(`DELETE /assets/training_logs/<id>.jsonl`) or every `.jsonl`
file in the directory
(`DELETE /assets/training_logs` whole-dir wipe).  No tombstone,
no revision bump — logs are an internal backstop, not workspace
state.

Response: **200 OK** + `{ "removed": <usize> }` (count of
`.jsonl` files unlinked; `0` is a legitimate response for "the
file / dir was already empty or the requested file did not
exist").  200 (vs the async path's 202) signals "done before
the response returns".

#### Errors

- 400 `bad_request` for an invalid path or a top-level outside
  `datasets` / `converters` / `training_logs` / `converter_logs`.
- 400 `bad_request` for a non-`.jsonl` file under a log dir, or
  a nested path under a log dir
  (e.g. `training_logs/sub/x.jsonl`).
- 404 `not_found` for a missing workspace.  Missing async-path
  targets surface as 404 `not_found` (the lib symlink-stat
  returns ENOENT); missing sync-path targets are not 404 — they
  reply `200 OK` with `removed: 0`.
- 409 `job_conflict`:
  - `WorkspaceDelete` in flight for this workspace (async path).
  - Active Train job for `training_logs` (sync path).
  - Active Convert job for `converter_logs` (sync path).

## Trained heads

Per-workspace head storage is bounded to **2 entries** (sliding
window by completion time, most-recent-first).  Producers
(`/train`, `/convert`) commit a head record only after the
on-disk bytes + manifest are durable and `heads.json` rewrites.

### `GET /api/v1/workspace/{id}/heads`

List heads with derived freshness.  Cache-only.

Response:
```json
{
  "heads": [
    {
      "head_id":           "<UUIDv4>",
      "workspace_revision": { "id": 5, "at": "2026-05-07T13:00:00Z" },
      "sha256":            "<hex>",
      "n_classes":         12,
      "size_bytes":        26112,
      "created_at":        "2026-05-07T13:01:00Z",
      "status":            "current"
    }
  ]
}
```

### `GET /api/v1/workspace/{id}/heads/{head_id}`

Per-head manifest.  Returns the
[`HeadManifest`](../modules/common/workspace.rs) body with
inline class labels.  Surfaces 404 if the `head_id` is not in
the cached `heads.json` index, even when an orphan
`<head_id>.json` happens to exist on disk (daemon-owned residue
swept by boot recovery).

Response:
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

The manifest carries 8 fields; the producer's input paths +
numeric training cfg live in the matching
`training_logs/<job_id>.jsonl` /
`converter_logs/<job_id>.jsonl` durable record only.

### `DELETE /api/v1/workspace/{id}/heads/{head_id}`

Synchronously remove a single head.  Updates `heads.json`,
atomic-rewrites `workspace.json` with the refreshed
`head_count`, and unlinks `heads/<head_id>.{mpk,json}` under
the per-workspace mutation mutex.

Response: `{ "deleted_head_id": "<UUIDv4>" }`

Errors: 404 `not_found`; 409 `job_conflict` if a running job
references this workspace.

## Active head

The active classifier head lives under `<workspace_root>/active/`
as a self-contained generation tree (`head.mpk`, `labels.txt`,
`manifest.json`).  Activation copies bytes from the source so
deleting the source workspace does not break inference.  Only
two on-disk generations are retained (current + previous);
older generations are pruned after `current.json` is durable.

### `POST /api/v1/active`

Activate a workspace head OR reset to the bundled default.
Idempotent force-set: re-activating the same head produces a
fresh `activation_id` so the operator can confirm the write
landed via `GET /active`.

Request body shape A (workspace head):
```json
{ "workspace_id": "<UUIDv4>", "head_id": "<UUIDv4>" }
```

Request body shape B (bundled default):
```json
{ "default": true }
```

`default: false` is rejected explicitly with 400.

Pipeline: acquire the global active mutex, stage the new
generation under `active/.tmp/<activation_id>/`, copy
`<head_id>.mpk` (or the bundled-default `head.mpk`),
materialize `labels.txt` from the manifest's inline `labels[]`,
compute both sha256s, write the new `manifest.json`, pre-load
+ validate the staged head on a blocking worker, atomically
rename the staging dir into `active/generations/<activation_id>/`,
fsync, atomically rewrite `active/current.json`, fsync `active/`,
install the prevalidated runtime candidate into `HotHead`,
then prune older generations (best-effort; failure leaves
residue for the next boot sweep).

Response (`ActiveResp`):
```json
{
  "sha256":                  "<hex>",
  "labels_sha256":           "<hex>",
  "n_classes":               12,
  "labels":                  ["cat", "dog", "..."],
  "runtime_head_id":         "<UUIDv4>",
  "activated_at":            "2026-05-07T13:05:00Z",
  "origin":                  "head",
  "source_workspace_id":     "<UUIDv4>",
  "source_workspace_revision": { "id": 5, "at": "2026-05-07T13:00:00Z" },
  "source_head_id":          "<UUIDv4>",
  "activation_id":           "<UUIDv4>"
}
```

For `origin: "default"` the `source_*` fields are absent and
`runtime_head_id` is the bundled-default UUID
(`00000000-0000-4000-8000-000000000000`).

Errors: 400 `bad_request` if the request body is invalid or
`default: false`; 404 `not_found` if the workspace or head
id is not in the cached `heads.json` index; 500 `internal` if
the staged head fails to load/validate (incompatible Burn
shape, sha256 mismatch).

### `GET /api/v1/active`

Read the current active generation's manifest, augmented with
`source_workspace_alive` (Head origin only — cheap stat of
the source workspace dir).  Wait-free; never takes the active
mutex.

Response (Head origin):
```json
{
  "sha256":                  "<hex>",
  "labels_sha256":           "<hex>",
  "n_classes":               12,
  "labels":                  ["cat", "dog", "..."],
  "runtime_head_id":         "<UUIDv4>",
  "activated_at":            "2026-05-07T13:05:00Z",
  "origin":                  "head",
  "source_workspace_id":     "<UUIDv4>",
  "source_workspace_revision": { "id": 5, "at": "2026-05-07T13:00:00Z" },
  "source_head_id":          "<UUIDv4>",
  "activation_id":           "<UUIDv4>",
  "source_workspace_alive":  true
}
```

For `origin: "default"` the `source_*` fields and
`source_workspace_alive` are absent.

Errors: 404 `not_found` if `current.json` is absent (no
activation has been performed and boot recovery has not yet
materialised the bundled default — should not happen on a
healthy daemon).

## Inference cadence + runtime view

### `GET /api/v1/inference`

Read the live inference cadence.

Response:
```json
{ "cfg": { "hop_samples": 1024, "top_k": 3 } }
```

### `POST /api/v1/inference`

Mutate `hop_samples`, `top_k`, or both.  Omitted fields keep
the current value.

Request:
```json
{ "hop_samples": 2048, "top_k": 5 }
```

Constraints: `hop_samples` in `1..=33024` (`WaveformLen * 3 / 4`);
`top_k` in `1..=64`.

Response: same shape as `GET /api/v1/inference`, with the
merged config.

## Producers (train, convert)

### `POST /api/v1/workspace/{id}/train`

Start the (single) train job.  Body is the *flattened*
`TrainingCfg` fields (no wrapper, no `dataset_path`).  The
trainer walks the fixed `<workspace>/datasets/` root.

Request:
```json
{
  "epochs":        12,
  "batch_size":    32,
  "learning_rate": 0.001,
  "seed":          42
}
```

The body shape is `{epochs: u32, batch_size: u32,
learning_rate: f32, seed?: u64}` with `deny_unknown_fields`,
so any unknown key is rejected at the wire boundary; `seed` is
optional and may be omitted.

The trainer discovers labels from non-hidden direct child
directories under `<workspace>/datasets/`, sorted by canonical
byte order.  Empty class folders, duplicate labels (under
ASCII case-insensitive comparison), unreadable folders, and
non-hidden non-directory root entries fail closed with
`bad_dataset` (HTTP 400).  Mid-walk file disappearance surfaces
as `dataset_read_failure` (HTTP 500 -- the daemon-owned
`datasets/` tree is not operator-recoverable mid-job).

Lazy per-batch FD opens cap concurrent FDs at
`batch_size × parallel_loaders` regardless of dataset size.
The deployment-bundled backbone at
`<workspace_root>/backbone/backbone.mpk` is the single source.

Response:
```json
{
  "head_id": "<UUIDv4>",
  "job_id":  "<UUIDv4>"
}
```

Errors: 400 `bad_request` for body validation failures
(out-of-range numerics, unknown keys) or `bad_dataset` for
dataset-shape rejections; 404 `not_found` if the workspace does
not exist; 409 `another_train_running` if a train job is
already unfinished daemon-wide; 409 `job_conflict` only if a
`WorkspaceDelete` is in flight for this workspace.

### `GET /api/v1/workspace/{id}/training`

List training jobs for this workspace. Returns exactly the jobs for the given workspace id.

Response: `{ "jobs": [ ... ] }` where entries are `JobSnapshot`s.

### `GET /api/v1/workspace/{id}/training/{job}`

Query the state of a single training job scoped to the workspace.

Response: A single `JobSnapshot` object.

### `DELETE /api/v1/workspace/{id}/training/{job}`

Cancel a running training job. Stops epoch processing as soon as possible.

Response: `{ "ok": true }`

### `POST /api/v1/workspace/{id}/convert`

Convert an operator-uploaded TFJS bundle into a workspace head.
Body is internally tagged on `converter_type`; each path field
is converter-rooted (slashless sub-path; a legacy leading `/`
is accepted and stripped).  Resolves below
`<workspace>/converters/`.

Request (`converter_type = "tfjs"`):
```json
{
  "converter_type":  "tfjs",
  "model_json_path": "tfjs/model.json",
  "shards":          ["tfjs/group1-shard1of2", "tfjs/group1-shard2of2"],
  "labels_path":     "tfjs/labels.txt",
  "labels_format":   "lines"
}
```

`converter_type` selects the converter and dispatches to a
per-variant payload struct that validates with
`deny_unknown_fields` after dispatch; an unknown or omitted
`converter_type` parse-fails with HTTP 400.  Each path is a
[`ConverterPath`](#converterpath) — a converter-rooted
[`AssetPath`](#assetpath)-shaped sub-path; an optional leading
`/` is stripped.  `labels_format` is `lines` (one label per
line) or `tfjs_metadata` (extract `wordLabels` from a
Teachable-Machine `metadata.json`).  Concurrent converts queue
on the convert semaphore (`max_convert_jobs`, default 1).

Response:
```json
{
  "head_id": "<UUIDv4>",
  "job_id":  "<UUIDv4>"
}
```

Errors: 400 `bad_request` for body validation failures
(missing or unknown `converter_type`, empty / lone-slash /
traversal on any path, shard cardinality); 404 `not_found` if
the workspace or any input file is missing;
409 `another_convert_running` if a convert job is already
unfinished daemon-wide; 409 `job_conflict` only if a
`WorkspaceDelete` is in flight for this workspace.

#### `ConverterPath`

Canonical wire form: `<sub>` where `<sub>` is a slash-joined
[`AssetPath`](#assetpath)-shaped path.  Internally the daemon
stores the workspace-rooted form `converters/<sub>` and
resolves via `<workspace>/converters/<sub>`.  A single leading
`/` is accepted and stripped before validation.  Empty input
(`""` or `"/"`) and traversal sequences fail closed at
deserialize.

## Jobs

The JobRegistry stores a memory-only bounded history of every
running job plus the newest terminal jobs (capped by
`max_recent_jobs = max_running_jobs + 1`, default 4).  `GET
/jobs` and `GET /jobs/{job_id}` never open log files; for
durable history, page the matching JSONL backstop via
`GET /workspace/{id}/assets/training_logs/{job_id}.jsonl?after_seq=&limit=`
or
`GET /workspace/{id}/assets/converter_logs/{job_id}.jsonl?after_seq=&limit=`.

### `GET /api/v1/jobs`

Memory-only list of recent typed job snapshots, newest-first.

Query:
- `limit` (optional usize) — clamps to `max_recent_jobs`
  server-side.

Response: `Vec<JobSnapshot>`.  A `dataset_delete` entry (the only
job type that carries `target_path`):
```json
{
  "job_id":      "<UUIDv4>",
  "job_type":    "dataset_delete",
  "workspace_id": "<UUIDv4>",
  "target_path":  "datasets/audio_dataset/cat",
  "state":        "running",
  "progress":    { "done": 3, "total": 10 },
  "result":       null,
  "last_seq":     42,
  "updated_at":  "2026-05-07T13:01:23Z"
}
```

A `train` / `convert` / `workspace_delete` entry omits
`target_path` entirely (the field is skipped when absent rather
than serialised as `null`):
```json
{
  "job_id":      "<UUIDv4>",
  "job_type":    "train",
  "workspace_id": "<UUIDv4>",
  "state":        "running",
  "progress":    { "done": 3, "total": 10 },
  "result":       null,
  "last_seq":     42,
  "updated_at":  "2026-05-07T13:01:23Z"
}
```

`job_type`: `train` | `convert` | `dataset_delete` |
`converter_delete` | `workspace_delete`.  `state`:
`queued` | `running` | `succeeded` | `failed` | `cancelled`.
`progress.total` is optional — clients render a percent only
when `total` is present.  `target_path` carries the
workspace-rooted display path for `dataset_delete` /
`converter_delete` jobs (e.g. `datasets/audio/cat`,
`converters/tfjs/model.json`); train, convert, and
workspace-delete jobs have no single primary target and the
field is absent from their wire shape.  Clients must treat
absent and `null` as equivalent for backwards compatibility.

### `GET /api/v1/jobs/{job_id}`

Single typed job snapshot.

Response: one `JobSnapshot` (same shape as the list entries).

Errors: 404 `not_found` if the job id is not retained.

### `GET /api/v1/jobs/{job_id}/events`

SSE stream for progress + log events.  Replays in-memory ring
events strictly after `after_seq`, then follows the broadcast
channel until terminal state OR client disconnect.  KeepAlive
fires every 15 s to defeat idle timeouts.

Query:
- `after_seq` (optional u64) — replay events with `seq >
  after_seq`.  Default 0 (fresh subscribe).
- `logs` (optional bool) — include log-line events.  Default
  `true`.

Response: `text/event-stream`.  Each frame:
```text
event: job
data: {"seq": 42, "at": "2026-05-07T13:01:23Z", "state": "running",
       "progress": {"done": 3, "total": 10}, "message": "..."}
```

Errors: 404 `not_found` if the job id is not retained;
409 `event_gap` if `after_seq` is older than the in-memory
ring (body carries `oldest_seq` / `latest_seq` so the client
can backfill via the JSONL log endpoint before reconnecting).

## Durable job logs

Train and convert events are appended as JSONL to
`<workspace>/training_logs/<job_id>.jsonl` and
`<workspace>/converter_logs/<job_id>.jsonl` respectively.
Writers flush at least once per second and on terminal state
(no per-line fsync).  Lines are capped at `max_log_line_bytes`
(default 8 KiB).

Reads and wipes go through the unified `/assets/{*path}`
surface; there is no dedicated `/training_logs` or
`/converter_logs` route.

| Operation                       | URL                                                                              |
|---------------------------------|----------------------------------------------------------------------------------|
| Page training-log JSONL         | `GET /workspace/{id}/assets/training_logs/{job_id}.jsonl?after_seq=&limit=`      |
| Page converter-log JSONL        | `GET /workspace/{id}/assets/converter_logs/{job_id}.jsonl?after_seq=&limit=`     |
| Wipe one training-log file      | `DELETE /workspace/{id}/assets/training_logs/{job_id}.jsonl`                     |
| Wipe one converter-log file     | `DELETE /workspace/{id}/assets/converter_logs/{job_id}.jsonl`                    |
| Wipe entire training-log dir    | `DELETE /workspace/{id}/assets/training_logs`                                    |
| Wipe entire converter-log dir   | `DELETE /workspace/{id}/assets/converter_logs`                                   |

Page reads return `{ "events": [...], "next_after_seq": N }`
(200 OK); wipes return `{ "removed": N }` (200 OK); both refuse
with 409 `job_conflict` while the matching producer (Train for
`training_logs`, Convert for `converter_logs`) is active in the
same workspace.  See
`GET /api/v1/workspace/{id}/assets/{*path}` and
`DELETE /api/v1/workspace/{id}/assets/{*path}` above for the
canonical surface.

## Mic surface

The mic surface has two layers: an immutable `catalogue` (which
mics and channels exist, set from the launch TOML at boot) and
a mutable `policy` (which mic and channel are active right
now).

`MicPolicy` has two independently-tagged sub-discriminators:

- `mic`: `{ "kind": "first_available" }` (walk candidates in
  declaration order) or `{ "kind": "fixed", "id": "<MicId>" }`.
- `channel`: `{ "kind": "auto" }` (RMS-arbitrate the candidate's
  channel whitelist) or `{ "kind": "fixed", "channel": <u16> }`.

### `GET /api/v1/mic`

Read both layers + version stamp.

Query:
- `min_version` (optional u64) — read-your-writes gate.

Response:
```json
{
  "catalogue": {
    "candidates": [
      { "id": "default-mock", "channels": [0],
        "source": { "kind": "mock", "period_size": 512, "sample_rate": 44100, "waveforms": [...] } }
    ]
  },
  "policy": {
    "mic":     { "kind": "first_available" },
    "channel": { "kind": "auto" }
  },
  "version": 42
}
```

### `POST /api/v1/mic`, `POST /api/v1/mic/policy`

Swap the active mic policy.  The two paths are aliases.

Request:
```json
{
  "policy": {
    "mic":     { "kind": "fixed", "id": "default-mock" },
    "channel": { "kind": "fixed", "channel": 0 }
  }
}
```

The submitted policy is validated against the live catalogue;
an unknown `id` or out-of-whitelist `channel` returns 400.

Response: same shape as `GET /api/v1/mic`, with the new
`version`.

## Status

### `GET /api/v1/status`

Daemon-wide snapshot.  Wait-free; the request path does not
touch sysinfo or take any lock.

Response (`StatusSnapshot`):
```json
{
  "cpu_pct":                              4.2,
  "mem_rss_kb":                           78848,
  "disk_free_kb":                         1234567,
  "metrics_age_ms":                       42,
  "metrics_stale":                        false,
  "uptime_s":                             12345,
  "subsystems":                           { "audio_capture": HeartbeatView, "...": "..." },
  "broadcast_audio_messages_dropped":     0,
  "broadcast_inference_messages_dropped": 0,
  "workspace": {
    "assets_uploaded_total":              42,
    "bytes_uploaded_total":               104857600,
    "workspace_core_writes_total":        87,
    "head_index_writes_total":            8,
    "dataset_mutations_rejected_total":   3,
    "converter_mutations_rejected_total": 1,
    "workspace_core_write_p99_us":        1234,
    "head_index_write_p99_us":            789,
    "job_events_dropped_total":           0,
    "sse_clients_current":                1,
    "boot_orphans_swept_total":           2
  }
}
```

`workspace.converter_mutations_rejected_total` mirrors
`dataset_mutations_rejected_total`: an admission rejection on
a `converters/<...>` upload or delete increments the converter
counter; a `datasets/<...>` rejection increments the dataset
counter.  The two trees are dispatched by
[`AssetTree`](../modules/file_mgr/dataset.rs) so operators can
distinguish per-tree contention.

`workspace.assets_uploaded_total` semantics are
workspace-rooted: `upload_workspace_file` increments it for
both `datasets/<...>` and `converters/<...>` paths.

`HeartbeatView`:
```json
{
  "healthy":         true,
  "detail":          "free-form one-line status",
  "age_ms":          123,
  "stale":           false,
  "degraded_reason": "no_backbone"
}
```

`degraded_reason` is omitted when the subsystem is not in a
degraded state.  `stale` is true iff `age_ms > HEALTH_STALE_AFTER`
(currently 5 s).  `metrics_stale` is true iff the host-metrics
sample is older than `METRICS_STALE_AFTER` OR no sample has
been published yet.

The `workspace` block holds workspace-side counters.  All
counters are cumulative since daemon start;
`workspace_core_write_p99_us` and `head_index_write_p99_us`
are derived on demand from a bounded ring of recent samples
(most-recent 256 writes).

## Streaming

Two WebSocket endpoints publish binary-protobuf `Envelope`
messages.  Each endpoint is also reachable over the daemon's
UDS listener with identical wire shape (transport policy may
differ).

The full wire-format contract — subprotocol negotiation,
framing, versioning — lives in [`PROTO.md`](PROTO.md).

### `GET /stream/audio` (WebSocket)

Subscribe to the Opus-encoded audio fan-out.

Headers:
- `Sec-WebSocket-Protocol: acoustics` (required by default; see
  `[stream.tcp_policy].require_subprotocol`).

Each WS binary message is one prost-encoded `Envelope`:
```text
Envelope {
  payload: AudioFrame {
    seq:                    u64,
    t_us_capture_monotonic: optional u64,
    t_us_publish_unix:      optional u64,
    sample_rate:            optional u32 (= 48000),
    frame_duration_ms:      optional u32 (= 20),
    codec.opus:             bytes (libopus, <= 4000 B per Opus spec),
  }
}
```

Backpressure: if the subscriber falls behind the broadcast
capacity (default 64 frames) the daemon closes the connection
with WS code 1011 and reason `"lagged N"`.

### `GET /stream/infer` (WebSocket)

Subscribe to the inference fan-out.

Wire shape:
```text
Envelope {
  payload: InferenceFrame {
    seq:                    u64,
    t_us_capture_monotonic: optional u64,
    t_us_publish_unix:      optional u64,
    top_k:                  repeated TopK { class_idx, label, prob },
    head_id:                optional string,
    head_version:           optional u64,
  }
}
```

`(head_id, head_version)` is captured atomically from the
engine's `HotHead::snapshot_with_version`, so receivers can
disambiguate generations of the same head.

Backpressure: same as `/stream/audio`.

## MIME type table for asset GET

| Extension | `Content-Type` |
|---|---|
| `.json` | `application/json` |
| `.txt` | `text/plain; charset=utf-8` |
| `.tar.gz`, `.tgz` | `application/gzip` |
| `.zip` | `application/zip` |
| `.wav` | `audio/wav` |
| `.mpk` | `application/octet-stream` |
| `.rknn` | `application/octet-stream` |
| `.bin` | `application/octet-stream` |
| (other) | `application/octet-stream` |

Single helper [`content_type_from_path`](../modules/file_mgr/mime.rs);
inspected only on extension, no MIME sniffing of bytes.

## Read-your-writes

Mutating endpoints that surface a monotonic `version: u64`
support `?min_version=N` on the matching read; the daemon
returns 425 `too_early` (without blocking) until the live
snapshot reaches `N`, and callers retry after a short delay.

Affects: `GET /api/v1/mic` (paired with `POST /api/v1/mic` and
`POST /api/v1/mic/policy`).

## Curl recipes

Health probe:
```bash
curl http://127.0.0.1:8787/api/v1/health
```

Create workspace, upload a dataset file, train, activate:
```bash
curl -X POST http://127.0.0.1:8787/api/v1/workspace \
  -H 'Content-Type: application/json' \
  -d '{"name": "pilot-1"}'
# -> { "id": "<WS>", "name": "pilot-1", ... }

curl -X POST http://127.0.0.1:8787/api/v1/workspace/<WS>/upload \
  -F 'path=datasets/cat/001.wav' \
  -F 'file=@./001.wav'

curl -X POST http://127.0.0.1:8787/api/v1/workspace/<WS>/train \
  -H 'Content-Type: application/json' \
  -d '{"epochs":12,"batch_size":32,"learning_rate":0.001}'
# -> { "head_id": "<HEAD>", "job_id": "<JOB>" }

curl -X POST http://127.0.0.1:8787/api/v1/active \
  -H 'Content-Type: application/json' \
  -d '{"workspace_id":"<WS>","head_id":"<HEAD>"}'
```

Convert a TFJS bundle:
```bash
# Upload converter inputs to the converters/ tree.
for f in model.json group1-shard1of2 group1-shard2of2 metadata.json; do
  curl -X POST http://127.0.0.1:8787/api/v1/workspace/<WS>/upload \
    -F "path=converters/tfjs/${f}" \
    -F "file=@./${f}"
done

# Convert; every path field is converter-rooted (slashless
# canonical form; a leading `/` is accepted and stripped).
curl -X POST http://127.0.0.1:8787/api/v1/workspace/<WS>/convert \
  -H 'Content-Type: application/json' \
  -d '{
    "converter_type":  "tfjs",
    "model_json_path": "tfjs/model.json",
    "shards":          ["tfjs/group1-shard1of2", "tfjs/group1-shard2of2"],
    "labels_path":     "tfjs/metadata.json",
    "labels_format":   "tfjs_metadata"
  }'
# -> { "head_id": "<HEAD>", "job_id": "<JOB>" }
```

Tail a job's events:
```bash
curl -N http://127.0.0.1:8787/api/v1/jobs/<JOB>/events?after_seq=0
```

Page durable train logs:
```bash
curl 'http://127.0.0.1:8787/api/v1/workspace/<WS>/assets/training_logs/<JOB>.jsonl?after_seq=0&limit=200'
```

Reset to the bundled default head:
```bash
curl -X POST http://127.0.0.1:8787/api/v1/active \
  -H 'Content-Type: application/json' \
  -d '{"default": true}'
```

Subscribe to inference frames over WebSocket:
```bash
websocat -B 65536 \
  --protocol acoustics \
  ws://127.0.0.1:8787/stream/infer
```
