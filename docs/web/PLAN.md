# Web Frontend Implementation Plan

> Implementation lives at [`web/`](../../web/). Spec at [`docs/web/ARCHITECTURE.md`](./ARCHITECTURE.md).

## Context

The acoustics_lab daemon (`acousticsd`) is mature: REST at `/api/v1/*`, two binary-protobuf WebSocket streams (`/stream/audio`, `/stream/infer`), SSE for job events, async job model, revision-tracked workspaces, 2-slot head storage with atomic hot-swap. Backend serves no static files. `web/` is a clean slate.

The frontend spec defines eight modules across three tabs (Dashboard, Workspace, Converter) plus a persistent Health Badge and floating Tiny Dashboard. The job is to build a slim SPA that lets a single operator record/edit audio clips, fine-tune classification heads, hot-swap weights, and observe live inference — all locally, with zero-latency dataset playback and no backend changes.

Pacing is deliberate: ample time, step-by-step. We will not pursue testing/benchmarking/CI infrastructure ([ARCHITECTURE.md §E.5](./ARCHITECTURE.md)).

## Locked decisions

- **Stack**: SvelteKit (latest) + TypeScript + Vite + Tailwind CSS v4. `@sveltejs/adapter-static`. Vite dev proxy → 127.0.0.1:8787.
- **Rendering**: pure-client SPA. Root layout sets `prerender = true; ssr = false` so build emits a static shell that hydrates browser-only.
- **No SSR/Node runtime**: build artifact is a static `web/build/` deployable to any static host or reverse-proxied.
- **Components**: hand-rolled Tailwind primitives. **Extracted only after a pattern is copy-pasted twice in real features.** No headless lib, no kit.
- **History**: memory ring via `GET /api/v1/jobs` for live state + JSONL backfill via `GET /api/v1/workspace/{id}/assets/training_logs/<job>.jsonl?after_seq=&limit=` for durability.
- **Converter workspaces**: marker tag `__converter__`, auto-name `converter-<uuid8>`. Client-side filter (backend treats all workspaces uniformly).
- **No tests, no benchmarks, no CI**.

## Strategy: vertical slices

We **do not** build full horizontal layers (audio infra → IDB → stores → components → modules). Each layer's correctness only becomes visible when an actual feature wires it end-to-end; building all three before the first feature compounds undetected integration risk.

Instead: five vertical slices, each delivering working user value. Shared foundations are extracted only when a second slice needs them.

### Slice A — Dashboard MVP (live audio + inference + health)

**User outcome**: open the app, see waveform/spectrogram of mic input, see live top-K classifications, see daemon health badge.

Deliverables:
1. SvelteKit scaffolding: `package.json`, `vite.config.ts`, `svelte.config.js`, `tsconfig.json`, `tailwind.config.js`, `postcss.config.js`, `app.html`, `app.css`, path aliases (`$lib`, `$proto`).
2. `src/routes/+layout.ts` → `prerender = true; ssr = false`. `+layout.svelte` → tab shell + badge slot.
3. Vite dev proxy for `/api`, `/stream/audio` (with `ws: true`), `/stream/infer` (with `ws: true`) to `127.0.0.1:8787`.
4. Proto decoder: a hand-written ~150 LOC decoder at [src/lib/stream/proto.ts](../../web/src/lib/stream/proto.ts). The schemas are tiny (4 messages, 14 fields total) and stable per [PROTO.md](../PROTO.md) (no third-party peers; wire-breaking changes ship as full replacements). Source-of-truth remains [modules/proto/*.proto](../../modules/proto); the TS decoder is kept in 1:1 correspondence by inspection — desync surfaces as a runtime decode warning, not a build break, but the cost of a codegen step + Long.js runtime + 200 KB of protobufjs/ts-proto outweighs the safety gain at this scope.
5. Fetch wrapper `src/lib/api/http.ts`: typed JSON helper with error envelope parsing (`{ error, code }`), default 5 s timeout, surfaces `code` on rejection for store-level handling.
6. WS client `src/lib/stream/socket.ts`: opens with `Sec-WebSocket-Protocol: acoustics`, auto-reconnect with capped exponential backoff (200 ms → 5 s), exposes a `MessagePort`-style consumer API. **Operates in a Web Worker** to keep envelope decode + Opus decode off the main thread. Main thread receives postMessage transferables.
7. Opus decoder: WebCodecs `AudioDecoder` with `{ codec: 'opus', sampleRate: 48000, numberOfChannels: 1 }`. Feature-detect `'AudioDecoder' in self`; if missing, show a clear "browser not supported" placeholder. **No WASM Opus fallback in v1** (defer until a real user has Safari ≤16).
8. Pair audio + inference by `t_us_capture_monotonic`: tiny ring buffer (last ~50 inference frames, ~12 s at 4 Hz) indexed by capture time; renderer queries nearest at RAF.
9. Renderers: `WaveformCanvas.svelte` (running PCM buffer, scrolling), `SpectrogramCanvas.svelte` (FFT via `AudioContext.AnalyserNode` for v1 — cheap and good enough; replace with Worker FFT only if quality demands), `TopKMeter.svelte` (bars for class probabilities).
10. Health badge: poll `GET /api/v1/status` at 2 Hz when visible, hover → popover with subsystem heartbeats. Single shared store.
11. Configuration drawer: mic policy (`GET/POST /api/v1/mic` with `?min_version=N` read-your-writes gate), inference cadence (`GET/POST /api/v1/inference`), active head selection (`GET/POST /api/v1/active` with `{ default: true }` option).

Critical files to create:
- [web/package.json](../../web/package.json), [web/vite.config.ts](../../web/vite.config.ts), [web/svelte.config.js](../../web/svelte.config.js), [web/tsconfig.json](../../web/tsconfig.json), [web/tailwind.config.js](../../web/tailwind.config.js)
- [web/src/app.html](../../web/src/app.html), [web/src/app.css](../../web/src/app.css)
- [web/src/routes/+layout.ts](../../web/src/routes/+layout.ts), [web/src/routes/+layout.svelte](../../web/src/routes/+layout.svelte)
- [web/src/routes/+page.svelte](../../web/src/routes/+page.svelte) (Dashboard)
- [web/src/lib/api/http.ts](../../web/src/lib/api/http.ts), [web/src/lib/api/types.ts](../../web/src/lib/api/types.ts), [web/src/lib/api/endpoints.ts](../../web/src/lib/api/endpoints.ts)
- [web/src/lib/stream/worker.ts](../../web/src/lib/stream/worker.ts), [web/src/lib/stream/client.ts](../../web/src/lib/stream/client.ts), [web/src/lib/stream/proto.ts](../../web/src/lib/stream/proto.ts)
- [web/src/lib/components/dashboard/](../../web/src/lib/components/dashboard/) — `WaveformCanvas.svelte`, `SpectrogramCanvas.svelte`, `TopKMeter.svelte`
- [web/src/lib/components/HealthBadge.svelte](../../web/src/lib/components/HealthBadge.svelte)
- [web/src/lib/stores/config.ts](../../web/src/lib/stores/config.ts), [web/src/lib/stores/streams.ts](../../web/src/lib/stores/streams.ts), [web/src/lib/stores/health.ts](../../web/src/lib/stores/health.ts)

Slice A done means: `pnpm dev` shows live waveform + spectrogram + top-K + health badge.

### Slice B — Workspace + Dataset Management

**User outcome**: create a workspace, record/upload audio, slice into labeled samples, commit to backend, see workspace revision update.

Deliverables:
1. Workspace list + detail routes: `/workspace`, `/workspace/[id]` (dynamic, client-rendered, no prerender).
2. CRUD via `POST/GET/PATCH/DELETE /api/v1/workspace[/{id}]`. Async delete: returns `{ job_id }` → subscribe to job events until terminal state. Workspace store tracks deletion-in-flight UI.
3. **Client-side filter**: workspaces tagged `__converter__` excluded from this tab's list.
4. IDB layer via `idb` (the minimal one, ~3 KB). Schema (one DB per origin):
   - `workspaces` — operator metadata cache (id, name, last seen revision).
   - `drafts` — recordings and segments. Fields: `id`, `workspace_id`, `class_label`, `blob` (WAV PCM-16 16 kHz mono), `duration_ms`, `created_at`, `state ∈ {draft, uploading, committed, failed}`, `upload_url`, `target_filename`, `last_error?`.
5. Recorder: `getUserMedia` → `AudioContext` (16 kHz target via `OfflineAudioContext` resample on stop) → PCM-16 WAV blob → IDB. Display recording timer + level meter via AnalyserNode.
6. Audio file import: drag-drop or file picker → decode via `AudioContext.decodeAudioData` → resample to 16 kHz mono → store in IDB.
7. Slicer UI: waveform editor with draggable boundaries, click-to-play segment, label per segment. Drafts persist across reloads from IDB. Spectrogram + waveform views per segment, basic quality flags (clipping, low SNR heuristics — defer if time pressed).
8. Commit flow (the load-bearing path):
   - Read `workspace.workspace_revision.id` as `base_rev`.
   - For each segment with `state = draft`: PUT to `/api/v1/workspace/{id}/assets/datasets/<class_label>/<uuid8>-<filename>.wav` via **XHR** (only place we use XHR — `upload.onprogress` is the only reliable upload progress).
   - On success: update IDB segment `state = committed`, record returned `workspace_revision_id`.
   - On failure: `state = failed`, store error code; show retry button per-segment.
   - **Filename safety**: prefix with uuid8 to avoid collisions; reject non-ASCII labels client-side per AssetPath rules.
   - **Resumability**: on page reload, scan for `state ∈ {uploading, failed}`, prompt "resume pending uploads?".
9. Revision UX: workspace card shows `rev N` chip. Heads list (when populated by Slice C) shows per-head `current` (green) or `stale (rev X → Y)` (amber) pill.

Critical files:
- [web/src/routes/workspace/+page.svelte](../../web/src/routes/workspace/+page.svelte), [web/src/routes/workspace/[id]/+page.svelte](../../web/src/routes/workspace/[id]/+page.svelte)
- [web/src/lib/idb/db.ts](../../web/src/lib/idb/db.ts), [web/src/lib/idb/drafts.ts](../../web/src/lib/idb/drafts.ts)
- [web/src/lib/audio/recorder.ts](../../web/src/lib/audio/recorder.ts), [web/src/lib/audio/wav.ts](../../web/src/lib/audio/wav.ts), [web/src/lib/audio/resample.ts](../../web/src/lib/audio/resample.ts)
- [web/src/lib/components/dataset/](../../web/src/lib/components/dataset/) — `Recorder.svelte`, `ImportZone.svelte`, `SegmentEditor.svelte`, `CommitDialog.svelte`
- [web/src/lib/api/upload.ts](../../web/src/lib/api/upload.ts) — XHR PUT with progress
- [web/src/lib/stores/workspace.ts](../../web/src/lib/stores/workspace.ts), [web/src/lib/stores/drafts.ts](../../web/src/lib/stores/drafts.ts)

### Slice C — Training + Job Events + History

**User outcome**: configure hyperparameters, launch a train job, watch real-time progress, browse past runs.

Deliverables:
1. Train form: epochs (1–1000), batch_size (1–4096), learning_rate (>0, ≤1.0), seed (optional), validation_split (0–1). Submit → `POST /api/v1/workspace/{id}/train` → receive `{ head_id, job_id }`.
2. **`subscribeJob(jobId)` helper** with the full state machine:
   - Open `GET /api/v1/jobs/{job_id}/events?after_seq=<cursor>` (SSE via `EventSource`).
   - On 409 `event_gap` response: parse body `{ oldest_seq, latest_seq }`, page through `GET /api/v1/workspace/{id}/assets/training_logs/<job_id>.jsonl?after_seq=<cursor>&limit=1000` until cursor ≥ `latest_seq`, then reopen SSE with `after_seq=<latest_seq>`.
   - On terminal state (`succeeded`/`failed`/`cancelled`): close stream, refresh heads list.
3. Progress UI: phase indicator (Loading → FeatureExtract → Train → Saving → Done), per-epoch loss/acc/val_acc line chart (canvas, no chart lib — small custom), virtualized log viewer.
4. Heads list (per workspace): `GET /api/v1/workspace/{id}/heads`. Per-head card: id (8 char), labels, n_classes, size, created_at, revision, status pill, "Activate" button → `POST /api/v1/active`, "Delete" button → `DELETE`.
5. Smart suggestion: if dataset revision matches an existing `current` head, show banner "head matches current dataset — skip training?" with "Activate instead" CTA. Implements ARCHITECTURE.md §B.3.
6. Cancel: `DELETE /api/v1/workspace/{id}/training/{job_id}`.
7. History view: `GET /api/v1/jobs` for memory ring; per workspace, also list `training_logs/` and `converter_logs/` via `GET /api/v1/workspace/{id}/assets/training_logs?after_seq=&limit=` directory listing, page through JSONL files for older entries. Merge + dedupe by `job_id`, sort by `created_at`. Filter by type (train / convert / dataset_delete / converter_delete / workspace_delete) and status.

Critical files:
- [web/src/lib/api/jobs.ts](../../web/src/lib/api/jobs.ts) — `subscribeJob`, JSONL pager, gap-recovery state machine
- [web/src/lib/components/training/](../../web/src/lib/components/training/) — `TrainForm.svelte`, `JobProgress.svelte`, `MetricsChart.svelte`, `LogViewer.svelte`, `HeadCard.svelte`
- [web/src/lib/components/history/HistoryList.svelte](../../web/src/lib/components/history/HistoryList.svelte)
- [web/src/lib/stores/jobs.ts](../../web/src/lib/stores/jobs.ts), [web/src/lib/stores/heads.ts](../../web/src/lib/stores/heads.ts)

### Slice D — Converter Tab + Tiny Dashboard

**User outcome**: drop TFJS bundle, convert to MPK, hot-swap converted head. Tiny Dashboard floats anywhere.

Deliverables:
1. Converter tab route `/converter`. Shows only workspaces tagged `__converter__`.
2. New conversion wizard: drop TFJS files (`model.json` + shards + labels) → auto-create workspace with name `converter-<uuid8>` and tags `['__converter__']` → PUT each file to `/api/v1/workspace/{id}/assets/converters/<filename>` → POST `/api/v1/workspace/{id}/convert` with `{ converter_type: 'tfjs', model_json_path, shards: [...], labels_path, labels_format }` → reuse `subscribeJob` from Slice C.
3. Post-conversion: download converted MPK button, "activate as inference head" button, "free up workspace" button (async DELETE with job tracking).
4. **Tiny Dashboard** as a floating, draggable card available on Workspace and Converter tabs. Reuses Slice A's streams + components (`WaveformCanvas`, `TopKMeter`). The shared Worker is always running; Tiny Dashboard just subscribes to its ring buffer. Toggle pinned/hidden via header icon. Persists open/closed state in `localStorage`.

Critical files:
- [web/src/routes/converter/+page.svelte](../../web/src/routes/converter/+page.svelte), [web/src/routes/converter/new/+page.svelte](../../web/src/routes/converter/new/+page.svelte)
- [web/src/lib/components/converter/](../../web/src/lib/components/converter/) — `ConvertWizard.svelte`, `ConverterCard.svelte`
- [web/src/lib/components/TinyDashboard.svelte](../../web/src/lib/components/TinyDashboard.svelte)

### Slice E — Polish + Primitive Extraction

**User outcome**: app feels finished. Polished feedback, keyboard navigation, responsive layout.

Deliverables:
1. Extract primitives **now** (after 4 slices of inline use): `Button`, `IconButton`, `Modal`, `Drawer`, `Toast`, `Tooltip`, `Tabs`, `Select`, `Slider`, `Toggle`, `Input`, `ProgressBar`, `Spinner`, `EmptyState`. Each tuned to the variants we actually used.
2. Global toast system: errors from API surface as toasts with `code`-aware copy ("Conflict: another training job is already running" for `another_train_running`, etc.).
3. Accessibility pass: semantic HTML, ARIA on interactive widgets, visible focus rings, full Tab/Enter/Space keyboard nav, contrast ≥ WCAG AA.
4. Responsive: target ≥1280 px desktop primary; ≥768 px tablet (collapsed nav, stacked cards); mobile not in scope per "single operator at the device" assumption.
5. Performance: virtualize logs and history (`@tanstack/svelte-virtual` or hand-rolled), debounce search inputs, RAF-throttle canvas renders, pool canvas contexts, lazy-import heavy routes.
6. Error boundary at the layout level — surfaces unrecovered errors instead of blank screen.
7. Empty states for every list (no workspaces, no heads, no jobs, no segments).
8. Optional: light/dark mode + i18n scaffolding deferred per [ARCHITECTURE.md §E.5.7](./ARCHITECTURE.md). Note locations where deferred work plugs in.

Critical files:
- [web/src/lib/components/ui/](../../web/src/lib/components/ui/) — extracted primitives
- [web/src/lib/components/Toast.svelte](../../web/src/lib/components/Toast.svelte), [web/src/lib/stores/toasts.ts](../../web/src/lib/stores/toasts.ts)
- [web/src/lib/utils/error-copy.ts](../../web/src/lib/utils/error-copy.ts) — code → message map

## Cross-cutting conventions

### Proto decoder
- Source-of-truth: [modules/proto/*.proto](../../modules/proto). Never copied into `web/`.
- Implementation: hand-written decoder at [src/lib/stream/proto.ts](../../web/src/lib/stream/proto.ts) covering `Envelope`, `AudioFrame`, `InferenceFrame`, `TopK`. ~150 LOC, zero deps.
- Wire-format dispatch on field tag: receiver silently drops unknown tags per proto3 unknown-field semantics ([PROTO.md](../PROTO.md)).
- When `.proto` files change, the TS decoder must be updated in lockstep. The total field count is tiny enough that visual inspection suffices; consider adding back codegen only if the schema grows.

### Error envelope handling
- All API responses parsed for `{ error, code }`. Fetch wrapper rejects with a typed `ApiError { status, code, message }`.
- Stores translate `ApiError.code` → user-facing copy via [src/lib/utils/error-copy.ts](../../web/src/lib/utils/error-copy.ts).
- The one special case: SSE `event_gap` (409) is **not** an error — it triggers JSONL backfill in `subscribeJob`.

### Read-your-writes — mic policy
- `POST /api/v1/mic` returns new `version`. Immediately re-GET with `?min_version=<new>`, retry on 425 `too_early` with backoff (≤3 attempts), then surface success. Without this, the UI flickers back to the pre-write state.

### WS lifecycle
- Single shared Worker owns both `/stream/audio` and `/stream/infer`. Drains continuously regardless of UI visibility (backpressure threshold 64 frames = 1.28 s at 50 Hz — never let the renderer be the consumer).
- Renderer subscribes to Worker's ring via `postMessage` transferables. Audio frames decoded in Worker; main thread receives `Float32Array` PCM windows.
- On WS close (1011 lagged): full reconnect, accept frame loss (no replay). Show momentary "stream interrupted" indicator.

### Active head provenance
- Header strip displays `head_id[:8]@v{head_version}` from latest inference frame. Hover → popover with `source_workspace_id`, `workspace_revision`, `labels`.
- If `GET /api/v1/active` returns `source_workspace_alive: false`, header strip turns amber: "active head's source workspace deleted".

### TypeScript types match Rust
- All domain types in [src/lib/api/types.ts](../../web/src/lib/api/types.ts). Use discriminated unions where Rust uses tagged enums (e.g., `ActiveResp` with `origin: 'head' | 'default'` discriminates on whether `source_*` fields exist).
- Keep `as const` enums for codes/states. No string literals scattered in feature code.

### Vite proxy
```ts
server: {
  proxy: {
    '/api':           { target: 'http://127.0.0.1:8787' },
    '/stream/audio':  { target: 'ws://127.0.0.1:8787', ws: true },
    '/stream/infer':  { target: 'ws://127.0.0.1:8787', ws: true },
  }
}
```
Forgetting `ws: true` is silent — upgrade fails and debugging takes an hour. Lock this in Slice A.

### `__converter__` tag convention
- Set on workspace creation in the converter wizard. Filtered out of Workspace Tab's list (`tags.includes('__converter__')` excludes). Shown in Converter Tab.
- Backend doesn't know about the marker; it's purely a frontend convention.

## Verification

End-to-end manual verification per slice. No automated tests.

**Slice A**:
1. Run daemon (`cargo run --release`) and `pnpm dev` (in `web/`).
2. Open `http://localhost:5173`. Confirm waveform draws live mic audio.
3. Confirm Top-K bars update at ~4 Hz with the bundled default head's classifications.
4. Hover Health Badge: confirm subsystem heartbeat list. Trip a degraded state (e.g., disconnect mic in mock mode) and confirm the badge color reflects.
5. Open dev tools: confirm exactly one `/stream/audio` and one `/stream/infer` WS open. Disconnect Wi-Fi, reconnect, confirm auto-reconnect.

**Slice B**:
1. Create workspace "test-1" via UI. Confirm appears in list with `rev 0`.
2. Open detail. Record 3 clips, slice into 2 segments each (6 total), label as `dog`/`cat` (3 each). Confirm drafts persist after page reload.
3. Commit. Watch per-segment progress. Confirm `GET /api/v1/workspace/{id}` shows `workspace_revision.id` advanced to 6, `head_count: 0`.
4. Kill the page mid-commit; reload; confirm "resume pending uploads" prompt, complete commit successfully.
5. Delete the workspace; confirm async job completes and workspace disappears from list.

**Slice C**:
1. With committed dataset from Slice B, open Train form. Submit defaults (e.g., 5 epochs, batch 32, lr 1e-3).
2. Confirm SSE progress streams in. Confirm per-epoch loss/acc chart updates. Confirm phase indicator advances Loading → FeatureExtract → Train → Saving → Done.
3. After completion, confirm head appears in Heads list with `current` pill.
4. Add 1 more segment to dataset, recommit. Confirm head pill flips to `stale (rev N → M)`.
5. Activate the head via "Activate" → confirm `/stream/infer` frames now carry the new `head_id`.
6. Force `event_gap`: pause the SSE subscription long enough for the ring to roll, reopen — confirm JSONL backfill catches up without losing log lines.
7. Cancel an in-flight train; confirm SSE terminates with `state: cancelled`.

**Slice D**:
1. In Converter Tab, run the wizard with a Teachable Machine TFJS export. Confirm `converter-<uuid8>` workspace appears here, **not** in Workspace Tab.
2. Confirm conversion job streams progress, completes; confirm download MPK works; confirm "Activate" updates inference head globally.
3. Open Tiny Dashboard on the Workspace Tab; confirm it shows the same live waveform + Top-K as the main Dashboard.
4. Free up the converter workspace; confirm async deletion.

**Slice E**:
1. Tab through every interactive control on every tab; confirm focus rings visible, Enter/Space activate correctly.
2. Force every error code path (e.g., POST `/train` while one is running → `another_train_running` 409) and confirm a toast surfaces with helpful copy.
3. Resize browser to 1280 px, 1024 px, 768 px; confirm layouts hold.
4. Open 50 segments + 5000 log lines; confirm scroll perf stays smooth (no jank).
5. Run Lighthouse: confirm no critical accessibility violations, perf score acceptable for a local SPA.

## Things explicitly NOT in scope

- Light/dark mode (deferred per [ARCHITECTURE.md §E.5.7](./ARCHITECTURE.md))
- Internationalization (deferred)
- Mobile-first responsive (≥768 px tablet only)
- Automated tests, benchmarks, CI
- Backend modifications (no static-file serving from daemon, no new endpoints)
- WebTransport / QUIC streams (WebSocket only)
- Multi-user / auth UI (operator-local trust model per [docs/API.md](../API.md))
- Telemetry / analytics

## Open questions worth revisiting after Slice A

- Spectrogram: stay with cheap `AnalyserNode` or migrate to Worker FFT? Decide based on visible quality after Slice A.
- WASM Opus fallback: skip in v1; revisit if a real user reports Safari ≤16 needs.
- COOP/COEP for `crossOriginIsolated` + SharedArrayBuffer: only needed if Worker→main transfer becomes a bottleneck. Skip in v1.
