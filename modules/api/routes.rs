//! Per-domain route modules.
//!
//! Each submodule (`mic`, `inference`, `workspace`, `training`,
//! `converter`, `status`, `health`) owns its handler fns, the
//! request / response DTOs unique to that domain, and a
//! `pub fn router() -> Router<AppState>` that wires the URL paths.
//! [`crate::api`] assembles them via `Router::new().merge(...)`.
//!
//! Workspace and asset routes share `workspace.rs`: every asset
//! route is nested under `/workspace/{id}/...`, and the handlers
//! share the `WorkspaceId::parse(&id)` boilerplate plus the same
//! `s.files` state.
//!
//! # Blocking-pool discipline
//!
//! Every `FsService` / `WorkspaceMgr` / `HotHead::install_*` call
//! does synchronous filesystem I/O and **must** run inside
//! `tokio::task::spawn_blocking` from the async handler.  Calling
//! them directly from the handler future would block the tokio
//! worker, starving the audio + inference broadcast loops that
//! share the same runtime.  The handler then `?`-propagates the
//! `JoinError` through `ApiError::Join` and the inner domain
//! result through its own `?`; the canonical pattern is
//! `task::spawn_blocking(move || ...).await??`.

// `POST /active` + `GET /active` -- active-head activation
// pipeline + read.  Atomic publish of
// `<root>/active/current.json` + the pointed generation;
// installs the prevalidated runtime candidate AFTER on-disk
// state is durable.
pub mod active;
pub mod converter;
// AssetPath-shaped dataset routes (`/upload`, `/assets`,
// `/assets/{*path}`).  Lives outside `workspace.rs` to keep the
// lifecycle file focused on the `workspace.json` shape.  Both
// modules share `s.files`.
pub mod dataset;
pub mod health;
// Trained-head routes (`GET /workspace/{id}/heads`,
// `GET /workspace/{id}/heads/{head_id}`,
// `DELETE /workspace/{id}/heads/{head_id}`).  Reads through the
// per-workspace cache; deletes synchronously rewrite
// `heads.json` + `workspace.json` and unlink the bytes.
pub mod heads;
pub mod inference;
// Memory-only `GET /jobs` + `GET /jobs/{job_id}` snapshots and
// the `GET /jobs/{job_id}/events` SSE stream.
pub mod jobs;
// JSONL log reads + wipes live on the unified `/assets/{*path}`
// surface (see [`dataset`]): paging on `?after_seq=&limit=` for
// `GET`s; sync wipe under `training_logs[/...]` /
// `converter_logs[/...]` for `DELETE`s, gated against the active
// producer.
pub mod mic;
pub mod status;
pub mod training;
pub mod workspace;
