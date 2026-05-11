// TypeScript types mirroring the acousticsd REST contract.
// Authoritative shapes live in docs/API.md and the Rust modules; this
// file is a thin restatement -- when backend types change, update here
// and the call sites will fail to compile.

export type Uuid = string;
export type Rfc3339 = string;

export interface WorkspaceRevision {
  id: number;
  at: Rfc3339;
}

export interface WorkspaceSummary {
  id: Uuid;
  name: string;
  tags: string[];
  created_at: Rfc3339;
  workspace_revision: WorkspaceRevision;
  head_count: number;
}

export type HeadStatus = 'current' | 'stale';

export interface HeadRecord {
  head_id: Uuid;
  workspace_revision: WorkspaceRevision;
  sha256: string;
  n_classes: number;
  size_bytes: number;
  created_at: Rfc3339;
  status: HeadStatus;
}

export interface HeadManifest extends HeadRecord {
  labels: string[];
}

export interface WorkspaceDetail extends WorkspaceSummary {
  heads: HeadRecord[];
}

export type ActiveOrigin = 'head' | 'default';

interface ActiveBase {
  sha256: string;
  labels_sha256: string;
  runtime_head_id: Uuid;
  n_classes: number;
  labels: string[];
  activated_at: Rfc3339;
  activation_id: Uuid;
}

export type ActiveResp =
  | (ActiveBase & {
      origin: 'head';
      source_workspace_id: Uuid;
      source_workspace_revision: WorkspaceRevision;
      source_head_id: Uuid;
      source_workspace_alive: boolean;
    })
  | (ActiveBase & { origin: 'default' });

export interface TrainParams {
  epochs: number;
  batch_size: number;
  learning_rate: number;
  seed?: number;
  validation_split?: number;
}

export type JobState = 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled';
export type JobType =
  | 'train'
  | 'convert'
  | 'dataset_delete'
  | 'converter_delete'
  | 'workspace_delete';

export interface JobProgress {
  done: number;
  total?: number;
}

export interface JobSnapshot {
  job_id: Uuid;
  job_type: JobType;
  workspace_id: Uuid;
  state: JobState;
  progress: JobProgress;
  result: unknown;
  last_seq: number;
  updated_at: Rfc3339;
  message?: string;
}

export interface JobEvent {
  seq: number;
  at: Rfc3339;
  state: JobState;
  progress?: JobProgress;
  message?: string;
  metrics?: Record<string, number>;
}

export interface SubsystemHealth {
  healthy: boolean;
  detail?: string;
  degraded_reason?: string;
  age_ms?: number;
  stale: boolean;
}

// The /status response is flat: process metrics, subsystem heartbeats,
// broadcast-drop counters, and workspace counters all live at the root.
// `inference_engine` is NOT in /status — query /active for the runtime
// head + labels.
export interface StatusSnapshot {
  cpu_pct: number;
  mem_rss_kb: number;
  disk_free_kb: number;
  metrics_age_ms: number;
  metrics_stale: boolean;
  uptime_s: number;
  subsystems: Record<string, SubsystemHealth>;
  broadcast_audio_messages_dropped: number;
  broadcast_inference_messages_dropped: number;
  workspace: Record<string, number>;
}

export interface InferenceCfg {
  hop_samples: number;
  top_k: number;
}

export interface MicSource {
  kind: string;
  sample_rate: number;
  [key: string]: unknown;
}

export interface MicCandidate {
  id: string;
  source: MicSource;
  channels: number[];
}

export interface MicCatalogue {
  candidates: MicCandidate[];
}

export type MicPolicyMic = { kind: 'first_available' } | { kind: 'fixed'; id: string };
export type MicPolicyChannel = { kind: 'auto' } | { kind: 'fixed'; channel: number };

export interface MicPolicy {
  mic: MicPolicyMic;
  channel: MicPolicyChannel;
}

export interface MicState {
  catalogue: MicCatalogue;
  policy: MicPolicy;
  version: number;
}

export interface TfjsConvertParams {
  converter_type: 'tfjs';
  model_json_path: string;
  shards: string[];
  labels_path: string;
  labels_format: 'lines' | 'tfjs_metadata';
}

export interface AssetReceipt {
  path: string;
  sha256: string;
  size_bytes: number;
  workspace_revision_id: number;
}

export interface ApiErrorBody {
  error: string;
  code: string;
  oldest_seq?: number;
  latest_seq?: number;
}
