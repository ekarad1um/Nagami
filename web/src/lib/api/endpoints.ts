import { api, ApiError } from './http';
import type {
  ActiveResp,
  InferenceCfg,
  MicPolicy,
  MicState,
  StatusSnapshot,
  Uuid,
  WorkspaceSummary
} from './types';

export const status = {
  get: () => api.get<StatusSnapshot>('/api/v1/status')
};

export const mic = {
  get: (minVersion?: number) => {
    const q = minVersion !== undefined ? `?min_version=${minVersion}` : '';
    return api.get<MicState>(`/api/v1/mic${q}`);
  },
  set: async (policy: MicPolicy): Promise<MicState> => {
    const fresh = await api.post<MicState>('/api/v1/mic', { policy });
    return readYourWrites(fresh.version);
  }
};

// Read-your-writes gate -- after a POST, fetch the policy again with
// ?min_version=N until the daemon agrees, then return the canonical state.
async function readYourWrites(minVersion: number, attempts = 0): Promise<MicState> {
  try {
    return await mic.get(minVersion);
  } catch (err) {
    if (err instanceof ApiError && err.status === 425 && attempts < 3) {
      await sleep(50 * 2 ** attempts);
      return readYourWrites(minVersion, attempts + 1);
    }
    throw err;
  }
}

const sleep = (ms: number) => new Promise((res) => setTimeout(res, ms));

export const inference = {
  get: () => api.get<{ cfg: InferenceCfg }>('/api/v1/inference').then((r) => r.cfg),
  set: (cfg: Partial<InferenceCfg>) =>
    api.post<{ cfg: InferenceCfg }>('/api/v1/inference', cfg).then((r) => r.cfg)
};

export const active = {
  get: () => api.get<ActiveResp>('/api/v1/active'),
  setHead: (workspace_id: Uuid, head_id: Uuid) =>
    api.post<ActiveResp>('/api/v1/active', { workspace_id, head_id }),
  setDefault: () => api.post<ActiveResp>('/api/v1/active', { default: true })
};

export const workspaces = {
  list: () => api.get<WorkspaceSummary[]>('/api/v1/workspace')
};
