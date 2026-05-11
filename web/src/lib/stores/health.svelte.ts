import { status as statusApi } from '$lib/api/endpoints';
import type { StatusSnapshot } from '$lib/api/types';

const POLL_INTERVAL_MS = 500;

class HealthStore {
  snapshot = $state<StatusSnapshot | null>(null);
  lastError = $state<string | null>(null);
  lastUpdated = $state<number | null>(null);

  private timer: ReturnType<typeof setInterval> | null = null;
  private inflight = false;

  start(): void {
    if (this.timer !== null) return;
    void this.tick();
    this.timer = setInterval(() => void this.tick(), POLL_INTERVAL_MS);
  }

  stop(): void {
    if (this.timer !== null) {
      clearInterval(this.timer);
      this.timer = null;
    }
  }

  get level(): 'unknown' | 'ok' | 'degraded' | 'down' {
    if (this.lastError) return 'down';
    if (!this.snapshot) return 'unknown';
    let degraded = this.snapshot.metrics_stale;
    for (const sub of Object.values(this.snapshot.subsystems)) {
      if (!sub.healthy) return 'down';
      if (sub.stale || sub.degraded_reason) degraded = true;
    }
    return degraded ? 'degraded' : 'ok';
  }

  private async tick(): Promise<void> {
    if (this.inflight) return;
    if (typeof document !== 'undefined' && document.hidden) return;
    this.inflight = true;
    try {
      this.snapshot = await statusApi.get();
      this.lastError = null;
      this.lastUpdated = Date.now();
    } catch (e) {
      this.lastError = e instanceof Error ? e.message : String(e);
    } finally {
      this.inflight = false;
    }
  }
}

export const health = new HealthStore();
