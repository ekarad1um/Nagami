import { createStreamClient, type SocketState, type StreamClient } from '$lib/stream/client';
import type { TopK } from '$lib/stream/proto';

// Singleton store wrapping the streaming worker.  Reactive fields (Svelte
// $state) update at human-readable rates (inference at ~4 Hz, status on
// change).  The PCM ring buffer is intentionally NON-reactive: at 50 Hz +
// 960 samples per frame, marking it $state would thrash every consumer.
// Renderers poll snapshot() at RAF instead.

const PCM_SAMPLE_RATE = 48_000;
const PCM_BUFFER_SECONDS = 10;

export interface HeadInfo {
  head_id: string | null;
  head_version: number | null;
}

class StreamsStore {
  audioStatus = $state<SocketState>('closed');
  inferStatus = $state<SocketState>('closed');
  latestTopK = $state<TopK[]>([]);
  head = $state<HeadInfo>({ head_id: null, head_version: null });
  unsupportedReason = $state<string | null>(null);
  inferenceFps = $state(0);

  readonly sampleRate = PCM_SAMPLE_RATE;
  private readonly ring = new Float32Array(PCM_SAMPLE_RATE * PCM_BUFFER_SECONDS);
  private writeIdx = 0;
  private totalSamples = 0;

  private client: StreamClient | null = null;
  private started = false;
  private inferenceTimes: number[] = [];

  start(): void {
    if (this.started) return;
    this.started = true;
    this.client = createStreamClient();
    this.client.audio.on(({ pcm }) => {
      this.pushPcm(pcm);
    });
    this.client.inference.on(({ top_k, head_id, head_version }) => {
      this.latestTopK = top_k;
      if (head_id !== this.head.head_id || head_version !== this.head.head_version) {
        this.head = { head_id, head_version };
      }
      this.trackInferenceFps();
    });
    this.client.status.on(({ channel, state }) => {
      if (channel === 'audio') this.audioStatus = state;
      else this.inferStatus = state;
    });
    this.client.unsupported.on((reason) => {
      this.unsupportedReason = reason;
    });
    this.client.start();
  }

  stop(): void {
    if (!this.started) return;
    this.client?.stop();
    this.client = null;
    this.started = false;
  }

  // Most recent `samples` PCM values, copied into a fresh array.  Caller
  // must not retain references between frames.
  snapshot(samples: number): Float32Array {
    const r = this.ring.length;
    const n = Math.min(samples, r);
    const out = new Float32Array(n);
    const start = (((this.writeIdx - n) % r) + r) % r;
    if (start + n <= r) {
      out.set(this.ring.subarray(start, start + n));
    } else {
      const head = r - start;
      out.set(this.ring.subarray(start), 0);
      out.set(this.ring.subarray(0, n - head), head);
    }
    return out;
  }

  get sampleCount(): number {
    return this.totalSamples;
  }

  private pushPcm(pcm: Float32Array): void {
    const n = pcm.length;
    const r = this.ring.length;
    const space = r - this.writeIdx;
    if (n <= space) {
      this.ring.set(pcm, this.writeIdx);
    } else {
      this.ring.set(pcm.subarray(0, space), this.writeIdx);
      this.ring.set(pcm.subarray(space), 0);
    }
    this.writeIdx = (this.writeIdx + n) % r;
    this.totalSamples += n;
  }

  private trackInferenceFps(): void {
    const now = performance.now();
    this.inferenceTimes.push(now);
    const cutoff = now - 2_000;
    while (this.inferenceTimes.length > 0 && this.inferenceTimes[0] < cutoff) {
      this.inferenceTimes.shift();
    }
    this.inferenceFps = this.inferenceTimes.length / 2;
  }
}

export const streams = new StreamsStore();
