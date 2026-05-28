import type { Config, Output, Report, PassReport } from "nagami-rs";
import MinifyWorker from "./minify.worker.ts?worker";

export type { Config, Output, Report, PassReport };

export interface RunResult {
  output: Output;
  error: null;
}

export interface RunError {
  output: null;
  error: string;
}

export type RunOutput = RunResult | RunError;

interface WorkerResponse {
  id: number;
  output: Output | null;
  error: string | null;
}

let worker: Worker | null = null;
let nextId = 0;
const pending = new Map<number, (r: RunOutput) => void>();

function failAllPending(reason: string): void {
  for (const resolve of pending.values()) {
    resolve({ output: null, error: reason });
  }
  pending.clear();
}

function handleMessage(e: MessageEvent<WorkerResponse>): void {
  const { id, output, error } = e.data;
  const resolve = pending.get(id);
  if (!resolve) return;
  pending.delete(id);
  if (output !== null) {
    resolve({ output, error: null });
  } else {
    resolve({ output: null, error: error ?? "unknown worker error" });
  }
}

function handleError(e: Event): void {
  const msg = e instanceof ErrorEvent && e.message ? e.message : "worker error";
  failAllPending(msg);
  // Drop the dead worker so the next call can spin up a fresh one.
  worker?.terminate();
  worker = null;
}

function ensureWorker(): Worker {
  if (worker) return worker;
  worker = new MinifyWorker();
  worker.addEventListener("message", handleMessage);
  worker.addEventListener("error", handleError);
  worker.addEventListener("messageerror", handleError);
  return worker;
}

export function run(source: string, config?: Config): Promise<RunOutput> {
  let w: Worker;
  try {
    w = ensureWorker();
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return Promise.resolve({ output: null, error: msg });
  }
  const id = ++nextId;
  return new Promise<RunOutput>((resolve) => {
    pending.set(id, resolve);
    w.postMessage({ id, source, config });
  });
}

// Eagerly construct the worker on module load so WASM streaming-compile
// happens in parallel with the rest of the app boot. A failure here (e.g.
// strict CSP without worker-src) must not crash module evaluation - the
// lazy path in run() will surface the error at call time.
try {
  ensureWorker();
} catch {
  // swallow; run() handles the same failure with a structured RunError
}
