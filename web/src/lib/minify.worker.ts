/// <reference lib="webworker" />
import init, { run as wasmRun, type Config, type Output } from "nagami-rs";

interface RunRequest {
  id: number;
  source: string;
  config?: Config;
}

interface RunResponse {
  id: number;
  output: Output | null;
  error: string | null;
}

const ctx = self as unknown as DedicatedWorkerGlobalScope;

let readyPromise: Promise<void> | null = null;
function ensureReady(): Promise<void> {
  if (!readyPromise) {
    readyPromise = init().then(
      () => undefined,
      (err) => {
        // Allow a future request to retry initialization
        readyPromise = null;
        throw err;
      },
    );
  }
  return readyPromise;
}

// Start WASM compile/instantiate as soon as the worker boots so the first
// request doesn't pay the full init latency.
ensureReady().catch(() => {});

ctx.onmessage = async (e: MessageEvent<RunRequest>) => {
  const { id, source, config } = e.data;
  try {
    await ensureReady();
    const output = wasmRun(source, config);
    const response: RunResponse = { id, output, error: null };
    ctx.postMessage(response);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    const response: RunResponse = { id, output: null, error: msg };
    ctx.postMessage(response);
  }
};
