/// <reference lib="webworker" />
import init, { initSync, run as wasmRun, type Config, type Output } from "nagami-rs";

interface RunRequest {
  id: number;
  source: string;
  config?: Config;
}

type InitMessage =
  | { type: "init-module"; module: WebAssembly.Module }
  | { type: "init-fallback" };

type IncomingMessage = RunRequest | InitMessage;

interface RunResponse {
  id: number;
  output: Output | null;
  error: string | null;
}

const ctx = self as unknown as DedicatedWorkerGlobalScope;

// Resolved by the init message; run() requests queue on this until then.
let resolveModule!: (m: WebAssembly.Module | null) => void;
const modulePromise = new Promise<WebAssembly.Module | null>((r) => {
  resolveModule = r;
});

let readyPromise: Promise<void> | null = null;
function ensureReady(): Promise<void> {
  if (!readyPromise) {
    readyPromise = modulePromise
      .then((module) => {
        if (module) {
          initSync({ module });
          return;
        }
        return init().then(() => undefined);
      })
      .then(
        () => undefined,
        (err) => {
          readyPromise = null; // allow a future request to retry
          throw err;
        },
      );
  }
  return readyPromise;
}

function isInitMessage(d: IncomingMessage): d is InitMessage {
  const t = (d as InitMessage).type;
  return t === "init-module" || t === "init-fallback";
}

ctx.onmessage = (e: MessageEvent<IncomingMessage>) => {
  const data = e.data;

  if (isInitMessage(data)) {
    resolveModule(data.type === "init-module" ? data.module : null);
    ensureReady().catch(() => { }); // instantiate eagerly so the first run is fast
    return;
  }

  const { id, source, config } = data;
  void (async () => {
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
  })();
};
