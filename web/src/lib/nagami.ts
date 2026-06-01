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

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

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

// URL of the wasm the document preloads. Fetching this exact URL consumes the
// <link rel="preload"> instead of wasting it. Absent in dev -> worker self-fetches.
function getPreloadedWasmUrl(): string | null {
  if (typeof document === "undefined") return null;
  const link = document.querySelector<HTMLLinkElement>(
    'link[rel="preload"][as="fetch"][href$=".wasm"]',
  );
  return link?.href ?? null;
}

// Fetch + compile the wasm on the main thread (consuming the preload, one download)
// and hand the compiled Module to the worker.
function primeWorkerWasm(w: Worker, url: string): void {
  void (async () => {
    try {
      // mode/credentials must match the crossorigin="anonymous" preload key.
      const resp = await fetch(url, { mode: "cors", credentials: "same-origin" });
      if (!resp.ok) throw new Error(`wasm fetch failed: ${resp.status}`);
      const ct = resp.headers.get("content-type") ?? "";
      const canStream =
        typeof WebAssembly.compileStreaming === "function" &&
        ct.includes("application/wasm");
      const module = canStream
        ? await WebAssembly.compileStreaming(resp)
        : await WebAssembly.compile(await resp.arrayBuffer());
      w.postMessage({ type: "init-module", module });
    } catch {
      // Couldn't compile on the main thread: let the worker fetch it itself.
      try {
        w.postMessage({ type: "init-fallback" });
      } catch {
        // Worker already torn down; nothing to do.
      }
    }
  })();
}

function ensureWorker(): Worker {
  if (worker) return worker;
  worker = new MinifyWorker();
  worker.addEventListener("message", handleMessage);
  worker.addEventListener("error", handleError);
  worker.addEventListener("messageerror", handleError);
  const wasmUrl = getPreloadedWasmUrl();
  if (wasmUrl) {
    primeWorkerWasm(worker, wasmUrl);
  } else {
    // No preload (dev mode): tell the worker to self-fetch the wasm.
    worker.postMessage({ type: "init-fallback" });
  }
  return worker;
}

export function run(source: string, config?: Config): Promise<RunOutput> {
  let w: Worker;
  try {
    w = ensureWorker();
  } catch (err) {
    return Promise.resolve({ output: null, error: errorMessage(err) });
  }
  const id = ++nextId;
  return new Promise<RunOutput>((resolve) => {
    pending.set(id, resolve);
    try {
      postRequest(w, { id, source, config });
    } catch (err) {
      // postMessage threw even after sanitising (e.g. a structurally
      // un-serialisable config). Honour run()'s never-reject contract: drop
      // the now-unanswerable pending entry and surface a structured error.
      pending.delete(id);
      resolve({ output: null, error: errorMessage(err) });
    }
  });
}

// Post a request to the worker, guarding the structuredClone boundary.
function postRequest(
  w: Worker,
  req: { id: number; source: string; config?: Config },
): void {
  try {
    w.postMessage(req);
  } catch {
    w.postMessage({
      id: req.id,
      source: req.source,
      config: req.config
        ? (JSON.parse(JSON.stringify(req.config)) as Config)
        : undefined,
    });
  }
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
