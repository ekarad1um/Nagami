import type { Config, Output, Report, PassReport, NameMap } from "nagami-rs";
import type { RunRequest, RunResponse } from "./worker-protocol";
import MinifyWorker from "./minify.worker.ts?worker";

export type { Config, Output, Report, PassReport, NameMap };

export interface RunResult {
  output: Output;
  error: null;
}

export interface RunError {
  output: null;
  error: string;
}

export type RunOutput = RunResult | RunError;

// A worker silent for this long while requests wait is dropped; one killed by
// the browser fires no event.
const STALL_MS = 30_000;

let worker: Worker | null = null;
let nextId = 0;
// Requests are kept whole so a replacement worker can replay them.
const pending = new Map<
  number,
  { req: RunRequest; resolve: (r: RunOutput) => void }
>();
// Fetched and compiled once on the main thread; every worker instantiates it.
let modulePromise: Promise<WebAssembly.Module> | null = null;
let stallTimer: ReturnType<typeof setTimeout> | undefined;

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

// Re-armed on every response, so a slow but progressing worker survives.
function armStall(): void {
  clearTimeout(stallTimer);
  if (pending.size > 0) stallTimer = setTimeout(onStall, STALL_MS);
}

function onStall(): void {
  failAllPending("minifier stopped responding");
  dropWorker();
}

function settle(id: number, result: RunOutput): void {
  const entry = pending.get(id);
  if (!entry) return;
  pending.delete(id);
  entry.resolve(result);
  armStall();
}

function failAllPending(reason: string): void {
  for (const { resolve } of pending.values()) {
    resolve({ output: null, error: reason });
  }
  pending.clear();
  clearTimeout(stallTimer);
}

// Events still queued from the old worker must not reach the handlers.
function dropWorker(): void {
  if (!worker) return;
  worker.removeEventListener("message", handleMessage);
  worker.removeEventListener("error", handleError);
  worker.removeEventListener("messageerror", handleError);
  worker.terminate();
  worker = null;
}

function handleMessage(e: MessageEvent<RunResponse>): void {
  const { id, output, error, fatal } = e.data;
  settle(
    id,
    output !== null
      ? { output, error: null }
      : { output: null, error: error ?? "unknown worker error" },
  );
  if (!fatal) return;
  // The instance is gone: replace the worker and replay what is still queued.
  dropWorker();
  try {
    const w = ensureWorker();
    for (const { req } of pending.values()) {
      try {
        w.postMessage(req);
      } catch (err) {
        settle(req.id, { output: null, error: errorMessage(err) });
      }
    }
  } catch (err) {
    failAllPending(errorMessage(err));
  }
}

// The worker script itself failed to load or run; replaying would repeat it.
function handleError(e: Event): void {
  const msg = e instanceof ErrorEvent && e.message ? e.message : "worker error";
  failAllPending(msg);
  dropWorker();
}

// Injected at build time by vite.config.ts; absent in dev.
function getPreloadedWasmUrl(): string | null {
  if (typeof document === "undefined") return null;
  const link = document.querySelector<HTMLLinkElement>(
    'link[rel="preload"][as="fetch"][href$=".wasm"]',
  );
  return link?.href ?? null;
}

// Consumes the preload: mode/credentials must match its crossorigin="anonymous" key.
async function compileWasm(url: string): Promise<WebAssembly.Module> {
  const resp = await fetch(url, { mode: "cors", credentials: "same-origin" });
  if (!resp.ok) throw new Error(`wasm fetch failed: ${resp.status}`);
  const ct = resp.headers.get("content-type") ?? "";
  return typeof WebAssembly.compileStreaming === "function" &&
    ct.includes("application/wasm")
    ? WebAssembly.compileStreaming(resp)
    : WebAssembly.compile(await resp.arrayBuffer());
}

function ensureWorker(): Worker {
  if (worker) return worker;
  const w = new MinifyWorker();
  w.addEventListener("message", handleMessage);
  w.addEventListener("error", handleError);
  w.addEventListener("messageerror", handleError);
  worker = w;
  const url = getPreloadedWasmUrl();
  if (!url) {
    w.postMessage({ type: "init-fallback" });
    return w;
  }
  // Posting to a worker dropped meanwhile is a no-op.
  (modulePromise ??= compileWasm(url)).then(
    (module) => w.postMessage({ type: "init-module", module }),
    () => {
      // This worker fetches it itself; the next one retries the compile.
      modulePromise = null;
      w.postMessage({ type: "init-fallback" });
    },
  );
  return w;
}

// Never rejects: a worker that cannot start (e.g. CSP) or an unclonable config
// comes back as a RunError.
export function run(source: string, config?: Config): Promise<RunOutput> {
  const req: RunRequest = { id: ++nextId, source, config };
  return new Promise<RunOutput>((resolve) => {
    pending.set(req.id, { req, resolve });
    try {
      ensureWorker().postMessage(req);
      if (pending.size === 1) armStall();
    } catch (err) {
      pending.delete(req.id);
      resolve({ output: null, error: errorMessage(err) });
    }
  });
}

// Eager so the wasm compiles during app boot.
try {
  ensureWorker();
} catch {
  /* a failure (e.g. CSP) resurfaces from run() */
}
