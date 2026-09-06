/// <reference lib="webworker" />
import init, { initSync, run as wasmRun } from "nagami-rs";
import type { InitMessage, RunRequest, RunResponse } from "./worker-protocol";

const ctx = self as unknown as DedicatedWorkerGlobalScope;

// Settled by the init message; requests wait on it.
let startInit!: (init: Promise<void>) => void;
const ready = new Promise<void>((resolve) => {
  startInit = resolve;
});
ready.catch(() => {}); // no unhandled rejection before the first request

// Once the instance is unusable this worker answers nothing further with it;
// the host replaces it. A trap, or any exception unwinding wasm frames (V8's
// stack-overflow RangeError included), leaves the shadow stack pointer and
// possibly the heap mid-operation.
let dead: string | null = null;

function isTrap(err: unknown): boolean {
  return err instanceof WebAssembly.RuntimeError || err instanceof RangeError;
}

function message(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

function die(id: number, reason: string): void {
  dead = reason;
  ctx.postMessage({
    id,
    output: null,
    error: reason,
    fatal: true,
  } satisfies RunResponse);
}

function isInitMessage(d: RunRequest | InitMessage): d is InitMessage {
  const t = (d as InitMessage).type;
  return t === "init-module" || t === "init-fallback";
}

ctx.onmessage = (e: MessageEvent<RunRequest | InitMessage>) => {
  const data = e.data;

  if (isInitMessage(data)) {
    startInit(
      (async () => {
        if (data.type === "init-module") initSync({ module: data.module });
        else await init();
      })(),
    );
    return;
  }

  const { id, source, config } = data;
  void (async () => {
    if (dead) return die(id, dead);
    try {
      await ready;
    } catch (err) {
      // Fatal so the host retries the load with a fresh worker.
      return die(id, `Minifier failed to start: ${message(err)}`);
    }
    try {
      const output = wasmRun(source, config);
      ctx.postMessage({ id, output, error: null } satisfies RunResponse);
    } catch (err) {
      if (isTrap(err)) {
        die(
          id,
          `Minifier crashed: ${message(err)}\nIt was restarted. Very long operator chains or deeply nested code exhaust its stack.`,
        );
      } else {
        ctx.postMessage({
          id,
          output: null,
          error: message(err),
        } satisfies RunResponse);
      }
    }
  })();
};
