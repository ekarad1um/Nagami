import init, {
  run as wasmRun,
  type Config,
  type Output,
  type Report,
  type PassReport,
} from "nagami-rs";

export type { Config, Output, Report, PassReport };

let initPromise: Promise<void> | null = null;
let initRetryDelay = 0;

function ensureInit(): Promise<void> {
  if (!initPromise) {
    initPromise = init().then(
      () => { initRetryDelay = 0; },
      (err) => {
        initRetryDelay = Math.min((initRetryDelay || 500) * 2, 30_000);
        setTimeout(() => { initPromise = null; }, initRetryDelay);
        throw err;
      },
    );
  }
  return initPromise;
}

export interface RunResult {
  output: Output;
  error: null;
}

export interface RunError {
  output: null;
  error: string;
}

export type RunOutput = RunResult | RunError;

export async function run(source: string, config?: Config): Promise<RunOutput> {
  try {
    await ensureInit();
    const output = wasmRun(source, config);
    return { output, error: null };
  } catch (e) {
    return { output: null, error: e instanceof Error ? e.message : String(e) };
  }
}

// Eagerly start loading WASM in the background
ensureInit();
