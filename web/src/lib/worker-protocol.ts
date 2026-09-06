import type { Config, Output } from "nagami-rs";

export interface RunRequest {
  id: number;
  source: string;
  config?: Config;
}

export type InitMessage =
  | { type: "init-module"; module: WebAssembly.Module }
  | { type: "init-fallback" }; // the worker fetches the wasm itself

export interface RunResponse {
  id: number;
  output: Output | null;
  error: string | null;
  // The instance is unusable (trap or failed init): replace the worker.
  fatal?: boolean;
}
