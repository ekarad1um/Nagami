import { registerSW } from "virtual:pwa-register";

// Long-lived tabs never navigate, so poll for a new SW.
const UPDATE_INTERVAL_MS = 60 * 60 * 1000;

// Silent: no prompt, no forced reload; a new SW serves the next load.
export function registerPWA(): void {
  registerSW({
    immediate: true,
    onRegisteredSW(_swUrl, registration) {
      if (!registration) return;
      caches.delete("nagami-wasm").catch(() => {});
      // update() rejects offline; expected.
      const check = () => registration.update().catch(() => {});
      check();
      setInterval(check, UPDATE_INTERVAL_MS);
      document.addEventListener("visibilitychange", () => {
        if (document.visibilityState === "visible") check();
      });
    },
  });
}
