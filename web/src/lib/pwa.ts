import { registerSW } from "virtual:pwa-register";

// How often to poll for a newer SW while a tab stays open.
const UPDATE_INTERVAL_MS = 60 * 60 * 1000; // 1 hour

// Register the SW for offline support + silent background updates, no prompt, no forced reload.
export function registerPWA(): void {
  registerSW({
    immediate: true,
    onRegisteredSW(_swUrl, registration) {
      if (!registration) return;
      // update() rejects when offline/unreachable; that's expected, so swallow it.
      const check = () => registration.update().catch(() => { });
      check(); // on load
      setInterval(check, UPDATE_INTERVAL_MS); // then hourly
      document.addEventListener("visibilitychange", () => {
        if (document.visibilityState === "visible") check();
      });
    },
  });
}
