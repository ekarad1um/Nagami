import { mount } from "svelte";
import "./app.css";
import App from "./App.svelte";
import { warmupHighlighter } from "./lib/highlight.svelte";
import { registerPWA } from "./lib/pwa";

mount(App, { target: document.getElementById("app")! });

// Highlighter off the critical path; service worker for offline + silent updates.
warmupHighlighter();
registerPWA();
