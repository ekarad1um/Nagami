import { mount } from 'svelte'
import './app.css'
import App from './App.svelte'
import { warmupHighlighter } from './lib/highlight'
import { registerPWA } from './lib/pwa'

const app = mount(App, {
  target: document.getElementById('app')!,
})

// Preload the syntax highlighter off the critical path, once the page is idle.
warmupHighlighter()

// Offline support + silent background updates (applied on the next manual refresh).
registerPWA()

export default app
