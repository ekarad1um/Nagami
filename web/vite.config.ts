import { sveltekit } from '@sveltejs/kit/vite';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig } from 'vite';

const DAEMON = '127.0.0.1:8787';

export default defineConfig({
  plugins: [tailwindcss(), sveltekit()],
  server: {
    proxy: {
      '/api': { target: `http://${DAEMON}`, changeOrigin: false },
      '/stream/audio': { target: `ws://${DAEMON}`, ws: true, changeOrigin: false },
      '/stream/infer': { target: `ws://${DAEMON}`, ws: true, changeOrigin: false }
    }
  },
  worker: {
    format: 'es'
  }
});
