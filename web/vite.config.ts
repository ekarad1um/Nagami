import { defineConfig, type Plugin, type HtmlTagDescriptor } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import tailwindcss from "@tailwindcss/vite";
import { ViteMinifyPlugin } from "vite-plugin-minify";
import { VitePWA } from "vite-plugin-pwa";

// Preloads the hashed wasm; nagami.ts reads this link's href to consume it.
// Build-only: ctx.bundle is undefined in dev.
function wasmPreload(): Plugin {
  return {
    name: "wasm-preload",
    transformIndexHtml: {
      order: "post",
      handler(_html, ctx): HtmlTagDescriptor[] {
        const wasm =
          ctx.bundle &&
          Object.keys(ctx.bundle).find((f) => f.endsWith(".wasm"));
        if (!wasm) return [];
        return [
          {
            tag: "link",
            attrs: {
              rel: "preload",
              href: `/${wasm}`,
              as: "fetch",
              crossorigin: "anonymous",
            },
            injectTo: "head",
          },
        ];
      },
    },
  };
}

export default defineConfig({
  plugins: [
    svelte(),
    tailwindcss(),
    wasmPreload(),
    VitePWA({
      registerType: "prompt",
      injectRegister: false,
      workbox: {
        skipWaiting: true,
        clientsClaim: false,
        globPatterns: ["**/*.{js,css,html,svg,wgsl}"],
        cleanupOutdatedCaches: true,
        // Single route: the precache already serves / offline, and a fallback
        // would answer /llms.txt, /robots.txt and friends with the app shell.
        navigateFallback: null,
        runtimeCaching: [
          {
            urlPattern: ({ url }) => url.pathname.endsWith(".wasm"),
            handler: "CacheFirst",
            options: {
              cacheName: "nagami-wasm",
              expiration: { maxEntries: 4, maxAgeSeconds: 60 * 60 * 24 },
              cacheableResponse: { statuses: [200] },
            },
          },
        ],
      },
      manifest: {
        name: "Nagami - WGSL Shader Minifier",
        short_name: "Nagami",
        description:
          "IR-level WGSL shader minifier - DCE, constant folding, inlining, and mangling. Runs entirely in the browser via WebAssembly.",
        theme_color: "#0c0c0c",
        background_color: "#0c0c0c",
        display: "standalone",
        start_url: "/",
        icons: [
          {
            src: "/favicon.svg",
            sizes: "any",
            type: "image/svg+xml",
            purpose: "any",
          },
        ],
      },
    }),
    ViteMinifyPlugin(),
  ],
  base: "/",
  worker: {
    format: "es",
  },
  optimizeDeps: {
    exclude: ["nagami-rs"],
  },
});
