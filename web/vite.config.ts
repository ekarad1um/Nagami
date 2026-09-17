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
        globPatterns: ["**/*.{js,css,html,svg,wgsl,wasm}"],
        maximumFileSizeToCacheInBytes: 4 * 1024 * 1024,
        manifestTransforms: [
          (entries) => {
            if (!entries.some((e) => e.url.endsWith(".wasm")))
              throw new Error("wasm missing from the precache manifest");
            return { manifest: entries };
          },
        ],
        cleanupOutdatedCaches: true,
        directoryIndex: null,
        navigationPreload: true,
        navigateFallback: null,
        runtimeCaching: [
          {
            urlPattern: ({ request, url }) =>
              request.mode === "navigate" &&
              (url.pathname === "/" || url.pathname === "/index.html"),
            handler: "NetworkFirst",
            options: {
              cacheName: "nagami-shell",
              networkTimeoutSeconds: 3,
              expiration: { maxEntries: 4 },
              precacheFallback: { fallbackURL: "index.html" },
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
