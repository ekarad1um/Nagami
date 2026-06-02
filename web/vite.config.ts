import { defineConfig, type Plugin, type HtmlTagDescriptor } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'
import tailwindcss from '@tailwindcss/vite'
import { ViteMinifyPlugin } from 'vite-plugin-minify'
import { VitePWA } from 'vite-plugin-pwa'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { seoFiles } from './vite-plugins/seo-files'

const pkg = JSON.parse(
  readFileSync(resolve(import.meta.dirname, 'package.json'), 'utf-8'),
) as { homepage?: string }
if (!pkg.homepage) {
  throw new Error('package.json must define a "homepage" URL for SEO files.')
}

// Preloads the hashed wasm (nagami.ts reads this link's href to consume it) and the
// sample shader. Build-only: ctx.bundle is undefined in dev.
function assetPreload(): Plugin {
  return {
    name: 'asset-preload',
    transformIndexHtml: {
      order: 'post',
      handler(_html, ctx) {
        const tags: HtmlTagDescriptor[] = []
        const wasmFile = ctx.bundle
          ? Object.keys(ctx.bundle).find(f => f.endsWith('.wasm'))
          : undefined
        if (wasmFile) {
          tags.push({
            tag: 'link',
            attrs: { rel: 'preload', href: `/${wasmFile}`, as: 'fetch', crossorigin: 'anonymous' },
            injectTo: 'head',
          })
        }
        // Preload sample shader to avoid waiting for JS to fetch it
        tags.push({
          tag: 'link',
          attrs: { rel: 'preload', href: '/example.wgsl', as: 'fetch', crossorigin: 'anonymous' },
          injectTo: 'head',
        })
        return tags
      },
    },
  }
}

export default defineConfig({
  plugins: [
    svelte(),
    tailwindcss(),
    assetPreload(),
    seoFiles({
      hostname: pkg.homepage,
      routes: ['/'],
      defaultChangefreq: 'weekly',
      defaultPriority: 1.0,
    }),
    VitePWA({
      registerType: 'prompt',
      injectRegister: false,
      workbox: {
        skipWaiting: true,
        clientsClaim: false,
        globPatterns: ['**/*.{js,css,html,svg,wgsl}'],
        cleanupOutdatedCaches: true,
        navigateFallback: '/index.html',
        navigateFallbackDenylist: [/^\/robots\.txt$/, /^\/sitemap\.xml$/],
        runtimeCaching: [
          {
            urlPattern: ({ url }) => url.pathname.endsWith('.wasm'),
            handler: 'CacheFirst',
            options: {
              cacheName: 'nagami-wasm',
              // A few entries cover old+new hashes during an update; LRU evicts stale.
              expiration: { maxEntries: 4, maxAgeSeconds: 60 * 60 * 24 * 30 },
              cacheableResponse: { statuses: [0, 200] },
            },
          },
        ],
      },
      manifest: {
        name: 'Nagami - WGSL Shader Minifier',
        short_name: 'Nagami',
        description:
          'IR-level WGSL shader minifier - DCE, constant folding, inlining, and mangling. Runs entirely in the browser via WebAssembly.',
        theme_color: '#0c0c0c',
        background_color: '#0c0c0c',
        display: 'standalone',
        start_url: '/',
        icons: [
          {
            src: '/favicon.svg',
            sizes: 'any',
            type: 'image/svg+xml',
            purpose: 'any',
          },
        ],
      },
    }),
    ViteMinifyPlugin(),
  ],
  base: '/',
  worker: {
    format: 'es',
  },
  optimizeDeps: {
    exclude: ['nagami-rs'],
  },
})
