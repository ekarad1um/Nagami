import { defineConfig, type Plugin } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'
import tailwindcss from '@tailwindcss/vite'
import { ViteMinifyPlugin } from 'vite-plugin-minify'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { seoFiles } from './vite-plugins/seo-files'

const pkg = JSON.parse(
  readFileSync(resolve(import.meta.dirname, 'package.json'), 'utf-8'),
) as { homepage?: string }
if (!pkg.homepage) {
  throw new Error('package.json must define a "homepage" URL for SEO files.')
}

function assetPreload(): Plugin {
  return {
    name: 'asset-preload',
    transformIndexHtml: {
      order: 'post',
      handler(_html, ctx) {
        const tags: ReturnType<typeof Array<any>> = []
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
    ViteMinifyPlugin(),
  ],
  base: '/',
  optimizeDeps: {
    exclude: ['nagami-rs'],
  },
})
