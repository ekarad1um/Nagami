import type { Plugin, HtmlTagDescriptor } from 'vite'
import { execSync } from 'node:child_process'

export type Changefreq =
  | 'always'
  | 'hourly'
  | 'daily'
  | 'weekly'
  | 'monthly'
  | 'yearly'
  | 'never'

export interface SitemapRoute {
  path: string
  lastmod?: string | Date
  changefreq?: Changefreq
  priority?: number
}

export interface RobotsRule {
  userAgent: string
  allow?: ReadonlyArray<string>
  disallow?: ReadonlyArray<string>
}

export interface SeoFilesOptions {
  hostname: string
  routes?: ReadonlyArray<string | SitemapRoute>
  defaultChangefreq?: Changefreq
  defaultPriority?: number
  robots?: { rules?: ReadonlyArray<RobotsRule> }
}

const SITEMAP_PATH = '/sitemap.xml'
const ROBOTS_PATH = '/robots.txt'

function escapeXml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;')
}

function normalizeOrigin(hostname: string): string {
  let url: URL
  try {
    url = new URL(hostname)
  } catch {
    throw new Error(`[seo-files] invalid hostname URL: ${hostname}`)
  }
  if (url.protocol !== 'https:' && url.protocol !== 'http:') {
    throw new Error(`[seo-files] hostname must use http(s): ${hostname}`)
  }
  if (
    url.pathname !== '/' ||
    url.search !== '' ||
    url.hash !== '' ||
    url.username !== '' ||
    url.password !== ''
  ) {
    throw new Error(
      `[seo-files] hostname must be an origin only (no path, query, fragment, or userinfo): ${hostname}`,
    )
  }
  return `${url.protocol}//${url.host}`
}

function joinUrl(origin: string, path: string): string {
  return origin + (path.startsWith('/') ? path : '/' + path)
}

function tryGitLastCommitISO(): string | null {
  try {
    const out = execSync('git log -1 --format=%cI', {
      encoding: 'utf-8',
      stdio: ['ignore', 'pipe', 'ignore'],
    }).trim()
    return out || null
  } catch {
    return null
  }
}

function asISODate(value: string | Date): string {
  const d = value instanceof Date ? value : new Date(value)
  if (Number.isNaN(d.getTime())) {
    throw new Error(`[seo-files] invalid lastmod: ${String(value)}`)
  }
  return d.toISOString()
}

function clampPriority(p: number): number {
  if (!Number.isFinite(p)) {
    throw new Error(`[seo-files] priority must be finite, got ${p}`)
  }
  return Math.max(0, Math.min(1, p))
}

function validatePath(path: string): void {
  if (typeof path !== 'string' || !path.startsWith('/')) {
    throw new Error(`[seo-files] route path must start with '/': ${path}`)
  }
}

function dedupeByPath(routes: ReadonlyArray<SitemapRoute>): SitemapRoute[] {
  const seen = new Set<string>()
  const out: SitemapRoute[] = []
  for (const r of routes) {
    if (seen.has(r.path)) continue
    seen.add(r.path)
    out.push(r)
  }
  return out
}

export function seoFiles(options: SeoFilesOptions): Plugin {
  if (!options || typeof options.hostname !== 'string' || !options.hostname) {
    throw new Error('[seo-files] options.hostname is required')
  }
  const origin = normalizeOrigin(options.hostname)

  const rawRoutes = options.routes ?? ['/']
  if (rawRoutes.length === 0) {
    throw new Error('[seo-files] at least one route is required')
  }
  const routes = dedupeByPath(
    rawRoutes.map(r => (typeof r === 'string' ? { path: r } : r)),
  )
  for (const r of routes) validatePath(r.path)

  const defaultChangefreq = options.defaultChangefreq
  const defaultPriority = options.defaultPriority

  let sitemapCache: string | null = null
  let robotsCache: string | null = null

  function renderSitemap(): string {
    const gitLastCommit = tryGitLastCommitISO()
    const fallbackLastmod = asISODate(gitLastCommit ?? new Date())
    const entries = routes.map(r => {
      const loc = escapeXml(joinUrl(origin, r.path))
      const lastmod = r.lastmod ? asISODate(r.lastmod) : fallbackLastmod
      const changefreq = r.changefreq ?? defaultChangefreq
      const priority = r.priority ?? defaultPriority
      const lines = [
        `    <loc>${loc}</loc>`,
        `    <lastmod>${lastmod}</lastmod>`,
      ]
      if (changefreq) lines.push(`    <changefreq>${changefreq}</changefreq>`)
      if (priority !== undefined) {
        lines.push(`    <priority>${clampPriority(priority).toFixed(1)}</priority>`)
      }
      return `  <url>\n${lines.join('\n')}\n  </url>`
    })
    return [
      '<?xml version="1.0" encoding="UTF-8"?>',
      '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
      ...entries,
      '</urlset>',
      '',
    ].join('\n')
  }

  function renderRobots(): string {
    const rules = options.robots?.rules ?? [{ userAgent: '*', allow: ['/'] }]
    const blocks: string[] = []
    for (const r of rules) {
      const lines = [`User-agent: ${r.userAgent}`]
      for (const p of r.allow ?? []) lines.push(`Allow: ${p}`)
      for (const p of r.disallow ?? []) lines.push(`Disallow: ${p}`)
      blocks.push(lines.join('\n'))
    }
    blocks.push(`Sitemap: ${joinUrl(origin, SITEMAP_PATH)}`)
    return blocks.join('\n\n') + '\n'
  }

  function getSitemap(): string {
    if (sitemapCache === null) sitemapCache = renderSitemap()
    return sitemapCache
  }
  function getRobots(): string {
    if (robotsCache === null) robotsCache = renderRobots()
    return robotsCache
  }

  return {
    name: 'seo-files',

    buildStart() {
      sitemapCache = null
      robotsCache = null
    },

    transformIndexHtml(): HtmlTagDescriptor[] {
      const canonical = joinUrl(origin, '/')
      return [
        { tag: 'link', attrs: { rel: 'canonical', href: canonical }, injectTo: 'head' },
        { tag: 'meta', attrs: { property: 'og:url', content: canonical }, injectTo: 'head' },
      ]
    },

    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (!req.url) return next()
        const path = req.url.split('?', 1)[0]
        if (path === SITEMAP_PATH) {
          res.setHeader('Content-Type', 'application/xml; charset=utf-8')
          res.end(getSitemap())
          return
        }
        if (path === ROBOTS_PATH) {
          res.setHeader('Content-Type', 'text/plain; charset=utf-8')
          res.end(getRobots())
          return
        }
        next()
      })
    },

    generateBundle() {
      this.emitFile({
        type: 'asset',
        fileName: SITEMAP_PATH.slice(1),
        source: getSitemap(),
      })
      this.emitFile({
        type: 'asset',
        fileName: ROBOTS_PATH.slice(1),
        source: getRobots(),
      })
    },
  }
}
