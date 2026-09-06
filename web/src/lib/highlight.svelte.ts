import type { HighlighterCore, ShikiTransformer } from "shiki/core";

// Shiki tokenizes about 2 MB/s on the main thread; larger inputs stay plain.
const MAX_HIGHLIGHT_CHARS = 256 * 1024;

// Reactive so highlighted views recompute once the lazy chunks land.
let highlighter: HighlighterCore | null = $state.raw(null);

const noTabindex: ShikiTransformer = {
  name: "no-tabindex",
  pre(node) {
    delete node.properties.tabindex;
  },
};

const shikiOpts = {
  lang: "wgsl" as const,
  theme: "github-dark-dimmed" as const,
  transformers: [noTabindex],
};

async function loadHighlighter(): Promise<HighlighterCore> {
  const [
    { createHighlighterCore },
    { createJavaScriptRegexEngine },
    wgsl,
    theme,
  ] = await Promise.all([
    import("shiki/core"),
    import("@shikijs/engine-javascript"),
    import("shiki/langs/wgsl.mjs"),
    import("shiki/themes/github-dark-dimmed.mjs"),
  ]);
  return createHighlighterCore({
    engine: createJavaScriptRegexEngine(),
    themes: [theme.default],
    langs: [wgsl.default],
  });
}

// Off the critical path. A failed chunk load (offline, stale deploy) is retried
// a few times with backoff; code stays plain meanwhile and forever after that.
export function warmupHighlighter(): void {
  const load = (attempt: number) => {
    loadHighlighter().then(
      (h) => (highlighter = h),
      () => {
        if (attempt < 4)
          setTimeout(() => load(attempt + 1), 5000 * (attempt + 1));
      },
    );
  };
  const start = () => load(0);
  if (typeof requestIdleCallback === "function") {
    requestIdleCallback(start, { timeout: 2000 });
  } else {
    setTimeout(start, 200);
  }
}

export function highlighterReady(): boolean {
  return highlighter !== null;
}

// Highlighted HTML, or null while shiki is loading or the code is over the cap.
export function highlightSync(code: string): string | null {
  if (!highlighter || !code || code.length > MAX_HIGHLIGHT_CHARS) return null;
  return highlighter.codeToHtml(code, shikiOpts);
}
