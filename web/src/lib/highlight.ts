import type { HighlighterCore, ShikiTransformer } from "shiki/core";

let highlighter: HighlighterCore | null = null;
let initPromise: Promise<HighlighterCore> | null = null;

async function loadHighlighter(): Promise<HighlighterCore> {
    const [{ createHighlighterCore }, { createJavaScriptRegexEngine }, wgsl, theme] =
        await Promise.all([
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

function ensureHighlighter(): Promise<HighlighterCore> {
    if (highlighter) return Promise.resolve(highlighter);
    if (!initPromise) {
        initPromise = loadHighlighter().then(
            (h) => {
                highlighter = h;
                return h;
            },
            (err) => {
                initPromise = null;
                throw err;
            },
        );
    }
    return initPromise;
}

const noTabindex: ShikiTransformer = {
    name: "no-tabindex",
    pre(node) { delete node.properties.tabindex; },
};

const shikiOpts = {
    lang: "wgsl" as const,
    theme: "github-dark-dimmed" as const,
    transformers: [noTabindex],
};

// Highlighted HTML if the highlighter is already loaded, else null.
export function highlightSync(code: string): string | null {
    if (!highlighter || !code) return null;
    return highlighter.codeToHtml(code, shikiOpts);
}

// Loads shiki if needed, then returns highlighted HTML.
export async function highlight(code: string): Promise<string> {
    const h = await ensureHighlighter();
    return h.codeToHtml(code, shikiOpts);
}

// Warm shiki up on idle so the first highlight is instant. Failures are swallowed;
// highlight() retries on demand.
export function warmupHighlighter(): void {
    const start = () => ensureHighlighter().catch(() => { });
    if (typeof requestIdleCallback === "function") {
        requestIdleCallback(start, { timeout: 2000 });
    } else {
        setTimeout(start, 200);
    }
}
