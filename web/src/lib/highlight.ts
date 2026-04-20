import { createHighlighterCore, type HighlighterCore, type ShikiTransformer } from "shiki/core";
import { createJavaScriptRegexEngine } from "@shikijs/engine-javascript";
import wgsl from "shiki/langs/wgsl.mjs";
import githubDarkDimmed from "shiki/themes/github-dark-dimmed.mjs";

let highlighter: HighlighterCore | null = null;
let initPromise: Promise<HighlighterCore> | null = null;

function ensureHighlighter(): Promise<HighlighterCore> {
    if (highlighter) return Promise.resolve(highlighter);
    if (!initPromise) {
        initPromise = createHighlighterCore({
            engine: createJavaScriptRegexEngine(),
            themes: [githubDarkDimmed],
            langs: [wgsl],
        }).then(
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

// Returns highlighted HTML synchronously if the highlighter is ready, otherwise null
export function highlightSync(code: string): string | null {
    if (!highlighter || !code) return null;
    return highlighter.codeToHtml(code, shikiOpts);
}

// Ensures shiki is loaded; returns highlighted HTML
export async function highlight(code: string): Promise<string> {
    const h = await ensureHighlighter();
    return h.codeToHtml(code, shikiOpts);
}

// Eagerly start loading the highlighter in the background
ensureHighlighter();
