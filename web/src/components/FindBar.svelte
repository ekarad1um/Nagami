<script lang="ts">
    import { untrack } from "svelte";

    interface Props {
        focusTrigger: number;
        inputContent: string;
        outputContent: string;
        onclose: () => void;
    }

    let { focusTrigger, inputContent, outputContent, onclose }: Props =
        $props();

    let query = $state("");
    let matchCount = $state(0);
    let currentIndex = $state(-1);
    let inputEl: HTMLInputElement | undefined = $state(undefined);

    let caseSensitive = $state(false);
    let useRegex = $state(false);
    let searchScope: "all" | "input" | "output" = $state("input");
    let regexError = $state(false);

    let allRanges: Range[] = [];

    const hasHighlightAPI = typeof CSS !== "undefined" && "highlights" in CSS;
    const scopeLabels = { all: "All", input: "In", output: "Out" } as const;

    function cycleScope() {
        const order: ("all" | "input" | "output")[] = [
            "all",
            "input",
            "output",
        ];
        searchScope = order[(order.indexOf(searchScope) + 1) % order.length];
    }

    // Focus input when opened or re-triggered
    $effect(() => {
        focusTrigger;
        inputEl?.focus();
        inputEl?.select();
    });

    // Rebuild search ranges when query, content, regex mode, or scope changes
    $effect(() => {
        inputContent;
        outputContent;
        caseSensitive;
        useRegex;
        searchScope;
        const q = query.trim();
        clearHighlights();
        allRanges = [];
        regexError = false;

        if (!q) {
            matchCount = 0;
            currentIndex = -1;
            return;
        }

        // Validate regex early
        const reFlags = caseSensitive ? "g" : "gi";
        if (useRegex) {
            try {
                new RegExp(q, reFlags);
            } catch {
                regexError = true;
                matchCount = 0;
                currentIndex = -1;
                return;
            }
        }

        // Select containers based on scope
        const selector =
            searchScope === "input"
                ? '[aria-label="Input"] [data-searchable]'
                : searchScope === "output"
                  ? '[aria-label="Output"] [data-searchable]'
                  : "[data-searchable]";
        const containers = document.querySelectorAll<HTMLElement>(selector);

        for (const container of containers) {
            // Build concatenated text + node mapping for cross-node matching
            const walker = document.createTreeWalker(
                container,
                NodeFilter.SHOW_TEXT,
                (node) =>
                    (node.parentElement as Element)?.closest(
                        "[data-placeholder]",
                    )
                        ? NodeFilter.FILTER_REJECT
                        : NodeFilter.FILTER_ACCEPT,
            );
            const nodeMap: { node: Text; start: number; end: number }[] = [];
            let fullText = "";
            let node: Text | null;
            while ((node = walker.nextNode() as Text | null)) {
                const text = node.textContent ?? "";
                nodeMap.push({
                    node,
                    start: fullText.length,
                    end: fullText.length + text.length,
                });
                fullText += text;
            }

            // Find all matches in concatenated text
            const matches: { start: number; end: number }[] = [];
            if (useRegex) {
                const re = new RegExp(q, reFlags);
                let m;
                while ((m = re.exec(fullText)) !== null) {
                    if (m[0].length === 0) {
                        re.lastIndex++;
                        continue;
                    }
                    matches.push({
                        start: m.index,
                        end: m.index + m[0].length,
                    });
                }
            } else {
                const haystack = caseSensitive
                    ? fullText
                    : fullText.toLowerCase();
                const needle = caseSensitive ? q : q.toLowerCase();
                let pos = 0;
                while (pos < fullText.length) {
                    const idx = haystack.indexOf(needle, pos);
                    if (idx === -1) break;
                    matches.push({ start: idx, end: idx + q.length });
                    pos = idx + q.length;
                }
            }

            // Create Range spanning potentially multiple text nodes
            for (const match of matches) {
                const range = new Range();
                let startSet = false;
                for (const nm of nodeMap) {
                    if (!startSet && nm.end > match.start) {
                        range.setStart(nm.node, match.start - nm.start);
                        startSet = true;
                    }
                    if (nm.end >= match.end) {
                        range.setEnd(nm.node, match.end - nm.start);
                        break;
                    }
                }
                allRanges.push(range);
            }
        }

        matchCount = allRanges.length;
        currentIndex = matchCount > 0 ? 0 : -1;
        untrack(() => applyHighlights());
    });

    function clearHighlights() {
        if (!hasHighlightAPI) return;
        CSS.highlights.delete("search-results");
        CSS.highlights.delete("search-current");
    }

    function applyHighlights() {
        if (!hasHighlightAPI || allRanges.length === 0) return;

        try {
            CSS.highlights.set("search-results", new Highlight(...allRanges));
            if (currentIndex >= 0 && currentIndex < allRanges.length) {
                CSS.highlights.set(
                    "search-current",
                    new Highlight(allRanges[currentIndex]),
                );
                scrollToMatch(allRanges[currentIndex]);
            }
        } catch {
            // Ranges became invalid (DOM changed), clear stale state
            clearHighlights();
            allRanges = [];
            matchCount = 0;
            currentIndex = -1;
        }
    }

    function scrollToMatch(range: Range) {
        const node = range.startContainer;
        const el = node instanceof Element ? node : node.parentElement;
        const scrollContainer = el?.closest(".code-panel");
        if (!(scrollContainer instanceof HTMLElement)) return;

        const rangeRect = range.getBoundingClientRect();
        const containerRect = scrollContainer.getBoundingClientRect();

        // Skip if already visible
        if (
            rangeRect.top >= containerRect.top &&
            rangeRect.bottom <= containerRect.bottom
        )
            return;

        const scrollTop =
            scrollContainer.scrollTop +
            (rangeRect.top - containerRect.top) -
            containerRect.height / 2;
        scrollContainer.scrollTo({
            top: Math.max(0, scrollTop),
            behavior: "smooth",
        });
    }

    function goNext() {
        if (matchCount === 0) return;
        currentIndex = (currentIndex + 1) % matchCount;
        clearHighlights();
        applyHighlights();
    }

    function goPrev() {
        if (matchCount === 0) return;
        currentIndex = (currentIndex - 1 + matchCount) % matchCount;
        clearHighlights();
        applyHighlights();
    }

    function handleKeydown(e: KeyboardEvent) {
        if (e.key === "Escape") {
            close();
        } else if (e.key === "Enter") {
            e.preventDefault();
            if (e.shiftKey) goPrev();
            else goNext();
        }
    }

    function close() {
        clearHighlights();
        onclose();
    }

    // Cleanup highlights when component is destroyed
    $effect(() => () => clearHighlights());
</script>

<!-- Injected via {@html} to bypass lightningcss, which doesn't recognize ::highlight() -->
<svelte:head>
    {@html `<style>
::highlight(search-results){background-color:rgba(250,200,50,.25)}
::highlight(search-current){background-color:rgba(250,200,50,.55)}
</style>`}
</svelte:head>

<div
    class="absolute top-2 left-1/2 -translate-x-1/2 z-50 flex items-center gap-1.5 bg-[#1a1a1a]/80 backdrop-blur-md border border-white/8 rounded-lg shadow-lg p-1"
>
    <!-- Search input group -->
    <div
        class="flex items-center bg-white/4 border border-white/6 focus-within:border-slate-500 rounded-md overflow-hidden transition-colors"
    >
        <svg
            class="w-3.5 h-3.5 text-slate-500 shrink-0 ml-2"
            viewBox="0 0 16 16"
            fill="currentColor"
        >
            <path
                d="M10.68 11.74a6 6 0 01-7.922-8.982 6 6 0 018.982 7.922l3.04 3.04a.749.749 0 01-.326 1.275.749.749 0 01-.734-.215zM11.5 7a4.499 4.499 0 10-8.997 0A4.499 4.499 0 0011.5 7z"
            />
        </svg>
        <input
            bind:this={inputEl}
            bind:value={query}
            type="text"
            placeholder="Find…"
            class="bg-transparent text-slate-200 text-xs outline-none w-32 px-1.5 py-1 placeholder:text-slate-600"
            onkeydown={handleKeydown}
            spellcheck="false"
        />
        <div class="flex items-center gap-px pr-1">
            <button
                onclick={() => (caseSensitive = !caseSensitive)}
                class="px-1 py-0.5 text-[10px] font-medium leading-none rounded-sm cursor-pointer transition-colors {caseSensitive
                    ? 'text-emerald-400 bg-emerald-400/15'
                    : 'text-slate-500 hover:text-slate-300 hover:bg-white/8'}"
                title="Match Case"
            >
                Aa
            </button>
            <button
                onclick={() => (useRegex = !useRegex)}
                class="px-1 py-0.5 text-[10px] font-mono leading-none rounded-sm cursor-pointer transition-colors {regexError
                    ? 'text-red-400 bg-red-400/15'
                    : useRegex
                      ? 'text-emerald-400 bg-emerald-400/15'
                      : 'text-slate-500 hover:text-slate-300 hover:bg-white/8'}"
                title="Use Regular Expression"
            >
                .*
            </button>
            <button
                onclick={cycleScope}
                class="px-1 py-0.5 text-[10px] font-medium leading-none rounded-sm cursor-pointer transition-colors whitespace-nowrap {searchScope !==
                'all'
                    ? 'text-emerald-400 bg-emerald-400/15'
                    : 'text-slate-500 hover:text-slate-300 hover:bg-white/8'}"
                title="Search scope: {scopeLabels[
                    searchScope
                ]} (click to cycle)"
            >
                {scopeLabels[searchScope]}
            </button>
        </div>
    </div>

    <!-- Match count -->
    <span
        class="text-[10px] leading-none tabular-nums whitespace-nowrap select-none text-slate-500 min-w-8 text-center"
    >
        {query.trim()
            ? matchCount > 0
                ? `${currentIndex + 1}/${matchCount}`
                : "0/0"
            : "N/A"}
    </span>

    <!-- Navigation -->
    <div class="flex items-center gap-px">
        <button
            onclick={goPrev}
            class="p-1 rounded transition-colors cursor-pointer {matchCount ===
            0
                ? 'text-slate-600'
                : 'text-slate-400 hover:text-slate-200 hover:bg-white/8'}"
            title="Previous match (Shift+Enter)"
            disabled={matchCount === 0}
        >
            <svg class="w-3.5 h-3.5" viewBox="0 0 16 16" fill="currentColor">
                <path
                    d="M3.22 9.78a.75.75 0 010-1.06l4.25-4.25a.75.75 0 011.06 0l4.25 4.25a.751.751 0 01-.018 1.042.751.751 0 01-1.042.018L8 6.06 4.28 9.78a.75.75 0 01-1.06 0z"
                />
            </svg>
        </button>
        <button
            onclick={goNext}
            class="p-1 rounded transition-colors cursor-pointer {matchCount ===
            0
                ? 'text-slate-600'
                : 'text-slate-400 hover:text-slate-200 hover:bg-white/8'}"
            title="Next match (Enter)"
            disabled={matchCount === 0}
        >
            <svg class="w-3.5 h-3.5" viewBox="0 0 16 16" fill="currentColor">
                <path
                    d="M12.78 6.22a.75.75 0 010 1.06l-4.25 4.25a.75.75 0 01-1.06 0L3.22 7.28a.751.751 0 01.018-1.042.751.751 0 011.042-.018L8 9.94l3.72-3.72a.75.75 0 011.06 0z"
                />
            </svg>
        </button>
    </div>

    <!-- Close -->
    <button
        onclick={close}
        class="p-1 text-slate-500 hover:text-slate-200 cursor-pointer rounded hover:bg-white/8 transition-colors"
        title="Close (Esc)"
    >
        <svg class="w-3.5 h-3.5" viewBox="0 0 16 16" fill="currentColor">
            <path
                d="M3.72 3.72a.75.75 0 011.06 0L8 6.94l3.22-3.22a.749.749 0 011.06 1.06L9.06 8l3.22 3.22a.749.749 0 11-1.06 1.06L8 9.06l-3.22 3.22a.749.749 0 11-1.06-1.06L6.94 8 3.72 4.78a.75.75 0 010-1.06z"
            />
        </svg>
    </button>
</div>
