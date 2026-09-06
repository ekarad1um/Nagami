<script lang="ts">
  import { highlightSync } from "../lib/highlight.svelte";
  import { downloadTextFile } from "../lib/filename";
  import type { NameMap } from "../lib/nagami";

  interface Props {
    value: string;
    loading: boolean;
    wrap: boolean;
    downloadName: string;
    nameMap: NameMap | null;
  }

  let { value, loading, wrap, downloadName, nameMap }: Props = $props();

  let copied = $state(false);
  let mapName = $derived(downloadName.replace(/\.min\.wgsl$/, ".map.json"));
  // Highlighted in the same flush as the new value: results arrive from the
  // worker, so unlike keystrokes there is no input latency to protect, and a
  // plain-then-coloured flash on every run would cost more than a rare stall.
  let html = $derived(highlightSync(value) ?? "");

  function copy() {
    navigator.clipboard.writeText(value).then(
      () => {
        copied = true;
        setTimeout(() => (copied = false), 1500);
      },
      () => {},
    );
  }

  function downloadMap() {
    if (nameMap) downloadTextFile(JSON.stringify(nameMap, null, 2), mapName);
  }
</script>

<div
  class="flex flex-col min-h-0 min-w-0 flex-1 relative"
  role="region"
  aria-label="Output"
>
  <div
    class="flex items-center justify-between px-3 h-8 border-b border-white/6 bg-[#111111] shrink-0"
  >
    <span
      class="text-[11px] font-medium text-slate-400 uppercase tracking-wider"
      >Output</span
    >
    {#if value}
      <div class="flex gap-1">
        <button
          onclick={copy}
          class="p-1 rounded text-slate-500 hover:text-slate-300 hover:bg-white/6 transition-colors cursor-pointer"
          title="Copy to clipboard"
        >
          {#if copied}
            <svg
              class="w-3.5 h-3.5 text-emerald-400"
              viewBox="0 0 16 16"
              fill="currentColor"
            >
              <path
                d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"
              />
            </svg>
          {:else}
            <svg class="w-3.5 h-3.5" viewBox="0 0 16 16" fill="currentColor">
              <path
                d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 010 1.5h-1.5a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-1.5a.75.75 0 011.5 0v1.5A1.75 1.75 0 019.25 16h-7.5A1.75 1.75 0 010 14.25z"
              />
              <path
                d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0114.25 11h-7.5A1.75 1.75 0 015 9.25zm1.75-.25a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-7.5a.25.25 0 00-.25-.25z"
              />
            </svg>
          {/if}
        </button>
        <button
          onclick={() => downloadTextFile(value, downloadName)}
          class="p-1 rounded text-slate-500 hover:text-slate-300 hover:bg-white/6 transition-colors cursor-pointer"
          title="Download .min.wgsl"
        >
          <svg class="w-3.5 h-3.5" viewBox="0 0 16 16" fill="currentColor">
            <path
              d="M2.75 14A1.75 1.75 0 011 12.25v-2.5a.75.75 0 011.5 0v2.5c0 .138.112.25.25.25h10.5a.25.25 0 00.25-.25v-2.5a.75.75 0 011.5 0v2.5A1.75 1.75 0 0113.25 14z"
            />
            <path
              d="M7.25 7.689V2a.75.75 0 011.5 0v5.689l1.97-1.969a.749.749 0 111.06 1.06l-3.25 3.25a.749.749 0 01-1.06 0L4.22 6.78a.749.749 0 111.06-1.06z"
            />
          </svg>
        </button>
        {#if nameMap}
          <button
            onclick={downloadMap}
            class="p-1 rounded text-slate-500 hover:text-slate-300 hover:bg-white/6 transition-colors cursor-pointer"
            title="Download .map.json (original → final names)"
          >
            <!-- Hand-drawn after SF Symbols rectangle.2.swap on the Octicons
                 16 grid: two rects exchanging = original <-> final names. -->
            <svg
              class="w-3.5 h-3.5"
              viewBox="0 0 16 16"
              fill="none"
              stroke="currentColor"
              stroke-width="1.5"
              stroke-linecap="round"
              stroke-linejoin="round"
            >
              <rect x="1.25" y="2.25" width="6" height="4" rx="1.25" />
              <rect x="8.75" y="9.75" width="6" height="4" rx="1.25" />
              <path
                d="M9 4.25h2.5A1.75 1.75 0 0 1 13.25 6v2.25M11.25 6.25l2 2 2-2"
              />
              <path
                d="M7 11.75H4.5A1.75 1.75 0 0 1 2.75 10V7.75M4.75 9.75l-2-2-2 2"
              />
            </svg>
          </button>
        {/if}
      </div>
    {/if}
  </div>

  <!-- Overlay, out of flow: toggling it must not shift the code area on every minify. -->
  {#if loading}
    <div
      class="loading-bar absolute left-0 right-0 top-8 z-10 pointer-events-none"
    ></div>
  {/if}

  <div
    data-searchable
    class="highlight-panel code-panel code-text flex-1 w-full overflow-auto p-3 selection:bg-slate-600/40 {wrap
      ? 'wrap-on'
      : ''}"
  >
    {#if html}
      <!-- eslint-disable-next-line svelte/no-at-html-tags -- shiki escapes its output -->
      {@html html}
    {:else if value}
      <pre
        class="m-0 p-0 text-slate-200"
        class:whitespace-pre={!wrap}
        class:whitespace-pre-wrap={wrap}
        class:break-all={wrap}>{value}</pre>
    {:else}
      <span class="text-slate-600" data-placeholder
        >{loading ? "Minifying..." : "Minified output appears here"}</span
      >
    {/if}
  </div>
</div>
