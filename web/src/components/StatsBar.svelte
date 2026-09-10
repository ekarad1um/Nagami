<script lang="ts">
  interface Props {
    inputBytes: number;
    outputBytes: number;
    error: string | null;
    bailout: string | null;
    fallback: string | null;
    loading: boolean;
    optionsOpen: boolean;
    onToggleOptions: () => void;
  }

  let {
    inputBytes,
    outputBytes,
    error,
    bailout,
    fallback,
    loading,
    optionsOpen,
    onToggleOptions,
  }: Props = $props();

  // Error (red), bailout or fallback (amber): a bailout ships the input
  // compacted, so byte stats would overstate the win; a fallback ships naga's
  // emitter text, which only the CLI otherwise warns about.
  let notice = $derived(
    error
      ? { text: error, amber: false }
      : bailout
        ? {
            text: `Not optimized: ${bailout}\nThe output is the input with comments removed and whitespace collapsed.`,
            amber: true,
          }
        : fallback
          ? {
              text: `Partially optimized: ${fallback}.\nThe output was minified by the IR passes but printed by naga's emitter, so it is larger than nagami's own text.`,
              amber: true,
            }
          : null,
  );

  let noticeExpanded = $state(false);
  let noticeContainer: HTMLDivElement | undefined = $state(undefined);

  $effect(() => {
    void notice;
    noticeExpanded = false;
  });

  function handleWindowClick(e: MouseEvent) {
    if (
      noticeExpanded &&
      noticeContainer &&
      !noticeContainer.contains(e.target as Node)
    ) {
      noticeExpanded = false;
    }
  }

  let savings = $derived(
    inputBytes > 0 && outputBytes > 0 && outputBytes < inputBytes
      ? ((1 - outputBytes / inputBytes) * 100).toFixed(1)
      : null,
  );

  function capitalize(s: string): string {
    return s.charAt(0).toUpperCase() + s.slice(1).trimEnd();
  }
</script>

<svelte:window onclick={handleWindowClick} />

<div
  class="flex items-center justify-between px-3 py-1.5 border-t border-white/6 bg-[#111111] text-xs shrink-0 select-none relative z-20"
>
  <div class="flex items-center gap-2 text-slate-400 tabular-nums">
    {#if notice}
      <div class="relative flex items-center" bind:this={noticeContainer}>
        <button
          class="flex items-center gap-1 transition-colors px-1.5 py-0.5 rounded hover:bg-white/6 cursor-pointer max-w-[60vw] {notice.amber
            ? 'text-amber-400 hover:text-amber-300'
            : 'text-red-400 hover:text-red-300'}"
          onclick={() => (noticeExpanded = !noticeExpanded)}
          aria-expanded={noticeExpanded}
          title={noticeExpanded ? "Collapse details" : "Expand details"}
        >
          <span class="truncate">{capitalize(notice.text.split("\n")[0])}</span>
          <svg
            class="w-3 h-3 shrink-0 transition-transform"
            style="transform: rotate({noticeExpanded ? '0' : '180'}deg)"
            viewBox="0 0 16 16"
            fill="currentColor"
          >
            <path
              d="M4.22 6.22a.75.75 0 011.06 0L8 8.94l2.72-2.72a.75.75 0 111.06 1.06l-3.25 3.25a.75.75 0 01-1.06 0L4.22 7.28a.75.75 0 010-1.06z"
            />
          </svg>
        </button>
        {#if noticeExpanded}
          <div
            class="absolute bottom-full left-0 mb-5 bg-[#1a1a1a] border border-white/8 rounded-lg shadow-xl p-3 whitespace-pre font-mono text-[11px] max-w-[calc(100vw-1.5rem)] max-h-48 overflow-auto z-50 select-text cursor-text {notice.amber
              ? 'text-amber-400'
              : 'text-red-400'}"
          >
            {capitalize(notice.text)}
          </div>
        {/if}
      </div>
    {:else if inputBytes > 0}
      <span>{inputBytes.toLocaleString()}</span>
      <span class="text-slate-600">→</span>
      <span>{outputBytes.toLocaleString()} bytes</span>
      {#if savings}
        <span class="text-emerald-400 font-medium">{savings}% smaller</span>
      {/if}
    {:else if loading}
      <span class="text-slate-500">Minifying...</span>
    {:else}
      <span class="text-slate-600">Paste or drop a WGSL shader</span>
    {/if}
  </div>

  <button
    class="flex items-center gap-1 text-slate-400 hover:text-slate-300 transition-colors px-1.5 py-0.5 rounded hover:bg-white/6 cursor-pointer"
    onclick={onToggleOptions}
    aria-expanded={optionsOpen}
    title={optionsOpen
      ? "Hide options (mangle, precision, preserve symbols, preamble)"
      : "Show options (mangle, precision, preserve symbols, preamble)"}
  >
    <svg
      class="w-3 h-3 transition-transform"
      style="transform: rotate({optionsOpen ? '0' : '180'}deg)"
      viewBox="0 0 16 16"
      fill="currentColor"
    >
      <path
        d="M4.22 6.22a.75.75 0 011.06 0L8 8.94l2.72-2.72a.75.75 0 111.06 1.06l-3.25 3.25a.75.75 0 01-1.06 0L4.22 7.28a.75.75 0 010-1.06z"
      />
    </svg>
    Options
  </button>
</div>
