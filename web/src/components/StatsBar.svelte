<script lang="ts">
  interface Props {
    inputBytes: number;
    outputBytes: number;
    error: string | null;
    loading: boolean;
    optionsOpen: boolean;
    onToggleOptions: () => void;
  }

  let {
    inputBytes,
    outputBytes,
    error,
    loading,
    optionsOpen,
    onToggleOptions,
  }: Props = $props();

  let errorExpanded = $state(false);
  let errorContainer: HTMLDivElement | undefined = $state(undefined);

  // Collapse error popover when error changes
  $effect(() => {
    error;
    errorExpanded = false;
  });

  function handleWindowClick(e: MouseEvent) {
    if (
      errorExpanded &&
      errorContainer &&
      !errorContainer.contains(e.target as Node)
    ) {
      errorExpanded = false;
    }
  }

  let savings = $derived(
    inputBytes > 0 && outputBytes > 0 && outputBytes < inputBytes
      ? ((1 - outputBytes / inputBytes) * 100).toFixed(1)
      : null,
  );

  function capitalize(s: string): string {
    if (!s) return s;
    let hasNewlineSuffix = s.endsWith("\n");
    return (
      s.charAt(0).toUpperCase() + s.slice(1, hasNewlineSuffix ? -1 : undefined)
    );
  }
</script>

<svelte:window onclick={handleWindowClick} />

<div
  class="flex items-center justify-between px-3 py-1.5 border-t border-white/6 bg-[#111111] text-xs shrink-0 select-none relative z-20"
>
  <div class="flex items-center gap-2 text-slate-400 tabular-nums">
    {#if error}
      <div class="relative flex items-center" bind:this={errorContainer}>
        <button
          class="flex items-center gap-1 text-red-400 hover:text-red-300 transition-colors px-1.5 py-0.5 rounded hover:bg-white/6 cursor-pointer max-w-[60vw]"
          onclick={() => (errorExpanded = !errorExpanded)}
          title={errorExpanded
            ? "Collapse error details"
            : "Expand error details"}
        >
          <span class="truncate">{capitalize(error.split("\n")[0])}</span>
          <svg
            class="w-3 h-3 shrink-0 transition-transform"
            style="transform: rotate({errorExpanded ? '0' : '180'}deg)"
            viewBox="0 0 16 16"
            fill="currentColor"
          >
            <path
              d="M4.22 6.22a.75.75 0 011.06 0L8 8.94l2.72-2.72a.75.75 0 111.06 1.06l-3.25 3.25a.75.75 0 01-1.06 0L4.22 7.28a.75.75 0 010-1.06z"
            />
          </svg>
        </button>
        {#if errorExpanded}
          <div
            class="absolute bottom-full left-0 mb-5 bg-[#1a1a1a] border border-white/8 rounded-lg shadow-xl p-3 whitespace-pre font-mono text-[11px] text-red-400 max-w-[calc(100vw-1.5rem)] max-h-48 overflow-auto z-50 select-text cursor-text"
          >
            {capitalize(error)}
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
      <span class="text-slate-500">Loading WASM...</span>
    {:else}
      <span class="text-slate-600">Paste or drop a WGSL shader</span>
    {/if}
  </div>

  <button
    class="flex items-center gap-1 text-slate-400 hover:text-slate-300 transition-colors px-1.5 py-0.5 rounded hover:bg-white/6 cursor-pointer"
    onclick={onToggleOptions}
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
