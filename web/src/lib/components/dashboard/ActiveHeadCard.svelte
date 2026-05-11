<script lang="ts">
  import { streams } from '$lib/stores/streams.svelte';
  import { config } from '$lib/stores/config.svelte';

  // Prefer the runtime head id stamped on the latest inference frame --
  // that value is atomic with the running weights generation.  Fall back
  // to the REST-reported active head while we wait for the first frame.
  let liveId = $derived<string | null>(
    streams.head.head_id ?? config.active?.runtime_head_id ?? null
  );
  let liveVersion = $derived<number | null>(streams.head.head_version);

  let origin = $derived(config.active?.origin ?? null);
  let nClasses = $derived(config.active?.n_classes ?? null);
  let orphaned = $derived(
    config.active?.origin === 'head' && !config.active.source_workspace_alive
  );
</script>

<aside
  class="rounded-lg border p-3.5"
  class:border-amber-200={orphaned}
  class:bg-amber-50={orphaned}
  class:border-zinc-200={!orphaned}
  class:bg-zinc-50={!orphaned}
>
  <header class="mb-3 flex items-center justify-between">
    <h4 class="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">Active head</h4>
    {#if origin}
      <span
        class="rounded-full px-2 py-0.5 text-[11px] font-medium capitalize tracking-wide"
        class:bg-zinc-200={origin === 'default'}
        class:text-zinc-700={origin === 'default'}
        class:bg-blue-100={origin === 'head'}
        class:text-blue-800={origin === 'head'}>{origin}</span
      >
    {/if}
  </header>

  {#if liveId === null}
    <p class="text-xs text-zinc-400">waiting for first inference frame…</p>
  {:else}
    <!-- All values use the same text-[10px] mono size so they read as a
         consistent metadata block.  `truncate` is the safety net: if a
         value ever exceeds the column width it shows an ellipsis instead
         of wrapping.  At common viewport widths every value (UUID, "v0",
         "20", an 8-char workspace prefix) fits without truncation. -->
    <dl class="grid grid-cols-[3.5rem_1fr] items-baseline gap-x-3 gap-y-1.5 text-xs">
      <dt class="text-zinc-500">id</dt>
      <dd class="truncate font-mono text-[10px] text-zinc-800" title={liveId}>{liveId}</dd>

      {#if liveVersion !== null}
        <dt class="text-zinc-500">version</dt>
        <dd class="truncate font-mono text-[10px] text-zinc-800">v{liveVersion}</dd>
      {/if}

      {#if nClasses !== null}
        <dt class="text-zinc-500">classes</dt>
        <dd class="truncate font-mono text-[10px] text-zinc-800">{nClasses}</dd>
      {/if}

      {#if config.active?.origin === 'head'}
        <dt class="text-zinc-500">workspace</dt>
        <dd
          class="truncate font-mono text-[10px] text-zinc-800"
          title={config.active.source_workspace_id}
        >
          {config.active.source_workspace_id.slice(0, 8)}<span class="text-zinc-400">…</span>
        </dd>

        <dt class="text-zinc-500">revision</dt>
        <dd class="truncate font-mono text-[10px] text-zinc-800">
          {config.active.source_workspace_revision.id}
        </dd>
      {/if}
    </dl>

    {#if orphaned}
      <p class="mt-3 text-[11px] text-amber-800">
        source workspace was deleted; inference continues on the orphaned activation.
      </p>
    {/if}
  {/if}
</aside>
