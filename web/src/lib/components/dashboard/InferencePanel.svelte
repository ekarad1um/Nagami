<script lang="ts">
  import { onMount } from 'svelte';
  import { streams } from '$lib/stores/streams.svelte';
  import TopKMeter from './TopKMeter.svelte';
  import ActiveHeadCard from './ActiveHeadCard.svelte';

  const STATE_LABEL: Record<string, string> = {
    connecting: 'connecting',
    open: 'live',
    closed: 'disconnected',
    error: 'error'
  };

  function pillClass(state: string): string {
    switch (state) {
      case 'open':
        return 'bg-emerald-100 text-emerald-700';
      case 'connecting':
        return 'bg-amber-100 text-amber-700';
      default:
        return 'bg-rose-100 text-rose-700';
    }
  }

  // Scroll-aware fade edges: the mask only applies on the side where
  // additional rows are hidden, so a tight list shows no fade at all and
  // an overflowing one cues the operator that more is available.
  let scrollEl = $state<HTMLDivElement | undefined>();
  let canScrollUp = $state(false);
  let canScrollDown = $state(false);

  function updateFades(el: HTMLDivElement): void {
    canScrollUp = el.scrollTop > 0;
    canScrollDown = el.scrollTop + el.clientHeight < el.scrollHeight - 1;
  }

  $effect(() => {
    // Re-measure whenever the Top-K list changes shape.
    void streams.latestTopK;
    const el = scrollEl;
    if (!el) return;
    queueMicrotask(() => {
      updateFades(el);
    });
  });

  onMount(() => {
    const el = scrollEl;
    if (!el) return;
    const onScroll = (): void => {
      updateFades(el);
    };
    el.addEventListener('scroll', onScroll, { passive: true });
    const ro = new ResizeObserver(() => {
      updateFades(el);
    });
    ro.observe(el);
    updateFades(el);
    return () => {
      el.removeEventListener('scroll', onScroll);
      ro.disconnect();
    };
  });
</script>

<section class="flex h-full flex-col rounded-xl border border-zinc-200 bg-white p-5 shadow-sm">
  <!-- Header pattern mirrors Visualization: title + small meta caption
       baseline-aligned on the left, status pill on the right.  Keeping
       font-mono on the Hz value prevents digit-width jitter as the
       inference rate fluctuates; the surrounding padding/align/color
       (text-[11px] text-zinc-400, gap-2, items-baseline) match Vis. -->
  <header class="mb-3 flex items-center justify-between gap-3">
    <div class="flex items-baseline gap-2">
      <h2 class="text-sm font-semibold text-zinc-900">Inference</h2>
      <span class="font-mono text-[11px] text-zinc-400">{streams.inferenceFps.toFixed(1)} Hz</span>
    </div>
    <span
      class="rounded-full px-2 py-0.5 text-[11px] font-medium capitalize tracking-wide {pillClass(
        streams.inferStatus
      )}"
    >
      {STATE_LABEL[streams.inferStatus] ?? streams.inferStatus}
    </span>
  </header>

  <!-- Top-K's max-h-56 (224px) matches the Visualization spectrogram's
       h-56, so the two main content blocks on left and right read as a
       balanced pair.  The cap still keeps the Inference panel's intrinsic
       height under Visualization's, and auto-overflow scrolls extra rows
       inside this fixed-height window.  Fade edges cue overflow direction. -->
  <div
    bind:this={scrollEl}
    class="max-h-56 overflow-y-auto pr-1"
    class:fade-edge-top={canScrollUp}
    class:fade-edge-bottom={canScrollDown}
  >
    <TopKMeter />
  </div>

  <div class="mt-auto pt-4">
    <ActiveHeadCard />
  </div>
</section>
