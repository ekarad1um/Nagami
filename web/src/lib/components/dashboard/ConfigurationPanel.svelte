<script lang="ts">
  import { config } from '$lib/stores/config.svelte';
  import type { MicPolicy } from '$lib/api/types';

  // Source select holds either the literal 'auto' (= first_available) or
  // a candidate id (= fixed device).  Channel select holds 'auto' or a
  // stringified channel index.  Collapsing the policy's two-field shape
  // into a single dropdown keeps the form intrinsically stable -- no
  // conditional rows that resize the column on toggle.
  let sourceSel = $state<string>('auto');
  let channelSel = $state<string>('auto');

  let hopSamples = $state(1024);
  let topK = $state(3);

  $effect(() => {
    const m = config.mic;
    if (!m) return;
    sourceSel = m.policy.mic.kind === 'fixed' ? m.policy.mic.id : 'auto';
    channelSel = m.policy.channel.kind === 'fixed' ? String(m.policy.channel.channel) : 'auto';
  });

  $effect(() => {
    const c = config.inference;
    if (!c) return;
    hopSamples = c.hop_samples;
    topK = c.top_k;
  });

  let candidates = $derived(config.mic?.catalogue.candidates ?? []);

  let channelOptions = $derived.by(() => {
    const m = config.mic;
    if (!m) return [] as number[];
    const targetId = sourceSel === 'auto' ? (m.catalogue.candidates[0]?.id ?? '') : sourceSel;
    const cand = m.catalogue.candidates.find((c) => c.id === targetId);
    return cand?.channels ?? [];
  });

  let micDirty = $derived.by(() => {
    const cur = config.mic?.policy;
    if (!cur) return false;
    const wantMic = sourceSel === 'auto' ? null : sourceSel;
    const wantCh = channelSel === 'auto' ? null : Number(channelSel);
    const curMic = cur.mic.kind === 'fixed' ? cur.mic.id : null;
    const curCh = cur.channel.kind === 'fixed' ? cur.channel.channel : null;
    return wantMic !== curMic || wantCh !== curCh;
  });

  let inferDirty = $derived.by(() => {
    const c = config.inference;
    if (!c) return false;
    return c.hop_samples !== hopSamples || c.top_k !== topK;
  });

  // Hop and Top-K sliders surface their progress as a CSS custom property
  // so the track can paint a coloured fill up to the thumb (see app.css
  // -- ::-webkit-slider-runnable-track gradient).  Firefox uses the
  // native ::-moz-range-progress and ignores --slider-percent.
  const HOP_MIN = 256;
  const HOP_MAX = 33_024;
  const TOPK_MIN = 1;
  const TOPK_MAX = 20;
  let hopPct = $derived(((hopSamples - HOP_MIN) / (HOP_MAX - HOP_MIN)) * 100);
  let topKPct = $derived(((topK - TOPK_MIN) / (TOPK_MAX - TOPK_MIN)) * 100);

  async function applyMic(): Promise<void> {
    const policy: MicPolicy = {
      mic: sourceSel === 'auto' ? { kind: 'first_available' } : { kind: 'fixed', id: sourceSel },
      channel:
        channelSel === 'auto' ? { kind: 'auto' } : { kind: 'fixed', channel: Number(channelSel) }
    };
    await config.setMicPolicy(policy);
  }

  async function applyInference(): Promise<void> {
    await config.setInferenceCfg({ hop_samples: hopSamples, top_k: topK });
  }

  function approxHz(hop: number): string {
    // Inference engine runs at 16 kHz internally; cadence ~= 16000 / hop Hz.
    const hz = 16_000 / Math.max(1, hop);
    if (hz >= 10) return `${hz.toFixed(0)} Hz`;
    if (hz >= 1) return `${hz.toFixed(1)} Hz`;
    return `${hz.toFixed(2)} Hz`;
  }

  const selectCls =
    'select-chevron block w-full rounded-md border border-zinc-200 bg-white px-2.5 py-1.5 text-xs transition disabled:cursor-not-allowed disabled:bg-zinc-50 disabled:text-zinc-400';
  const primaryBtn =
    'inline-flex items-center justify-center rounded-md bg-zinc-900 px-3.5 py-1.5 text-xs font-medium text-white transition hover:bg-zinc-800 disabled:cursor-not-allowed disabled:bg-zinc-100 disabled:text-zinc-400 disabled:hover:bg-zinc-100';
</script>

<section class="rounded-xl border border-zinc-200 bg-white p-5 shadow-sm">
  <header class="mb-4 flex items-baseline justify-between">
    <h2 class="text-sm font-semibold text-zinc-900">Configuration</h2>
    {#if config.error}
      <span class="truncate text-xs text-rose-700">{config.error}</span>
    {/if}
  </header>

  <div class="grid grid-cols-1 gap-x-10 gap-y-6 md:grid-cols-2">
    <!-- Microphone ============================================ -->
    <div class="flex flex-col">
      <h3 class="mb-3 text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
        Microphone
      </h3>

      {#if !config.mic}
        <p class="text-xs text-zinc-400">loading…</p>
      {:else}
        <div class="space-y-3">
          <label for="mic-source" class="block text-xs">
            <span class="mb-1 block text-zinc-600">Source</span>
            <select id="mic-source" name="mic-source" bind:value={sourceSel} class={selectCls}>
              <option value="auto">auto · first available</option>
              {#each candidates as cand (cand.id)}
                <option value={cand.id}
                  >{cand.id} · {cand.source.kind} · {cand.source.sample_rate} Hz</option
                >
              {/each}
            </select>
          </label>

          <label for="mic-channel" class="block text-xs">
            <span class="mb-1 block text-zinc-600">Channel</span>
            <select id="mic-channel" name="mic-channel" bind:value={channelSel} class={selectCls}>
              <option value="auto">auto</option>
              {#each channelOptions as ch (ch)}
                <option value={String(ch)}>{ch}</option>
              {/each}
            </select>
          </label>
        </div>

        <div class="mt-auto flex items-center justify-between pt-4">
          <span class="font-mono text-[10px] text-zinc-400">policy v{config.mic.version}</span>
          <button
            type="button"
            class={primaryBtn}
            disabled={!micDirty || config.loading}
            onclick={applyMic}>Apply</button
          >
        </div>
      {/if}
    </div>

    <!-- Inference cadence ====================================== -->
    <div class="flex flex-col">
      <h3 class="mb-3 text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
        Inference cadence
      </h3>

      {#if !config.inference}
        <p class="text-xs text-zinc-400">loading…</p>
      {:else}
        <!-- space-y-3 matches the Microphone column so row baselines on
             the left (selects) line up with row baselines on the right
             (sliders).  Slider blocks pull the meta caption inline with
             the value, so each block is the same height as a select. -->
        <div class="space-y-3">
          <div>
            <div class="flex items-baseline justify-between">
              <label for="hop-samples" class="text-xs text-zinc-600">Hop samples</label>
              <span class="text-[11px] text-zinc-500">
                <span class="font-mono text-zinc-700">{hopSamples}</span>
                <span class="text-zinc-400">· ≈ {approxHz(hopSamples)}</span>
              </span>
            </div>
            <input
              id="hop-samples"
              type="range"
              min={HOP_MIN}
              max={HOP_MAX}
              step="256"
              bind:value={hopSamples}
              style="--slider-percent: {hopPct}%"
              class="mt-1"
            />
          </div>

          <div>
            <div class="flex items-baseline justify-between">
              <label for="top-k" class="text-xs text-zinc-600">Top-K</label>
              <span class="font-mono text-[11px] text-zinc-700">{topK}</span>
            </div>
            <input
              id="top-k"
              type="range"
              min={TOPK_MIN}
              max={TOPK_MAX}
              step="1"
              bind:value={topK}
              style="--slider-percent: {topKPct}%"
              class="mt-1"
            />
          </div>
        </div>

        <div class="mt-auto flex items-center justify-end pt-4">
          <button
            type="button"
            class={primaryBtn}
            disabled={!inferDirty || config.loading}
            onclick={applyInference}>Apply</button
          >
        </div>
      {/if}
    </div>
  </div>
</section>
