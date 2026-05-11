<script lang="ts">
  import { streams } from '$lib/stores/streams.svelte';
  import WaveformCanvas from './WaveformCanvas.svelte';
  import SpectrogramCanvas from './SpectrogramCanvas.svelte';

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
</script>

<section class="flex h-full flex-col rounded-xl border border-zinc-200 bg-white p-5 shadow-sm">
  <!-- Metadata folds into the header so the panel bottom edge matches the
       sides at exactly p-5 (20px).  Previous footer pushed bottom whitespace
       to ~48px, breaking corner symmetry. -->
  <header class="mb-3 flex items-center justify-between gap-3">
    <div class="flex items-baseline gap-2">
      <h2 class="text-sm font-semibold text-zinc-900">Visualization</h2>
      <span class="text-[11px] text-zinc-400">48 kHz · mono · opus · 3 s window</span>
    </div>
    <span
      class="rounded-full px-2 py-0.5 text-[11px] font-medium capitalize tracking-wide {pillClass(
        streams.audioStatus
      )}"
    >
      {STATE_LABEL[streams.audioStatus] ?? streams.audioStatus}
    </span>
  </header>

  <!-- Fixed canvas heights keep the section's intrinsic height predictable
       so the grid row doesn't inflate via flex-grow chains.  Waveform h-32
       (128px) + spectrogram h-56 (224px) give the spec enough vertical
       presence that the symmetric p-5 (20px) top/bottom whitespace reads
       as balanced; with a shorter spectrogram the panel felt top-heavy. -->
  <div class="space-y-2">
    <div class="h-32">
      <WaveformCanvas seconds={3} />
    </div>
    <div class="h-56">
      <SpectrogramCanvas seconds={3} />
    </div>
  </div>
</section>
