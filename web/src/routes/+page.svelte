<script lang="ts">
  import { streams } from '$lib/stores/streams.svelte';
  import VisualizationPanel from '$lib/components/dashboard/VisualizationPanel.svelte';
  import InferencePanel from '$lib/components/dashboard/InferencePanel.svelte';
  import ConfigurationPanel from '$lib/components/dashboard/ConfigurationPanel.svelte';
</script>

{#if streams.unsupportedReason}
  <div class="mb-4 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
    <p class="font-medium">Limited browser support</p>
    <p class="mt-1 text-xs">{streams.unsupportedReason}</p>
  </div>
{/if}

<div class="space-y-5">
  <div class="grid grid-cols-1 gap-5 lg:grid-cols-3">
    <div class="lg:col-span-2">
      <VisualizationPanel />
    </div>
    <div class="lg:col-span-1">
      <InferencePanel />
    </div>
  </div>

  <ConfigurationPanel />
</div>
