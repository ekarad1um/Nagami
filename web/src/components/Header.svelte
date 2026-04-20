<script lang="ts">
  import type { Config } from "../lib/nagami";

  interface Props {
    profile: Config["profile"];
    onProfileChange: (p: Config["profile"]) => void;
  }

  let { profile, onProfileChange }: Props = $props();

  const profiles: Config["profile"][] = ["baseline", "aggressive", "max"];

  const profileTooltips: Record<"baseline" | "aggressive" | "max", string> = {
    baseline:
      "Baseline: DCE, constant folding, dead parameter elimination, emit-merge, rename. Fast and safe. Mangle: off.",
    aggressive:
      "Aggressive: baseline + function inlining (24 nodes / 3 sites), load-dedup, variable coalescing. Mangle: off.",
    max: "Max (default): aggressive + CSE, higher inlining limits (48 nodes / 6 sites). Mangle: on.",
  };
</script>

<header
  class="flex items-center justify-between px-4 py-2 border-b border-white/6 bg-[#111111] shrink-0"
>
  <div class="flex items-center gap-2">
    <span class="text-emerald-400 text-lg font-bold select-none">◆</span>
    <span class="text-sm font-semibold text-slate-100 tracking-tight"
      >Nagami<span class="text-emerald-400">[n]</span></span
    >
  </div>

  <div class="flex items-center gap-3">
    <!-- Profile segmented control -->
    <div class="flex rounded-md bg-white/6 p-0.5 text-xs">
      {#each profiles as p}
        <button
          class="px-2.5 py-1 rounded-[5px] font-medium capitalize transition-colors cursor-pointer {profile ===
          p
            ? 'bg-white/8 text-slate-100'
            : 'text-slate-400 hover:text-slate-300'}"
          onclick={() => onProfileChange(p)}
          title={profileTooltips[p as "baseline" | "aggressive" | "max"]}
        >
          {p}
        </button>
      {/each}
    </div>

    <!-- GitHub link -->
    <a
      href="https://github.com/ekarad1um/Nagami"
      target="_blank"
      rel="noopener"
      class="text-slate-500 hover:text-slate-300 transition-colors"
      aria-label="GitHub"
      title="View source code on GitHub"
    >
      <svg class="w-4.5 h-4.5" viewBox="0 0 16 16" fill="currentColor">
        <path
          d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"
        />
      </svg>
    </a>
  </div>
</header>
