<script lang="ts">
  import type { Config } from "../lib/nagami";

  interface Props {
    open: boolean;
    options: Config;
    onOptionsChange: (options: Config) => void;
  }

  let { open, options, onOptionsChange }: Props = $props();

  function set<K extends keyof Config>(key: K, value: Config[K]) {
    onOptionsChange({ ...options, [key]: value });
  }

  function parseSymbols(raw: string): string[] {
    return raw
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
  }

  function handlePreserve(e: Event) {
    const symbols = parseSymbols((e.target as HTMLInputElement).value);
    set("preserveSymbols", symbols.length > 0 ? symbols : undefined);
  }

  // Written back only when the field disagrees with options semantically, so
  // a separator being typed ("a, ") survives the round trip through options.
  let preserveEl: HTMLInputElement | undefined = $state(undefined);
  $effect(() => {
    const want = options.preserveSymbols?.join(", ") ?? "";
    if (preserveEl && parseSymbols(preserveEl.value).join(", ") !== want)
      preserveEl.value = want;
  });

  function profileDefaultMangle(): boolean {
    return (options.profile ?? "max") === "max";
  }

  function effectiveMangle(): boolean {
    return options.mangle ?? profileDefaultMangle();
  }

  // Flat <select> state: "full" | d{N} (decimal places) | s{N} (sig figs).
  function precisionDropdownValue(fp: Config["floatPrecision"]): string {
    if (fp == null || fp === "full") return "full";
    if (typeof fp === "number") return `d${fp}`;
    if ("decimalPlaces" in fp) return `d${fp.decimalPlaces}`;
    if ("significantFigures" in fp) return `s${fp.significantFigures}`;
    if ("sigFigs" in fp) return `s${fp.sigFigs}`;
    return "full"; // per-type configs have no flat entry; the next edit replaces them
  }

  function precisionFromDropdown(v: string): Config["floatPrecision"] {
    if (v === "full") return undefined;
    const n = Number(v.slice(1));
    if (v.startsWith("d")) return { decimalPlaces: n };
    if (v.startsWith("s")) return { significantFigures: n };
    return undefined;
  }
</script>

<div
  class="border-t border-white/6 bg-[#111111] shrink-0 relative z-10 options-tray"
  class:options-open={open}
  inert={!open}
>
  <div class="options-inner">
    <div class="px-4 py-3">
      <div class="grid grid-cols-2 md:grid-cols-4 gap-x-6 gap-y-3 text-xs">
        <label
          class="flex items-center justify-between gap-2 text-slate-400"
          title="Shorten names to reduce size. Default follows profile (on for max, off for baseline/aggressive). Under max, CSE and vector-constant hoisting run only while mangle is on."
        >
          Mangle
          <input
            type="checkbox"
            checked={effectiveMangle()}
            onchange={(e) => {
              const checked = (e.target as HTMLInputElement).checked;
              const profileDefault = profileDefaultMangle();
              set("mangle", checked === profileDefault ? undefined : checked);
            }}
            class="accent-emerald-400 cursor-pointer"
          />
        </label>

        <label
          class="flex items-center justify-between gap-2 text-slate-400"
          title="Format output with indentation and newlines for readability."
        >
          Beautify
          <input
            type="checkbox"
            checked={options.beautify === true}
            onchange={(e) =>
              set(
                "beautify",
                (e.target as HTMLInputElement).checked || undefined,
              )}
            class="accent-emerald-400 cursor-pointer"
          />
        </label>

        <label
          class="flex items-center justify-between gap-2 text-slate-400"
          title="Round float literals to limit emission size. 'full' preserves original precision. 'd' = decimal places after the dot, 'sf' = significant figures regardless of magnitude."
        >
          Precision
          <select
            value={precisionDropdownValue(options.floatPrecision)}
            onchange={(e) => {
              const v = (e.target as HTMLSelectElement).value;
              set("floatPrecision", precisionFromDropdown(v));
            }}
            class="bg-white/6 text-slate-300 rounded px-1.5 py-0.5 border border-white/6 outline-none text-xs cursor-pointer focus:border-emerald-400/30 transition-colors"
          >
            <option value="full">full</option>
            {#each [1, 2, 3, 4, 5, 6] as n (n)}
              <option value={`d${n}`}>{n}d</option>
            {/each}
            {#each [1, 2, 3, 4] as n (n)}
              <option value={`s${n}`}>{n}sf</option>
            {/each}
          </select>
        </label>

        <label
          class="flex items-center justify-between gap-2 text-slate-400"
          title="Comma-separated names to exclude from renaming, struct members included (e.g. main, uniforms)."
        >
          Preserve
          <input
            bind:this={preserveEl}
            type="text"
            oninput={handlePreserve}
            placeholder="sym1, sym2"
            class="bg-white/6 text-slate-300 rounded px-2 py-1 sm:px-1.5 sm:py-0.5 border border-white/6 outline-none text-xs w-28 sm:w-24 placeholder:text-slate-600 focus:border-emerald-400/30 transition-colors"
          />
        </label>
      </div>

      <div class="mt-3 border-t border-white/6 pt-3">
        <label
          for="preamble"
          class="text-xs text-slate-400 flex items-center gap-1 mb-1.5"
          title="External WGSL declarations (uniforms, structs) prepended for parsing but excluded from output."
        >
          Preamble
          {#if options.preamble}
            <span
              class="w-1 h-1 rounded-full bg-emerald-400 shrink-0 translate-y-px"
            ></span>
          {/if}
        </label>
        <textarea
          id="preamble"
          value={options.preamble ?? ""}
          oninput={(e) => {
            const v = (e.target as HTMLTextAreaElement).value;
            set("preamble", v.trim() ? v : undefined);
          }}
          placeholder={"// External declarations, e.g.:\nstruct Inputs { time: f32, size: vec2f }\n@group(0) @binding(0) var<uniform> inputs: Inputs;"}
          spellcheck={false}
          class="w-full bg-white/4 text-slate-300 rounded-md px-2.5 py-2.5 sm:px-3 sm:py-2 border border-white/6 outline-none text-xs leading-5 font-mono resize-y min-h-18 max-h-48 placeholder:text-slate-600 focus:border-emerald-400/30 transition-colors"
          rows={3}></textarea>
      </div>
    </div>
  </div>
</div>
