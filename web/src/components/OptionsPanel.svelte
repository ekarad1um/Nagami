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

  function handlePreserve(e: Event) {
    const raw = (e.target as HTMLInputElement).value;
    const symbols = raw
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    set("preserveSymbols", symbols.length > 0 ? symbols : undefined);
  }

  function profileDefaultMangle(): boolean {
    return (options.profile ?? "max") === "max";
  }

  function effectiveMangle(): boolean {
    return options.mangle ?? profileDefaultMangle();
  }

  // The precision dropdown stores its state as a single string so the
  // <select> can list a flat menu.  "full" / `d{N}` (decimal places) /
  // `s{N}` (significant figures) round-trip cleanly to the
  // `floatPrecision` Config field below.
  function precisionDropdownValue(fp: Config["floatPrecision"]): string {
    // Loose `==` catches both `undefined` and `null`; the latter is not in
    // the published type but, if an embedder passes it, the `"f16" in fp`
    // checks below would throw a TypeError on `null`.
    if (fp == undefined || fp === "full") return "full";
    if (typeof fp === "number") return `d${fp}`;
    if (
      typeof fp === "object" &&
      !("f16" in fp || "f32" in fp || "f64" in fp || "abstractFloat" in fp)
    ) {
      if ("decimalPlaces" in fp) return `d${fp.decimalPlaces}`;
      if ("significantFigures" in fp) return `s${fp.significantFigures}`;
      if ("sigFigs" in fp) return `s${fp.sigFigs}`;
    }
    // Per-type configurations have no single flat-menu entry.  The app's
    // own UI never produces one (precisionFromDropdown only emits the
    // forms above), so this branch is reached only by an external
    // embedder that pre-sets a per-type floatPrecision; we show "full"
    // and editing the dropdown then replaces the per-type config wholesale.
    return "full";
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
        <!-- Mangle -->
        <label
          class="flex items-center justify-between gap-2 text-slate-400"
          title="Shorten names to reduce size. Default follows profile (on for max, off for baseline/aggressive)."
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

        <!-- Beautify -->
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

        <!-- Float precision -->
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
            {#each [1, 2, 3, 4, 5, 6] as n}
              <option value={`d${n}`}>{n}d</option>
            {/each}
            {#each [1, 2, 3, 4] as n}
              <option value={`s${n}`}>{n}sf</option>
            {/each}
          </select>
        </label>

        <!-- Preserve symbols -->
        <label
          class="flex items-center justify-between gap-2 text-slate-400"
          title="Comma-separated names to exclude from renaming (e.g. main, uniforms)."
        >
          Preserve
          <input
            type="text"
            value={options.preserveSymbols?.join(", ") ?? ""}
            oninput={handlePreserve}
            placeholder="sym1, sym2"
            class="bg-white/6 text-slate-300 rounded px-2 py-1 sm:px-1.5 sm:py-0.5 border border-white/6 outline-none text-xs w-28 sm:w-24 placeholder:text-slate-600 focus:border-emerald-400/30 transition-colors"
          />
        </label>
      </div>

      <!-- Preamble -->
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
          rows={3}
        ></textarea>
      </div>
    </div>
  </div>
</div>
