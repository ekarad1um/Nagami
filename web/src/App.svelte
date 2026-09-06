<script lang="ts">
  import Header from "./components/Header.svelte";
  import InputPanel from "./components/InputPanel.svelte";
  import OutputPanel from "./components/OutputPanel.svelte";
  import StatsBar from "./components/StatsBar.svelte";
  import OptionsPanel from "./components/OptionsPanel.svelte";
  import FindBar from "./components/FindBar.svelte";
  import { run, type Config, type NameMap } from "./lib/nagami";
  import { downloadTextFile } from "./lib/filename";

  interface Result {
    output: string;
    inputBytes: number;
    outputBytes: number;
    bailout: string | null;
    nameMap: NameMap | null;
    wrap: boolean;
  }
  const EMPTY: Result = {
    output: "",
    inputBytes: 0,
    outputBytes: 0,
    bailout: null,
    nameMap: null,
    wrap: true,
  };

  let input = $state("");
  let inputFileName = $state("");
  let userTouched = false;
  // Replaced wholesale per run, so every consumer changes together.
  let result: Result = $state.raw(EMPTY);
  let error: string | null = $state(null);
  let loading = $state(false);
  let options: Config = $state({ profile: "max" });
  let optionsOpen = $state(false);
  let findOpen = $state(false);
  let findFocusTrigger = $state(0);

  // The sample stays out of the bundle; an HTML body is a SPA fallback, not a
  // shader. Network throws are retried with backoff.
  (async () => {
    for (let attempt = 0; attempt < 3 && !userTouched; ++attempt) {
      try {
        const r = await fetch(import.meta.env.BASE_URL + "example.wgsl");
        const ct = r.headers.get("content-type") ?? "";
        if (!r.ok || ct.startsWith("text/html")) return;
        const text = await r.text();
        if (!userTouched) input = text;
        return;
      } catch {
        await new Promise((r) => setTimeout(r, 1000 * (attempt + 1)));
      }
    }
  })();

  let downloadName = $derived(
    inputFileName
      ? inputFileName.replace(/\.(wgsl|glsl)$/i, "") + ".min.wgsl"
      : "shader.min.wgsl",
  );
  let debounceTimer: ReturnType<typeof setTimeout>;
  let minifyGen = 0;

  async function minify() {
    const gen = ++minifyGen;
    const config = $state.snapshot(options) as Config;
    loading = true;
    error = null;
    const r = await run(input, config);
    if (gen !== minifyGen) return;
    loading = false;
    error = r.error;
    result = r.output
      ? {
          output: r.output.source,
          inputBytes: r.output.report.inputBytes,
          outputBytes: r.output.report.outputBytes,
          bailout: r.output.report.bailout,
          nameMap: r.output.nameMap,
          wrap: config.beautify !== true,
        }
      : EMPTY;
  }

  $effect(() => {
    void options;
    if (input.trim()) {
      debounceTimer = setTimeout(minify, 300);
    } else {
      ++minifyGen;
      result = EMPTY;
      error = null;
      loading = false;
    }
    return () => clearTimeout(debounceTimer);
  });

  function handleGlobalKeydown(e: KeyboardEvent) {
    if (e.repeat) return;
    if ((e.metaKey || e.ctrlKey) && e.key === "s") {
      e.preventDefault();
      if (result.output) downloadTextFile(result.output, downloadName);
    }
    if ((e.metaKey || e.ctrlKey) && e.key === "f") {
      e.preventDefault();
      findOpen = true;
      ++findFocusTrigger;
    }
    if ((e.metaKey || e.ctrlKey) && e.key === "a") {
      const el = document.activeElement;
      if (el?.tagName !== "TEXTAREA" && el?.tagName !== "INPUT") {
        const target = document.querySelector(
          '[aria-label="Output"] [data-searchable]',
        );
        if (target) {
          e.preventDefault();
          const sel = window.getSelection();
          const range = document.createRange();
          range.selectNodeContents(target);
          sel?.removeAllRanges();
          sel?.addRange(range);
        }
      }
    }
  }

  // A file dropped outside the input panel would otherwise navigate the tab to
  // it; a refused drop (dropEffect "none") navigates too, so the copy cursor stays.
  function blockFileDrop(e: DragEvent) {
    if (e.dataTransfer?.types.includes("Files")) e.preventDefault();
  }
</script>

<svelte:window
  onkeydown={handleGlobalKeydown}
  ondragover={blockFileDrop}
  ondrop={blockFileDrop}
/>

<Header
  profile={options.profile}
  onProfileChange={(p) => (options = { ...options, profile: p })}
/>

<main
  class="flex-1 flex flex-col md:flex-row min-h-0 divide-y md:divide-y-0 md:divide-x divide-white/6 relative"
>
  {#if findOpen}
    <FindBar
      focusTrigger={findFocusTrigger}
      inputContent={input}
      outputContent={result.output}
      onclose={() => (findOpen = false)}
    />
  {/if}
  <InputPanel
    value={input}
    oninput={(v, name) => {
      userTouched = true;
      input = v;
      if (name) inputFileName = name;
    }}
    onreject={(reason) => (error = reason)}
  />
  <OutputPanel
    value={result.output}
    {loading}
    wrap={result.wrap}
    {downloadName}
    nameMap={result.nameMap}
  />
</main>

<OptionsPanel
  open={optionsOpen}
  {options}
  onOptionsChange={(o) => (options = o)}
/>

<StatsBar
  inputBytes={result.inputBytes}
  outputBytes={result.outputBytes}
  {error}
  bailout={result.bailout}
  {loading}
  {optionsOpen}
  onToggleOptions={() => (optionsOpen = !optionsOpen)}
/>
