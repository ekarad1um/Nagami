<script lang="ts">
  import Header from "./components/Header.svelte";
  import CodePanel from "./components/CodePanel.svelte";
  import StatsBar from "./components/StatsBar.svelte";
  import OptionsPanel from "./components/OptionsPanel.svelte";
  import FindBar from "./components/FindBar.svelte";
  import { run, type Config } from "./lib/nagami";
  import { downloadTextFile } from "./lib/filename";

  let input = $state("");
  let output = $state("");
  let inputBytes = $state(0);
  let outputBytes = $state(0);
  let error: string | null = $state(null);
  let loading = $state(false);
  let userTouched = false;

  // Deferred load with retry - keeps the sample shader out of the JS bundle
  (async () => {
    for (let attempt = 0; attempt < 3; ++attempt) {
      if (userTouched) return;
      try {
        const r = await fetch(import.meta.env.BASE_URL + "example.wgsl");
        const ct = r.headers.get("content-type") ?? "";
        if (!r.ok || ct.startsWith("text/html")) return;
        const text = await r.text();
        if (!userTouched) input = text;
        return;
      } catch {
        if (attempt < 2)
          await new Promise((r) => setTimeout(r, 1000 * (attempt + 1)));
      }
    }
  })();
  let optionsOpen = $state(false);
  let options: Config = $state({ profile: "max" });
  let outputWrap = $state(true);
  let inputFileName = $state("");
  let findOpen = $state(false);
  let findFocusTrigger = $state(0);

  let downloadName = $derived(
    inputFileName
      ? inputFileName.replace(/\.(wgsl|glsl)$/i, "") + ".min.wgsl"
      : "shader.min.wgsl",
  );

  let debounceTimer: ReturnType<typeof setTimeout>;
  let minifyGen = 0;

  function snapshotOptions(config: Config): Config {
    return {
      ...config,
      preserveSymbols: config.preserveSymbols
        ? [...config.preserveSymbols]
        : undefined,
    };
  }

  function scheduleMinify() {
    clearTimeout(debounceTimer);
    if (!input.trim()) {
      ++minifyGen;
      output = "";
      outputWrap = options.beautify !== true;
      inputBytes = 0;
      outputBytes = 0;
      error = null;
      loading = false;
      return;
    }
    debounceTimer = setTimeout(doMinify, 300);
  }

  async function doMinify() {
    const gen = ++minifyGen;
    const config = snapshotOptions(options);
    loading = true;
    error = null;
    const result = await run(input, config);
    if (gen !== minifyGen) return;
    loading = false;
    if (result.error !== null) {
      error = result.error;
      output = "";
      outputWrap = config.beautify !== true;
      inputBytes = 0;
      outputBytes = 0;
    } else {
      output = result.output.source;
      outputWrap = config.beautify !== true;
      inputBytes = result.output.report.inputBytes;
      outputBytes = result.output.report.outputBytes;
      error = null;
    }
  }

  $effect(() => {
    // Re-run when input or options change
    input;
    options;
    scheduleMinify();
    return () => clearTimeout(debounceTimer);
  });

  function downloadOutput() {
    if (!output) return;
    downloadTextFile(output, downloadName);
  }

  function handleGlobalKeydown(e: KeyboardEvent) {
    if ((e.metaKey || e.ctrlKey) && e.key === "s") {
      e.preventDefault();
      downloadOutput();
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
</script>

<svelte:window onkeydown={handleGlobalKeydown} />

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
      outputContent={output}
      onclose={() => (findOpen = false)}
    />
  {/if}
  <CodePanel
    label="Input"
    value={input}
    placeholder="Paste or drop a WGSL shader here..."
    oninput={(v) => {
      userTouched = true;
      input = v;
    }}
    ondrop={(text, name) => {
      userTouched = true;
      input = text;
      if (name) inputFileName = name;
    }}
  />
  <CodePanel
    label="Output"
    value={output}
    readonly
    {loading}
    {downloadName}
    wrap={outputWrap}
    placeholder={loading ? "Loading WASM..." : "Minified output appears here"}
  />
</main>

<OptionsPanel
  open={optionsOpen}
  {options}
  onOptionsChange={(o) => (options = o)}
/>

<StatsBar
  {inputBytes}
  {outputBytes}
  {error}
  {loading}
  {optionsOpen}
  onToggleOptions={() => (optionsOpen = !optionsOpen)}
/>
