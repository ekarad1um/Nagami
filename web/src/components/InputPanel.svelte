<script lang="ts">
  import { tick } from "svelte";
  import { highlighterReady, highlightSync } from "../lib/highlight.svelte";

  interface Props {
    value: string;
    // Drops pass the file name along.
    oninput: (value: string, fileName?: string) => void;
    onreject: (reason: string) => void;
  }

  let { value, oninput, onreject }: Props = $props();

  let dragover = $state(false);
  let html = $state("");

  // Deferred a frame so a keystroke paints before shiki runs; cleared first so
  // the overlay never shows stale text over the textarea.
  $effect(() => {
    const code = value;
    html = "";
    if (!highlighterReady()) return;
    const raf = requestAnimationFrame(() => (html = highlightSync(code) ?? ""));
    return () => cancelAnimationFrame(raf);
  });

  let lineNumbers = $derived.by(() => {
    let count = 1;
    for (let i = value.indexOf("\n"); i !== -1; i = value.indexOf("\n", i + 1))
      ++count;
    return Array.from({ length: count }, (_, i) => i + 1).join("\n");
  });

  function handleDragOver(e: DragEvent) {
    e.preventDefault();
    dragover = true;
  }

  function handleDragLeave(e: DragEvent) {
    const related = e.relatedTarget as Node | null;
    if (related && (e.currentTarget as HTMLElement).contains(related)) return;
    dragover = false;
  }

  function handleDrop(e: DragEvent) {
    dragover = false;
    const file = e.dataTransfer?.files[0];
    // Plain text is left to the textarea's native drop.
    if (!file) return;
    e.preventDefault();
    if (file.size > 32 * 1024 * 1024) {
      onreject(`${file.name} is larger than 32 MB`);
      return;
    }
    file.text().then(
      (text) => oninput(text, file.name),
      () => onreject(`could not read ${file.name}`),
    );
  }

  const INDENT = "  ";

  // The differing span is replaced through execCommand so the browser's undo
  // stack keeps the edit (assigning value resets it). Its input event already
  // delivered newValue, so the oninput below is a no-op unless the command
  // failed. Selection is restored after Svelte re-renders.
  function applyEdit(
    ta: HTMLTextAreaElement,
    newValue: string,
    selStart: number,
    selEnd: number,
  ) {
    let from = 0;
    const shorter = Math.min(value.length, newValue.length);
    while (from < shorter && value[from] === newValue[from]) ++from;
    let toOld = value.length;
    let toNew = newValue.length;
    while (
      toOld > from &&
      toNew > from &&
      value[toOld - 1] === newValue[toNew - 1]
    ) {
      --toOld;
      --toNew;
    }
    try {
      ta.setSelectionRange(from, toOld);
      document.execCommand("insertText", false, newValue.slice(from, toNew));
    } catch {
      /* oninput below delivers newValue instead */
    }
    oninput(newValue);
    tick().then(() => ta.setSelectionRange(selStart, selEnd));
  }

  // Span of the lines the selection touches; a selection ending at column 0
  // does not cover that line.
  function lineRange(start: number, end: number): [number, number] {
    // lastIndexOf clamps a negative position to 0, so guard offset 0.
    const from = start > 0 ? value.lastIndexOf("\n", start - 1) + 1 : 0;
    const last = end > start && value[end - 1] === "\n" ? end - 1 : end;
    const nl = value.indexOf("\n", last);
    return [from, nl === -1 ? value.length : nl];
  }

  // Rewrites the selected lines, keeping the selection on the same text.
  function editLines(
    ta: HTMLTextAreaElement,
    start: number,
    end: number,
    transform: (lines: string[]) => string[],
  ) {
    const [from, to] = lineRange(start, end);
    const lines = value.slice(from, to).split("\n");
    const out = transform(lines);
    const block = out.join("\n");
    if (block.length === to - from && block === value.slice(from, to)) return;
    const newStart = Math.max(from, start + out[0].length - lines[0].length);
    applyEdit(
      ta,
      value.slice(0, from) + block + value.slice(to),
      newStart,
      Math.max(newStart, end + block.length - (to - from)),
    );
  }

  const isComment = (line: string) => /^\s*(\/\/|$)/.test(line);
  const dedent = (line: string) =>
    line.startsWith(INDENT) ? line.slice(INDENT.length) : line;

  function handleKeydown(e: KeyboardEvent) {
    if (e.isComposing) return;
    const ta = e.currentTarget as HTMLTextAreaElement;
    const { selectionStart: start, selectionEnd: end } = ta;
    const mod = e.metaKey || e.ctrlKey;

    if (mod && e.key === "/") {
      e.preventDefault();
      editLines(ta, start, end, (lines) =>
        lines.every(isComment)
          ? lines.map((l) => l.replace(/^(\s*)\/\/\s?/, "$1"))
          : lines.map((l) => "// " + l),
      );
    } else if (mod && e.shiftKey && e.key === "D") {
      e.preventDefault();
      const [from, to] = lineRange(start, end);
      const copy = "\n" + value.slice(from, to);
      applyEdit(
        ta,
        value.slice(0, to) + copy + value.slice(to),
        start + copy.length,
        end + copy.length,
      );
    } else if (e.key === "Tab") {
      e.preventDefault();
      if (start === end && !e.shiftKey) {
        applyEdit(
          ta,
          value.slice(0, start) + INDENT + value.slice(end),
          start + INDENT.length,
          start + INDENT.length,
        );
      } else {
        editLines(ta, start, end, (lines) =>
          lines.map((l) => (e.shiftKey ? dedent(l) : INDENT + l)),
        );
      }
    } else if (e.key === "Enter" && !mod && !e.shiftKey) {
      e.preventDefault();
      const [from] = lineRange(start, start);
      const indent = value.slice(from, start).match(/^\s*/)![0];
      const extra = value.slice(0, start).trimEnd().endsWith("{") ? INDENT : "";
      const insertion = "\n" + indent + extra;
      applyEdit(
        ta,
        value.slice(0, start) + insertion + value.slice(end),
        start + insertion.length,
        start + insertion.length,
      );
    }
  }
</script>

<div
  class="flex flex-col min-h-0 min-w-0 flex-1 relative {dragover
    ? 'ring-2 ring-emerald-400/50 ring-inset'
    : ''}"
  role="region"
  aria-label="Input"
  ondragover={handleDragOver}
  ondragleave={handleDragLeave}
  ondrop={handleDrop}
>
  <div
    class="flex items-center justify-between px-3 h-8 border-b border-white/6 bg-[#111111] shrink-0"
  >
    <span
      class="text-[11px] font-medium text-slate-400 uppercase tracking-wider"
      >Input</span
    >
  </div>

  <div class="flex-1 overflow-auto min-h-0 min-w-0 code-panel">
    <div class="flex min-h-full">
      <div
        class="line-gutter code-text sticky left-0 z-1 shrink-0 select-none whitespace-pre text-right py-3 pr-2 pl-3 min-w-8 text-slate-400"
        aria-hidden="true"
      >
        {lineNumbers}
      </div>
      <!-- Stacked layers: highlight (on top) + textarea (behind) -->
      <div class="editor-stack flex-1 min-w-0">
        <textarea
          class="code-text p-3 bg-transparent text-transparent caret-slate-200 resize-none outline-none placeholder:text-slate-600 selection:bg-slate-600/40"
          {value}
          placeholder="Paste or drop a WGSL shader here..."
          spellcheck="false"
          autocomplete="off"
          autocapitalize="off"
          wrap="off"
          oninput={(e) => oninput(e.currentTarget.value)}
          onkeydown={handleKeydown}></textarea>
        <div
          data-searchable
          class="highlight-input code-text p-3 pointer-events-none whitespace-pre"
          aria-hidden="true"
        >
          <!-- eslint-disable-next-line svelte/no-at-html-tags -- shiki escapes its output -->
          {#if html}{@html html}{:else}{value}{/if}
        </div>
      </div>
    </div>
  </div>
</div>
