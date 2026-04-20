<script lang="ts">
  import { tick } from "svelte";
  import { highlightSync, highlight } from "../lib/highlight";
  import { downloadTextFile } from "../lib/filename";

  interface Props {
    label: string;
    value: string;
    readonly?: boolean;
    placeholder?: string;
    loading?: boolean;
    wrap?: boolean;
    downloadName?: string;
    oninput?: (value: string) => void;
    ondrop?: (text: string, fileName?: string) => void;
  }

  let {
    label,
    value,
    readonly = false,
    placeholder = "",
    loading = false,
    wrap = false,
    downloadName = "shader.min.wgsl",
    oninput,
    ondrop,
  }: Props = $props();

  let copied = $state(false);
  let dragover = $state(false);
  let highlightedHtml = $state("");
  let highlightedCode = $state("");
  let highlightSeq = 0;

  // Reactively highlight when value changes (readonly output only)
  $effect(() => {
    const seq = ++highlightSeq;
    const code = value;
    if (!readonly || !code) {
      highlightedHtml = "";
      highlightedCode = "";
      return;
    }
    const sync = highlightSync(code);
    if (sync) {
      highlightedHtml = sync;
      highlightedCode = code;
      return;
    }
    highlightedHtml = "";
    highlightedCode = "";
    highlight(code).then((html) => {
      if (seq === highlightSeq) {
        highlightedHtml = html;
        highlightedCode = code;
      }
    });
  });

  // Input highlighting (editable mode)
  let inputHighlightedHtml = $state("");
  let inputHighlightedCode = $state("");
  let inputHighlightSeq = 0;

  $effect(() => {
    const seq = ++inputHighlightSeq;
    if (readonly) return;
    const code = value;
    if (!code) {
      inputHighlightedHtml = "";
      inputHighlightedCode = "";
      return;
    }
    const sync = highlightSync(code);
    if (sync) {
      inputHighlightedHtml = sync;
      inputHighlightedCode = code;
      return;
    }
    inputHighlightedHtml = "";
    inputHighlightedCode = "";
    highlight(code).then((html) => {
      if (seq === inputHighlightSeq) {
        inputHighlightedHtml = html;
        inputHighlightedCode = code;
      }
    });
  });

  let lineCount = $derived.by(() => {
    if (!value) return 1;
    let count = 1;
    let idx = -1;
    while ((idx = value.indexOf("\n", idx + 1)) !== -1) ++count;
    return count;
  });
  let lineNumbers = $derived(
    Array.from({ length: lineCount }, (_, i) => i + 1).join("\n"),
  );

  function handleCopy() {
    navigator.clipboard.writeText(value).then(
      () => {
        copied = true;
        setTimeout(() => (copied = false), 1500);
      },
      () => {},
    );
  }

  function handleDownload() {
    downloadTextFile(value, downloadName);
  }

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
    e.preventDefault();
    dragover = false;
    const file = e.dataTransfer?.files[0];
    if (!file) return;
    if (file.size > 32 * 1024 * 1024) {
      console.warn("Dropped file too large (>32 MB):", file.name);
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === "string") {
        ondrop?.(reader.result, file.name);
      }
    };
    reader.onerror = () => {
      console.warn("Failed to read dropped file:", file.name);
    };
    reader.readAsText(file);
  }

  const INDENT = "  ";

  // Apply an edit to the textarea, update value, and restore selection
  function applyEdit(
    ta: HTMLTextAreaElement,
    newValue: string,
    selStart: number,
    selEnd: number,
  ) {
    if (oninput) {
      oninput(newValue);
      tick().then(() => {
        ta.selectionStart = selStart;
        ta.selectionEnd = selEnd;
      });
    }
  }

  // Get the line start/end indices that cover the current selection
  function getSelectedLineRange(
    text: string,
    selStart: number,
    selEnd: number,
  ): { lineStart: number; lineEnd: number } {
    const lineStart = text.lastIndexOf("\n", selStart - 1) + 1;
    let lineEnd = text.indexOf("\n", selEnd);
    if (lineEnd === -1) lineEnd = text.length;
    return { lineStart, lineEnd };
  }

  function handleKeydown(e: KeyboardEvent) {
    if (readonly) return;
    const ta = e.target as HTMLTextAreaElement;
    const start = ta.selectionStart;
    const end = ta.selectionEnd;
    const mod = e.metaKey || e.ctrlKey;

    // Toggle line comment: Cmd/Ctrl + /
    if (mod && e.key === "/") {
      e.preventDefault();
      const { lineStart, lineEnd } = getSelectedLineRange(value, start, end);
      const block = value.substring(lineStart, lineEnd);
      const lines = block.split("\n");

      // Determine if we should uncomment (all non-empty lines already commented)
      const allCommented = lines.every(
        (l) => l.trimStart().startsWith("//") || l.trim() === "",
      );

      let newLines: string[];
      let startDelta = 0;
      let endDelta = 0;

      if (allCommented) {
        // Uncomment: remove first "// " or "//" from each line
        newLines = lines.map((l, i) => {
          const match = l.match(/^(\s*)\/\/\s?/);
          if (!match) return l;
          const removed = match[0].length - match[1].length;
          if (i === 0) startDelta = -removed;
          endDelta += -removed;
          return l.substring(0, match[1].length) + l.substring(match[0].length);
        });
      } else {
        // Comment: prepend "// " to each line
        newLines = lines.map((l, i) => {
          if (i === 0) startDelta = 3;
          endDelta += 3;
          return "// " + l;
        });
      }

      const newBlock = newLines.join("\n");
      const newValue =
        value.substring(0, lineStart) + newBlock + value.substring(lineEnd);
      const newStart = Math.max(lineStart, start + startDelta);
      const newEnd = Math.max(newStart, end + endDelta);
      applyEdit(ta, newValue, newStart, newEnd);
      return;
    }

    // Duplicate line: Cmd/Ctrl + Shift + D
    if (mod && e.shiftKey && e.key === "D") {
      e.preventDefault();
      const { lineStart, lineEnd } = getSelectedLineRange(value, start, end);
      const block = value.substring(lineStart, lineEnd);
      const newValue =
        value.substring(0, lineEnd) + "\n" + block + value.substring(lineEnd);
      const offset = block.length + 1;
      applyEdit(ta, newValue, start + offset, end + offset);
      return;
    }

    // Tab / Shift+Tab indentation
    if (e.key === "Tab") {
      e.preventDefault();

      // Multi-line selection: indent/dedent all selected lines
      if (start !== end) {
        const { lineStart, lineEnd } = getSelectedLineRange(value, start, end);
        const block = value.substring(lineStart, lineEnd);
        const lines = block.split("\n");

        let newLines: string[];
        let startDelta = 0;
        let endDelta = 0;

        if (e.shiftKey) {
          // Dedent
          newLines = lines.map((l, i) => {
            if (l.startsWith(INDENT)) {
              if (i === 0) startDelta = -INDENT.length;
              endDelta += -INDENT.length;
              return l.substring(INDENT.length);
            }
            return l;
          });
        } else {
          // Indent
          newLines = lines.map((_, i) => {
            if (i === 0) startDelta = INDENT.length;
            endDelta += INDENT.length;
            return INDENT + lines[i];
          });
        }

        const newBlock = newLines.join("\n");
        const newValue =
          value.substring(0, lineStart) + newBlock + value.substring(lineEnd);
        applyEdit(
          ta,
          newValue,
          Math.max(lineStart, start + startDelta),
          end + endDelta,
        );
      } else if (e.shiftKey) {
        // Shift+Tab with cursor: dedent current line
        const lineStart = value.lastIndexOf("\n", start - 1) + 1;
        const lineText = value.substring(lineStart);
        if (lineText.startsWith(INDENT)) {
          const newValue =
            value.substring(0, lineStart) + lineText.substring(INDENT.length);
          applyEdit(
            ta,
            newValue,
            Math.max(lineStart, start - INDENT.length),
            Math.max(lineStart, start - INDENT.length),
          );
        }
      } else {
        // Tab with cursor: insert indent
        const newValue =
          value.substring(0, start) + INDENT + value.substring(end);
        applyEdit(ta, newValue, start + INDENT.length, start + INDENT.length);
      }
      return;
    }

    // Enter: auto-indent to match current line
    if (e.key === "Enter" && !mod && !e.shiftKey) {
      e.preventDefault();
      const lineStart = value.lastIndexOf("\n", start - 1) + 1;
      const currentLine = value.substring(lineStart, start);
      const indent = currentLine.match(/^\s*/)?.[0] ?? "";
      // If previous non-whitespace char is '{', add one level
      const trimmedBefore = value.substring(0, start).trimEnd();
      const extra = trimmedBefore.endsWith("{") ? INDENT : "";
      const insertion = "\n" + indent + extra;
      const newValue =
        value.substring(0, start) + insertion + value.substring(end);
      const newPos = start + insertion.length;
      applyEdit(ta, newValue, newPos, newPos);
      return;
    }
  }
</script>

<div
  class="flex flex-col min-h-0 min-w-0 flex-1 {dragover
    ? 'ring-2 ring-emerald-400/50 ring-inset'
    : ''}"
  role="region"
  aria-label={label}
  ondragover={!readonly ? handleDragOver : undefined}
  ondragleave={!readonly ? handleDragLeave : undefined}
  ondrop={!readonly ? handleDrop : undefined}
>
  <!-- Panel header -->
  <div
    class="flex items-center justify-between px-3 h-8 border-b border-white/6 bg-[#111111] shrink-0"
  >
    <span
      class="text-[11px] font-medium text-slate-400 uppercase tracking-wider"
      >{label}</span
    >
    {#if readonly && value}
      <div class="flex gap-1">
        <button
          onclick={handleCopy}
          class="p-1 rounded text-slate-500 hover:text-slate-300 hover:bg-white/6 transition-colors cursor-pointer"
          title="Copy to clipboard"
        >
          {#if copied}
            <svg
              class="w-3.5 h-3.5 text-emerald-400"
              viewBox="0 0 16 16"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
            >
              <polyline points="3 8 7 12 13 4" />
            </svg>
          {:else}
            <svg class="w-3.5 h-3.5" viewBox="0 0 16 16" fill="currentColor">
              <path
                d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 010 1.5h-1.5a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-1.5a.75.75 0 011.5 0v1.5A1.75 1.75 0 019.25 16h-7.5A1.75 1.75 0 010 14.25z"
              />
              <path
                d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0114.25 11h-7.5A1.75 1.75 0 015 9.25zm1.75-.25a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-7.5a.25.25 0 00-.25-.25z"
              />
            </svg>
          {/if}
        </button>
        <button
          onclick={handleDownload}
          class="p-1 rounded text-slate-500 hover:text-slate-300 hover:bg-white/6 transition-colors cursor-pointer"
          title="Download .min.wgsl"
        >
          <svg class="w-3.5 h-3.5" viewBox="0 0 16 16" fill="currentColor">
            <path
              d="M2.75 14A1.75 1.75 0 011 12.25v-2.5a.75.75 0 011.5 0v2.5c0 .138.112.25.25.25h10.5a.25.25 0 00.25-.25v-2.5a.75.75 0 011.5 0v2.5A1.75 1.75 0 0113.25 14z"
            />
            <path
              d="M7.25 7.689V2a.75.75 0 011.5 0v5.689l1.97-1.969a.749.749 0 111.06 1.06l-3.25 3.25a.749.749 0 01-1.06 0L4.22 6.78a.749.749 0 111.06-1.06z"
            />
          </svg>
        </button>
      </div>
    {/if}
  </div>

  <!-- Loading indicator -->
  {#if loading}
    <div class="loading-bar shrink-0"></div>
  {/if}

  <!-- Code area -->
  {#if readonly}
    <div
      data-searchable
      class="highlight-panel code-panel code-text flex-1 w-full overflow-auto p-3 {wrap
        ? 'wrap-on'
        : ''}"
    >
      {#if highlightedHtml && highlightedCode === value}
        {@html highlightedHtml}
      {:else if value}
        <pre
          class="m-0 p-0 text-slate-200"
          class:whitespace-pre={!wrap}
          class:whitespace-pre-wrap={wrap}
          class:break-all={wrap}>{value}</pre>
      {:else}
        <span class="text-slate-600" data-placeholder>{placeholder}</span>
      {/if}
    </div>
  {:else}
    <div class="flex-1 overflow-auto min-h-0 min-w-0 code-panel">
      <div class="flex min-h-full">
        <!-- Line number gutter (sticky, GitHub-style) -->
        <div
          class="line-gutter code-text sticky left-0 z-1 shrink-0 select-none whitespace-pre text-right py-3 pr-2 pl-3 min-w-8 text-slate-400"
          aria-hidden="true"
        >
          {lineNumbers}
        </div>
        <!-- Stacked layers: highlight (behind) + textarea (on top) -->
        <div class="editor-stack flex-1 min-w-0">
          <div
            data-searchable
            class="highlight-input code-text p-3 pointer-events-none whitespace-pre"
            aria-hidden="true"
          >
            {#if inputHighlightedHtml && inputHighlightedCode === value}
              {@html inputHighlightedHtml}
            {:else}
              {value}
            {/if}
          </div>
          <textarea
            class="code-text p-3 bg-transparent text-transparent caret-slate-200 resize-none outline-none placeholder:text-slate-600 selection:bg-slate-600/40"
            {value}
            {placeholder}
            spellcheck="false"
            autocomplete="off"
            autocapitalize="off"
            wrap="off"
            oninput={(e) => oninput?.((e.target as HTMLTextAreaElement).value)}
            onkeydown={handleKeydown}
          ></textarea>
        </div>
      </div>
    </div>
  {/if}
</div>
