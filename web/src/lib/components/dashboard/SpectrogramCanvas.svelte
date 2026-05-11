<script lang="ts">
  import { onMount } from 'svelte';
  import { streams } from '$lib/stores/streams.svelte';
  import { Fft, hannWindow } from '$lib/audio/fft';

  interface Props {
    fftSize?: number;
    hopSize?: number;
    seconds?: number;
    minDb?: number;
    maxDb?: number;
    maxHz?: number;
  }
  let {
    fftSize = 512,
    hopSize = 256,
    seconds = 3,
    minDb = -90,
    maxDb = -10,
    maxHz = 12_000
  }: Props = $props();

  let canvas: HTMLCanvasElement | undefined = $state();

  onMount(() => {
    const el = canvas;
    if (!el) return;
    const ctx = el.getContext('2d');
    if (!ctx) return;

    const fft = new Fft(fftSize);
    const winFn = hannWindow(fftSize);
    const re = new Float32Array(fftSize);
    const im = new Float32Array(fftSize);

    const useBins = Math.min(fftSize >>> 1, Math.ceil((maxHz / streams.sampleRate) * fftSize));
    const historyColumns = Math.max(1, Math.ceil((seconds * streams.sampleRate) / hopSize));

    // Circular column buffer.  Each column has `useBins` log-magnitude
    // values, oldest at writeCol, newest at writeCol-1 (mod historyColumns).
    const cols = new Float32Array(historyColumns * useBins);
    let writeCol = 0;
    let colsWritten = 0;
    let lastSampleIdx = 0;

    let pixelW = 1;
    let pixelH = 1;
    let img: ImageData | null = null;

    const resize = () => {
      const r = el.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const w = Math.max(1, Math.floor(r.width * dpr));
      const h = Math.max(1, Math.floor(r.height * dpr));
      if (el.width !== w) el.width = w;
      if (el.height !== h) el.height = h;
      pixelW = w;
      pixelH = h;
      img = ctx.createImageData(w, h);
    };

    const ingest = () => {
      const have = streams.sampleCount;
      const newSamples = have - lastSampleIdx;
      if (newSamples < fftSize) return;
      const pcm = streams.snapshot(Math.min(newSamples + fftSize, fft.size * 256));
      const total = pcm.length;
      const advances = Math.floor((have - lastSampleIdx) / hopSize);
      const used = advances * hopSize;
      lastSampleIdx += used;
      const startInPcm = total - used - fftSize;
      if (startInPcm < 0) return;
      for (let h = 0; h < advances; h++) {
        const off = startInPcm + h * hopSize;
        if (off + fftSize > total) break;
        for (let i = 0; i < fftSize; i++) {
          re[i] = pcm[off + i] * winFn[i];
          im[i] = 0;
        }
        fft.forward(re, im);
        const dest = writeCol * useBins;
        for (let k = 0; k < useBins; k++) {
          const mag = Math.sqrt(re[k] * re[k] + im[k] * im[k]) + 1e-12;
          cols[dest + k] = 20 * Math.log10(mag);
        }
        writeCol = (writeCol + 1) % historyColumns;
        if (colsWritten < historyColumns) colsWritten++;
      }
    };

    const palette = (norm: number): [number, number, number] => {
      // viridis-ish ramp (dark indigo -> teal -> yellow)
      const t = Math.max(0, Math.min(1, norm));
      const r = Math.round(255 * Math.min(1, Math.max(0, t * 3 - 1.4)));
      const g = Math.round(255 * Math.min(1, Math.max(0, t * 1.7)));
      const b = Math.round(255 * Math.min(1, Math.max(0, 0.6 + (0.5 - t) * 1.4)));
      return [r, g, b];
    };

    const render = () => {
      if (!img) return;
      const data = img.data;
      data.fill(0);
      if (colsWritten === 0) {
        ctx.putImageData(img, 0, 0);
        return;
      }
      // Always stretch the available history across the full pixel width.
      // The oldest column lives at (writeCol - colsWritten) mod historyColumns;
      // we render colsWritten columns scaled to pixelW.
      const startCol = (writeCol - colsWritten + historyColumns) % historyColumns;
      for (let x = 0; x < pixelW; x++) {
        const frac = (x / pixelW) * colsWritten;
        const colOff = Math.floor(frac);
        const col = (startCol + colOff) % historyColumns;
        const colBase = col * useBins;
        for (let y = 0; y < pixelH; y++) {
          const bin = useBins - 1 - Math.floor((y / pixelH) * useBins);
          if (bin < 0 || bin >= useBins) continue;
          const db = cols[colBase + bin];
          const norm = (db - minDb) / (maxDb - minDb);
          const [r, g, b] = palette(norm);
          const i = (y * pixelW + x) * 4;
          data[i] = r;
          data[i + 1] = g;
          data[i + 2] = b;
          data[i + 3] = 255;
        }
      }
      ctx.putImageData(img, 0, 0);
    };

    const ro = new ResizeObserver(resize);
    ro.observe(el);
    resize();

    let raf: number | null = null;
    const tick = () => {
      ingest();
      render();
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);

    return () => {
      if (raf !== null) cancelAnimationFrame(raf);
      ro.disconnect();
    };
  });
</script>

<canvas bind:this={canvas} class="block h-full w-full rounded-md bg-zinc-950"></canvas>
