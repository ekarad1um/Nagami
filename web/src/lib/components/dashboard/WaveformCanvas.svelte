<script lang="ts">
  import { onMount } from 'svelte';
  import { streams } from '$lib/stores/streams.svelte';

  interface Props {
    seconds?: number;
    color?: string;
    background?: string;
  }
  // Background uses the recessed nested-data tier (zinc-50 / #fafafa).
  // The waveform sits one tonal step below the white panel so it reads
  // as a viewport into the audio stream; pairs visually with the Active
  // Head card's matching zinc-50 surface for a coherent nested tier.
  let { seconds = 3, color = '#3b82f6', background = '#fafafa' }: Props = $props();

  let canvas: HTMLCanvasElement | undefined = $state();
  let raf: number | null = null;

  onMount(() => {
    const el = canvas;
    if (!el) return;
    const ctx = el.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    let hiBuf: Float32Array = new Float32Array(0);
    let loBuf: Float32Array = new Float32Array(0);

    const resize = () => {
      const r = el.getBoundingClientRect();
      const w = Math.max(1, Math.floor(r.width * dpr));
      const h = Math.max(1, Math.floor(r.height * dpr));
      if (el.width !== w) el.width = w;
      if (el.height !== h) el.height = h;
      hiBuf = new Float32Array(w);
      loBuf = new Float32Array(w);
    };

    const ro = new ResizeObserver(resize);
    ro.observe(el);
    resize();

    const fillRgba = hexToRgba(color, 0.15);
    const gridStroke = '#e4e4e7';

    const draw = () => {
      const w = el.width;
      const h = el.height;
      const mid = h / 2;
      const amp = mid * 0.92;

      ctx.fillStyle = background;
      ctx.fillRect(0, 0, w, h);

      ctx.strokeStyle = gridStroke;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, mid + 0.5);
      ctx.lineTo(w, mid + 0.5);
      ctx.stroke();

      const pcm = streams.snapshot(streams.sampleRate * seconds);
      if (pcm.length === 0) {
        raf = requestAnimationFrame(draw);
        return;
      }

      const samplesPerPixel = pcm.length / w;
      for (let x = 0; x < w; x++) {
        const start = Math.floor(x * samplesPerPixel);
        const end = Math.min(pcm.length, Math.floor((x + 1) * samplesPerPixel));
        let lo = 0;
        let hi = 0;
        for (let i = start; i < end; i++) {
          const v = pcm[i];
          if (v < lo) lo = v;
          if (v > hi) hi = v;
        }
        hiBuf[x] = hi;
        loBuf[x] = lo;
      }

      // Filled envelope (translucent).
      ctx.fillStyle = fillRgba;
      ctx.beginPath();
      ctx.moveTo(0, mid - hiBuf[0] * amp);
      for (let x = 1; x < w; x++) ctx.lineTo(x, mid - hiBuf[x] * amp);
      for (let x = w - 1; x >= 0; x--) ctx.lineTo(x, mid - loBuf[x] * amp);
      ctx.closePath();
      ctx.fill();

      // Top and bottom contour lines.
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.25;
      ctx.lineJoin = 'round';
      ctx.beginPath();
      ctx.moveTo(0, mid - hiBuf[0] * amp);
      for (let x = 1; x < w; x++) ctx.lineTo(x, mid - hiBuf[x] * amp);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(0, mid - loBuf[0] * amp);
      for (let x = 1; x < w; x++) ctx.lineTo(x, mid - loBuf[x] * amp);
      ctx.stroke();

      raf = requestAnimationFrame(draw);
    };

    raf = requestAnimationFrame(draw);

    return () => {
      if (raf !== null) cancelAnimationFrame(raf);
      raf = null;
      ro.disconnect();
    };
  });

  function hexToRgba(hex: string, alpha: number): string {
    const h = hex.replace('#', '');
    const r = parseInt(h.slice(0, 2), 16);
    const g = parseInt(h.slice(2, 4), 16);
    const b = parseInt(h.slice(4, 6), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }
</script>

<canvas bind:this={canvas} class="block h-full w-full rounded-md"></canvas>
