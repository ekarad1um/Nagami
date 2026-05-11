// In-place radix-2 Cooley-Tukey FFT with precomputed twiddle factors.
// Size fixed at construction; reusable across frames.

export class Fft {
  readonly size: number;
  private readonly cos: Float32Array;
  private readonly sin: Float32Array;

  constructor(size: number) {
    if (size < 2 || (size & (size - 1)) !== 0) {
      throw new Error('fft: size must be a power of 2 ≥ 2');
    }
    this.size = size;
    const half = size >>> 1;
    this.cos = new Float32Array(half);
    this.sin = new Float32Array(half);
    for (let i = 0; i < half; i++) {
      const a = (-2 * Math.PI * i) / size;
      this.cos[i] = Math.cos(a);
      this.sin[i] = Math.sin(a);
    }
  }

  forward(re: Float32Array, im: Float32Array): void {
    const n = this.size;
    let j = 0;
    for (let i = 1; i < n; i++) {
      let bit = n >>> 1;
      while (j & bit) {
        j ^= bit;
        bit >>>= 1;
      }
      j ^= bit;
      if (i < j) {
        let t = re[i];
        re[i] = re[j];
        re[j] = t;
        t = im[i];
        im[i] = im[j];
        im[j] = t;
      }
    }
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >>> 1;
      const step = n / size;
      for (let i = 0; i < n; i += size) {
        let tw = 0;
        for (let k = 0; k < half; k++) {
          const c = this.cos[tw];
          const s = this.sin[tw];
          const a = i + k;
          const b = i + k + half;
          const tRe = re[b] * c - im[b] * s;
          const tIm = re[b] * s + im[b] * c;
          re[b] = re[a] - tRe;
          im[b] = im[a] - tIm;
          re[a] += tRe;
          im[a] += tIm;
          tw += step;
        }
      }
    }
  }
}

export function hannWindow(size: number): Float32Array {
  const w = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (size - 1)));
  }
  return w;
}
