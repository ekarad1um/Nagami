//! Zero-allocation CPU kernels for the inference hot loop.
//!
//! All three of these are caller-buffer; no heap activity
//! in steady state.  They are split out from
//! [`crate::inference::engine`] so they can be unit-tested
//! in isolation (no `Session`, no Burn, no I/O) and so the
//! engine module reads as a single linear pipeline.
//!
//! Behavior is pinned by the parity tests in
//! `tests/preproc_parity.rs` against the reference NPY
//! captures under `reference/`.
//!
//! [`top_k_indices_into`] performs a partial sort and may
//! allocate when `out`'s capacity is insufficient; the
//! engine's hot loop pre-reserves it, so steady state has
//! zero allocations.

/// `logits[c] = bias[c] + sum_i features[i] * weight[i, c]`,
/// row-major weight of shape `[in_features, n_classes]`.
/// Zero alloc.
///
/// The inner loop is a contiguous `n_classes`-wide FMA
/// accumulation that LLVM auto-vectorizes on aarch64 NEON.
/// Bypasses Burn's Tensor path which heap-allocates several
/// times per call.
pub fn head_forward(features: &[f32], weight: &[f32], bias: &[f32], logits: &mut [f32]) {
    let n = logits.len();
    // These are *runtime* `assert_eq!`, not `debug_assert_eq!`.
    // LOAD-BEARING for aarch64 codegen: with the runtime asserts
    // in place, LLVM proves `bias.len() == logits.len()` (so
    // `copy_from_slice` lowers to a tight `memcpy`) AND
    // `weight.len() == features.len() * n` (so `chunks_exact(n)`
    // proves the inner-loop slice length, the `logits.iter_mut()
    // .zip(row.iter())` zip elides per-iteration bounds checks,
    // and the inner FMA accumulation auto-vectorizes to NEON
    // `fmla` without panic-edge poisoning).
    //
    // FUTURE CONTRIBUTOR: do NOT downgrade these to
    // `debug_assert_eq!` without re-checking the release-build
    // codegen on aarch64 Linux.  Without the runtime asserts,
    // the head matmul re-emits per-iteration bounds-check
    // trampolines (the most expensive CPU-side op in the loop).
    assert_eq!(bias.len(), n, "bias shape mismatches logits");
    assert_eq!(
        weight.len(),
        features.len() * n,
        "weight shape mismatches features x logits",
    );
    logits.copy_from_slice(bias);
    for (f, row) in features.iter().zip(weight.chunks_exact(n)) {
        for (out, &w) in logits.iter_mut().zip(row.iter()) {
            *out += f * w;
        }
    }
}

/// Numerically-stable softmax: `probs[i] = exp(logits[i] -
/// max) / sum`.  Caller-owned buffers; `probs.len()` must
/// equal `logits.len()`.  Subtracts the max before `exp` to
/// avoid overflow.
///
/// Length contract is `assert_eq!` (not `debug_assert_eq!`):
/// an out-of-sync pair would silently truncate or scale stale
/// tail values, which the caller may later read as valid
/// probabilities.
pub fn softmax_into(logits: &[f32], probs: &mut [f32]) {
    assert_eq!(
        logits.len(),
        probs.len(),
        "softmax shape mismatch: logits.len()={}, probs.len()={}",
        logits.len(),
        probs.len(),
    );
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    // If logits is all `-inf` (extreme silence), `m` is `-inf` and every
    // `exp(0) = 1` -> uniform distribution.  Catching this explicitly so we
    // don't propagate `-inf - -inf = NaN` into the exponent.
    if !m.is_finite() {
        let n = probs.len();
        let p = if n == 0 { 0.0 } else { 1.0 / n as f32 };
        probs.fill(p);
        return;
    }
    let mut sum = 0.0f32;
    for (p, &l) in probs.iter_mut().zip(logits.iter()) {
        let e = (l - m).exp();
        *p = e;
        sum += e;
    }
    // sum is guaranteed > 0 here: at least the entry with `l == m`
    // contributes `exp(0) = 1`.
    let inv = 1.0 / sum;
    for p in probs.iter_mut() {
        *p *= inv;
    }
}

/// Fill `out` with the top-`k` indices of `xs`, sorted by descending
/// value.  `k = 0` yields `out.clear()`; `k > xs.len()` is silently capped.
///
/// `out` is cleared and re-populated.  The function uses
/// `partial_cmp(...).unwrap_or(Equal)` so NaN sorts as equal to
/// everything (rather than panicking) -- defensive against an upstream
/// bug; the engine drops NaN frames before this is reached.
///
/// Algorithm: `select_nth_unstable_by` partitions in O(n) so that the
/// top-`k` indices occupy the first `k` slots in unspecified order;
/// then `sort_unstable_by` sorts just those `k`.  Total: O(n + k log k).
/// For typical heads (n=3..17, k=3) this is the same as the previous
/// O(n log n) full sort to within a constant factor; the win shows up
/// at the catalogue's `MAX_N_CLASSES = 100_000` ceiling.
///
/// Steady-state allocation: zero, IF `out.capacity() >= xs.len()`.  The
/// engine reserves up to `MAX_N_CLASSES` once, before the loop.
pub fn top_k_indices_into(xs: &[f32], k: usize, out: &mut Vec<usize>) {
    out.clear();
    let n = xs.len();
    let k_capped = k.min(n);
    if k_capped == 0 {
        return;
    }
    out.extend(0..n);
    // Comparator orders by descending value: returning `Less` for
    // higher-valued indices puts them earlier.  NaN compares Equal
    // to everything for the same defensive-sort reason as before.
    let cmp = |&a: &usize, &b: &usize| {
        xs[b]
            .partial_cmp(&xs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    };
    if k_capped < n {
        // After this call, indices `[0..k_capped]` are exactly the
        // top-`k` (in unspecified order).
        out.select_nth_unstable_by(k_capped - 1, cmp);
        out.truncate(k_capped);
    }
    // Sort only the surviving k_capped elements descending.
    out.sort_unstable_by(cmp);
}

/// Transpose a frame-major `[N_FRAMES][N_BINS]` spectrogram
/// into bin-major `[bin][frame]` layout for librknnrt's NHWC
/// ingestion.
///
/// The librknnrt runtime reports `dims=[1, 232, 1, 43]`
/// (`[N, H=bins, W=1, C=frames]`) on rv1126b /
/// RKNN-toolkit2 2.4.0, even though the source ONNX
/// declared `[1, 43, 232, 1]` NHWC; librknnrt transposes
/// internally to NCHW `[1, 1, 43, 232]`.  Feeding the
/// frame-major layout (`frame * N_BINS + bin`) would
/// silently produce a near-uniform softmax with zero NPU
/// activations.
///
/// `out.len()` must equal `spec.len() * spec[0].len()`.
pub fn transpose_frame_major_to_bin_major<const N_FRAMES: usize, const N_BINS: usize>(
    spec: &[[f32; N_BINS]; N_FRAMES],
    out: &mut [f32],
) {
    debug_assert_eq!(out.len(), N_FRAMES * N_BINS);
    for (f, row) in spec.iter().enumerate() {
        for (b, &v) in row.iter().enumerate() {
            out[b * N_FRAMES + f] = v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bias-only forward (zero features) returns the bias.
    #[test]
    fn head_forward_bias_only() {
        let features = vec![0.0; 4];
        let weight = vec![999.0; 4 * 3];
        let bias = vec![1.0, 2.0, 3.0];
        let mut logits = vec![0.0; 3];
        head_forward(&features, &weight, &bias, &mut logits);
        assert_eq!(logits, vec![1.0, 2.0, 3.0]);
    }

    /// Identity-ish forward: features = [1, 0, 0], weight = [[a,b,c],
    /// [d,e,f], [g,h,i]] -> logits ~= [a, b, c] + bias.
    #[test]
    fn head_forward_first_feature_picks_first_row() {
        let features = vec![1.0, 0.0, 0.0];
        // Row-major [in=3, out=3]: weight[i * 3 + c]
        let weight = vec![
            10.0, 20.0, 30.0, // i=0
            -1.0, -2.0, -3.0, // i=1
            100.0, 200.0, 300.0, // i=2
        ];
        let bias = vec![0.5, 0.5, 0.5];
        let mut logits = vec![0.0; 3];
        head_forward(&features, &weight, &bias, &mut logits);
        assert_eq!(logits, vec![10.5, 20.5, 30.5]);
    }

    #[test]
    fn softmax_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0];
        let mut probs = vec![0.0; 3];
        softmax_into(&logits, &mut probs);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum={sum}");
        // Monotonicity: largest logit -> largest prob.
        assert!(probs[2] > probs[1] && probs[1] > probs[0]);
    }

    /// Large logits (~=1000) would overflow `exp` without max-subtraction.
    #[test]
    fn softmax_stable_under_huge_logits() {
        let logits = vec![1000.0, 1001.0, 999.0];
        let mut probs = vec![0.0; 3];
        softmax_into(&logits, &mut probs);
        assert!(probs.iter().all(|p| p.is_finite()));
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum={sum}");
    }

    /// All -inf -> uniform (the silence-edge case).
    #[test]
    fn softmax_uniform_on_neg_inf_logits() {
        let logits = vec![f32::NEG_INFINITY; 4];
        let mut probs = vec![0.0; 4];
        softmax_into(&logits, &mut probs);
        for &p in probs.iter() {
            assert!((p - 0.25).abs() < 1e-6, "p={p}");
        }
    }

    #[test]
    fn top_k_basic() {
        let xs = vec![0.1, 0.5, 0.3, 0.05, 0.05];
        let mut out = Vec::with_capacity(8);
        top_k_indices_into(&xs, 3, &mut out);
        assert_eq!(out, vec![1, 2, 0]);
    }

    /// K > len caps to len (no panic, no extra entries).
    #[test]
    fn top_k_caps_at_len() {
        let xs = vec![0.4, 0.6];
        let mut out = Vec::with_capacity(8);
        top_k_indices_into(&xs, 100, &mut out);
        assert_eq!(out, vec![1, 0]);
    }

    /// K = 0 yields empty.
    #[test]
    fn top_k_zero_yields_empty() {
        let xs = vec![0.4, 0.6];
        let mut out = vec![999; 4];
        top_k_indices_into(&xs, 0, &mut out);
        assert!(out.is_empty());
    }

    /// Reusing `out` across calls does not allocate.
    #[test]
    fn top_k_reuse_does_not_allocate_when_capacity_reserved() {
        let xs = vec![0.2; 32];
        let mut out: Vec<usize> = Vec::with_capacity(32);
        let initial_cap = out.capacity();
        for _ in 0..10 {
            top_k_indices_into(&xs, 5, &mut out);
        }
        assert!(
            out.capacity() <= initial_cap,
            "capacity grew across reuses (initial={initial_cap}, now={})",
            out.capacity(),
        );
    }

    /// Transpose: spec[f][b] -> flat[b * N_FRAMES + f].
    #[test]
    fn transpose_basic() {
        let mut spec = [[0.0f32; 3]; 2]; // N_FRAMES=2, N_BINS=3
        spec[0] = [1.0, 2.0, 3.0];
        spec[1] = [10.0, 20.0, 30.0];
        let mut flat = vec![0.0f32; 6];
        transpose_frame_major_to_bin_major::<2, 3>(&spec, &mut flat);
        // Bin 0: frames [1, 10]; bin 1: [2, 20]; bin 2: [3, 30]
        assert_eq!(flat, vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]);
    }
}
