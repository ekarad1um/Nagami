//! Tests covering WGSL texture built-ins: stores, gathers, sample
//! variants (bias, grad, compare, compare-level, base-clamp-to-edge),
//! array-indexed sampling, and image-atomic operations.  Each MARK
//! block targets one built-in so regressions point directly at the
//! offending emission path.

use super::helpers::*;

// MARK: textureStore (ImageStore)

#[test]
fn texture_store_roundtrip() {
    let src = r#"
            @group(0) @binding(0) var tex: texture_storage_2d<rgba8unorm, write>;
            @compute @workgroup_size(1)
            fn main() {
                textureStore(tex, vec2u(0u, 0u), vec4f(1.0, 0.0, 0.0, 1.0));
            }
        "#;
    let out = compact(src);
    assert!(
        out.contains("textureStore("),
        "textureStore should be present: {out}"
    );
    assert_valid_wgsl(&out);
}

// MARK: textureGather

#[test]
fn texture_gather_roundtrip() {
    let src = r#"
        @group(0) @binding(0) var tex: texture_2d<f32>;
        @group(0) @binding(1) var samp: sampler;
        @fragment fn main() -> @location(0) vec4f {
            return textureGather(0, tex, samp, vec2f(0.5, 0.5));
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("textureGather("),
        "textureGather should be emitted: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn texture_gather_compare_roundtrip() {
    let src = r#"
        @group(0) @binding(0) var tex: texture_depth_2d;
        @group(0) @binding(1) var samp: sampler_comparison;
        @fragment fn main() -> @location(0) vec4f {
            return textureGatherCompare(tex, samp, vec2f(0.5, 0.5), 0.5);
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("textureGatherCompare("),
        "textureGatherCompare should be emitted: {out}"
    );
    assert_valid_wgsl(&out);
}

// MARK: textureSampleBaseClampToEdge

#[test]
fn texture_sample_base_clamp_to_edge_roundtrip() {
    let src = r#"
        @group(0) @binding(0) var tex: texture_2d<f32>;
        @group(0) @binding(1) var samp: sampler;
        @fragment fn main() -> @location(0) vec4f {
            return textureSampleBaseClampToEdge(tex, samp, vec2f(0.5, 0.5));
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("textureSampleBaseClampToEdge("),
        "textureSampleBaseClampToEdge should be emitted: {out}"
    );
    assert_valid_wgsl(&out);
}

// MARK: textureSampleBias

#[test]
fn texture_sample_bias_roundtrip() {
    let src = r#"
        @group(0) @binding(0) var tex: texture_2d<f32>;
        @group(0) @binding(1) var samp: sampler;
        @fragment fn main() -> @location(0) vec4f {
            return textureSampleBias(tex, samp, vec2f(0.5, 0.5), 2.0);
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("textureSampleBias("),
        "textureSampleBias should be emitted: {out}"
    );
    assert_valid_wgsl(&out);
}

// MARK: textureSampleGrad

#[test]
fn texture_sample_grad_roundtrip() {
    let src = r#"
        @group(0) @binding(0) var tex: texture_2d<f32>;
        @group(0) @binding(1) var samp: sampler;
        @fragment fn main() -> @location(0) vec4f {
            return textureSampleGrad(tex, samp, vec2f(0.5, 0.5), vec2f(1.0, 0.0), vec2f(0.0, 1.0));
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("textureSampleGrad("),
        "textureSampleGrad should be emitted: {out}"
    );
    assert_valid_wgsl(&out);
}

// MARK: textureSampleCompare / textureSampleCompareLevel

#[test]
fn texture_sample_compare_roundtrip() {
    let src = r#"
        @group(0) @binding(0) var tex: texture_depth_2d;
        @group(0) @binding(1) var samp: sampler_comparison;
        @fragment fn main() -> @location(0) vec4f {
            let d = textureSampleCompare(tex, samp, vec2f(0.5, 0.5), 0.5);
            return vec4f(d, d, d, 1.0);
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("textureSampleCompare("),
        "textureSampleCompare should be emitted: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn texture_sample_compare_level_roundtrip() {
    let src = r#"
        @group(0) @binding(0) var tex: texture_depth_2d;
        @group(0) @binding(1) var samp: sampler_comparison;
        @fragment fn main() -> @location(0) vec4f {
            let d = textureSampleCompareLevel(tex, samp, vec2f(0.5, 0.5), 0.5);
            return vec4f(d, d, d, 1.0);
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("textureSampleCompareLevel("),
        "textureSampleCompareLevel should be emitted: {out}"
    );
    assert_valid_wgsl(&out);
}

// MARK: ImageSample with array_index

#[test]
fn texture_sample_2d_array_roundtrip() {
    let src = r#"
        @group(0) @binding(0) var tex: texture_2d_array<f32>;
        @group(0) @binding(1) var samp: sampler;
        @fragment fn main() -> @location(0) vec4f {
            return textureSampleLevel(tex, samp, vec2f(0.5, 0.5), 0, 0.0);
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("textureSampleLevel("),
        "textureSampleLevel should be emitted: {out}"
    );
    assert_valid_wgsl(&out);
}

// MARK: ImageAtomic

#[test]
fn image_atomic_add_roundtrip() {
    let src = r#"
        @group(0) @binding(0) var t: texture_storage_2d<r32uint, atomic>;
        @compute @workgroup_size(1) fn main() {
            var v = 1u;
            textureAtomicAdd(t, vec2u(0, 0), v);
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("textureAtomicAdd"),
        "textureAtomicAdd should be emitted: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn image_atomic_multi_use_coord_roundtrip() {
    // Exercise ImageAtomic with a shared coordinate expression to verify
    // ref counts are computed correctly for ImageAtomic operands.
    let src = r#"
        @group(0) @binding(0) var t: texture_storage_2d<r32uint, atomic>;
        @compute @workgroup_size(1) fn main() {
            let c = vec2u(0, 0);
            var v1 = 1u;
            var v2 = 2u;
            textureAtomicAdd(t, c, v1);
            textureAtomicAdd(t, c, v2);
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("textureAtomicAdd"),
        "textureAtomicAdd should be emitted: {out}"
    );
    assert_valid_wgsl(&out);
}
