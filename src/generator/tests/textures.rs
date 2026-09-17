//! WGSL texture built-ins: stores, gathers, sample variants, array-indexed
//! sampling, image atomics; one MARK block per built-in.

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

/// A single-use image operation bound before a loop keeps its `let`
/// (`loop_sunk_work`), and so does the arithmetic over a buffer load beside
/// it: the platform compiler does not hoist either back out.
#[test]
fn image_op_is_not_sunk_into_a_loop() {
    let src = r#"
        @group(0) @binding(0) var tex: texture_2d<f32>;
        @group(0) @binding(1) var samp: sampler;
        @group(0) @binding(2) var<storage, read_write> o: array<f32>;
        @compute @workgroup_size(1) fn main() {
            let c = textureSampleLevel(tex, samp, vec2f(0.5), 0.0);
            let l = textureLoad(tex, vec2i(1), 0);
            let sz = textureDimensions(tex);
            let d = o[1] * 2.0;
            var total = 0.0;
            for (var i = 0; i < 4; i++) {
                total = total + c.x + l.y + f32(sz.x) + d;
            }
            o[0] = total;
        }
    "#;
    let out = compact(src);
    let loop_at = out.find("for(").expect("for loop emitted: {out}");
    let (before, body) = out.split_at(loop_at);
    for op in ["textureSampleLevel(", "textureLoad(", "textureDimensions("] {
        assert!(
            before.contains(op) && !body.contains(op),
            "{op} must stay bound before the loop: {out}"
        );
    }
    assert!(
        before.contains("*2") && !body.contains("*2"),
        "invariant arithmetic is bound before the loop too: {out}"
    );
    assert_valid_wgsl(&out);

    // Emitted and consumed at the same loop depth, the op crosses no loop
    // boundary and renders at its use.
    let same_depth = r#"
        @group(0) @binding(0) var tex: texture_2d<f32>;
        @group(0) @binding(2) var<storage, read_write> o: array<f32>;
        @compute @workgroup_size(1) fn main() {
            var total = 0.0;
            for (var i = 0; i < 4; i++) {
                let l = textureLoad(tex, vec2i(i), 0);
                total = total + l.y;
            }
            o[0] = total;
        }
    "#;
    let out = compact(same_depth);
    assert!(
        out.contains("+=textureLoad("),
        "same-depth use is inlined: {out}"
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
    // Shared coordinate: ref counts must cover ImageAtomic operands.
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

// MARK: Guards over image operations

/// End to end: the pipeline neither hoists an implicit-LOD sample out of
/// the `&&` it was guarded by nor speculates the resolve's loads into a
/// `select`.  A merge or a hoist would put a fetch in the body's top-level
/// statements; only the tail load behind `D`'s early return belongs there.
#[test]
fn image_operations_keep_their_guards_through_the_pipeline() {
    let src = r#"
        @group(0) @binding(0) var t: texture_2d<f32>;
        @group(0) @binding(1) var s: sampler;
        @group(0) @binding(2) var<storage, read_write> o: array<vec4f>;
        fn resolve(c: bool, p: vec2i) -> vec4f {
            if (c) { return textureLoad(t, p, 0) + textureLoad(t, p + 1, 0); }
            return textureLoad(t, p * 2, 0);
        }
        @fragment fn fs(@location(0) uv: vec2f) -> @location(0) vec4f {
            var lit: bool;
            if (uv.x > 0.5) { lit = textureSample(t, s, uv).x > 0.5; } else { lit = false; }
            let base = resolve(lit, vec2i(uv));
            o[0] = base;
            return select(base, vec4f(1.0), lit);
        }
    "#;
    let out = compact_with_passes(src, Profile::Max);
    assert_valid_wgsl(&out);
    let module = naga::front::wgsl::parse_str(&out).expect("re-parse");
    let image_ops = |f: &naga::Function, block: &naga::Block| {
        block
            .iter()
            .filter_map(|stmt| match stmt {
                naga::Statement::Emit(range) => Some(range.clone()),
                _ => None,
            })
            .flatten()
            .filter(|h| crate::passes::expr_util::is_expensive_expr(&f.expressions[*h]))
            .count()
    };
    let (mut top_level, mut total) = (0usize, 0usize);
    for f in crate::ir::visit::all_functions(&module) {
        top_level += image_ops(f, &f.body);
        total += f
            .expressions
            .iter()
            .filter(|(_, e)| crate::passes::expr_util::is_expensive_expr(e))
            .count();
    }
    assert_eq!(total, 4, "the four fetches survive: {out}");
    assert_eq!(top_level, 1, "only the tail load runs unguarded: {out}");
}

/// A fetch read once through a consumer read twice: the consumer's `let`
/// would carry the fetch ahead of the `&&`, so the guard stays an `if`
/// and the load renders inside it.
#[test]
fn a_fetch_behind_a_bound_consumer_keeps_its_guard() {
    let src = r#"
        @group(0) @binding(0) var t: texture_2d<f32>;
        @fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
            var ok: bool;
            if (p.x > 0.5) {
                let s = textureLoad(t, vec2i(p.xy), 0).xy;
                ok = s.x + s.y > 1.0;
            } else { ok = false; }
            return select(vec4f(0.0), vec4f(1.0), ok);
        }
    "#;
    let out = compact_with_passes(src, Profile::Max);
    assert_valid_wgsl(&out);
    let load = out.find("textureLoad(").expect("the fetch");
    let guard = out.find("if ").expect("the guard survives: {out}");
    assert!(guard < load, "the fetch renders inside the guard: {out}");
    assert!(!out.contains("&&"), "{out}");
}
