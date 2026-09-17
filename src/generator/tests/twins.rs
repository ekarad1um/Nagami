//! Structurally equal values (`module_emit::twins`): a later spelling of a
//! value in scope renders as the first's `let` name, the two priced as one.

use super::helpers::*;

/// The front-end gives each spelling its own tree; the binder prices the
/// value by both and the second spelling takes the name.
#[test]
fn a_value_spelled_twice_binds_once() {
    let out = compact(
        "fn f(x: f32, y: f32) -> f32 { let a = sqrt(x * x + y * y) * 0.5; let b = sqrt(x * x + y * y) * 0.25; return a + b; }\
         @fragment fn main(@location(0) p: vec2f) -> @location(0) vec4f { return vec4f(f(p.x, p.y)); }",
    );
    assert!(
        out.contains("let A=sqrt(x*x+y*y);return A*.5+A*.25;"),
        "{out}"
    );
}

/// Two spellings of a short value stay inline: the `let` is priced on the
/// uses of both (2 * 5 < 5 + 1 + 6 + 2), as one value's would be.
#[test]
fn two_spellings_of_a_short_value_stay_inline() {
    let out = compact(
        "fn f(x: f32, y: f32) -> f32 { let a = x * y + 1; let b = x * y + 1; return a * b; }\
         @fragment fn main(@location(0) p: vec2f) -> @location(0) vec4f { return vec4f(f(p.x, p.y)); }",
    );
    assert!(out.contains("return (x*y+1)*(x*y+1);"), "{out}");
}

/// A spelling in a nested block reads the name of a first spelling above
/// it; one in a sibling block does not see it, and spells the value.
#[test]
fn a_twin_takes_the_name_only_where_the_first_is_in_scope() {
    let nested = compact(
        "fn f(x: f32, y: f32, c: bool) -> f32 { var r = sqrt(x * x + y * y); if c { r = sqrt(x * x + y * y) * 2; } return r; }\
         @fragment fn main(@location(0) p: vec2f) -> @location(0) vec4f { return vec4f(f(p.x, p.y, p.x > 0)); }",
    );
    assert!(
        nested.contains("let A=sqrt(x*x+y*y);var r=A;if c{r=A*2;}"),
        "{nested}"
    );
    let sibling = compact(
        "fn f(x: f32, y: f32, c: bool) -> f32 { var r = 0.0; if c { r = sqrt(x * x + y * y) * 2; } else { r = sqrt(x * x + y * y) * 3; } return r; }\
         @fragment fn main(@location(0) p: vec2f) -> @location(0) vec4f { return vec4f(f(p.x, p.y, p.x > 0)); }",
    );
    assert!(
        sibling.contains("if c{r=sqrt(x*x+y*y)*2;}else{r=sqrt(x*x+y*y)*3;}"),
        "{sibling}"
    );
    assert_valid_wgsl(&sibling);
}

/// A loop's `continuing` starts afresh: the `for` header renders it ahead
/// of the body's `let`s, so a twin there of a body value spells itself.
#[test]
fn a_twin_in_the_continuing_block_does_not_see_the_body() {
    let out = compact(
        "@group(0) @binding(0) var<storage, read_write> o: array<f32>;\
         @compute @workgroup_size(1) fn main(@builtin(local_invocation_index) x: u32) {\
           for (var i = 0u; i < 8u; i += u32(sqrt(f32(x * x + 3))) + 1) { o[i] = sqrt(f32(x * x + 3)) * f32(i); } }",
    );
    assert!(
        out.contains("i+=u32(sqrt(f32(x*x+3)))+1){o[i]=sqrt(f32(x*x+3))*f32(i);}"),
        "{out}"
    );
    assert_valid_wgsl(&out);
}

/// A load is one value at its `Emit` (the emitter binds it ahead of the
/// write), so a spelling over a fresh load of the place is no twin.
#[test]
fn a_spelling_over_a_later_load_is_not_a_twin() {
    let out = compact(
        "fn f(x: f32) -> f32 { var v = x; let a = sqrt(v * 2 + 1); v = 3; let b = sqrt(v * 2 + 1); return a + b; }\
         @fragment fn main(@location(0) p: f32) -> @location(0) vec4f { return vec4f(f(p)); }",
    );
    assert!(
        out.contains("let A=v;v=3;return sqrt(A*2+1)+sqrt(v*2+1);"),
        "{out}"
    );
}

/// A splat's inline text is its consumer's: under a mixed arithmetic
/// operator it is the bare scalar.  Three of them are no `let`.
#[test]
fn a_splat_under_a_mixed_operator_is_not_a_candidate() {
    let out = compact(
        "fn f(c: vec2i, b: i32) -> vec2u { return vec2u((c % b + b) % b); }\
         @fragment fn main(@location(0) p: vec2f) -> @location(0) vec4f { return vec4f(vec2f(f(vec2i(p), 3)), 0, 1); }",
    );
    assert!(out.contains("return vec2u((c%b+b)%b);"), "{out}");
}

/// Lanes read off twins of one vector fold as lanes of that vector: the
/// four `m[0].x .. m[0].w` of a copied column are `m[0]`.
#[test]
fn lanes_read_off_twins_of_a_vector_fold_to_the_vector() {
    let out = compact(
        "fn f(m: mat2x4f) -> mat2x4f { return mat2x4f(vec4f(m[0].x, m[0].y, m[0].z, m[0].w), vec4f(m[1].x, m[1].y, m[1].z, m[1].w)); }\
         @fragment fn main(@location(0) p: vec4f) -> @location(0) vec4f { return f(mat2x4f(p, p))[0]; }",
    );
    assert!(out.contains("return mat2x4f(m[0],m[1]);"), "{out}");
}
