//! The binder's byte rule: a `let` where the text it saves at the uses
//! exceeds its own bytes, the counts the text holds as the judge.

use super::helpers::*;

/// `let N=E;` costs `E + N + 6` and each use spells `N` for `E`: two uses
/// of a five-byte value inline (10 < 12 with the name), two of a
/// nine-byte one bind (18 > 16).
#[test]
fn a_two_use_value_binds_only_past_the_let_it_pays() {
    let short = compact(
        "@group(0) @binding(0) var<uniform> u: vec4f;\
         @fragment fn main() -> @location(0) vec4f { let s = u.x + u.y; return vec4f(s, s, 0, 0); }",
    );
    assert!(
        short.contains("vec4(u.x+u.y,u.x+u.y,0,0)"),
        "a short value inlines at both uses: {short}"
    );
    let long = compact(
        "@group(0) @binding(0) var<uniform> u: vec4f;\
         @fragment fn main() -> @location(0) vec4f { let s = u.x + u.y * u.z; return vec4f(s, s, 0, 0); }",
    );
    assert!(
        long.contains("let A=u.x+u.y*u.z;return vec4(A,A,0,0)"),
        "a long value binds: {long}"
    );
}

/// The parentheses a use adds around an inlined operator value are part of
/// its price: `x+y` twice under `*` is `(x+y)` twice, which a name is not.
#[test]
fn the_parentheses_of_a_use_price_the_inlined_value() {
    let bare = compact(
        "@group(0) @binding(0) var<uniform> u: vec4f;\
         @fragment fn main() -> @location(0) vec4f { let s = u.x + u.y; return vec4f(s + 1, s + 2, 0, 0); }",
    );
    assert!(
        bare.contains("vec4(u.x+u.y+1,u.x+u.y+2,0,0)"),
        "bare uses inline: {bare}"
    );
    let wrapped = compact(
        "@group(0) @binding(0) var<uniform> u: vec4f;\
         @fragment fn main() -> @location(0) vec4f { let s = u.x + u.y; return vec4f(s * 2, s * 3, 0, 0); }",
    );
    assert!(
        wrapped.contains("let A=u.x+u.y;return vec4(A*2,A*3,0,0)"),
        "parenthesised uses bind: {wrapped}"
    );
}

/// A two-byte value pays its `let` at ten uses (10 * 1 > 2 + 1 + 6).
#[test]
fn a_two_byte_value_binds_at_ten_uses() {
    let src = |uses: usize| {
        let sum = (0..uses).map(|_| "n").collect::<Vec<_>>().join("+");
        format!(
            "fn f(x: f32) -> f32 {{ let n = -x; return {sum}; }}\
             @fragment fn main(@location(0) p: f32) -> @location(0) vec4f {{ return vec4f(f(p)); }}"
        )
    };
    let nine = compact(&src(9));
    assert!(nine.contains("return -x+-x+"), "nine uses inline: {nine}");
    let ten = compact(&src(10));
    assert!(ten.contains("let A=-x;return A+A+"), "ten uses bind: {ten}");
}

/// A consumer the rule inlines renders its operand at each of its own
/// uses: `1/b` read by two two-use products appears four times, which the
/// census counting each consumer once prices as two.  The render's own
/// counts are the judge, and the function renders again by them.
#[test]
fn a_value_is_priced_by_the_uses_its_inlined_consumers_render() {
    let out = compact(
        "fn f(x: f32, y: f32) -> f32 {\
           let r = 1 / x; let c = y * r; let d = (y + 1) * r;\
           return c * c + d; }\
         @fragment fn main(@location(0) p: vec2f) -> @location(0) vec4f { return vec4f(f(p.x, p.y)); }",
    );
    assert!(
        out.contains("let A=1/x;return y*A*(y*A)+(y+1)*A;"),
        "the shared operand binds once its consumer inlines: {out}"
    );
}

/// Work a two-use value the bytes inline would carry into a loop is pinned
/// where the render found it sunk, and the function renders again: the
/// uniform load stays before the loop, the swizzle over its name inlines.
#[test]
fn a_two_use_value_inlined_into_a_loop_keeps_its_load_outside() {
    let src = "@group(0) @binding(0) var<uniform> u: vec4f;\
               @group(0) @binding(1) var<storage, read_write> o: array<vec2f>;\
               @compute @workgroup_size(1) fn main() {\
                 let d = u.zw;\
                 for (var i = 0; i < 4; i++) { o[i] = d * f32(i) + d; } }";
    let out = compact_with_passes(src, Profile::Max);
    let uniform = {
        let at = out.find("var<uniform> ").expect("the uniform") + "var<uniform> ".len();
        out[at..].split(':').next().unwrap().to_string()
    };
    let loop_at = out.find("for (").expect("the loop");
    assert!(
        out[..loop_at].contains(&format!("let b = {uniform};")),
        "the load is bound before the loop: {out}"
    );
    assert!(
        !out[loop_at..].contains(&format!("{uniform}.")),
        "no uniform read inside the loop: {out}"
    );
}
