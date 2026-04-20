//! Tests for compound-assignment folding (`x = x + y` rewritten to
//! `x += y`) and related boolean-branch flips that exploit compact
//! compound forms.  Sections below pin the structural-pointer
//! equality predicate, the NaN-safety gate on ordered comparisons,
//! and the accept/reject-flip heuristic that yields shorter if/else
//! shapes.

use super::helpers::*;

// MARK: Compound assignment

#[test]
fn compound_add_assignment() {
    let out = compact("fn f() -> f32 { var x = 1.0f; x = x + 2.0; return x; }");
    assert!(out.contains("x+="), "expected compound +=: {out}");
    assert!(
        !out.contains("x=(x+"),
        "should not have expanded form: {out}"
    );
}

#[test]
fn compound_sub_only_when_lhs_is_self() {
    // x = x - 2 -> x -= 2
    let out = compact("fn f() -> f32 { var x = 5.0f; x = x - 2.0; return x; }");
    assert!(out.contains("x-="), "expected compound -=: {out}");
}

#[test]
fn no_compound_for_non_commutative_rhs() {
    // x = 2 - x  is NOT  x -= 2
    let out = compact("fn f() -> f32 { var x = 5.0f; x = 2.0 - x; return x; }");
    assert!(
        !out.contains("x-="),
        "should not compound when self is on RHS: {out}"
    );
}

#[test]
fn compound_mul_commutative_rhs() {
    // x = 2.0 * x -> x *= 2.0 (multiplication is commutative,
    // so RHS-position pointer should still compound).
    let out = compact(
        r#"
            fn f() -> f32 {
                var x: f32 = 1.0;
                x = 2.0 * x;
                return x;
            }
        "#,
    );
    assert!(
        out.contains("x*=2") || out.contains("x *= 2"),
        "commutative rhs should become compound assign: {out}"
    );
}

#[test]
fn compound_div_assignment() {
    let out = compact("fn f() -> f32 { var x = 10f; x = x / 2.0; return x; }");
    assert!(out.contains("/="), "x = x / 2 should become x /= 2: {out}");
}

#[test]
fn compound_mod_assignment() {
    let out = compact("fn f() -> i32 { var x = 10i; x = x % 3; return x; }");
    assert!(out.contains("%="), "x = x % 3 should become x %= 3: {out}");
}

#[test]
fn compound_bitwise_and_assignment() {
    let out = compact("fn f() -> u32 { var x = 0xFFu; x = x & 0x0Fu; return x; }");
    assert!(
        out.contains("&="),
        "x = x & mask should become x &= mask: {out}"
    );
}

#[test]
fn compound_bitwise_or_commutative() {
    // Bitwise OR is commutative, so `x = 0x0F | x` should also compound.
    let out = compact("fn f() -> u32 { var x = 0xF0u; x = 0x0Fu | x; return x; }");
    assert!(
        out.contains("|="),
        "commutative OR should become x |= val: {out}"
    );
}

#[test]
fn no_compound_for_div_rhs() {
    // Division is non-commutative: `x = 2 / x` should NOT become `x /= 2`.
    let out = compact("fn f() -> f32 { var x = 10f; x = 2.0 / x; return x; }");
    assert!(
        !out.contains("/="),
        "x = 2 / x should not become x /= 2: {out}"
    );
}

// MARK: Compound assignment with structural pointer equality

#[test]
fn compound_assign_array_index() {
    // buf[0] = buf[0] + 1.0 should fold to buf[0] += 1.0
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> buf: array<f32, 4>;
        @compute @workgroup_size(1) fn main() {
            buf[0] = buf[0] + 1.0;
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("+="),
        "array index compound assign should fold: {out}"
    );
    assert!(
        !out.contains("]=") || out.contains("]+="),
        "should use compound, not expanded form: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn compound_assign_struct_member_via_index() {
    // particles[i].pos = particles[i].pos + vel should fold
    let src = r#"
        struct Particle { pos: vec3<f32>, vel: vec3<f32> }
        @group(0) @binding(0) var<storage, read_write> p: array<Particle, 64>;
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let i = gid.x;
            p[i].pos = p[i].pos + p[i].vel;
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains(".pos+="),
        "nested struct member compound assign should fold: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn compound_assign_dynamic_index_in_function() {
    // Function with dynamic index: buf[i] = buf[i] + v -> buf[i] += v
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> buf: array<f32, 4>;
        fn add_at(i: u32, v: f32) {
            buf[i] = buf[i] + v;
        }
        @compute @workgroup_size(1) fn main() { add_at(0u, 1.0); }
    "#;
    let out = compact(src);
    assert!(
        out.contains("+="),
        "dynamic index compound assign in function should fold: {out}"
    );
    assert_valid_wgsl(&out);
}

// MARK: Empty accept flip: comparison operators

#[test]
fn empty_accept_flips_comparison() {
    // `if c < 10 {} else { break; }` should become `if c >= 10 { break; }`
    // for integers (comparison flipped, empty true-branch eliminated),
    // OR the loop should be reconstructed as a for-loop with the original condition.
    let out = compact(
        r#"
            fn f() -> i32 {
                var i: i32 = 0;
                loop {
                    if i < 10 {
                    } else {
                        break;
                    }
                    i = i + 1;
                }
                return i;
            }
        "#,
    );
    assert!(
        !out.contains("{}else"),
        "empty true-branch with else should be eliminated: {out}"
    );
    assert!(
        out.contains(">=10")
            || out.contains(">= 10")
            || out.contains("for(") && out.contains("<10"),
        "comparison should be flipped to >= or loop reconstructed as for: {out}"
    );
}

#[test]
fn empty_accept_flips_equality() {
    // `if a == b {} else { x = 1; }` -> `if a != b { x = 1; }`
    let out = compact(
        r#"
            fn f(a: i32, b: i32) -> i32 {
                var x: i32 = 0;
                if a == b {
                } else {
                    x = 1;
                }
                return x;
            }
        "#,
    );
    assert!(
        !out.contains("{}else"),
        "empty true-branch with else should be eliminated: {out}"
    );
    assert!(out.contains("!="), "== should be flipped to !=: {out}");
}

#[test]
fn empty_accept_flips_less_equal() {
    let out = compact("fn f(a: i32) -> i32 { var x = 0i; if a <= 5 {} else { x = 1; } return x; }");
    assert!(out.contains(">"), "<= should flip to >: {out}");
    assert!(!out.contains("{}"), "empty branch should be removed: {out}");
}

#[test]
fn empty_accept_flips_greater() {
    let out = compact("fn f(a: i32) -> i32 { var x = 0i; if a > 5 {} else { x = 1; } return x; }");
    assert!(out.contains("<="), "> should flip to <=: {out}");
}

#[test]
fn empty_accept_flips_greater_equal() {
    let out = compact("fn f(a: i32) -> i32 { var x = 0i; if a >= 5 {} else { x = 1; } return x; }");
    // >= flips to <
    assert!(
        out.contains("<") && !out.contains("<=") && !out.contains(">="),
        ">= should flip to <: {out}"
    );
}

#[test]
fn empty_accept_flips_not_equal() {
    let out = compact(
        "fn f(a: i32, b: i32) -> i32 { var x = 0i; if a != b {} else { x = 1; } return x; }",
    );
    assert!(out.contains("=="), "!= should flip to ==: {out}");
}

// MARK: NaN-safe comparison semantics

#[test]
fn negated_float_comparison_preserves_nan_semantics() {
    // Float ordered comparisons must NOT be flipped when negated,
    // because !(x < y) != (x >= y) when NaN is involved.
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> buf: array<f32, 4>;
        @compute @workgroup_size(1)
        fn main() {
            let x = buf[0];
            if x < 0.0 {
            } else {
                buf[1] = 1.0;
            }
            buf[2] = 2.0;
        }
    "#;
    let out = compact(src);
    // Must use !(...<...) rather than flipping to >=
    assert!(
        !out.contains(">="),
        "float < must not flip to >= (NaN semantics): {out}"
    );
    assert!(
        out.contains("!("),
        "float negation should use !() wrapper: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn negated_int_comparison_still_flips() {
    // Integer comparisons are safe to flip (no NaN).
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> buf: array<i32, 4>;
        @compute @workgroup_size(1)
        fn main() {
            let x = buf[0];
            if x < 0 {
            } else {
                buf[1] = 1;
            }
            buf[2] = 2;
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains(">="),
        "integer < should flip to >= (no NaN concern): {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn negated_float_equality_still_flips() {
    // Float == / != flips are NaN-safe: !(x==y) is equivalent to (x!=y).
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> buf: array<f32, 4>;
        @compute @workgroup_size(1)
        fn main() {
            let x = buf[0];
            if x == 0.0 {
            } else {
                buf[1] = 1.0;
            }
            buf[2] = 2.0;
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("!="),
        "float == should flip to != (NaN-safe): {out}"
    );
    assert_valid_wgsl(&out);
}
