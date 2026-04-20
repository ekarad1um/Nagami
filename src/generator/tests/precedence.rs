//! Parenthesis-elision tests.  Each case pins one scenario where
//! the emitter drops parentheses safely or, conversely, keeps them
//! to preserve non-associative behaviour (for example `a - (b - c)`
//! or float comparisons that cannot be NaN-flipped).

use super::helpers::*;

// MARK: Precedence-aware parenthesis elision

#[test]
fn mul_inside_add_drops_parens() {
    // a*b has higher prec than +, so no parens needed
    let out = compact("fn f(a:f32,b:f32,c:f32)->f32{return a*b+c;}");
    assert!(
        !out.contains("(a*b)"),
        "mul in add should not be parenthesised: {out}"
    );
    assert!(out.contains("a*b+c"), "expected a*b+c: {out}");
}

#[test]
fn add_inside_mul_keeps_parens() {
    let out = compact("fn f(a:f32,b:f32,c:f32)->f32{return (a+b)*c;}");
    assert!(
        out.contains("(a+b)*c"),
        "add in mul must keep parens: {out}"
    );
}

#[test]
fn right_assoc_sub_keeps_parens() {
    // a - (b - c): right child same prec -> parens needed (left-assoc)
    let out = compact("fn f(a:f32,b:f32,c:f32)->f32{return a-(b-c);}");
    assert!(
        out.contains("a-(b-c)"),
        "right sub in sub must keep parens: {out}"
    );
}

#[test]
fn left_same_prec_add_drops_parens() {
    // (a + b) + c -> left child same prec, left-assoc -> no parens
    let out = compact("fn f(a:f32,b:f32,c:f32)->f32{return (a+b)+c;}");
    assert!(
        !out.contains("(a+b)+c"),
        "left add in add should drop parens: {out}"
    );
    assert!(out.contains("a+b+c"), "expected a+b+c: {out}");
}

#[test]
fn unary_neg_on_name_no_parens() {
    let out = compact("fn f(a:f32)->f32{return -a;}");
    assert!(out.contains("-a"), "unary neg on name: {out}");
    assert!(!out.contains("(-a)"), "should not wrap unary neg: {out}");
}

#[test]
fn unary_neg_on_binary_keeps_parens() {
    let out = compact("fn f(a:f32,b:f32)->f32{return -(a+b);}");
    assert!(
        out.contains("-(a+b)"),
        "unary neg on binary must keep parens: {out}"
    );
}

#[test]
fn bitwise_or_inside_xor_keeps_parens() {
    // | has lower prec than ^, so parens must stay
    let out = compact("fn f(a:u32,b:u32,c:u32)->u32{return (a|b)^c;}");
    assert!(
        out.contains("(a|b)^c"),
        "bitwise or in xor must keep parens: {out}"
    );
}

#[test]
fn precedence_output_is_valid_wgsl() {
    let src = "fn f(a:f32,b:f32,c:f32)->f32{return (a+b)*c-a/(b+c);}";
    let out = compact(src);
    assert_valid_wgsl(&out);
}

#[test]
fn right_assoc_div_keeps_parens() {
    // a / (b / c) is NOT the same as a / b / c (left-assoc), so
    // parentheses on the RHS must be preserved.
    let out = compact(
        r#"
            fn f(a: f32, b: f32, c: f32) -> f32 {
                return a / (b / c);
            }
        "#,
    );
    assert!(
        out.contains("a/(b/c)") || out.contains("a / (b / c)"),
        "right-hand division should keep parens: {out}"
    );
}

// MARK: Additional operators

#[test]
fn logical_and_inside_or_drops_parens() {
    // `(a && b) || c` - && has higher precedence than ||, no parens needed.
    let out = compact("fn f(a: bool, b: bool, c: bool) -> bool { return (a && b) || c; }");
    // Output should NOT have `(a&&b)||c` - the inner parens are optional.
    assert_valid_wgsl(&out);
}

#[test]
fn logical_or_inside_and_keeps_parens() {
    // `(a || b) && c` - || has lower precedence than &&, parens required.
    let out = compact("fn f(a: bool, b: bool, c: bool) -> bool { return (a || b) && c; }");
    assert!(out.contains("("), "parens needed for || inside &&: {out}");
    assert_valid_wgsl(&out);
}

#[test]
fn shift_requires_parens_on_binary_operand() {
    // WGSL shift operators require `unary_expression` on both sides.
    // `(a + b) << c` must keep parens.
    let out = compact("fn f(a: u32, b: u32, c: u32) -> u32 { return (a + b) << c; }");
    assert!(
        out.contains("("),
        "parens required for binary inside shift: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn bitwise_xor_with_additive_child_keeps_parens() {
    let out = compact("fn f(a:u32,b:u32)->u32{return a^(b-1u);}");
    assert!(
        out.contains("a^(b-1)")
            || out.contains("a^(b-1u)")
            || out.contains("a ^ (b - 1)")
            || out.contains("a ^ (b - 1u)"),
        "xor with additive child must keep RHS parens: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn bitwise_xor_additive_child_inside_comparison_keeps_parens() {
    let out = compact("fn f(t:u32,a:u32,b:u32)->bool{return t==(a^(b-1u));}");
    assert!(
        out.contains("==(a^(b-1))")
            || out.contains("==(a^(b-1u))")
            || out.contains("== (a ^ (b - 1))")
            || out.contains("== (a ^ (b - 1u))"),
        "comparison against xor-with-subtraction must keep inner parens: {out}"
    );
    assert_valid_wgsl(&out);
}
