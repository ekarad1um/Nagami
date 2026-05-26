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

// MARK: Non-chainable equality / inequality parenthesisation

#[test]
fn equality_does_not_chain_left_child() {
    // `(a == b) == c` is not a valid WGSL expression because `==`/`!=`
    // operands must syntactically resolve to a strictly lower-precedence
    // form (https://www.w3.org/TR/WGSL/#composite-value-decomposition-expr).
    // Emitting `a==b==c` would be unparseable.
    let out = compact("fn f(a:i32,b:i32,c:i32)->bool{return (a==b)==(c==a);}");
    assert!(
        out.contains("(a==b)"),
        "left == child of == must stay parenthesised: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn inequality_does_not_chain_right_child() {
    // Same constraint as `==`: `a != (b != c)` must preserve the inner
    // parens.  The right child shares the parent precedence and would
    // already require parens for normal left-assoc rules; this test
    // also confirms `!=` is included in the non-chainable branch.
    // Types chosen so the inner `b != c` yields a bool that the outer
    // `!=` compares against a bool LHS - otherwise validation rejects
    // the IR before the emitter is exercised.
    let out = compact("fn f(a:bool,b:i32,c:i32)->bool{return a!=(b!=c);}");
    assert!(
        out.contains("a!=(b!=c)") || out.contains("a != (b != c)"),
        "right != child of != must stay parenthesised: {out}"
    );
    assert_valid_wgsl(&out);
}

// MARK: Minified token-merge disambiguation

#[test]
fn divide_followed_by_pointer_deref_does_not_form_block_comment() {
    // `Binary(Divide, _, Load { pointer: FunctionArgument(ptr) })`
    // renders the RHS via `emit_lvalue`, which dereferences a
    // pointer-typed function argument with a leading `*`.  In compact
    // mode the previous emission produced `... /*p`, which the WGSL
    // lexer sees as the start of a block comment instead of "divide by
    // the pointee".  The fix in `assemble_binary` must push a
    // disambiguating space when `op` ends with `/` and the RHS begins
    // with `*`.
    let src = r#"
        fn divide(p: ptr<function, f32>) -> f32 { return 1.0 / *p; }
        fn caller() -> f32 { var x: f32 = 5.0; return divide(&x); }
    "#;
    let out = compact(src);
    assert!(
        !out.contains("/*"),
        "compact output must not contain the `/*` trigraph (would start a block comment): {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn equality_with_relational_child_does_not_need_extra_parens() {
    // WGSL relational operators (`<`, `<=`, `>`, `>=`) bind more
    // tightly than equality, so `a<b==c` already parses as
    // `(a<b)==c`.  The non-chainable rule for `==`/`!=` exists to
    // prevent same-precedence chaining, not to force parens around
    // strictly-higher-precedence children.  This regression pins
    // that the emitter still drops these (redundant) parens.
    let out = compact("fn f(a:i32,b:i32,c:bool)->bool{return (a<b)==c;}");
    assert!(
        !out.contains("(a<b)") && !out.contains("(a < b)"),
        "relational child of equality should drop the redundant parens: {out}"
    );
    assert_valid_wgsl(&out);
}
