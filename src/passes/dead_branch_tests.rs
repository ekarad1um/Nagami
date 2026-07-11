use super::*;
use crate::config::Config;

fn run_pass(source: &str) -> (bool, naga::Module) {
    let mut module = naga::front::wgsl::parse_str(source).expect("source should parse");
    let mut pass = DeadBranchPass;
    let config = Config::default();
    let ctx = PassContext {
        config: &config,
        trace_run_dir: None,
    };
    let changed = pass.run(&mut module, &ctx).expect("pass should run");

    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("module should remain valid after pass");

    (changed, module)
}

/// Count If statements recursively in a block.
fn count_ifs(block: &naga::Block) -> usize {
    let mut n = 0;
    for stmt in block.iter() {
        if matches!(stmt, naga::Statement::If { .. }) {
            n += 1;
        }
        for nested in nested_blocks(stmt) {
            n += count_ifs(nested);
        }
    }
    n
}

/// Count Switch statements recursively in a block.
fn count_switches(block: &naga::Block) -> usize {
    let mut n = 0;
    for stmt in block.iter() {
        if matches!(stmt, naga::Statement::Switch { .. }) {
            n += 1;
        }
        for nested in nested_blocks(stmt) {
            n += count_switches(nested);
        }
    }
    n
}

// If: condition folded to true

/// Collapsing a constant-selector `switch` whose DROPPED case produced a
/// statement result (a non-inlined call) would orphan that result
/// expression - invalid IR.  The guard keeps the switch intact instead.
/// `run_pass` validates the post-pass module and panics on invalid IR, so
/// this test fails if the guard is removed.
#[test]
fn constant_switch_with_result_producing_dropped_case_stays_valid() {
    let src = r#"
fn helper() -> i32 { return 7; }
@compute @workgroup_size(1)
fn main() {
    var x: i32 = 0;
    switch 1 {
        case 0: { x = helper(); }
        default: { x = 5; }
    }
    _ = x;
}
"#;
    // Must not panic: the pass leaves valid IR (switch kept, result
    // expression keeps its producer).
    let (_changed, module) = run_pass(src);
    // The switch survives (was NOT collapsed) because case 0 carries a
    // call result the collapse cannot safely drop.
    let main = module
        .entry_points
        .iter()
        .find(|e| e.name == "main")
        .expect("main entry point");
    fn has_switch(block: &naga::Block) -> bool {
        block.iter().any(|s| {
            matches!(s, naga::Statement::Switch { .. }) || nested_blocks(s).any(has_switch)
        })
    }
    assert!(
        has_switch(&main.function.body),
        "switch with a result-producing dropped case must be kept intact"
    );
}

#[test]
fn eliminates_if_true_accept_branch() {
    let src = r#"
@fragment
fn fs() -> @location(0) vec4f {
    var x = 0.0;
    if true { x = 1.0; } else { x = 2.0; }
    return vec4f(x);
}
"#;
    let (changed, module) = run_pass(src);
    assert!(changed, "pass should report a change");
    let body = &module.entry_points[0].function.body;
    assert_eq!(count_ifs(body), 0, "if should be eliminated");
}

// If: condition folded to false

#[test]
fn eliminates_if_false_reject_branch() {
    let src = r#"
@fragment
fn fs() -> @location(0) vec4f {
    var x = 0.0;
    if false { x = 1.0; } else { x = 2.0; }
    return vec4f(x);
}
"#;
    let (changed, module) = run_pass(src);
    assert!(changed, "pass should report a change");
    let body = &module.entry_points[0].function.body;
    assert_eq!(count_ifs(body), 0, "if should be eliminated");
}

// If: true with no else (empty reject)

#[test]
fn eliminates_if_true_no_else() {
    let src = r#"
@fragment
fn fs() -> @location(0) vec4f {
    var x = 0.0;
    if true { x = 1.0; }
    return vec4f(x);
}
"#;
    let (changed, module) = run_pass(src);
    assert!(changed);
    assert_eq!(count_ifs(&module.entry_points[0].function.body), 0);
}

// If: false with no else -> entire if dropped

#[test]
fn eliminates_if_false_no_else() {
    let src = r#"
@fragment
fn fs() -> @location(0) vec4f {
    var x = 0.0;
    if false { x = 1.0; }
    return vec4f(x);
}
"#;
    let (changed, module) = run_pass(src);
    assert!(changed);
    assert_eq!(count_ifs(&module.entry_points[0].function.body), 0);
}

// Nested dead branches

#[test]
fn eliminates_nested_dead_branches() {
    let src = r#"
@fragment
fn fs() -> @location(0) vec4f {
    var x = 0.0;
    if true {
        if false { x = 1.0; } else { x = 2.0; }
    } else {
        x = 3.0;
    }
    return vec4f(x);
}
"#;
    let (changed, module) = run_pass(src);
    assert!(changed);
    assert_eq!(
        count_ifs(&module.entry_points[0].function.body),
        0,
        "both nested ifs should be eliminated"
    );
}

// Non-constant condition is preserved

#[test]
fn preserves_non_constant_if() {
    let src = r#"
@fragment
fn fs(@location(0) v: f32) -> @location(0) vec4f {
    var x = 0.0;
    if v > 0.5 { x = 1.0; } else { x = 2.0; }
    return vec4f(x);
}
"#;
    let (changed, _) = run_pass(src);
    assert!(!changed, "non-constant condition should not be eliminated");
}

// Switch with constant selector

#[test]
fn eliminates_switch_with_constant_selector() {
    let src = r#"
@fragment
fn fs() -> @location(0) vec4f {
    var x = 0.0;
    switch 1i {
        case 1i { x = 1.0; }
        case 2i { x = 2.0; }
        default { x = 3.0; }
    }
    return vec4f(x);
}
"#;
    let (changed, module) = run_pass(src);
    assert!(changed);
    assert_eq!(count_switches(&module.entry_points[0].function.body), 0);
}

// Switch: no match falls to default

#[test]
fn switch_constant_falls_to_default() {
    let src = r#"
@fragment
fn fs() -> @location(0) vec4f {
    var x = 0.0;
    switch 99i {
        case 1i { x = 1.0; }
        default { x = 3.0; }
    }
    return vec4f(x);
}
"#;
    let (changed, module) = run_pass(src);
    assert!(changed);
    assert_eq!(count_switches(&module.entry_points[0].function.body), 0);
}

// Switch: degenerate `case X, default` splicing and its safety guards.

#[test]
fn degenerate_case_list_to_default_is_spliced() {
    // `case 0, default {...}` lowers to an empty fall-through `case 0` plus
    // the `default` carrying the body, so every selector value runs the body
    // once - the switch wrapper is dead.
    let src = r#"
@fragment
fn fs(@location(0) v: f32) -> @location(0) vec4f {
    var x = 0.0;
    switch i32(v) {
        case 0, default { x = 5.0; }
    }
    return vec4f(x);
}
"#;
    let (changed, module) = run_pass(src);
    assert!(changed);
    assert_eq!(count_switches(&module.entry_points[0].function.body), 0);
}

#[test]
fn non_fallthrough_empty_case_is_not_degenerate() {
    // `case 1 {}` does NOT fall through, so selector==1 runs the empty body
    // and EXITS the switch (x stays 0); only other values hit the default.
    // The switch distinguishes selector 1 from the rest and must be kept -
    // splicing the default body would wrongly run it for selector 1.
    let src = r#"
@fragment
fn fs(@location(0) v: f32) -> @location(0) vec4f {
    var x = 0.0;
    switch i32(v) {
        case 1 { }
        default { x = 9.0; }
    }
    return vec4f(x);
}
"#;
    let (_, module) = run_pass(src);
    assert_eq!(
        count_switches(&module.entry_points[0].function.body),
        1,
        "non-fall-through empty case must keep the switch (distinguishes its selector)"
    );
}

#[test]
fn degenerate_default_with_bare_break_keeps_switch() {
    // The default body holds a bare `break` targeting the switch; splicing it
    // into the parent block would mis-target the break, so the switch is kept.
    let src = r#"
@fragment
fn fs(@location(0) v: f32) -> @location(0) vec4f {
    var x = 0.0;
    switch i32(v) {
        case 0, default { x = 1.0; if v > 0.5 { break; } x = 2.0; }
    }
    return vec4f(x);
}
"#;
    let (_, module) = run_pass(src);
    assert_eq!(
        count_switches(&module.entry_points[0].function.body),
        1,
        "default body with a bare break must keep the switch"
    );
}

// Pass reports no change when nothing to do

#[test]
fn reports_no_change_when_nothing_to_eliminate() {
    let src = r#"
@fragment
fn fs(@location(0) v: f32) -> @location(0) vec4f {
    var x = 0.0;
    if v > 0.5 { x = 1.0; }
    return vec4f(x);
}
"#;
    let (changed, _) = run_pass(src);
    assert!(!changed);
}

// The constant-condition branch arm must also resolve
// `Expression::Constant(c)` whose init is a bool literal, not just
// raw `Literal::Bool` - otherwise a branch on a named `const X:
// bool = false;` slips past until const_fold has inlined it.
#[test]
fn eliminates_branch_with_const_bool_condition() {
    let src = r#"
const ENABLE_FEATURE: bool = false;

@fragment
fn fs() -> @location(0) vec4f {
    if ENABLE_FEATURE {
        return vec4f(1.0, 0.0, 0.0, 1.0);
    }
    return vec4f(0.0);
}
"#;
    let (changed, module) = run_pass(src);
    assert!(
        changed,
        "branch on const false-bool should be statically eliminated"
    );
    // The entry point's body should have no If statements; the
    // dead `if ENABLE_FEATURE { ... }` branch should be removed.
    let ep_body = &module.entry_points[0].function.body;
    assert_eq!(
        count_ifs(ep_body),
        0,
        "if/else on a const-bool selector must be eliminated"
    );
}

#[test]
fn eliminates_switch_with_const_int_selector() {
    let src = r#"
const SELECTOR: i32 = 1;

@fragment
fn fs() -> @location(0) vec4f {
    var v = 0.0;
    switch SELECTOR {
        case 0: { v = 1.0; }
        case 1: { v = 2.0; }
        default: { v = 3.0; }
    }
    return vec4f(v);
}
"#;
    let (changed, module) = run_pass(src);
    assert!(
        changed,
        "switch on const integer should be statically resolved"
    );
    // Walk the entry point looking for any Switch statement; there
    // shouldn't be one (the matching case has been spliced in).
    fn has_switch(block: &naga::Block) -> bool {
        block.iter().any(|s| {
            matches!(s, naga::Statement::Switch { .. }) || nested_blocks(s).any(has_switch)
        })
    }
    assert!(
        !has_switch(&module.entry_points[0].function.body),
        "switch on const-int selector must be resolved away"
    );
}

// Function (not just entry point)

#[test]
fn eliminates_dead_branch_in_regular_function() {
    let src = r#"
fn helper() -> f32 {
    if true { return 1.0; } else { return 2.0; }
}

@fragment
fn fs() -> @location(0) vec4f {
    return vec4f(helper());
}
"#;
    let (changed, module) = run_pass(src);
    assert!(changed);
    // The helper function's body should have no If statements.
    for (_, func) in module.functions.iter() {
        assert_eq!(count_ifs(&func.body), 0);
    }
}

// contains_bare_break / contains_bare_continue helpers

#[test]
fn bare_break_detected_at_top_level() {
    let mut block = naga::Block::new();
    block.push(naga::Statement::Break, Default::default());
    assert!(contains_bare_break(&block));
}

#[test]
fn bare_break_detected_inside_if_block() {
    // Parse a shader that has a break inside an if inside a loop,
    // then inspect the loop body.
    let src = r#"
@fragment
fn fs() -> @location(0) vec4f {
    var x = 0.0;
    loop {
        if x > 1.0 { break; }
        x = x + 1.0;
        continuing { break if x > 10.0; }
    }
    return vec4f(x);
}
"#;
    let module = naga::front::wgsl::parse_str(src).unwrap();
    let loop_body = module.entry_points[0]
        .function
        .body
        .iter()
        .find_map(|s| {
            if let naga::Statement::Loop { body, .. } = s {
                Some(body)
            } else {
                None
            }
        })
        .unwrap();
    assert!(contains_bare_break(loop_body));
}

#[test]
fn bare_break_not_detected_inside_nested_switch() {
    // A break inside a nested switch targets that switch, not the parent.
    let src = r#"
@fragment
fn fs(@location(0) v: i32) -> @location(0) vec4f {
    var x = 0.0;
    loop {
        switch v {
            case 1i {
                x = 1.0;
                break;
            }
            default {
                x = 2.0;
            }
        }
        continuing { break if x > 0.0; }
    }
    return vec4f(x);
}
"#;
    let module = naga::front::wgsl::parse_str(src).unwrap();
    // The loop body contains a switch with a break - but that break
    // targets the switch, NOT the loop.
    let loop_body = module.entry_points[0].function.body.iter().find_map(|s| {
        if let naga::Statement::Loop { body, .. } = s {
            Some(body)
        } else {
            None
        }
    });
    assert!(loop_body.is_some());
    // contains_bare_break should NOT see through the nested switch
    assert!(!contains_bare_break(loop_body.unwrap()));
}

#[test]
fn bare_continue_detected_inside_switch() {
    // In naga IR, Switch does NOT capture Continue - it still targets
    // the enclosing loop.  contains_bare_continue must find it.
    let src = r#"
@fragment
fn fs(@location(0) v: i32) -> @location(0) vec4f {
    var x = 0.0;
    loop {
        switch v {
            case 1i { continue; }
            default { x = 2.0; }
        }
        x = x + 1.0;
        continuing { break if x > 5.0; }
    }
    return vec4f(x);
}
"#;
    let module = naga::front::wgsl::parse_str(src).unwrap();
    let loop_body = module.entry_points[0].function.body.iter().find_map(|s| {
        if let naga::Statement::Loop { body, .. } = s {
            Some(body)
        } else {
            None
        }
    });
    assert!(loop_body.is_some());
    assert!(
        contains_bare_continue(loop_body.unwrap()),
        "continue inside switch should be detected by contains_bare_continue"
    );
    // contains_bare_loop_control should also detect it
    assert!(
        contains_bare_loop_control(loop_body.unwrap()),
        "continue inside switch should be detected by contains_bare_loop_control"
    );
}

#[test]
fn bare_continue_not_detected_inside_nested_loop() {
    // Continue inside a nested loop targets that inner loop.
    let src = r#"
@fragment
fn fs() -> @location(0) vec4f {
    var x = 0.0;
    loop {
        loop {
            if x > 1.0 { continue; }
            x = x + 1.0;
            continuing { break if x > 3.0; }
        }
        x = x + 1.0;
        continuing { break if x > 10.0; }
    }
    return vec4f(x);
}
"#;
    let module = naga::front::wgsl::parse_str(src).unwrap();
    let outer_loop_body = module.entry_points[0].function.body.iter().find_map(|s| {
        if let naga::Statement::Loop { body, .. } = s {
            Some(body)
        } else {
            None
        }
    });
    assert!(outer_loop_body.is_some());
    // The outer loop body contains a nested loop with continue.
    // contains_bare_continue should NOT see through the nested loop.
    assert!(
        !contains_bare_continue(outer_loop_body.unwrap()),
        "continue inside nested loop should NOT be detected"
    );
}

/// Count non-empty reject blocks in If statements (recursive).
fn count_non_empty_rejects(block: &naga::Block) -> usize {
    let mut n = 0;
    for stmt in block.iter() {
        if let naga::Statement::If { reject, .. } = stmt
            && !reject.is_empty()
        {
            n += 1;
        }
        for nested in nested_blocks(stmt) {
            n += count_non_empty_rejects(nested);
        }
    }
    n
}

/// Count If statements with non-empty accept blocks recursively.
fn count_non_empty_accepts(block: &naga::Block) -> usize {
    let mut n = 0;
    for stmt in block.iter() {
        if let naga::Statement::If { accept, .. } = stmt
            && !accept.is_empty()
        {
            n += 1;
        }
        for nested in nested_blocks(stmt) {
            n += count_non_empty_accepts(nested);
        }
    }
    n
}

// Pattern A: condition is Load(d), so d is false in the reject branch.
// Storing `false` to d in the else is a no-op.

#[test]
fn redundant_else_pattern_a_load_condition() {
    let src = r#"
fn f(a: bool, b: bool) -> bool {
    var d: bool;
    if a { d = b; } else { d = false; }
    if d { d = true; } else { d = false; }
    return d;
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(f32(f(true, true))); }
"#;
    let (changed, module) = run_pass(src);
    assert!(changed, "pass should report a change");
    for (_, func) in module.functions.iter() {
        assert_eq!(
            count_non_empty_rejects(&func.body),
            0,
            "all else {{ d = false; }} branches should be cleared"
        );
    }
}

// Pattern B: var d: bool (zero-init), first if's else stores false
// before any modification.

#[test]
fn redundant_else_pattern_b_zero_init() {
    let src = r#"
fn f(a: bool) -> bool {
    var d: bool;
    if a { d = true; } else { d = false; }
    return d;
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(f32(f(true))); }
"#;
    let (changed, module) = run_pass(src);
    assert!(changed);
    for (_, func) in module.functions.iter() {
        assert_eq!(count_non_empty_rejects(&func.body), 0);
    }
}

// Negative: else stores a non-zero value - must be preserved.

#[test]
fn preserves_else_storing_non_zero() {
    let src = r#"
fn f(a: bool) -> bool {
    var d: bool;
    if a { d = false; } else { d = true; }
    return d;
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(f32(f(true))); }
"#;
    let (changed, module) = run_pass(src);
    // The accept stores `false` to d - redundant (d is zero-init) ->
    // accept cleared.  The else stores `true` -> NOT redundant -> kept.
    assert!(
        changed,
        "accept zero-store to zero-init var should be cleared"
    );
    for (_, func) in module.functions.iter() {
        // The if should still exist (reject is non-empty).
        assert!(
            count_ifs(&func.body) >= 1,
            "if should be preserved because reject stores non-zero"
        );
    }
}

// Negative: variable is not known-zero (has non-zero init), so
// else { d = 0 } is meaningful.

#[test]
fn preserves_else_when_var_not_known_zero() {
    let src = r#"
fn f(a: bool) -> f32 {
    var x: f32 = 1.0;
    if a { x = 2.0; } else { x = 0.0; }
    return x;
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(f(true)); }
"#;
    let (changed, _module) = run_pass(src);
    assert!(
        !changed,
        "else {{ x = 0.0 }} is not redundant when x was 1.0"
    );
}

// Short-circuit desugaring subsumes the "prior store" edge-case:
// both ifs match the && pattern and are replaced with Binary stores.

#[test]
fn short_circuit_desugars_despite_prior_store() {
    let src = r#"
fn f(a: bool, b: bool) -> bool {
    var d: bool;
    d = true;
    if a { d = b; } else { d = false; }
    if d { d = true; } else { d = false; }
    return d;
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(f32(f(true, true))); }
"#;
    let (changed, module) = run_pass(src);
    // Phase 0 desugars both ifs into d = a&&b and d = d&&true.
    assert!(changed);
    for (_, func) in module.functions.iter() {
        assert_eq!(
            count_ifs(&func.body),
            0,
            "both ifs should be desugared by short-circuit pass"
        );
    }
}

// Chained `&&` pattern: multiple if-else in sequence.

#[test]
fn redundant_else_chained_and() {
    let src = r#"
fn f(a: bool, b: bool, c: bool) -> bool {
    var d: bool;
    if a { d = b; } else { d = false; }
    if d { d = c; } else { d = false; }
    return d;
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(f32(f(true, true, true))); }
"#;
    let (changed, module) = run_pass(src);
    assert!(changed);
    for (_, func) in module.functions.iter() {
        assert_eq!(
            count_non_empty_rejects(&func.body),
            0,
            "all 2 else branches should be cleared"
        );
    }
}

// Loop: short-circuit desugaring still applies inside loops
// (if cond { d = val; } else { d = false; } -> d = cond && val
// is valid regardless of d's prior value).

#[test]
fn short_circuit_inside_loop() {
    let src = r#"
fn f(a: bool, b: bool) -> bool {
    var d: bool;
    loop {
        if a { d = true; } else { d = false; }
        continuing { break if b; }
    }
    return d;
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(f32(f(true, true))); }
"#;
    let (changed, module) = run_pass(src);
    // Phase 0 desugars the if inside the loop to d = a && true.
    assert!(changed);
    for (_, func) in module.functions.iter() {
        assert_eq!(
            count_non_empty_rejects(&func.body),
            0,
            "short-circuit desugaring should remove the if"
        );
    }
}

// Short-circuit ||: if(!cond) { d = val; } else { d = true; }

#[test]
fn short_circuit_basic_or_replacement() {
    let src = r#"
fn f(a: bool, b: bool) -> bool {
    var d: bool;
    if !a { d = b; } else { d = true; }
    return d;
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(f32(f(true, true))); }
"#;
    let (changed, module) = run_pass(src);
    assert!(changed);
    for (_, func) in module.functions.iter() {
        assert_eq!(
            count_ifs(&func.body),
            0,
            "if should be desugared to d = a || b"
        );
    }
}

// Negative: non-boolean types must NOT be matched.

#[test]
fn short_circuit_preserves_non_bool_if_else() {
    // f32 stores: reject stores 0.0, but LogicalAnd only works on bool.
    let src = r#"
fn f(a: bool) -> f32 {
    var d: f32;
    if a { d = 1.0; } else { d = 0.0; }
    return d;
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(f(true)); }
"#;
    let (_, module) = run_pass(src);
    for (_, func) in module.functions.iter() {
        // The if must be preserved (not a bool short-circuit pattern).
        assert!(
            count_ifs(&func.body) >= 1,
            "non-bool if-else must not be desugared"
        );
    }
}

// Negative: reject stores non-false value -> not a short-circuit.

#[test]
fn short_circuit_preserves_reject_storing_non_false() {
    let src = r#"
fn f(a: bool, b: bool, c: bool) -> bool {
    var d: bool;
    if a { d = b; } else { d = c; }
    return d;
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(f32(f(true, true, false))); }
"#;
    let (_, module) = run_pass(src);
    for (_, func) in module.functions.iter() {
        assert!(
            count_ifs(&func.body) >= 1,
            "reject stores c (not false), must not desugar"
        );
    }
}

// No change when there is nothing to eliminate.

#[test]
fn redundant_else_no_change_when_nothing_to_do() {
    let src = r#"
@fragment
fn fs(@location(0) v: f32) -> @location(0) vec4f {
    var x = 0.0;
    if v > 0.5 { x = 1.0; } else { x = 2.0; }
    return vec4f(x);
}
"#;
    let (changed, _) = run_pass(src);
    assert!(!changed, "no redundant else stores present");
}

// Entry point (not just regular function).

#[test]
fn redundant_else_in_entry_point() {
    let src = r#"
@fragment
fn fs(@location(0) v: f32) -> @location(0) vec4f {
    var d: bool;
    if v > 0.5 { d = true; } else { d = false; }
    return vec4f(f32(d));
}
"#;
    let (changed, module) = run_pass(src);
    assert!(changed);
    assert_eq!(
        count_non_empty_rejects(&module.entry_points[0].function.body),
        0,
        "else in entry point should be cleared"
    );
}

// OR pattern (accept side): condition Load(d) -> d is true in accept.
// `if d { d = true; } else { d = b; }` - the accept `d = true` is
// redundant because d is already true.

#[test]
fn redundant_accept_true_store_when_condition_is_load() {
    // Manually write the pattern: `if d { d = true; } else { d = b; }`
    // The accept branch stores `true` to a variable that the condition
    // already proves is `true`.
    let src = r#"
fn f(a: bool, b: bool) -> bool {
    var d: bool;
    if a { d = true; } else { d = b; }
    if d { d = true; } else { d = b; }
    return d;
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(f32(f(true, true))); }
"#;
    let (changed, module) = run_pass(src);
    assert!(changed, "pass should clear redundant accept `d = true`");
    for (_, func) in module.functions.iter() {
        // The first if: no pattern fires (a is a function arg, not a local).
        // The second if: condition = Load(d), accept = {d = true} -> redundant.
        // After clearing, the second if should have an empty accept.
        let non_empty_accepts = count_non_empty_accepts(&func.body);
        assert!(
            non_empty_accepts <= 1,
            "second if's accept `d = true` should be cleared, got {non_empty_accepts} non-empty accepts"
        );
    }
}

// Regression: condition narrowing must NOT fire on a STALE forwarded
// load.  `let t = d; d = false; if t { d = true; }` - the condition `t`
// captured d's value BEFORE the `d = false` store, so the loaded value
// (true) does not reflect d's current value (false).  Narrowing would
// clobber the correct post-store known value and wrongly classify the
// live `d = true` store as redundant, dropping it and returning false
// instead of true.  The fresh-load tracking must keep the branch.
#[test]
fn narrowing_skips_stale_forwarded_load() {
    let src = r#"
fn f() -> bool {
    var d: bool = true;
    let t = d;
    d = false;
    if t { d = true; }
    return d;
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(f32(f())); }
"#;
    let (_, module) = run_pass(src);
    // The `if t { d = true; }` accept must survive: the store is live.
    let total_non_empty_accepts: usize = module
        .functions
        .iter()
        .map(|(_, func)| count_non_empty_accepts(&func.body))
        .sum();
    assert!(
        total_non_empty_accepts >= 1,
        "live `d = true` store behind a stale-load condition must be preserved"
    );
}

// Positive companion: when the SAME local is re-loaded fresh for the
// condition (no intervening store), narrowing still fires.  Distinguishes
// the fix from a blanket disable of load-condition narrowing.
#[test]
fn narrowing_fires_on_fresh_load_after_store() {
    let src = r#"
fn f(a: bool) -> bool {
    var d: bool = true;
    if a { d = false; } else { d = true; }
    if d { d = true; }
    return d;
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(f32(f(true))); }
"#;
    let (changed, _) = run_pass(src);
    assert!(
        changed,
        "fresh `if d {{ d = true; }}` after a branch write should still be eliminated"
    );
}

#[test]
fn short_circuit_chained_and_with_non_literal_value() {
    // Both ifs match the && pattern (reject stores false to same local).
    // Phase 0 desugars them into Binary(LogicalAnd) stores.
    let src = r#"
fn f(a: bool, b: bool) -> bool {
    var d: bool;
    if a { d = b; } else { d = false; }
    if d { d = b; } else { d = false; }
    return d;
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(f32(f(true, true))); }
"#;
    let (changed, module) = run_pass(src);
    assert!(changed);
    for (_, func) in module.functions.iter() {
        assert_eq!(
            count_ifs(&func.body),
            0,
            "both ifs should be desugared by short-circuit pass"
        );
    }
}

// Empty construct elimination: if both branches are empty after
// recursion, the whole If is dropped.

#[test]
fn eliminates_empty_if_after_both_branches_cleared() {
    // Both branches store zero to a zero-initialized local -> both get
    // cleared by the redundant-else-store pass.  The resulting empty If
    // should then be discarded.
    let src = r#"
fn f(a: bool) -> bool {
    var d: bool;
    if a { d = false; } else { d = false; }
    return d;
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(f32(f(true))); }
"#;
    let (changed, module) = run_pass(src);
    assert!(changed, "pass should report a change");
    // After clearing both branches, the if should be gone.
    for (_, func) in module.functions.iter() {
        assert_eq!(count_ifs(&func.body), 0, "empty if should be eliminated");
    }
}

#[test]
fn short_circuit_basic_and_replacement() {
    // The && lowered pattern: if a { d = b; } else { d = false; }
    // should be entirely desugared into d = a && b (no If left).
    let src = r#"
fn f(a: bool, b: bool) -> bool {
    var d: bool;
    if a { d = b; } else { d = false; }
    return d;
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(f32(f(true, true))); }
"#;
    let (changed, module) = run_pass(src);
    assert!(changed);
    for (_, func) in module.functions.iter() {
        assert_eq!(
            count_ifs(&func.body),
            0,
            "if should be desugared by short-circuit pass"
        );
    }
}

#[test]
fn short_circuit_folds_branch_with_emit_in_value() {
    // The accept branch stores a computed expression (`arr[idx]` needs
    // an `Emit` to load the element).  The re-sugar HOISTS that leading
    // emit and folds `if a { d = arr[idx]; } else { d = false; }` into
    // `d = a && arr[idx]`.  Sound because the load is side-effect-free,
    // WGSL bounds-checks the (possibly out-of-range) index rather than
    // trapping, the value is discarded by the `&&` when `a` is false,
    // and lifting it out of the branch only reduces non-uniformity.
    let src = r#"
fn f(a: bool, idx: u32) -> bool {
    let arr = array<bool, 4>(true, false, true, false);
    var d: bool;
    if a { d = arr[idx]; } else { d = false; }
    return d;
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(f32(f(true, 0u))); }
"#;
    let (changed, module) = run_pass(src);
    assert!(changed, "the emit-in-value branch should be re-sugared");
    let f_function = module
        .functions
        .iter()
        .find(|(_, func)| func.name.as_deref() == Some("f"))
        .map(|(_, func)| func)
        .expect("function `f` should survive the pass");
    // The lowered short-circuit `If` is folded away ...
    assert_eq!(
        count_ifs(&f_function.body),
        0,
        "the lowered short-circuit If should fold into `&&`"
    );
    // ... into a synthesized `a && arr[idx]` LogicalAnd.
    assert!(
        f_function.expressions.iter().any(|(_, e)| matches!(
            e,
            naga::Expression::Binary {
                op: naga::BinaryOperator::LogicalAnd,
                ..
            }
        )),
        "a LogicalAnd should be synthesized for `a && arr[idx]`"
    );
}

// Empty construct elimination for Switch: all cases empty.

#[test]
fn eliminates_empty_switch_all_cases_cleared() {
    // Switch with a constant selector resolves to a case with an empty
    // body -> entire switch is removed.  For a non-constant selector,
    // if all cases are empty after recursion, the switch is a no-op.
    let src = r#"
@fragment
fn fs() -> @location(0) vec4f {
    var x = 0.0;
    switch 1i { case 1i: { x = 1.0; } default: {} }
    return vec4f(x);
}
"#;
    let (changed, module) = run_pass(src);
    assert!(changed, "constant-selector switch should be eliminated");
    assert_eq!(
        count_switches(&module.entry_points[0].function.body),
        0,
        "switch should be eliminated"
    );
}

// Degenerate switch: one Default case with runtime selector -> unwrap.

#[test]
fn unwraps_degenerate_switch_default_only() {
    // Switch with only a Default case and a runtime selector.
    // The body always executes, so it should be spliced directly.
    let src = r#"
@fragment
fn fs(@location(0) v: f32) -> @location(0) vec4f {
    var x = 0.0;
    switch i32(v) { default: { x = 1.0; } }
    return vec4f(x);
}
"#;
    let (changed, module) = run_pass(src);
    assert!(changed, "degenerate switch should be unwrapped");
    assert_eq!(
        count_switches(&module.entry_points[0].function.body),
        0,
        "switch wrapper should be removed"
    );
}

#[test]
fn preserves_degenerate_switch_with_bare_break() {
    // Default-only switch whose body has a bare Break - unsafe to
    // splice because the Break targets the switch.
    let src = r#"
@fragment
fn fs(@location(0) v: f32) -> @location(0) vec4f {
    var x = 0.0;
    switch i32(v) { default: { x = 1.0; break; } }
    return vec4f(x);
}
"#;
    let (_changed, module) = run_pass(src);
    // Should NOT unwrap because of the bare Break.
    assert_eq!(
        count_switches(&module.entry_points[0].function.body),
        1,
        "switch with bare break must be preserved"
    );
}

#[test]
fn preserves_switch_with_multiple_cases() {
    // Switch with multiple cases - not degenerate.
    let src = r#"
@fragment
fn fs(@location(0) v: f32) -> @location(0) vec4f {
    var x = 0.0;
    switch i32(v) { case 0i: { x = 1.0; } default: { x = 2.0; } }
    return vec4f(x);
}
"#;
    let (_changed, module) = run_pass(src);
    assert_eq!(
        count_switches(&module.entry_points[0].function.body),
        1,
        "multi-case switch must be preserved"
    );
}

// MARK: Else block elision (CFG flattening) tests

#[test]
fn else_elision_when_accept_returns() {
    // if (cond) { return ...; } else { x = 2.0; }
    // After elision: if (cond) { return ...; }  x = 2.0;
    let src = r#"
@fragment
fn fs(@location(0) v: f32) -> @location(0) vec4f {
    var x = 0.0;
    if v > 0.5 { return vec4f(1.0); } else { x = 2.0; }
    return vec4f(x);
}
"#;
    let (changed, module) = run_pass(src);
    assert!(changed, "else elision should fire");
    let body = &module.entry_points[0].function.body;
    assert_eq!(
        count_non_empty_rejects(body),
        0,
        "reject block should be empty after elision"
    );
}

#[test]
fn else_elision_when_accept_breaks() {
    // Inside a loop: if (cond) { break; } else { x = 2.0; }
    let src = r#"
@fragment
fn fs(@location(0) v: f32) -> @location(0) vec4f {
    var x = 0.0;
    loop {
        if v > 0.5 { break; } else { x = 2.0; }
        continuing { break if x > 10.0; }
    }
    return vec4f(x);
}
"#;
    let (changed, module) = run_pass(src);
    assert!(changed, "else elision should fire when accept has break");
    let body = &module.entry_points[0].function.body;
    assert_eq!(
        count_non_empty_rejects(body),
        0,
        "reject block should be empty after elision"
    );
}

#[test]
fn no_else_elision_when_accept_does_not_terminate() {
    // Accept block does NOT definitely terminate.
    let src = r#"
@fragment
fn fs(@location(0) v: f32) -> @location(0) vec4f {
    var x = 0.0;
    if v > 0.5 { x = 1.0; } else { x = 2.0; }
    return vec4f(x);
}
"#;
    let (changed, _) = run_pass(src);
    assert!(!changed, "no elision when accept doesn't terminate");
}

#[test]
fn no_else_elision_when_reject_is_empty() {
    // Accept terminates but reject is already empty - nothing to hoist.
    let src = r#"
@fragment
fn fs(@location(0) v: f32) -> @location(0) vec4f {
    var x = 0.0;
    if v > 0.5 { return vec4f(1.0); }
    return vec4f(x);
}
"#;
    let (changed, _) = run_pass(src);
    assert!(!changed, "no change when reject is already empty");
}

/// Else-elision fires even when both arms terminate with
/// value-bearing returns - the saved `else{...}` scaffolding
/// outweighs the residual `if c { return v1; }`.  Gating on
/// `!reject.terminates` made the corpus strictly larger on real
/// shaders; the only collapse the elision misses is symmetric
/// `return;` arms, which we don't fold anyway.
#[test]
fn else_elision_fires_when_both_arms_return_values() {
    let src = r#"
@fragment
fn fs(@location(0) v: f32) -> @location(0) vec4f {
    if v > 0.5 { return vec4f(1.0); } else { return vec4f(0.0); }
}
"#;
    let (changed, module) = run_pass(src);
    assert!(
        changed,
        "else-elision should fire even when both arms terminate \
             (value-bearing returns still save the `else{{}}` overhead)"
    );
    let body = &module.entry_points[0].function.body;
    assert_eq!(
        count_non_empty_rejects(body),
        0,
        "reject block must be hoisted out even when it terminates - \
             the resulting `if c {{ return v1; }} return v2;` is shorter \
             than `if c {{ return v1; }} else {{ return v2; }}`"
    );
}

/// Nested `Block(empty)` (either authored `{ }` or one drained
/// by an upstream fold) must be dropped after recursion; otherwise
/// a vacuous `{}` leaks into the output.
#[test]
fn empty_nested_block_is_dropped() {
    // WGSL source has no syntax for a bare nested `{ }` inside
    // a function body, so build the IR directly.
    let mut module = naga::Module::default();
    let f32_ty = module.types.insert(
        naga::Type {
            name: None,
            inner: naga::TypeInner::Scalar(naga::Scalar::F32),
        },
        naga::Span::UNDEFINED,
    );
    let bool_ty = module.types.insert(
        naga::Type {
            name: None,
            inner: naga::TypeInner::Scalar(naga::Scalar::BOOL),
        },
        naga::Span::UNDEFINED,
    );
    let _ = f32_ty; // silence unused if naga ever changes the API

    let mut func = naga::Function::default();
    func.arguments.push(naga::FunctionArgument {
        name: Some("c".to_string()),
        ty: bool_ty,
        binding: None,
    });
    let cond = func
        .expressions
        .append(naga::Expression::FunctionArgument(0), naga::Span::UNDEFINED);
    // The accept block holds a single Block(empty) - exactly the
    // shape the elision targets.
    let mut accept = naga::Block::new();
    accept.push(
        naga::Statement::Block(naga::Block::new()),
        naga::Span::UNDEFINED,
    );
    func.body.push(
        naga::Statement::If {
            condition: cond,
            accept,
            reject: naga::Block::new(),
        },
        naga::Span::UNDEFINED,
    );
    module.functions.append(func, naga::Span::UNDEFINED);

    // Pre-condition: the hand-built input must satisfy naga's
    // validator.  Without this, a stricter future validator would
    // silently turn this into an invalid-IR exercise.
    crate::io::validate_module(&module).expect("hand-built input must satisfy naga's validator");

    let mut pass = DeadBranchPass;
    let config = Config::default();
    let ctx = PassContext {
        config: &config,
        trace_run_dir: None,
    };
    let changed = pass.run(&mut module, &ctx).expect("pass should run");
    assert!(
        changed,
        "empty nested Block elision must report `changed = true`"
    );

    // Post-condition every IR pass must uphold.
    crate::io::validate_module(&module).expect("module must remain valid after the elision");

    // Walk the resulting body and verify no `Block(empty)` remains.
    fn contains_empty_block(block: &naga::Block) -> bool {
        block.iter().any(|stmt| {
            matches!(stmt, naga::Statement::Block(inner) if inner.is_empty())
                || nested_blocks(stmt).any(contains_empty_block)
        })
    }

    let (_, f) = module.functions.iter().next().unwrap();
    assert!(
        !contains_empty_block(&f.body),
        "empty Block(_) statements must be elided after the pass"
    );
}

// MARK: Generalized redundant store elimination tests

#[test]
fn redundant_store_same_literal_i32() {
    // Storing the same i32 literal a variable already holds is a no-op.
    let src = r#"
fn f(a: bool) -> i32 {
    var x: i32 = 42;
    if a { x = 42; } else { x = 42; }
    return x;
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(f32(f(true))); }
"#;
    let (changed, module) = run_pass(src);
    assert!(changed, "redundant i32 stores should be eliminated");
    for (_, func) in module.functions.iter() {
        assert_eq!(
            count_ifs(&func.body),
            0,
            "entire if should be removed after both branches cleared"
        );
    }
}

#[test]
fn redundant_store_after_explicit_store() {
    // After storing a literal, a subsequent if-else storing the same
    // value should be eliminated.
    let src = r#"
fn f(a: bool) -> f32 {
    var x: f32;
    x = 3.0;
    if a { x = 3.0; } else { x = 3.0; }
    return x;
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(f(true)); }
"#;
    let (changed, module) = run_pass(src);
    assert!(
        changed,
        "redundant stores after explicit store should be eliminated"
    );
    for (_, func) in module.functions.iter() {
        assert_eq!(count_ifs(&func.body), 0);
    }
}

#[test]
fn preserves_store_of_different_literal() {
    // Storing a different literal should be preserved.
    let src = r#"
fn f(a: bool) -> i32 {
    var x: i32 = 42;
    if a { x = 99; } else { x = 42; }
    return x;
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(f32(f(true))); }
"#;
    let (changed, module) = run_pass(src);
    // reject stores 42 which matches init -> cleared.  But accept stores
    // 99 which differs -> preserved.  The if should still exist.
    assert!(changed);
    for (_, func) in module.functions.iter() {
        assert!(
            count_ifs(&func.body) >= 1,
            "if with non-redundant accept store must be preserved"
        );
    }
}

// Regression: dead_branch_elimination must not treat a switch where all
// cases end with `break` as a terminator of the outer block.  Pattern: phi
// variables are assigned inside the switch cases, then captured with
// `let ev = phi;` after the switch.  The continuing block stores `phi = ev`
// (phi-assignment), forcing naga to emit the `ev` expressions in the loop
// body scope (not lazily in continuing).  Treating the switch as a
// terminator drops the Emit covering those let-bindings, causing
// `Expression NotInScope` in the continuing block on naga validation.
#[test]
fn switch_with_all_break_cases_does_not_drop_continuing_emits() {
    let src = r#"
@compute @workgroup_size(1) fn main() {
    var phi_a: u32;
    var phi_b: bool;
    loop {
        switch phi_a {
            case 10u: { phi_b = false; break; }
            default: { phi_b = true; break; }
        }
        let ev_b = phi_b;
        let ev_a = phi_a + 1u;
        continue;
        continuing {
            phi_a = ev_a;
            break if !ev_b;
        }
    }
}
"#;
    // run_pass validates after the pass; panics with NotInScope if bug still present.
    let (_, module) = run_pass(src);
    let ep = &module.entry_points[0];
    let has_loop = ep
        .function
        .body
        .iter()
        .any(|s| matches!(s, naga::Statement::Loop { .. }));
    assert!(has_loop, "loop must be preserved");
}

// Regression: code after a switch-with-all-break-cases must not be dead-code-removed.
#[test]
fn switch_break_cases_not_terminator_of_outer_block() {
    let src = r#"
@compute @workgroup_size(1) fn main() {
    var x: u32 = 0u;
    var y: u32 = 0u;
    switch x {
        case 0u: { y = 1u; break; }
        default: { y = 2u; break; }
    }
    x = y + 1u;
}
"#;
    let (_, module) = run_pass(src);
    let ep = &module.entry_points[0];
    let store_count = ep
        .function
        .body
        .iter()
        .filter(|s| matches!(s, naga::Statement::Store { .. }))
        .count();
    assert!(store_count >= 1, "stores after switch must not be removed");
}

// MARK: Switch fall-through edge in definitely_terminates
//
// Round-3 review flagged that the `definitely_terminates` fix
// for the case `cases.last().fall_through == true` had no
// regression test because the unsafe shape can't be produced via
// a WGSL source - naga's frontend never emits a final case with
// `fall_through: true`.  Hand-build the IR directly to assert
// the fix's behaviour.

/// Build a single-Default-case `Switch` whose body terminates.
/// Helper for the fall-through tests below.
fn build_terminating_default_switch(fall_through: bool) -> naga::Statement {
    let mut body = naga::Block::new();
    body.push(
        naga::Statement::Return { value: None },
        naga::Span::UNDEFINED,
    );
    naga::Statement::Switch {
        // Selector handle is irrelevant - `definitely_terminates`
        // looks only at the case structure.  We construct a fake
        // expression arena just to obtain one Handle.
        selector: {
            let mut arena: naga::Arena<naga::Expression> = naga::Arena::new();
            arena.append(
                naga::Expression::Literal(naga::Literal::U32(0)),
                naga::Span::UNDEFINED,
            )
        },
        cases: vec![naga::SwitchCase {
            value: naga::SwitchValue::Default,
            body,
            fall_through,
        }],
    }
}

#[test]
fn switch_with_default_terminator_and_no_fallthrough_definitely_terminates() {
    let stmt = build_terminating_default_switch(/*fall_through=*/ false);
    assert!(
        definitely_terminates(&stmt),
        "Default case with terminating body and no fall-through must terminate"
    );
}

#[test]
fn switch_with_last_case_fallthrough_does_not_terminate() {
    // The fall_through-on-last-case shape is the regression
    // target: fall-through past the last case is Break-equivalent
    // (execution resumes after the switch), so the switch as a
    // whole does NOT terminate the outer block.  naga's WGSL
    // frontend won't emit this shape, but the inliner / CSE
    // could, and the previous version of `definitely_terminates`
    // would have mis-classified it as terminating.
    let stmt = build_terminating_default_switch(/*fall_through=*/ true);
    assert!(
        !definitely_terminates(&stmt),
        "last-case fall-through must not classify the switch as terminating \
             (fall-through past the last case is Break-equivalent and execution \
             resumes after the switch)"
    );
}

/// `break if NAMED_CONST` must fold the same sweep as `if
/// NAMED_CONST` / `switch NAMED_CONST` - all three flow through
/// `resolve_to_literal`.  Pre-fix the break_if arm only matched
/// raw `Expression::Literal`, so the fold lagged until
/// `const_fold` inlined the constant on a later sweep.
#[test]
fn break_if_with_named_const_true_unwraps_loop() {
    let src = r#"
            const STOP: bool = true;
            fn f() -> i32 {
                var i: i32 = 0;
                loop {
                    i = i + 1;
                    continuing {
                        break if STOP;
                    }
                }
                return i;
            }
            @compute @workgroup_size(1) fn main() { _ = f(); }
        "#;
    let (changed, module) = run_pass(src);
    assert!(
        changed,
        "break_if with a named-const-true selector must unwrap the loop"
    );
    // Confirm the loop is gone in the helper function.
    let helper = module
        .functions
        .iter()
        .find(|(_, f)| f.name.as_deref() == Some("f"))
        .expect("helper function `f` survives the pass");
    fn has_loop(block: &naga::Block) -> bool {
        block.iter().any(|stmt| {
            matches!(stmt, naga::Statement::Loop { .. }) || nested_blocks(stmt).any(has_loop)
        })
    }
    assert!(
        !has_loop(&helper.1.body),
        "loop with `break if true` selector must have been unwrapped"
    );
}

/// `break if true` must not unwrap a loop whose body has a bare
/// `break`/`continue` targeting that loop - unwrapping would
/// re-target the bare statement at the surrounding scope.  The
/// `contains_bare_loop_control` guard must also fire on
/// named-const-true selectors, not just literal `true`.
#[test]
fn break_if_with_named_const_true_preserves_loop_with_bare_break() {
    let src = r#"
            const STOP: bool = true;
            fn f(c: bool) -> i32 {
                var i: i32 = 0;
                loop {
                    i = i + 1;
                    if c {
                        break;
                    }
                    continuing {
                        break if STOP;
                    }
                }
                return i;
            }
            @compute @workgroup_size(1) fn main() { _ = f(true); }
        "#;
    let (_, module) = run_pass(src);
    fn has_loop(block: &naga::Block) -> bool {
        block.iter().any(|stmt| {
            matches!(stmt, naga::Statement::Loop { .. }) || nested_blocks(stmt).any(has_loop)
        })
    }
    let helper = module
        .functions
        .iter()
        .find(|(_, f)| f.name.as_deref() == Some("f"))
        .expect("helper function `f` survives the pass");
    assert!(
        has_loop(&helper.1.body),
        "loop with bare `break` in body must NOT be unwrapped even when \
             `break if STOP` selects a named-const-true (the bare break would \
             mis-target the surrounding scope after splice)"
    );
}

/// Counterpart: a `const NEVER: bool = false;` selector must
/// rewrite the loop's `break_if` to `None` (the loop is
/// non-terminating via that path but otherwise preserved).
#[test]
fn break_if_with_named_const_false_drops_break_if() {
    let src = r#"
            const NEVER: bool = false;
            fn f() -> i32 {
                var i: i32 = 0;
                loop {
                    i = i + 1;
                    if i > 10 {
                        break;
                    }
                    continuing {
                        break if NEVER;
                    }
                }
                return i;
            }
            @compute @workgroup_size(1) fn main() { _ = f(); }
        "#;
    let (changed, module) = run_pass(src);
    assert!(
        changed,
        "break_if with a named-const-false selector must be dropped"
    );
    // Confirm the surviving loop has no break_if.
    fn first_loop_break_if(block: &naga::Block) -> Option<Option<naga::Handle<naga::Expression>>> {
        for stmt in block.iter() {
            if let naga::Statement::Loop { break_if, .. } = stmt {
                return Some(*break_if);
            }
            for nested in nested_blocks(stmt) {
                if let Some(bi) = first_loop_break_if(nested) {
                    return Some(bi);
                }
            }
        }
        None
    }
    let helper = module
        .functions
        .iter()
        .find(|(_, f)| f.name.as_deref() == Some("f"))
        .expect("helper function `f` survives the pass");
    let bi = first_loop_break_if(&helper.1.body)
        .expect("loop should survive (only the break_if is dropped)");
    assert!(
        bi.is_none(),
        "break_if with a named-const-false selector must be rewritten to None"
    );
}

/// Same fall-through gate must apply at the NESTED level via
/// `case_body_terminates_beyond_switch`.  Build an outer switch
/// whose default case body contains *another* switch whose last
/// case falls through.  The inner switch falls past its cases
/// (Break-equivalent), so the OUTER switch's default case body
/// does NOT terminate beyond the switch - it falls through to
/// whatever the outer block does after the switch.
///
/// Pre-fix: `case_body_terminates_beyond_switch` missed the
/// `!last_falls_through` gate and would classify the outer
/// switch's case body as terminating, letting upstream callers
/// drop reachable statements after the outer switch.
#[test]
fn nested_switch_with_last_case_fallthrough_does_not_terminate_beyond() {
    let inner = build_terminating_default_switch(/*fall_through=*/ true);
    let mut outer_body = naga::Block::new();
    outer_body.push(inner, naga::Span::UNDEFINED);
    // The outer switch wraps the inner switch in its default case.
    // If `case_body_terminates_beyond_switch` correctly applies
    // the fall-through gate to nested switches, the outer switch
    // should NOT be classified as terminating either.
    let outer = naga::Statement::Switch {
        selector: {
            let mut arena: naga::Arena<naga::Expression> = naga::Arena::new();
            arena.append(
                naga::Expression::Literal(naga::Literal::U32(0)),
                naga::Span::UNDEFINED,
            )
        },
        cases: vec![naga::SwitchCase {
            value: naga::SwitchValue::Default,
            body: outer_body,
            fall_through: false,
        }],
    };
    assert!(
        !definitely_terminates(&outer),
        "nested switch whose inner switch has last-case fall-through must \
             propagate that non-termination through \
             `case_body_terminates_beyond_switch`, NOT classify the outer \
             switch as terminating"
    );
}
