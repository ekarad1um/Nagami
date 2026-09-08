use super::*;
use crate::config::Config;

fn run_pass(source: &str) -> (bool, naga::Module) {
    let mut module = naga::front::wgsl::parse_str(source).expect("source should parse");
    let mut pass = DeadBranchPass;
    let config = Config::default();
    let ctx = PassContext {
        config: &config,
        name_log: None,
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

/// A value-position fold rebuilds the arena; an emitted initializer
/// (`OV * 3.0`) must come out single and forwardable, or load_dedup's init
/// forward hands naga an expression no `Emit` introduced and its whole run
/// rolls back.
#[test]
fn value_position_fold_keeps_an_emitted_initializer_forwardable() {
    let src = r#"
override OV: f32 = 2.0;
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {
    var x = OV * 3.0;
    let d = id.x > 1u && id.y > 2u;
    if d { out[1] = 7.0; }
    out[0] = x;
}
"#;
    let (changed, mut module) = run_pass(src);
    assert!(changed, "the `&&` join folds");
    let config = Config::default();
    let ctx = PassContext {
        config: &config,
        name_log: None,
    };
    crate::passes::load_dedup::LoadDedupPass
        .run(&mut module, &ctx)
        .expect("load_dedup should run");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("the forwarded init must be in scope");
    let function = &module.entry_points[0].function;
    assert!(
        function
            .local_variables
            .iter()
            .all(|(_, l)| l.init.is_none()),
        "the init was forwarded to its only load"
    );
}

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

/// Collapsing a constant-selector switch whose dropped case produced a
/// statement result (a non-inlined call) would orphan that expression
/// (invalid IR); the guard keeps the switch.  `run_pass` validates, so
/// removing the guard fails here.
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
    let (_changed, module) = run_pass(src);
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

/// The dead-tail drop after a definite terminator needs the same
/// result-producer guard: dropping an unreachable `Call { result: Some }`
/// orphans its expression, and one poisoned function rolls the whole pass
/// back every sweep, freezing every fold.  The `if true` fold in `main`
/// proves the pass still lands elsewhere.
#[test]
fn dead_tail_with_result_producer_stays_valid_and_folds_elsewhere() {
    let src = r#"
fn helper() -> i32 { return 7; }
fn poisoned() -> i32 {
    return 1;
    let x = helper();
    return x;
}
@compute @workgroup_size(1)
fn main() {
    var v: i32 = 0;
    if true { v = 42; } else { v = 7; }
    v = v + poisoned();
    _ = v;
}
"#;
    let (changed, module) = run_pass(src);
    assert!(changed, "folds elsewhere must land, not roll back");
    let main = module
        .entry_points
        .iter()
        .find(|e| e.name == "main")
        .expect("main entry point");
    assert_eq!(
        count_ifs(&main.function.body),
        0,
        "if true must fold despite the poisoned dead tail in another function"
    );
}

/// Arms that are empty or a lone bare `break` do nothing (the breaks only
/// exit the switch and expressions carry no side effects), so the whole
/// statement goes; the splice paths cannot remove it because their
/// bare-break guard refuses the body.
#[test]
fn deletes_switch_whose_arms_are_empty_or_bare_break() {
    let src = r#"
@group(0) @binding(0) var<storage, read_write> s: array<u32>;
@compute @workgroup_size(1)
fn main() {
    let x = s[0];
    switch (x) {
        case 0u: { break; }
        default: { break; }
    }
    s[1] = 9u;
}
"#;
    let (changed, module) = run_pass(src);
    assert!(changed, "deletion must be reported");
    let body = &module.entry_points[0].function.body;
    assert_eq!(count_switches(body), 0, "trivial switch must be deleted");
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

#[test]
fn degenerate_case_list_to_default_is_spliced() {
    // `case 0, default {...}` lowers to an empty fall-through `case 0` plus
    // the body-carrying `default`, so every selector runs the body once.
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
    // `case 1 {}` exits the switch for selector 1, so the switch
    // distinguishes that value and must stay.
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
    // The bare `break` targets the switch; splicing would mis-target it.
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

// `Expression::Constant` with a bool-literal init must resolve too, or a
// branch on `const X: bool = false;` slips past until const_fold inlines it.
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
    for (_, func) in module.functions.iter() {
        assert_eq!(count_ifs(&func.body), 0);
    }
}

#[test]
fn bare_break_detected_at_top_level() {
    let mut block = naga::Block::new();
    block.push(naga::Statement::Break, Default::default());
    assert!(contains_bare_break(&block));
}

#[test]
fn bare_break_detected_inside_if_block() {
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
    let loop_body = module.entry_points[0].function.body.iter().find_map(|s| {
        if let naga::Statement::Loop { body, .. } = s {
            Some(body)
        } else {
            None
        }
    });
    assert!(loop_body.is_some());
    assert!(!contains_bare_break(loop_body.unwrap()));
}

#[test]
fn bare_continue_detected_inside_switch() {
    // A naga `Switch` does not capture `Continue`; it still targets the
    // enclosing loop.
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
        contains_loop_control(loop_body.unwrap(), false, true),
        "continue inside switch should be detected by the continue-only search"
    );
    assert!(
        contains_bare_loop_control(loop_body.unwrap()),
        "continue inside switch should be detected by contains_bare_loop_control"
    );
}

#[test]
fn bare_continue_not_detected_inside_nested_loop() {
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
    assert!(
        !contains_loop_control(outer_loop_body.unwrap(), false, true),
        "continue inside nested loop should NOT be detected"
    );
}

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

// The condition `Load(d)` proves `d` false in the reject arm, so its
// `d = false` is a no-op.
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

// `d` is zero-init, so the else's `d = false` is a no-op.
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
    // The accept's `false` store clears (zero-init); the else's `true` keeps
    // the if.
    assert!(
        changed,
        "accept zero-store to zero-init var should be cleared"
    );
    for (_, func) in module.functions.iter() {
        assert!(
            count_ifs(&func.body) >= 1,
            "if should be preserved because reject stores non-zero"
        );
    }
}

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

// Both ifs match the `&&` pattern regardless of the prior store.
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
    assert!(changed);
    for (_, func) in module.functions.iter() {
        assert_eq!(
            count_ifs(&func.body),
            0,
            "both ifs should be desugared by short-circuit pass"
        );
    }
}

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

// `d = cond && val` is valid regardless of `d`'s prior value, so the loop
// does not matter.
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
    assert!(changed);
    for (_, func) in module.functions.iter() {
        assert_eq!(
            count_non_empty_rejects(&func.body),
            0,
            "short-circuit desugaring should remove the if"
        );
    }
}

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

/// const_fold's De Morgan turns the lowering's `!(x==y)` into `x!=y` before
/// this pass sees it, so the un-flip path must recover the negation and
/// build `d = (x==y) || b`; the left-operand op assertion pins the polarity.
#[test]
fn short_circuit_or_with_equality_left_operand_desugars() {
    let src = r#"
fn f(x: u32, y: u32, b: bool) -> bool {
    var d: bool;
    if x != y { d = b; } else { d = true; }
    return d;
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(f32(f(1u, 2u, true))); }
"#;
    let (changed, module) = run_pass(src);
    assert!(changed);
    for (_, func) in module.functions.iter() {
        assert_eq!(
            count_ifs(&func.body),
            0,
            "equality-left || must desugar to d = (x==y) || b"
        );
        let or_left_op = func.expressions.iter().find_map(|(_, e)| match e {
            naga::Expression::Binary {
                op: naga::BinaryOperator::LogicalOr,
                left,
                ..
            } => Some(match func.expressions[*left] {
                naga::Expression::Binary { op, .. } => Some(op),
                _ => None,
            }),
            _ => None,
        });
        assert_eq!(
            or_left_op,
            Some(Some(naga::BinaryOperator::Equal)),
            "the || left operand must be the UN-FLIPPED comparison (source \
             had x != y, so the recovered negation is x == y)"
        );
    }
}

#[test]
fn short_circuit_preserves_non_bool_if_else() {
    // `LogicalAnd` needs bool; the f32 shape must not match.
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
        assert!(
            count_ifs(&func.body) >= 1,
            "non-bool if-else must not be desugared"
        );
    }
}

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

// The condition `Load(d)` proves `d` true in the accept arm, so its
// `d = true` is a no-op.
#[test]
fn redundant_accept_true_store_when_condition_is_load() {
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
        // The first if's condition is an argument, so only the second accept
        // clears.
        let non_empty_accepts = count_non_empty_accepts(&func.body);
        assert!(
            non_empty_accepts <= 1,
            "second if's accept `d = true` should be cleared, got {non_empty_accepts} non-empty accepts"
        );
    }
}

// `t` captured `d` before `d = false`, so narrowing on it would clobber the
// known value and drop the live `d = true`; fresh-load tracking must keep
// the branch.
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

// A fresh re-load (no intervening store) must still narrow; the stale-load
// guard is not a blanket disable.
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

#[test]
fn eliminates_empty_if_after_both_branches_cleared() {
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
    for (_, func) in module.functions.iter() {
        assert_eq!(count_ifs(&func.body), 0, "empty if should be eliminated");
    }
}

#[test]
fn short_circuit_basic_and_replacement() {
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
    // The accept value needs an `Emit` (`arr[idx]`), which the re-sugar
    // hoists: sound because the load is side-effect-free, WGSL bounds-checks
    // the index, `&&` discards the value when `a` is false, and lifting it
    // only reduces non-uniformity.
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
    assert_eq!(
        count_ifs(&f_function.body),
        0,
        "the lowered short-circuit If should fold into `&&`"
    );
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

#[test]
fn eliminates_empty_switch_all_cases_cleared() {
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

#[test]
fn unwraps_degenerate_switch_default_only() {
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
    // The bare `break` targets the switch, so splicing would mis-target it.
    let src = r#"
@fragment
fn fs(@location(0) v: f32) -> @location(0) vec4f {
    var x = 0.0;
    switch i32(v) { default: { x = 1.0; break; } }
    return vec4f(x);
}
"#;
    let (_changed, module) = run_pass(src);
    assert_eq!(
        count_switches(&module.entry_points[0].function.body),
        1,
        "switch with bare break must be preserved"
    );
}

#[test]
fn preserves_switch_with_multiple_cases() {
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

/// Both arms terminating still elides: the saved `else{...}` outweighs the
/// residual `if c { return v1; }`, and gating on `!reject.terminates` made
/// the corpus larger; the only missed collapse is symmetric `return;` arms,
/// which are not folded anyway.
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

/// A nested `Block(empty)` (authored `{ }` or drained by an upstream fold)
/// must be dropped after recursion or a vacuous `{}` leaks into the output.
#[test]
fn empty_nested_block_is_dropped() {
    // WGSL has no syntax for a bare nested `{ }`, so build the IR directly.
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

    // A stricter future validator would otherwise silently turn this into an
    // invalid-IR exercise.
    crate::io::validate_module(&module).expect("hand-built input must satisfy naga's validator");

    let mut pass = DeadBranchPass;
    let config = Config::default();
    let ctx = PassContext {
        config: &config,
        name_log: None,
    };
    let changed = pass.run(&mut module, &ctx).expect("pass should run");
    assert!(
        changed,
        "empty nested Block elision must report `changed = true`"
    );

    crate::io::validate_module(&module).expect("module must remain valid after the elision");

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
    let src = r#"
fn f(a: bool) -> i32 {
    var x: i32 = 42;
    if a { x = 99; } else { x = 42; }
    return x;
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(f32(f(true))); }
"#;
    let (changed, module) = run_pass(src);
    // The reject's 42 matches the init and clears; the accept's 99 keeps the
    // if.
    assert!(changed);
    for (_, func) in module.functions.iter() {
        assert!(
            count_ifs(&func.body) >= 1,
            "if with non-redundant accept store must be preserved"
        );
    }
}

// A switch whose cases all end in `break` is not a terminator: the
// continuing block's phi stores force naga to emit the post-switch `let`
// bindings in the loop body, and treating the switch as terminating drops
// that Emit ("Expression NotInScope").
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
    let (_, module) = run_pass(src);
    let ep = &module.entry_points[0];
    let has_loop = ep
        .function
        .body
        .iter()
        .any(|s| matches!(s, naga::Statement::Loop { .. }));
    assert!(has_loop, "loop must be preserved");
}

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

// MARK: Switch fall-through edge in tail_terminates
//
// naga's frontend never emits a final case with `fall_through: true`, so the
// shape is hand-built.

fn build_terminating_default_switch(fall_through: bool) -> naga::Statement {
    let mut body = naga::Block::new();
    body.push(
        naga::Statement::Return { value: None },
        naga::Span::UNDEFINED,
    );
    naga::Statement::Switch {
        // `tail_terminates` ignores the selector; a throwaway arena
        // yields a handle.
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
    let stmt = build_terminating_default_switch(false);
    assert!(
        tail_terminates(&stmt, /*bare_break_terminates=*/ true),
        "Default case with terminating body and no fall-through must terminate"
    );
}

#[test]
fn switch_with_last_case_fallthrough_does_not_terminate() {
    // Fall-through past the last case is Break-equivalent (execution resumes
    // after the switch); the inliner or CSE could produce this shape.
    let stmt = build_terminating_default_switch(true);
    assert!(
        !tail_terminates(&stmt, /*bare_break_terminates=*/ true),
        "last-case fall-through must not classify the switch as terminating \
             (fall-through past the last case is Break-equivalent and execution \
             resumes after the switch)"
    );
}

/// `break if NAMED_CONST` must fold the same sweep as `if`/`switch` on a
/// named const (all via `resolve_to_literal`), not lag until `const_fold`
/// inlines it.
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

/// Unwrapping a loop whose body has a bare `break`/`continue` would
/// re-target it at the surrounding scope; the `contains_bare_loop_control`
/// guard must fire on named-const selectors too.
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

/// A named-const-false selector rewrites the loop's `break_if` to `None`,
/// keeping the loop.
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

/// `case_body_terminates_beyond_switch` must apply the fall-through gate to
/// a nested switch: the inner switch falls past its cases
/// (Break-equivalent), so the outer default body does not terminate beyond
/// the switch.
#[test]
fn nested_switch_with_last_case_fallthrough_does_not_terminate_beyond() {
    let inner = build_terminating_default_switch(true);
    let mut outer_body = naga::Block::new();
    outer_body.push(inner, naga::Span::UNDEFINED);
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
        !tail_terminates(&outer, /*bare_break_terminates=*/ true),
        "nested switch whose inner switch has last-case fall-through must \
             propagate that non-termination through \
             `case_body_terminates_beyond_switch`, NOT classify the outer \
             switch as terminating"
    );
}

/// The single-store forward substitutes the stored value for the one load;
/// a `-0.0` literal stays a runtime read (Dawn on Metal flushes the
/// literal but not the runtime negation).
#[test]
fn single_store_forward_keeps_a_negative_zero_store() {
    for (value, forwarded) in [("-(0.0)", false), ("0.5", true)] {
        let source = format!(
            "@group(0) @binding(0) var<storage, read_write> out: array<u32>;\n\
             @compute @workgroup_size(1) fn main() {{\n\
               var z: f32;\n\
               z = {value};\n\
               out[0] = bitcast<u32>(-z);\n\
             }}"
        );
        let (_, module) = run_pass(&source);
        let function = &module.entry_points[0].function;
        let mut stores = 0;
        crate::passes::expr_util::for_each_statement(&function.body, &mut |stmt| {
            if let naga::Statement::Store { pointer, .. } = stmt
                && matches!(
                    function.expressions[*pointer],
                    naga::Expression::LocalVariable(_)
                )
            {
                stores += 1;
            }
        });
        assert_eq!(stores == 0, forwarded, "value {value}: {stores} stores");
    }
}
