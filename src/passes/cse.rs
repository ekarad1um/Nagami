//! Common-subexpression elimination across a single function body.
//!
//! The pass walks each statement tree with a dominance-scoped map from
//! structural expression keys to the first (canonical) occurrence of
//! that expression.  Duplicate evaluations are rewritten to reference
//! the canonical handle, and the `Emit` ranges around the replaced
//! handles are rebuilt in a single fused traversal.
//!
//! Only side-effect-free expressions are eligible; loads, image ops,
//! derivatives, and statement-attached result expressions remain
//! unique because their value can depend on program state beyond
//! their operands.

use super::scoped_map::ScopedMap;
use crate::pipeline::{Pass, PassContext};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use crate::error::Error;

use super::expr_util::{flatten_replacement_chains, try_map_expression_handles_in_place};

/// Replace duplicate pure expression sub-DAGs with a reference to the
/// first dominating evaluation.
#[derive(Debug, Default)]
pub struct CSEPass;

impl Pass for CSEPass {
    fn name(&self) -> &'static str {
        "cse"
    }

    fn run(&mut self, module: &mut naga::Module, _ctx: &PassContext<'_>) -> Result<bool, Error> {
        let mut changed = false;
        for (_, function) in module.functions.iter_mut() {
            changed |= cse_function(function);
        }
        for entry in module.entry_points.iter_mut() {
            changed |= cse_function(&mut entry.function);
        }
        Ok(changed)
    }
}

// MARK: Key type

/// Hashable structural key for an expression.  Child handles are
/// pre-resolved through the replacement map so two expressions that
/// differ only in which canonical operand they reference hash equal.
#[derive(Clone, Eq, PartialEq)]
enum CseKey {
    Compose {
        ty: naga::Handle<naga::Type>,
        components: Vec<naga::Handle<naga::Expression>>,
    },
    Access {
        base: naga::Handle<naga::Expression>,
        index: naga::Handle<naga::Expression>,
    },
    AccessIndex {
        base: naga::Handle<naga::Expression>,
        index: u32,
    },
    Splat {
        size: naga::VectorSize,
        value: naga::Handle<naga::Expression>,
    },
    Swizzle {
        size: naga::VectorSize,
        vector: naga::Handle<naga::Expression>,
        // Stored as `[u8; 4]` because `SwizzleComponent` lacks `Hash`.
        pattern: [u8; 4],
    },
    Unary {
        op: naga::UnaryOperator,
        expr: naga::Handle<naga::Expression>,
    },
    Binary {
        op: naga::BinaryOperator,
        left: naga::Handle<naga::Expression>,
        right: naga::Handle<naga::Expression>,
    },
    Select {
        condition: naga::Handle<naga::Expression>,
        accept: naga::Handle<naga::Expression>,
        reject: naga::Handle<naga::Expression>,
    },
    Relational {
        fun: naga::RelationalFunction,
        argument: naga::Handle<naga::Expression>,
    },
    Math {
        fun: naga::MathFunction,
        arg: naga::Handle<naga::Expression>,
        arg1: Option<naga::Handle<naga::Expression>>,
        arg2: Option<naga::Handle<naga::Expression>>,
        arg3: Option<naga::Handle<naga::Expression>>,
    },
    As {
        expr: naga::Handle<naga::Expression>,
        kind: naga::ScalarKind,
        convert: Option<naga::Bytes>,
    },
    ArrayLength(naga::Handle<naga::Expression>),
}

impl Hash for CseKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Mix in the discriminant so keys from different variants with
        // structurally identical payloads cannot collide.
        std::mem::discriminant(self).hash(state);
        match self {
            CseKey::Compose { ty, components } => {
                ty.hash(state);
                components.hash(state);
            }
            CseKey::Access { base, index } => {
                base.hash(state);
                index.hash(state);
            }
            CseKey::AccessIndex { base, index } => {
                base.hash(state);
                index.hash(state);
            }
            CseKey::Splat { size, value } => {
                size.hash(state);
                value.hash(state);
            }
            CseKey::Swizzle {
                size,
                vector,
                pattern,
            } => {
                size.hash(state);
                vector.hash(state);
                pattern.hash(state);
            }
            CseKey::Unary { op, expr } => {
                op.hash(state);
                expr.hash(state);
            }
            CseKey::Binary { op, left, right } => {
                op.hash(state);
                left.hash(state);
                right.hash(state);
            }
            CseKey::Select {
                condition,
                accept,
                reject,
            } => {
                condition.hash(state);
                accept.hash(state);
                reject.hash(state);
            }
            CseKey::Relational { fun, argument } => {
                fun.hash(state);
                argument.hash(state);
            }
            CseKey::Math {
                fun,
                arg,
                arg1,
                arg2,
                arg3,
            } => {
                fun.hash(state);
                arg.hash(state);
                arg1.hash(state);
                arg2.hash(state);
                arg3.hash(state);
            }
            CseKey::As {
                expr,
                kind,
                convert,
            } => {
                expr.hash(state);
                kind.hash(state);
                convert.hash(state);
            }
            CseKey::ArrayLength(h) => {
                h.hash(state);
            }
        }
    }
}

// MARK: Key construction

/// Resolve `handle` through the replacement map, returning the handle
/// unchanged when no replacement has been registered yet.
#[inline]
fn resolve(
    handle: naga::Handle<naga::Expression>,
    replacements: &HashMap<naga::Handle<naga::Expression>, naga::Handle<naga::Expression>>,
) -> naga::Handle<naga::Expression> {
    replacements.get(&handle).copied().unwrap_or(handle)
}

/// Build a [`CseKey`] for `expr`, or `None` when the expression is
/// ineligible for CSE.  Every child handle is resolved first so
/// duplicates that already reference a canonical operand hash equal.
fn build_cse_key(
    expr: &naga::Expression,
    replacements: &HashMap<naga::Handle<naga::Expression>, naga::Handle<naga::Expression>>,
) -> Option<CseKey> {
    let r = |h: naga::Handle<naga::Expression>| resolve(h, replacements);
    let ro = |h: &Option<naga::Handle<naga::Expression>>| h.map(|h| resolve(h, replacements));

    match expr {
        naga::Expression::Compose { ty, components } => Some(CseKey::Compose {
            ty: *ty,
            components: components.iter().map(|h| r(*h)).collect(),
        }),
        naga::Expression::Access { base, index } => Some(CseKey::Access {
            base: r(*base),
            index: r(*index),
        }),
        naga::Expression::AccessIndex { base, index } => Some(CseKey::AccessIndex {
            base: r(*base),
            index: *index,
        }),
        naga::Expression::Splat { size, value } => Some(CseKey::Splat {
            size: *size,
            value: r(*value),
        }),
        naga::Expression::Swizzle {
            size,
            vector,
            pattern,
        } => Some(CseKey::Swizzle {
            size: *size,
            vector: r(*vector),
            pattern: [
                pattern[0] as u8,
                pattern[1] as u8,
                pattern[2] as u8,
                pattern[3] as u8,
            ],
        }),
        naga::Expression::Unary { op, expr } => Some(CseKey::Unary {
            op: *op,
            expr: r(*expr),
        }),
        naga::Expression::Binary { op, left, right } => Some(CseKey::Binary {
            op: *op,
            left: r(*left),
            right: r(*right),
        }),
        naga::Expression::Select {
            condition,
            accept,
            reject,
        } => Some(CseKey::Select {
            condition: r(*condition),
            accept: r(*accept),
            reject: r(*reject),
        }),
        naga::Expression::Relational { fun, argument } => Some(CseKey::Relational {
            fun: *fun,
            argument: r(*argument),
        }),
        naga::Expression::Math {
            fun,
            arg,
            arg1,
            arg2,
            arg3,
        } => Some(CseKey::Math {
            fun: *fun,
            arg: r(*arg),
            arg1: ro(arg1),
            arg2: ro(arg2),
            arg3: ro(arg3),
        }),
        naga::Expression::As {
            expr,
            kind,
            convert,
        } => Some(CseKey::As {
            expr: r(*expr),
            kind: *kind,
            convert: *convert,
        }),
        naga::Expression::ArrayLength(h) => Some(CseKey::ArrayLength(r(*h))),

        // Ineligible variants: loads (state-dependent), image and
        // derivative ops (side effects and execution-mask coupling),
        // `*Result` variants (tied to their originating statement),
        // non-emittable declarative expressions, and literals or
        // constants (trivially unique or canonicalised elsewhere).
        _ => None,
    }
}

// MARK: Per-function driver

/// Run CSE across the function body: collect replacements under a
/// scoped map, flatten any transitive chains, rewrite the expression
/// arena, then fuse emit-range surgery with statement-level handle
/// remapping in a single traversal.  Returns `true` when at least one
/// replacement fired.
fn cse_function(function: &mut naga::Function) -> bool {
    let mut replacements: HashMap<naga::Handle<naga::Expression>, naga::Handle<naga::Expression>> =
        HashMap::new();

    let mut cse_map: ScopedMap<CseKey, naga::Handle<naga::Expression>> = ScopedMap::new();

    collect_cse_replacements(
        &function.body,
        &function.expressions,
        &mut cse_map,
        &mut replacements,
    );

    if replacements.is_empty() {
        return false;
    }

    // Flatten `A -> B -> C` into `A -> C` and `B -> C` so the arena
    // walk below resolves every handle in a single step.
    flatten_replacement_chains(&mut replacements);

    // Rewrite each expression's children to reference canonical handles.
    for (_, expr) in function.expressions.iter_mut() {
        let _ = try_map_expression_handles_in_place(expr, &mut |h| {
            Some(replacements.get(&h).copied().unwrap_or(h))
        });
    }

    // Fused traversal: rebuild `Emit` ranges around survivors and
    // remap statement-level handles in one descent (formerly two
    // separate walks via `apply_replacements_to_block` and
    // `rebuild_emit_ranges_after_removal`).  The "replaced" predicate
    // is `replacements.contains_key(h)` directly, avoiding the extra
    // `HashSet<Handle>` the previous shape allocated.
    apply_and_rebuild(&mut function.body, &replacements);

    // Drop named-expression entries whose handle was replaced so the
    // generator does not emit dangling `let` bindings.
    function
        .named_expressions
        .retain(|h, _| !replacements.contains_key(h));

    true
}

// MARK: Dominance-scoped collection

/// Walk the statement tree, populating `cse_map` with canonical
/// expressions and `replacements` with each duplicate's redirection.
///
/// `cse_map` is a [`ScopedMap`] keyed by [`CseKey`]; every control-flow
/// boundary takes a checkpoint and rolls back on exit so the map always
/// reflects the current dominator set rather than whatever sibling
/// branches happened to register.  Cost is `O(in-scope writes)` per
/// scope instead of `O(map_size)`.
fn collect_cse_replacements(
    block: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    cse_map: &mut ScopedMap<CseKey, naga::Handle<naga::Expression>>,
    replacements: &mut HashMap<naga::Handle<naga::Expression>, naga::Handle<naga::Expression>>,
) {
    for statement in block {
        match statement {
            naga::Statement::Emit(range) => {
                for handle in range.clone() {
                    let expr = &expressions[handle];
                    if let Some(key) = build_cse_key(expr, replacements) {
                        if let Some(canonical) = cse_map.get(&key) {
                            // Duplicate: redirect to the dominating handle.
                            replacements.insert(handle, *canonical);
                        } else {
                            // First occurrence: register as canonical.
                            cse_map.insert(key, handle);
                        }
                    }
                }
            }

            naga::Statement::If { accept, reject, .. } => {
                let checkpoint = cse_map.checkpoint();

                collect_cse_replacements(accept, expressions, cse_map, replacements);
                cse_map.rollback_to(checkpoint);

                collect_cse_replacements(reject, expressions, cse_map, replacements);
                cse_map.rollback_to(checkpoint);
            }

            naga::Statement::Switch { cases, .. } => {
                let checkpoint = cse_map.checkpoint();

                for case in cases {
                    collect_cse_replacements(&case.body, expressions, cse_map, replacements);
                    cse_map.rollback_to(checkpoint);
                }
            }

            naga::Statement::Loop {
                body, continuing, ..
            } => {
                // Parent-scope entries dominate the first iteration,
                // but loop-interior entries must be discarded for
                // post-loop code because the iteration count is
                // unknown at pass time.
                let checkpoint = cse_map.checkpoint();

                collect_cse_replacements(body, expressions, cse_map, replacements);
                collect_cse_replacements(continuing, expressions, cse_map, replacements);

                cse_map.rollback_to(checkpoint);
            }

            naga::Statement::Block(inner) => {
                collect_cse_replacements(inner, expressions, cse_map, replacements);
            }

            _ => {}
        }
    }
}

// MARK: Fused fixup walk

/// Single-pass fixup: rebuild every `Emit` range so the replaced
/// handles drop out, splitting into contiguous sub-ranges around
/// survivors and discarding emits that become empty, while also
/// remapping statement-level expression handles to their canonical
/// replacements.  Recurses into every control-flow sub-block once.
///
/// Mirrors `load_dedup::apply_to_block`.  Predates the fused shape
/// used to exist as two separate walks (`apply_replacements_to_block`
/// followed by `rebuild_emit_ranges_after_removal`); the single
/// traversal saves one full statement-tree descent per function.
fn apply_and_rebuild(
    block: &mut naga::Block,
    replacements: &HashMap<naga::Handle<naga::Expression>, naga::Handle<naga::Expression>>,
) {
    let original = std::mem::take(block);
    for (mut statement, span) in original.span_into_iter() {
        match &mut statement {
            naga::Statement::Emit(range) => {
                let surviving: Vec<_> = range
                    .clone()
                    .filter(|h| !replacements.contains_key(h))
                    .collect();
                if surviving.is_empty() {
                    continue;
                }
                // Emit the surviving handles as contiguous sub-ranges.
                let mut start = surviving[0];
                let mut end = surviving[0];
                for &h in &surviving[1..] {
                    if h.index() == end.index() + 1 {
                        end = h;
                    } else {
                        block.push(
                            naga::Statement::Emit(naga::Range::new_from_bounds(start, end)),
                            span,
                        );
                        start = h;
                        end = h;
                    }
                }
                block.push(
                    naga::Statement::Emit(naga::Range::new_from_bounds(start, end)),
                    span,
                );
                continue;
            }
            naga::Statement::Block(inner) => {
                apply_and_rebuild(inner, replacements);
            }
            naga::Statement::If { accept, reject, .. } => {
                apply_and_rebuild(accept, replacements);
                apply_and_rebuild(reject, replacements);
            }
            naga::Statement::Switch { cases, .. } => {
                for case in cases.iter_mut() {
                    apply_and_rebuild(&mut case.body, replacements);
                }
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => {
                apply_and_rebuild(body, replacements);
                apply_and_rebuild(continuing, replacements);
            }
            _ => {}
        }

        // Remap statement-level expression handles to canonical
        // replacements (covers `Store`, `Call`, `If::condition`,
        // `Switch::selector`, `Atomic`, `ImageStore`, and friends).
        super::expr_util::remap_statement_handles(&mut statement, &mut |h| {
            replacements.get(&h).copied().unwrap_or(h)
        });

        block.push(statement, span);
    }
}

// MARK: Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn run_pass(source: &str) -> (bool, naga::Module) {
        let mut module = naga::front::wgsl::parse_str(source).expect("source should parse");
        let mut pass = CSEPass;
        let config = Config::default();
        let ctx = PassContext {
            config: &config,
            trace_run_dir: None,
        };
        let changed = pass.run(&mut module, &ctx).expect("pass should succeed");
        let _ = crate::io::validate_module(&module).expect("module should remain valid");
        (changed, module)
    }

    // `ScopedMap` invariant tests live under `passes::scoped_map::tests`.
    // The integration tests below (`does_not_cse_across_if_branches`,
    // `cse_dominates_into_if`, `pre_loop_expression_deduped_in_loop_body`)
    // exercise the CSE-side scope semantics end-to-end.

    #[test]
    fn eliminates_duplicate_binary_expressions() {
        // `a * a` appears twice - CSE should unify them.
        let source = r#"
fn f(a: f32) -> f32 {
    let x = a * a;
    let y = a * a;
    return x + y;
}
"#;
        let (changed, _module) = run_pass(source);
        assert!(changed, "CSE should find duplicate a*a");
    }

    #[test]
    fn eliminates_duplicate_math_calls() {
        // `sin(a)` appears twice.
        let source = r#"
fn f(a: f32) -> f32 {
    let x = sin(a);
    let y = sin(a);
    return x + y;
}
"#;
        let (changed, _module) = run_pass(source);
        assert!(changed, "CSE should find duplicate sin(a)");
    }

    #[test]
    fn no_change_when_no_duplicates() {
        let source = r#"
fn f(a: f32, b: f32) -> f32 {
    let x = a * b;
    let y = a + b;
    return x + y;
}
"#;
        let (changed, _module) = run_pass(source);
        assert!(!changed, "no duplicates should mean no change");
    }

    #[test]
    fn does_not_cse_across_if_branches() {
        // The same expression in accept and reject branches should NOT be
        // unified because neither dominates the other.
        let source = r#"
fn f(a: f32, cond: bool) -> f32 {
    if cond {
        let x = a * a;
        return x;
    } else {
        let y = a * a;
        return y;
    }
}
"#;
        let (changed, _module) = run_pass(source);
        // Both `a * a` are in separate branches.  The CSE pass should NOT
        // unify them (neither dominates the other).
        assert!(!changed, "CSE should not unify across if/else branches");
    }

    #[test]
    fn cse_dominates_into_if() {
        // An expression before the if dominates both branches.
        let source = r#"
fn f(a: f32, cond: bool) -> f32 {
    let x = a * a;
    if cond {
        let y = a * a;
        return x + y;
    }
    return x;
}
"#;
        let (changed, _module) = run_pass(source);
        assert!(changed, "pre-if expression should dominate into if body");
    }

    #[test]
    fn cse_nested_expressions() {
        // duplicate nested tree: sin(a * a)
        let source = r#"
fn f(a: f32) -> f32 {
    let x = sin(a * a);
    let y = sin(a * a);
    return x + y;
}
"#;
        let (changed, _module) = run_pass(source);
        assert!(changed, "CSE should unify nested sin(a*a)");
    }

    #[test]
    fn does_not_cse_loads() {
        // Loads should not be CSE'd (handled by load_dedup).
        let source = r#"
fn f(a: f32) -> f32 {
    var x = a;
    let v1 = x;
    let v2 = x;
    return v1 + v2;
}
"#;
        let (changed, _module) = run_pass(source);
        // Loads from `x` should not be eliminated by CSE.
        assert!(!changed, "CSE should not eliminate Loads");
    }

    #[test]
    fn cse_compose_expressions() {
        let source = r#"
fn f(a: f32, b: f32) -> f32 {
    let v1 = vec2f(a, b);
    let v2 = vec2f(a, b);
    return v1.x + v2.y;
}
"#;
        let (changed, _module) = run_pass(source);
        assert!(changed, "CSE should unify duplicate Compose expressions");
    }

    #[test]
    fn cse_select_expressions() {
        let source = r#"
fn f(a: f32, b: f32, c: bool) -> f32 {
    let x = select(a, b, c);
    let y = select(a, b, c);
    return x + y;
}
"#;
        let (changed, _module) = run_pass(source);
        assert!(changed, "CSE should unify duplicate select expressions");
    }

    #[test]
    fn cse_within_loop_body() {
        // Duplicate expressions within the same loop iteration should be unified.
        let source = r#"
fn f(a: f32) -> f32 {
    var sum = 0.0;
    for (var i = 0u; i < 10u; i++) {
        let x = a * a;
        let y = a * a;
        sum += x + y;
    }
    return sum;
}
"#;
        let (changed, _module) = run_pass(source);
        assert!(changed, "CSE should unify duplicates within loop body");
    }

    #[test]
    fn pre_loop_expression_deduped_in_loop_body() {
        // Expression before loop dominates loop body (first iteration).
        let source = r#"
fn f(a: f32) -> f32 {
    let pre = a * a;
    var sum = pre;
    for (var i = 0u; i < 10u; i++) {
        let inner = a * a;
        sum += inner;
    }
    return sum;
}
"#;
        let (changed, _module) = run_pass(source);
        assert!(
            changed,
            "pre-loop expression should be available inside loop"
        );
    }

    #[test]
    fn validates_complex_shader() {
        // A more complex shader to stress-test validation.
        let source = r#"
fn calc(p: vec3f) -> f32 {
    let d = dot(p, p);
    let n = normalize(p);
    let r = reflect(n, vec3f(0.0, 1.0, 0.0));
    return dot(r, r) + dot(p, p) + dot(n, n);
}

@fragment
fn main() -> @location(0) vec4f {
    let a = calc(vec3f(1.0, 2.0, 3.0));
    return vec4f(a, a, a, 1.0);
}
"#;
        let (changed, _module) = run_pass(source);
        // There should be CSE opportunities (duplicate dot(p,p), dot(n,n)).
        assert!(changed, "complex shader should have CSE opportunities");
    }
}
