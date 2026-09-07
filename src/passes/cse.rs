//! Common-subexpression elimination within one function body.
//!
//! A dominance-scoped map from structural expression keys to the first
//! (canonical) occurrence redirects every later duplicate to it; the
//! `Emit` ranges around replaced handles are then rebuilt in one walk.
//! Only pure expressions are eligible: loads, image ops, derivatives and
//! statement-attached results depend on state beyond their operands.

use super::scoped_map::ScopedMap;
use crate::pipeline::{Pass, PassContext};
use std::hash::{Hash, Hasher};

use crate::error::Error;

use super::expr_util::{flatten_replacement_chains, try_map_expression_handles_in_place};
use crate::handle_set::HandleMap;

/// Replace duplicate pure expressions with the first dominating evaluation.
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

/// Structural expression key.  Child handles are pre-resolved through
/// the replacement map so duplicates that already reference a canonical
/// operand compare equal.
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
        // `SwizzleComponent` lacks `Hash`; it is `#[repr(u8)]`, so `as u8`
        // is exact and a future discriminant can only cause a key miss,
        // never a wrong fold.
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
        // Discriminant first so identical payloads of different variants
        // cannot collide.
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

#[inline]
fn resolve(
    handle: naga::Handle<naga::Expression>,
    replacements: &HandleMap<naga::Expression, naga::Handle<naga::Expression>>,
) -> naga::Handle<naga::Expression> {
    replacements.get(handle).copied().unwrap_or(handle)
}

fn build_cse_key(
    expr: &naga::Expression,
    replacements: &HandleMap<naga::Expression, naga::Handle<naga::Expression>>,
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

        // Impure or statement-bound (Load, image / derivative ops,
        // `*Result`), pre-emit declaratives, and literals / constants
        // (canonicalised elsewhere).
        _ => None,
    }
}

// MARK: Per-function driver

fn cse_function(function: &mut naga::Function) -> bool {
    let mut replacements: HandleMap<naga::Expression, naga::Handle<naga::Expression>> =
        Default::default();

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

    // The arena walk resolves one level, so chains must be flat.
    flatten_replacement_chains(&mut replacements);

    for (_, expr) in function.expressions.iter_mut() {
        let _ = try_map_expression_handles_in_place(expr, &mut |h| {
            Some(replacements.get(h).copied().unwrap_or(h))
        });
    }

    apply_and_rebuild(&mut function.body, &replacements);

    // A name on a replaced handle would become a dangling `let`.
    function
        .named_expressions
        .retain(|h, _| !replacements.contains_key(h));

    true
}

// MARK: Dominance-scoped collection

/// Checkpoint / rollback at every nested block keeps `cse_map` equal to
/// the dominator set of the current statement, never to what sibling
/// branches registered.
fn collect_cse_replacements(
    block: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    cse_map: &mut ScopedMap<CseKey, naga::Handle<naga::Expression>>,
    replacements: &mut HandleMap<naga::Expression, naga::Handle<naga::Expression>>,
) {
    for statement in block {
        match statement {
            naga::Statement::Emit(range) => {
                for handle in range.clone() {
                    let expr = &expressions[handle];
                    if let Some(key) = build_cse_key(expr, replacements) {
                        if let Some(canonical) = cse_map.get(&key) {
                            replacements.insert(handle, *canonical);
                        } else {
                            cse_map.insert(key, handle);
                        }
                    }
                }
            }

            // Roll back after EVERY nested block: a canonical registered
            // inside one must not reach code after it.  If / Switch arms
            // and `Block` bodies are lexical scopes in the output, so a
            // leaked canonical references an out-of-scope `let` - naga's
            // flow-insensitive validator accepts it, re-parse and tint do
            // not.  A loop body does not dominate its continuing block
            // (`continue` skips the body's later `Emit`s), so a continuing
            // expression redirected to a body canonical reads a value that
            // iteration never produced - a silent miscompile naga
            // validates; forgoing body->continuing CSE in `continue`-free
            // loops is the price.  Leaf statements iterate zero blocks,
            // and a future block-bearing variant inherits this scoping.
            _ => {
                let checkpoint = cse_map.checkpoint();
                for nested in super::expr_util::nested_blocks(statement) {
                    collect_cse_replacements(nested, expressions, cse_map, replacements);
                    cse_map.rollback_to(checkpoint);
                }
            }
        }
    }
}

// MARK: Fused fixup walk

/// Rebuild `Emit` ranges around surviving handles and remap statement
/// operands to their canonicals in one walk.
fn apply_and_rebuild(
    block: &mut naga::Block,
    replacements: &HandleMap<naga::Expression, naga::Handle<naga::Expression>>,
) {
    let original = std::mem::take(block);
    for (mut statement, span) in original.span_into_iter() {
        if let naga::Statement::Emit(range) = &statement {
            let surviving: Vec<_> = range
                .clone()
                .filter(|h| !replacements.contains_key(h))
                .collect();
            super::expr_util::push_emit_runs(block, &surviving, span);
            continue;
        }
        for nested in super::expr_util::nested_blocks_mut(&mut statement) {
            apply_and_rebuild(nested, replacements);
        }

        super::expr_util::remap_statement_handles(&mut statement, &mut |h| {
            replacements.get(h).copied().unwrap_or(h)
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
            name_log: None,
        };
        let changed = pass.run(&mut module, &ctx).expect("pass should succeed");
        let _ = crate::io::validate_module(&module).expect("module should remain valid");
        (changed, module)
    }

    #[test]
    fn eliminates_duplicate_binary_expressions() {
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
        assert!(!changed, "CSE should not unify across if/else branches");
    }

    #[test]
    fn cse_dominates_into_if() {
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
        let source = r#"
fn f(a: f32) -> f32 {
    var x = a;
    let v1 = x;
    let v2 = x;
    return v1 + v2;
}
"#;
        let (changed, _module) = run_pass(source);
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
        assert!(changed, "complex shader should have CSE opportunities");
    }
}
