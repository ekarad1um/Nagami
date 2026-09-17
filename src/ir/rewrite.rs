//! Rewrites of a function's arena and blocks that every pass would otherwise
//! carry its own copy of: the take-rewrite-push block rebuild, `Emit`-range
//! surgery around dropped handles, replacement maps and their application,
//! and the emission-order arena rebuild.

use crate::handle_set::HandleMap;
use crate::ir::visit::{
    Scope, nested_blocks_mut, remap_block_handles, remap_statement_handles,
    try_map_expression_handles_in_place,
};
use crate::passes::expr_util::expression_needs_emit;

// MARK: Block rewriting

/// Rebuild `block` statement by statement: each `(statement, span)` is
/// taken out and handed to `f` with the block under construction, and `f`
/// pushes whatever stands in its place - nothing to drop it, several to
/// expand it, the statement itself to keep it.  Nested blocks are `f`'s to
/// descend into ([`rewrite_block`] does it bottom-up); this is the flat form
/// for a rewrite whose descent has its own order.
pub(crate) fn rewrite_statements(
    block: &mut naga::Block,
    f: &mut dyn FnMut(naga::Statement, naga::Span, &mut naga::Block),
) {
    let original = std::mem::take(block);
    let mut rebuilt = naga::Block::with_capacity(original.len());
    for (statement, span) in original.span_into_iter() {
        f(statement, span, &mut rebuilt);
    }
    *block = rebuilt;
}

/// [`rewrite_statements`] over `block` and every block nested in it,
/// bottom-up: `f` sees each statement after its own nested blocks have been
/// rewritten, so a match on an arm's shape (the short-circuit re-sugar
/// matching an `if` whose arms were just re-sugared) reads the final one,
/// and the `Scope` handed along is the statement's own.
pub(crate) fn rewrite_block(
    block: &mut naga::Block,
    scope: Scope,
    f: &mut dyn FnMut(naga::Statement, naga::Span, &mut naga::Block, Scope),
) {
    rewrite_statements(block, &mut |mut statement, span, out| {
        let inner = scope.inner(&statement);
        for nested in nested_blocks_mut(&mut statement) {
            rewrite_block(nested, inner, f);
        }
        f(statement, span, out, scope);
    });
}

// MARK: Emit-range surgery

/// Push `surviving` as `Emit` statements, one per contiguous run.  Shared by
/// every pass that drops handles out of a range, so the run-splitting exists
/// once.
pub(crate) fn push_emit_runs(
    block: &mut naga::Block,
    surviving: &[naga::Handle<naga::Expression>],
    span: naga::Span,
) {
    let Some(&first) = surviving.first() else {
        return;
    };
    let mut start = first;
    let mut end = first;
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
}

/// Drop every handle `removed` accepts from every `Emit` range in `block`
/// (nested control flow included), rebuilding contiguous sub-ranges around
/// the survivors and discarding `Emit` statements left empty.  One
/// implementation for every pass that rewrites handles in place (folded
/// literals, hoisted constants, forwarded loads), so a new control-flow
/// statement is handled once.
pub(crate) fn rebuild_emit_ranges_after_removal(
    block: &mut naga::Block,
    removed: &dyn Fn(naga::Handle<naga::Expression>) -> bool,
) {
    rewrite_block(block, Scope::default(), &mut |statement, span, out, _| {
        if let naga::Statement::Emit(range) = &statement {
            // An emit that lost every handle disappears.
            let surviving: Vec<_> = range.clone().filter(|&h| !removed(h)).collect();
            push_emit_runs(out, &surviving, span);
        } else {
            out.push(statement, span);
        }
    });
}

// MARK: Replacement maps

/// The expression that will stand where `handle` does once `map` is
/// applied: the end of its chain, or `handle` itself when it is not a key.
/// Passes accumulate `A -> B` entries and can produce `A -> B -> C` (a load
/// forwarded to a load that was itself forwarded; a spliced call whose
/// argument was an earlier splice's result), and every gate that judges a
/// value "as the arena will read" has to look through the chain.  Every
/// builder keeps its map acyclic; as defence in depth the walk is bounded by
/// the map size (an acyclic chain over `N` entries has at most `N` hops) and
/// a longer one debug-asserts and stops where it is, so a debug build fails
/// loudly and a release build degrades to a partial resolution instead of
/// hanging.
///
/// Never inlined: a dozen call sites, most of them closures, would each
/// carry a copy of the loop.
#[inline(never)]
pub(crate) fn follow(
    map: &HandleMap<naga::Expression, naga::Handle<naga::Expression>>,
    mut handle: naga::Handle<naga::Expression>,
) -> naga::Handle<naga::Expression> {
    let mut hops = map.len();
    while let Some(&next) = map.get(handle) {
        if hops == 0 {
            debug_assert!(
                false,
                "follow: cycle detected (chain exceeded {} hops); the pass \
                 produced a cyclic replacement map",
                map.len()
            );
            break;
        }
        handle = next;
        hops -= 1;
    }
    handle
}

/// Point every key of `replacements` at its terminal target
/// ([`follow`]), so one [`try_map_expression_handles_in_place`] walk, which
/// resolves a single level, applies the whole chain; callers MUST flatten
/// before applying a map or dangling references survive.
pub(crate) fn flatten_replacement_chains(
    replacements: &mut HandleMap<naga::Expression, naga::Handle<naga::Expression>>,
) {
    let keys: Vec<_> = replacements.keys().copied().collect();
    for key in keys {
        let target = follow(replacements, key);
        replacements.insert(key, target);
    }
}

/// One replacement map applied to a whole function: every reference to a
/// key, in the arena and in the statements, is pointed at the end of its
/// chain ([`follow`]).  The two switches are the two things the three
/// forwarding passes do differently, named so the difference is a line and
/// not a fourth walk.
pub(crate) struct Rewrite<'m> {
    pub(crate) map: &'m HandleMap<naga::Expression, naga::Handle<naga::Expression>>,
    /// Rewrite an arena use only where its target precedes it: naga forbids
    /// a forward reference, and a forward whose target trails the use is
    /// left to read the original.  Off for a map whose targets always
    /// precede (a splice's values are cloned ahead of the call).
    pub(crate) backward_only: bool,
    /// The keys stop being expressions of their own: their `Emit` slots and
    /// names go.  Off where an arena rebuild follows and drops them anyway,
    /// or where the dead-expression cleanup is left to a later sweep.
    pub(crate) retire_replaced: bool,
}

impl Rewrite<'_> {
    pub(crate) fn apply(&self, function: &mut naga::Function) {
        let map = self.map;
        for (user, expr) in function.expressions.iter_mut() {
            let _ = try_map_expression_handles_in_place(expr, &mut |h| {
                let target = follow(map, h);
                Some(if self.backward_only && target >= user {
                    h
                } else {
                    target
                })
            });
        }
        remap_block_handles(&mut function.body, &mut |h| follow(map, h));
        if self.retire_replaced {
            rebuild_emit_ranges_after_removal(&mut function.body, &|h| map.contains_key(h));
            function
                .named_expressions
                .retain(|h, _| !map.contains_key(h));
        }
    }
}

// MARK: Arena rebuild

/// Rebuild `function.expressions` in emission order: only expressions
/// reachable from the body survive, each appended after its operands and
/// after everything emitted before it, with the body, the locals' inits,
/// and the named expressions remapped (names of dropped expressions go
/// too).  Callers rely on the ORDER as much as on the garbage collection: a
/// pass that appends a synthesized expression puts it at the END of the
/// arena, behind consumers that already exist, and a later pass forwarding
/// it into one of them would create a forward reference naga rejects.
pub fn rebuild_function_expressions(function: &mut naga::Function) {
    let old_expressions = std::mem::take(&mut function.expressions);
    let mut new_expressions = naga::Arena::new();
    let mut handle_map = HandleMap::default();

    // A declarative init (`var c = BG;`) sits in no `Emit` range, so the
    // body walk never reaches it; cloned after the body it would land
    // BEHIND every consumer of the local, and a later store-to-load forward
    // of the init value would read as a forward reference and be declined.
    // An emitted init (`var x = OV * 3.0;`) is cloned by the body walk at
    // its own `Emit`, ahead of the local's loads; cloning it here would
    // leave the init on an un-emitted duplicate, which a forward then hands
    // to a statement and naga rejects as out of scope.
    let mut emitted_inits = Vec::new();
    for (lh, local) in function.local_variables.iter_mut() {
        if let Some(init) = &mut local.init {
            if expression_needs_emit(&old_expressions[*init]) {
                emitted_inits.push(lh);
            } else {
                *init = clone_expression_handle(
                    *init,
                    &old_expressions,
                    &mut new_expressions,
                    &mut handle_map,
                );
            }
        }
    }

    rebuild_block_expressions(
        &mut function.body,
        &old_expressions,
        &mut new_expressions,
        &mut handle_map,
    );

    for lh in emitted_inits {
        if let Some(init) = &mut function.local_variables[lh].init {
            *init = clone_expression_handle(
                *init,
                &old_expressions,
                &mut new_expressions,
                &mut handle_map,
            );
        }
    }

    let named = std::mem::take(&mut function.named_expressions);
    function.named_expressions = named
        .into_iter()
        .filter_map(|(h, name)| handle_map.get(h).map(|&m| (m, name)))
        .collect();

    function.expressions = new_expressions;
}

/// [`rebuild_function_expressions`] for one block: every expression the
/// block references is cloned from `old_expressions` into `new_expressions`
/// through `handle_map`, in emission order.  A map pre-seeded with an entry
/// for a declarative expression substitutes the entry for that expression,
/// which is how a spliced callee's parameter reads become the caller's
/// arguments; `Emit` ranges must not be pre-seeded (an emitted expression is
/// cloned exactly once, at its own `Emit`).
pub(crate) fn rebuild_block_expressions(
    block: &mut naga::Block,
    old_expressions: &naga::Arena<naga::Expression>,
    new_expressions: &mut naga::Arena<naga::Expression>,
    handle_map: &mut HandleMap<naga::Expression, naga::Handle<naga::Expression>>,
) {
    rewrite_statements(block, &mut |mut statement, span, rebuilt| {
        // Cloning a child may append a non-emittable dependency between two
        // emitted handles, so the range is split around it.
        if let naga::Statement::Emit(ref range) = statement {
            let mut mapped_handles = Vec::new();
            for handle in range.clone() {
                // A clone from an earlier walk would leave this copy dead
                // and the map on the wrong one.
                debug_assert!(!handle_map.contains_key(handle));
                let mut expression = old_expressions[handle].clone();
                let _ = try_map_expression_handles_in_place(&mut expression, &mut |child| {
                    Some(clone_expression_handle(
                        child,
                        old_expressions,
                        new_expressions,
                        handle_map,
                    ))
                });
                let mapped = new_expressions.append(expression, old_expressions.get_span(handle));
                handle_map.insert(handle, mapped);
                mapped_handles.push(mapped);
            }

            push_emit_runs(rebuilt, &mapped_handles, span);
            return;
        }

        // A loop's `break_if` is emitted inside `body` / `continuing`, so
        // those are rebuilt first and the `break_if` clone memo-hits the copy
        // its owning Emit produced; the generic remap would clone it first,
        // appending an un-emitted duplicate that sits in no Emit range.
        if matches!(statement, naga::Statement::Loop { .. }) {
            if let naga::Statement::Loop {
                body,
                continuing,
                break_if,
            } = &mut statement
            {
                rebuild_block_expressions(body, old_expressions, new_expressions, handle_map);
                rebuild_block_expressions(continuing, old_expressions, new_expressions, handle_map);
                if let Some(handle) = break_if {
                    *handle = clone_expression_handle(
                        *handle,
                        old_expressions,
                        new_expressions,
                        handle_map,
                    );
                }
            }
            rebuilt.push(statement, span);
            return;
        }

        remap_statement_handles(&mut statement, &mut |h| {
            clone_expression_handle(h, old_expressions, new_expressions, handle_map)
        });

        debug_assert!(!matches!(statement, naga::Statement::Loop { .. }));
        for nested in nested_blocks_mut(&mut statement) {
            rebuild_block_expressions(nested, old_expressions, new_expressions, handle_map);
        }

        rebuilt.push(statement, span);
    });
}

/// Clone `handle` with its cone from `old_expressions` into
/// `new_expressions`, memoised in `handle_map` so a shared sub-DAG stays
/// shared and a pre-seeded entry stands in for its expression.
pub(crate) fn clone_expression_handle(
    handle: naga::Handle<naga::Expression>,
    old_expressions: &naga::Arena<naga::Expression>,
    new_expressions: &mut naga::Arena<naga::Expression>,
    handle_map: &mut HandleMap<naga::Expression, naga::Handle<naga::Expression>>,
) -> naga::Handle<naga::Expression> {
    if let Some(mapped) = handle_map.get(handle).copied() {
        return mapped;
    }

    let mut expression = old_expressions[handle].clone();
    let _ = try_map_expression_handles_in_place(&mut expression, &mut |child| {
        Some(clone_expression_handle(
            child,
            old_expressions,
            new_expressions,
            handle_map,
        ))
    });

    let mapped = new_expressions.append(expression, old_expressions.get_span(handle));
    handle_map.insert(handle, mapped);
    mapped
}

// MARK: Tests

#[cfg(test)]
mod tests {
    use super::*;

    /// `n` expression handles, so replacement maps in tests use real keys.
    fn expr_handles(n: usize) -> Vec<naga::Handle<naga::Expression>> {
        let mut arena = naga::Arena::new();
        (0..n)
            .map(|i| {
                arena.append(
                    naga::Expression::Literal(naga::Literal::U32(i as u32)),
                    naga::Span::UNDEFINED,
                )
            })
            .collect()
    }

    /// `Loop { body: [Block [Break], Break], continuing: [Continue] }`,
    /// `Kill` after it: every shape of nesting the walks distinguish.
    fn nested_body() -> naga::Block {
        let inner = naga::Block::from_vec(vec![naga::Statement::Break]);
        let body =
            naga::Block::from_vec(vec![naga::Statement::Block(inner), naga::Statement::Break]);
        let continuing = naga::Block::from_vec(vec![naga::Statement::Continue]);
        naga::Block::from_vec(vec![
            naga::Statement::Loop {
                body,
                continuing,
                break_if: None,
            },
            naga::Statement::Kill,
        ])
    }

    #[test]
    fn rewrite_block_visits_bottom_up_with_the_statements_own_scope() {
        let mut block = nested_body();
        let mut seen = Vec::new();
        rewrite_block(
            &mut block,
            Scope::default(),
            &mut |stmt, span, out, scope| {
                seen.push((
                    match stmt {
                        naga::Statement::Block(_) => "block",
                        naga::Statement::Break => "break",
                        naga::Statement::Continue => "continue",
                        naga::Statement::Loop { .. } => "loop",
                        naga::Statement::Kill => "kill",
                        _ => "other",
                    },
                    scope.loop_depth,
                ));
                out.push(stmt, span);
            },
        );
        assert_eq!(
            seen,
            [
                ("break", 1),
                ("block", 1),
                ("break", 1),
                ("continue", 1),
                ("loop", 0),
                ("kill", 0)
            ]
        );
        assert_eq!(
            block.len(),
            2,
            "a rewrite that keeps everything changes nothing"
        );
    }

    #[test]
    fn rewrite_block_drops_and_expands_at_any_depth() {
        let mut block = nested_body();
        rewrite_block(
            &mut block,
            Scope::default(),
            &mut |stmt, span, out, _| match stmt {
                naga::Statement::Break => {}
                naga::Statement::Continue => {
                    out.push(naga::Statement::Kill, span);
                    out.push(naga::Statement::Kill, span);
                }
                other => out.push(other, span),
            },
        );
        let naga::Statement::Loop {
            body, continuing, ..
        } = &block[0]
        else {
            panic!("the loop survives");
        };
        assert_eq!(body.len(), 1, "the loop body keeps only the inner block");
        let naga::Statement::Block(inner) = &body[0] else {
            panic!("the inner block survives");
        };
        assert!(inner.is_empty(), "its break is dropped");
        assert_eq!(continuing.len(), 2, "one continue became two kills");
    }

    #[test]
    fn push_emit_runs_splits_on_a_gap_and_pushes_nothing_for_no_survivors() {
        let h = expr_handles(6);
        let mut block = naga::Block::new();
        push_emit_runs(&mut block, &[], naga::Span::UNDEFINED);
        assert!(block.is_empty());
        push_emit_runs(&mut block, &[h[0], h[1], h[3], h[5]], naga::Span::UNDEFINED);
        let runs: Vec<Vec<_>> = block
            .iter()
            .map(|s| match s {
                naga::Statement::Emit(r) => r.clone().collect(),
                _ => panic!("only emits"),
            })
            .collect();
        assert_eq!(runs, [vec![h[0], h[1]], vec![h[3]], vec![h[5]]]);
    }

    #[test]
    fn follow_reaches_the_chain_end_and_leaves_a_non_key_alone() {
        let h = expr_handles(5);
        let mut m = HandleMap::default();
        m.insert(h[0], h[1]);
        m.insert(h[1], h[2]);
        assert_eq!(follow(&m, h[0]), h[2]);
        assert_eq!(follow(&m, h[1]), h[2]);
        assert_eq!(follow(&m, h[2]), h[2]);
        assert_eq!(follow(&m, h[4]), h[4]);
    }

    #[test]
    fn flatten_replacement_chains_collapses_transitive_edges() {
        let h = expr_handles(7);
        let mut m = HandleMap::default();
        m.insert(h[1], h[2]);
        m.insert(h[2], h[3]);
        m.insert(h[3], h[4]);
        m.insert(h[5], h[6]);
        flatten_replacement_chains(&mut m);
        assert_eq!(m[h[1]], h[4]);
        assert_eq!(m[h[2]], h[4]);
        assert_eq!(m[h[3]], h[4]);
        assert_eq!(m[h[5]], h[6]);
    }

    #[test]
    fn flatten_replacement_chains_is_noop_on_direct_edges() {
        let h = expr_handles(4);
        let mut m = HandleMap::default();
        m.insert(h[0], h[2]);
        m.insert(h[1], h[3]);
        flatten_replacement_chains(&mut m);
        assert_eq!(m[h[0]], h[2]);
        assert_eq!(m[h[1]], h[3]);
    }

    /// Every caller builds an acyclic map; a cyclic one must still
    /// terminate (debug builds may assert, release builds return).
    #[test]
    fn flatten_replacement_chains_terminates_on_cycles() {
        for cycle in [2usize, 3] {
            let h = expr_handles(cycle);
            let mut m = HandleMap::default();
            for i in 0..cycle {
                m.insert(h[i], h[(i + 1) % cycle]);
            }
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                flatten_replacement_chains(&mut m);
            }));
        }
    }

    /// `var t: f32; t = 2.0; let a = t; out[0] = a + 1.0;` with the load of
    /// `t` forwarded to the literal it stores, plus a forward whose target
    /// trails its use.
    fn forwarding_fixture() -> (
        naga::Module,
        HandleMap<naga::Expression, naga::Handle<naga::Expression>>,
    ) {
        let src = r#"
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1) fn main() {
    var t: f32;
    t = 2.0;
    let a = t;
    out[0] = a + 1.0;
}
"#;
        let module = naga::front::wgsl::parse_str(src).expect("source should parse");
        let function = &module.entry_points[0].function;
        let load = function
            .expressions
            .iter()
            .find(|(_, e)| matches!(e, naga::Expression::Load { .. }))
            .map(|(h, _)| h)
            .expect("the load of t");
        let literal = function
            .expressions
            .iter()
            .find(
                |(_, e)| matches!(e, naga::Expression::Literal(naga::Literal::F32(v)) if *v == 2.0),
            )
            .map(|(h, _)| h)
            .expect("the stored literal");
        let mut map = HandleMap::default();
        map.insert(load, literal);
        (module, map)
    }

    fn function_of(module: &naga::Module) -> &naga::Function {
        &module.entry_points[0].function
    }

    #[test]
    fn rewrite_points_every_use_at_the_target_and_can_retire_the_key() {
        let (mut module, map) = forwarding_fixture();
        let (load, literal) = map.iter().next().map(|(h, &t)| (*h, t)).unwrap();
        Rewrite {
            map: &map,
            backward_only: true,
            retire_replaced: false,
        }
        .apply(&mut module.entry_points[0].function);
        let function = function_of(&module);
        let uses_literal = function
            .expressions
            .iter()
            .any(|(_, e)| matches!(e, naga::Expression::Binary { left, .. } if *left == literal));
        assert!(uses_literal, "the sum reads the literal, not the load");
        assert!(
            function.named_expressions.contains_key(&load),
            "without retirement the load keeps its name"
        );
        let emitted = function
            .body
            .iter()
            .any(|s| matches!(s, naga::Statement::Emit(r) if r.clone().any(|h| h == load)));
        assert!(emitted, "and its Emit slot");

        let (mut module, map) = forwarding_fixture();
        Rewrite {
            map: &map,
            backward_only: true,
            retire_replaced: true,
        }
        .apply(&mut module.entry_points[0].function);
        let function = function_of(&module);
        assert!(!function.named_expressions.contains_key(&load));
        let emitted = function
            .body
            .iter()
            .any(|s| matches!(s, naga::Statement::Emit(r) if r.clone().any(|h| h == load)));
        assert!(!emitted, "retirement drops the load out of its Emit");
    }

    #[test]
    fn backward_only_leaves_a_use_that_precedes_its_target() {
        let (mut module, _) = forwarding_fixture();
        let function = &mut module.entry_points[0].function;
        // Forward the literal `1.0` (an operand of the sum) to a handle
        // appended AFTER the sum: illegal as a forward reference.
        let one = function
            .expressions
            .iter()
            .find(
                |(_, e)| matches!(e, naga::Expression::Literal(naga::Literal::F32(v)) if *v == 1.0),
            )
            .map(|(h, _)| h)
            .expect("the literal 1.0");
        let late = function.expressions.append(
            naga::Expression::Literal(naga::Literal::F32(3.0)),
            naga::Span::UNDEFINED,
        );
        let mut map = HandleMap::default();
        map.insert(one, late);
        Rewrite {
            map: &map,
            backward_only: true,
            retire_replaced: false,
        }
        .apply(function);
        let still_one = function
            .expressions
            .iter()
            .any(|(_, e)| matches!(e, naga::Expression::Binary { right, .. } if *right == one));
        assert!(still_one, "the sum keeps reading `1.0`");
        Rewrite {
            map: &map,
            backward_only: false,
            retire_replaced: false,
        }
        .apply(function);
        let now_late = function
            .expressions
            .iter()
            .any(|(_, e)| matches!(e, naga::Expression::Binary { right, .. } if *right == late));
        assert!(now_late, "unguarded, the forward reference is written");
    }

    #[test]
    fn rebuild_keeps_one_copy_of_an_emitted_initializer() {
        // `x`'s init is emitted by the body, `c`'s is a declarative reference
        // no `Emit` covers: `x` must keep the emitted copy (an un-emitted
        // second one is what a later init forward hands to statements naga
        // rejects) and `c`'s must precede the local's loads so that forward
        // is not declined.
        let src = r#"
const BG: f32 = 2.0;
override OV: f32 = 3.0;
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1) fn main() {
    var x = OV * 3.0;
    var c = BG;
    out[0] = x + c;
}
"#;
        let mut module = naga::front::wgsl::parse_str(src).expect("source should parse");
        rebuild_function_expressions(&mut module.entry_points[0].function);
        let function = &module.entry_points[0].function;
        let products = function
            .expressions
            .iter()
            .filter(|(_, e)| {
                matches!(
                    e,
                    naga::Expression::Binary {
                        op: naga::BinaryOperator::Multiply,
                        ..
                    }
                )
            })
            .count();
        assert_eq!(products, 1, "an emitted init is cloned once");
        let init_of = |name: &str| {
            function
                .local_variables
                .iter()
                .find(|(_, l)| l.name.as_deref() == Some(name))
                .and_then(|(_, l)| l.init)
                .expect("local keeps its init")
        };
        let x_init = init_of("x");
        let emitted = function.body.iter().any(
            |s| matches!(s, naga::Statement::Emit(range) if range.clone().any(|h| h == x_init)),
        );
        assert!(emitted, "the init is the copy an `Emit` covers");
        let first_load = function
            .expressions
            .iter()
            .find(|(_, e)| matches!(e, naga::Expression::Load { .. }))
            .map(|(h, _)| h)
            .expect("the body loads its locals");
        assert!(
            init_of("c").index() < first_load.index(),
            "a declarative init precedes the loads"
        );
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("module stays valid after the rebuild");
    }
}
