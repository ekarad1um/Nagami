//! Function inlining for expression-only helpers: a body of the shape
//! `[Emit*, Return { value }]` with no locals, stores, calls, or other
//! side-effecting statements becomes an `InlineTemplate` expression DAG
//! cloned into each call site in place of the call.  Global / texture reads
//! in the body are fine; re-emission at the call site keeps their call-time
//! timing.  `max_node_count` caps a template's DAG, `max_call_sites` its
//! reuse, and `MAX_MULTI_SITE_EXPANSION` the nodes added by duplicating a
//! body across sites.

use crate::error::Error;
use crate::handle_set::{HandleMap, HandleSet};
use crate::pipeline::{Pass, PassContext};

use super::expr_util::{
    const_expression_leaf, const_index_value, expression_needs_emit, for_each_function_mut,
    has_negative_zero_leaf, is_disallowed_inline_expression, is_float_zero_literal,
    nested_blocks_mut, rebuild_function_expressions, remap_statement_handles,
    try_map_expression_handles_in_place, visit_expression_children, zero_value_is_float,
};

/// Const-ness and zero-reachability of a caller argument, in one walk.  A
/// `Constant` / `Override` hides its value from this pass, so it counts as a
/// possible zero, exactly as [`has_negative_zero_leaf`] treats it.
fn argument_facts(
    caller: &naga::Arena<naga::Expression>,
    types: &naga::UniqueArena<naga::Type>,
    root: naga::Handle<naga::Expression>,
) -> (bool, bool) {
    let (mut is_const, mut reaches_zero) = (true, false);
    let mut seen = HandleSet::default();
    let mut stack = vec![root];
    while let Some(h) = stack.pop() {
        if !seen.insert(h) {
            continue;
        }
        let expr = &caller[h];
        match expr {
            naga::Expression::Literal(lit) => reaches_zero |= is_float_zero_literal(lit),
            naga::Expression::ZeroValue(ty) => reaches_zero |= zero_value_is_float(types, *ty),
            naga::Expression::Constant(_) | naga::Expression::Override(_) => reaches_zero = true,
            _ => {}
        }
        match const_expression_leaf(expr) {
            Some(true) => {}
            Some(false) => is_const = false,
            None => visit_expression_children(expr, |c| stack.push(c)),
        }
    }
    (is_const, reaches_zero)
}

/// `true` when substituting `arguments` turns a float `-x`, `x * y` or
/// `x / y` the CALLEE left runtime into a const-expression with a zero in it.
/// A parameter read is runtime by construction, so inlining is the moment
/// that const-ness can appear; Dawn on Metal then flushes the `-0.0` the call
/// computed.  Same crossing test as `load_dedup`, rooted at the operator so a
/// zero already const in one operand is seen when the OTHER is what crosses -
/// `fn g(a: f32) -> f32 { return a * 0.; }` called as `g(-1.)`.
fn inlining_crosses_sign_sensitive(
    template: &InlineTemplate,
    arguments: &[naga::Handle<naga::Expression>],
    caller: &naga::Arena<naga::Expression>,
    types: &naga::UniqueArena<naga::Type>,
) -> bool {
    if template.sign_sensitive_ops.is_empty() {
        return false;
    }
    let n = template.expressions.len();
    // Const-ness with every parameter read left runtime (the call's own
    // meaning) against const-ness after substitution, plus whether a zero the
    // operator could re-sign is reachable.
    let (mut before, mut after, mut zero) = (vec![false; n], vec![false; n], vec![false; n]);
    for (handle, expr) in template.expressions.iter() {
        let i = handle.index();
        match expr {
            naga::Expression::FunctionArgument(idx) => {
                let (is_const, reaches_zero) = arguments
                    .get(*idx as usize)
                    .map_or((true, true), |&a| argument_facts(caller, types, a));
                (before[i], after[i], zero[i]) = (false, is_const, reaches_zero);
            }
            _ => match const_expression_leaf(expr) {
                Some(known) => {
                    (before[i], after[i]) = (known, known);
                    // A `ZeroValue` IS a zero; a `Constant` / `Override`
                    // hides its value, so both count, as in `load_dedup`.
                    zero[i] = match expr {
                        naga::Expression::Literal(lit) => is_float_zero_literal(lit),
                        naga::Expression::ZeroValue(ty) => zero_value_is_float(types, *ty),
                        naga::Expression::Constant(_) | naga::Expression::Override(_) => true,
                        _ => false,
                    };
                }
                None => {
                    let (mut b, mut a, mut z) = (true, true, false);
                    visit_expression_children(expr, |c| {
                        b &= before[c.index()];
                        a &= after[c.index()];
                        z |= zero[c.index()];
                    });
                    (before[i], after[i], zero[i]) = (b, a, z);
                }
            },
        }
    }
    template
        .sign_sensitive_ops
        .iter()
        .any(|op| after[op.index()] && !before[op.index()] && zero[op.index()])
}

/// The three operators whose zero result takes its sign from the operands.
fn is_sign_sensitive_op(expr: &naga::Expression) -> bool {
    matches!(
        expr,
        naga::Expression::Binary {
            op: naga::BinaryOperator::Multiply | naga::BinaryOperator::Divide,
            ..
        } | naga::Expression::Unary {
            op: naga::UnaryOperator::Negate,
            ..
        }
    )
}

/// Default inlining budgets (used by [`super::Profile::Aggressive`]).
pub const DEFAULT_MAX_INLINE_NODE_COUNT: usize = 24;
/// Default inlining call-site budget (used by [`super::Profile::Aggressive`]).
pub const DEFAULT_MAX_INLINE_CALL_SITES: usize = 3;
/// Widened node budget when running under [`super::Profile::Max`].
pub const MAX_PROFILE_MAX_INLINE_NODE_COUNT: usize = 48;
/// Widened call-site budget when running under [`super::Profile::Max`].
pub const MAX_PROFILE_MAX_INLINE_CALL_SITES: usize = 6;

/// Cap on `node_count * (call_sites - 1)`, the nodes added by duplicating a
/// template across call sites: post-mangle call syntax is 1-2 characters,
/// so cloning a non-trivial body over several sites is almost always a net
/// regression (single-site inlining adds zero).  Hard-coded rather than a
/// `Config` knob because it follows from the mangled call-site length, not
/// a profile choice; multi-sweep convergence catches single-site cases that
/// mature only after later simplification.
const MAX_MULTI_SITE_EXPANSION: usize = 6;

/// Inlining pass with per-run node and call-site budgets.
#[derive(Debug)]
pub struct InliningPass {
    max_node_count: usize,
    max_call_sites: usize,
}

impl Default for InliningPass {
    fn default() -> Self {
        Self {
            max_node_count: DEFAULT_MAX_INLINE_NODE_COUNT,
            max_call_sites: DEFAULT_MAX_INLINE_CALL_SITES,
        }
    }
}

impl InliningPass {
    /// A pass with explicit budgets; the profile picks among the
    /// `DEFAULT_*` / `MAX_PROFILE_*` constants.
    pub fn new(max_node_count: usize, max_call_sites: usize) -> Self {
        Self {
            max_node_count,
            max_call_sites,
        }
    }
}

/// An inlinable function's expression arena and return handle, cloned into
/// each caller.  `argument_types` lets the pre-clone OOB gate size an
/// `Access` base from the callee's declared parameter type, independent of
/// the caller argument's shape.
#[derive(Clone)]
struct InlineTemplate {
    argument_types: Vec<naga::Handle<naga::Type>>,
    return_expr: naga::Handle<naga::Expression>,
    expressions: naga::Arena<naga::Expression>,
    /// Float `-x` / `x * y` / `x / y` REACHABLE from `return_expr`: the only
    /// ones a clone reproduces, and empty for most callees, which is what
    /// keeps the per-call-site crossing test off the hot path.
    sign_sensitive_ops: Vec<naga::Handle<naga::Expression>>,
}

impl Pass for InliningPass {
    fn name(&self) -> &'static str {
        "function_inlining"
    }

    fn run(&mut self, module: &mut naga::Module, ctx: &PassContext<'_>) -> Result<bool, Error> {
        // A call to an empty function is a no-op, but the template machinery
        // excludes every void function, so without this `fn e(){} ... e();`
        // survives every pass, pinned alive by its own call sites.  Calls to
        // preserve-listed functions stay: an intact call site is the
        // cheapest proof their external-contract declaration stays live.
        let mut changed = delete_calls_to_empty_functions(module, &ctx.config.preserve_symbols);

        let templates = collect_inline_templates(
            module,
            self.max_node_count,
            self.max_call_sites,
            &ctx.config.preserve_symbols,
        );
        if templates.is_empty() {
            return Ok(changed);
        }

        let mut inlined = 0usize;
        let types = &module.types;
        for_each_function_mut(&mut module.functions, &mut module.entry_points, &mut |f| {
            inlined += inline_in_function(f, &templates, types);
        });
        changed |= inlined > 0;

        Ok(changed)
    }
}

/// Remove every `Call` to a function whose body is empty (or a lone bare
/// `return;`), so the next compaction can drop the callee itself.
fn delete_calls_to_empty_functions(module: &mut naga::Module, preserve: &[String]) -> bool {
    let empty: HandleSet<naga::Function> = module
        .functions
        .iter()
        .filter(|(_, f)| {
            let body_is_empty = matches!(
                f.body.iter().collect::<Vec<_>>().as_slice(),
                [] | [naga::Statement::Return { value: None }]
            );
            body_is_empty
                && f.result.is_none()
                && !f
                    .name
                    .as_deref()
                    .is_some_and(|n| preserve.iter().any(|p| p == n))
        })
        .map(|(h, _)| h)
        .collect();
    if empty.is_empty() {
        return false;
    }

    let mut changed = false;
    for_each_function_mut(&mut module.functions, &mut module.entry_points, &mut |f| {
        changed |= drop_empty_calls_in_block(&mut f.body, &empty);
    });
    changed
}

fn drop_empty_calls_in_block(block: &mut naga::Block, empty: &HandleSet<naga::Function>) -> bool {
    let original = std::mem::take(block);
    let mut rebuilt = naga::Block::with_capacity(original.len());
    let mut changed = false;
    for (mut stmt, span) in original.span_into_iter() {
        if let naga::Statement::Call {
            function,
            result: None,
            ..
        } = &stmt
            && empty.contains(function)
        {
            changed = true;
            continue;
        }
        for nested in nested_blocks_mut(&mut stmt) {
            changed |= drop_empty_calls_in_block(nested, empty);
        }
        rebuilt.push(stmt, span);
    }
    *block = rebuilt;
    changed
}

// MARK: Template collection

/// Build an [`InlineTemplate`] for every eligible function.  Eligibility is
/// the soundness invariant: a body of exactly `[Emit*, Return { value }]`
/// with no locals, stores, calls, or other side-effecting statements, so the
/// expression DAG clones into a caller without disturbing any assumption
/// about stores or control flow between statements.
fn collect_inline_templates(
    module: &naga::Module,
    max_node_count: usize,
    max_call_sites: usize,
    preserve: &[String],
) -> HandleMap<naga::Function, InlineTemplate> {
    let mut templates = Default::default();
    // A library module (no entry points) keeps every function (compact runs
    // with `KeepUnused::Yes`), so inlining only trades a call for a second
    // copy of the body.
    if module.entry_points.is_empty() {
        return templates;
    }
    let call_counts = collect_call_counts(module);

    for (function_handle, function) in module.functions.iter() {
        // Preserved functions are never templates: a `--preamble` input
        // carries only a STUB body whose real definition arrives when the
        // consumer concatenates the preamble, so baking the stub into callers
        // bypasses that definition, and a plain `--preserve-symbol`
        // declaration survives only through intact call sites once entry
        // points exist.
        if function
            .name
            .as_deref()
            .is_some_and(|n| preserve.iter().any(|p| p == n))
        {
            continue;
        }
        let call_sites = call_counts.get(function_handle).copied().unwrap_or(0);
        if call_sites == 0 || call_sites > max_call_sites {
            continue;
        }
        if !function.local_variables.is_empty() || function.result.is_none() {
            continue;
        }

        let Some(return_expr) = extract_inline_return_expression(&function.body) else {
            continue;
        };

        // Dense over the callee's arena and re-checked at every node, so a
        // `Vec<bool>` beats a hash set.
        let mut visited = vec![false; function.expressions.len()];
        let Some(node_count) = analyze_inline_expression(
            return_expr,
            &function.expressions,
            function.arguments.len(),
            &mut visited,
        ) else {
            continue;
        };

        if node_count == 0 || node_count > max_node_count {
            continue;
        }

        if call_sites > 1 {
            let expansion = node_count * (call_sites - 1);
            if expansion > MAX_MULTI_SITE_EXPANSION {
                continue;
            }
            // Node counts undercount TEXT: a `Math` node renders its full
            // builtin name (`faceForward(` is 12 characters), so duplicating
            // one across sites grows bytes inside the node budget.  The
            // minority case though - the veto measures net-negative on both
            // corpora, and wants replacing by a per-node text-cost estimate
            // rather than a veto on the node KIND.
            let has_char_heavy_node = function.expressions.iter().any(|(h, e)| {
                visited[h.index()]
                    && matches!(
                        e,
                        naga::Expression::Math { .. }
                            | naga::Expression::ImageSample { .. }
                            | naga::Expression::ImageLoad { .. }
                            | naga::Expression::ImageQuery { .. }
                    )
            });
            if has_char_heavy_node {
                continue;
            }
        }

        templates.insert(
            function_handle,
            InlineTemplate {
                argument_types: function.arguments.iter().map(|a| a.ty).collect(),
                return_expr,
                sign_sensitive_ops: function
                    .expressions
                    .iter()
                    .filter(|(h, e)| visited[h.index()] && is_sign_sensitive_op(e))
                    .map(|(h, _)| h)
                    .collect(),
                expressions: function.expressions.clone(),
            },
        );
    }

    templates
}

fn collect_call_counts(module: &naga::Module) -> HandleMap<naga::Function, usize> {
    let mut counts = Default::default();

    for (_, function) in module.functions.iter() {
        collect_call_counts_in_block(&function.body, &mut counts);
    }
    for entry in module.entry_points.iter() {
        collect_call_counts_in_block(&entry.function.body, &mut counts);
    }

    counts
}

fn collect_call_counts_in_block(
    block: &naga::Block,
    counts: &mut HandleMap<naga::Function, usize>,
) {
    super::expr_util::for_each_statement(block, &mut |statement| {
        if let naga::Statement::Call { function, .. } = statement {
            *counts.entry(*function).or_insert(0) += 1;
        }
    });
}

/// The value handle of a body shaped exactly `[Emit*, Return { value }]`,
/// `None` for anything else: the pass's single purity gate.
fn extract_inline_return_expression(block: &naga::Block) -> Option<naga::Handle<naga::Expression>> {
    let mut return_value = None;
    let mut seen_return = false;

    for statement in block {
        match statement {
            naga::Statement::Emit(_) if !seen_return => {}
            naga::Statement::Return { value: Some(value) } if !seen_return => {
                return_value = Some(*value);
                seen_return = true;
            }
            _ => return None,
        }
    }

    return_value
}

/// Node count of the DAG under `handle`, or `None` when it holds an
/// inline-disallowed expression or an out-of-range `FunctionArgument`.
fn analyze_inline_expression(
    handle: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
    argument_count: usize,
    visited: &mut [bool],
) -> Option<usize> {
    let slot = visited.get_mut(handle.index())?;
    if *slot {
        return Some(0);
    }
    *slot = true;

    let expr = &expressions[handle];
    if is_disallowed_inline_expression(expr) {
        return None;
    }

    if let naga::Expression::FunctionArgument(index) = expr {
        return ((*index as usize) < argument_count).then_some(1);
    }

    // `ok` stops descending siblings after a failure, so the `visited` marks
    // match an early-return walk exactly.
    let mut total = 1usize;
    let mut ok = true;
    visit_expression_children(expr, |child| {
        if !ok {
            return;
        }
        match analyze_inline_expression(child, expressions, argument_count, visited) {
            Some(n) => total += n,
            None => ok = false,
        }
    });
    ok.then_some(total)
}

// MARK: Call-site rewriting

/// Replace every eligible call in `function` with its template's cloned
/// return expression; returns the number of call sites rewritten.
fn inline_in_function(
    function: &mut naga::Function,
    templates: &HandleMap<naga::Function, InlineTemplate>,
    types: &naga::UniqueArena<naga::Type>,
) -> usize {
    let arena_len = function.expressions.len();
    let (changed, _) = inline_in_block(
        &mut function.body,
        &mut function.expressions,
        templates,
        types,
        &mut Default::default(),
    );

    if changed > 0 {
        rebuild_function_expressions(function);
        function.named_expressions.clear();
    } else if function.expressions.len() > arena_len {
        // Only declined clones appended anything; a pass that changes
        // nothing must leave the arena as it found it (the driver replays
        // on that).
        truncate_expressions(&mut function.expressions, arena_len);
    }

    changed
}

/// `naga::Arena` has no truncate; drain and re-append the prefix (handles
/// are positional, so they survive unchanged).
fn truncate_expressions(arena: &mut naga::Arena<naga::Expression>, len: usize) {
    let kept: Vec<_> = arena
        .drain()
        .take(len)
        .map(|(_, expression, span)| (expression, span))
        .collect();
    for (expression, span) in kept {
        arena.append(expression, span);
    }
}

/// Inline eligible calls in `block`, returning the change count and the
/// `CallResult` keys this scope added to the shared `replacements` map.  A
/// `CallResult` belongs to one call in one block, so a scope's keys are new
/// and its caller restores the map by removing exactly them - cheaper than
/// handing every nested block its own copy.  naga lets a loop's `continuing`
/// block and `break_if` reference body-defined expressions, so the loop
/// keeps its body's keys live until both are rewritten; otherwise a
/// body-inlined call leaves them pointing at an orphaned `CallResult`
/// (invalid IR, a whole-module rollback).
fn inline_in_block(
    block: &mut naga::Block,
    expressions: &mut naga::Arena<naga::Expression>,
    templates: &HandleMap<naga::Function, InlineTemplate>,
    types: &naga::UniqueArena<naga::Type>,
    replacements: &mut HandleMap<naga::Expression, naga::Handle<naga::Expression>>,
) -> (usize, Vec<naga::Handle<naga::Expression>>) {
    let mut changed = 0usize;
    let mut added = Vec::new();

    let original = std::mem::take(block);
    let mut rebuilt = naga::Block::with_capacity(original.len());

    for (mut statement, span) in original.span_into_iter() {
        apply_replacements_to_statement(&mut statement, expressions, replacements);

        match statement {
            naga::Statement::Call {
                function,
                arguments,
                result: Some(result_handle),
            } => {
                if let Some(template) = templates.get(function)
                    && template.argument_types.len() == arguments.len()
                    && !arguments
                        .iter()
                        .any(|&a| has_negative_zero_leaf(expressions, a))
                    && !inlining_crosses_sign_sensitive(template, &arguments, expressions, types)
                {
                    let old_len = expressions.len();
                    // Indexed by template handle: the template arena is the
                    // densest address space and the memo lookup is the hot
                    // spot of the clone.
                    let mut memo: Vec<Option<naga::Handle<naga::Expression>>> =
                        vec![None; template.expressions.len()];
                    if let Some(root_handle) = clone_inline_expression(
                        template.return_expr,
                        template,
                        &arguments,
                        expressions,
                        types,
                        &mut memo,
                    ) {
                        push_emit_ranges_for_new_expressions(
                            &mut rebuilt,
                            expressions,
                            old_len,
                            span,
                        );

                        let shadowed = replacements.insert(result_handle, root_handle);
                        debug_assert!(shadowed.is_none(), "call results are unique per block");
                        added.push(result_handle);
                        changed += 1;
                        continue;
                    }
                }

                rebuilt.push(
                    naga::Statement::Call {
                        function,
                        arguments,
                        result: Some(result_handle),
                    },
                    span,
                );
            }
            naga::Statement::Loop {
                mut body,
                mut continuing,
                mut break_if,
            } => {
                let (cb, body_added) =
                    inline_in_block(&mut body, expressions, templates, types, replacements);
                changed += cb;
                // `break_if` may also reference a `CallResult` defined in
                // `continuing` itself, so both scopes stay in the map until it
                // is rewritten.
                let (cc, continuing_added) =
                    inline_in_block(&mut continuing, expressions, templates, types, replacements);
                changed += cc;
                if let Some(handle) = break_if {
                    break_if = Some(resolve_replacement(handle, replacements));
                }
                for key in body_added.into_iter().chain(continuing_added) {
                    replacements.remove(key);
                }
                rebuilt.push(
                    naga::Statement::Loop {
                        body,
                        continuing,
                        break_if,
                    },
                    span,
                );
            }
            // Nothing after an If / Switch / Block may reference its interior
            // expressions, so their maps are dropped.
            mut other => {
                for nested in nested_blocks_mut(&mut other) {
                    let (c, nested_added) =
                        inline_in_block(nested, expressions, templates, types, replacements);
                    changed += c;
                    for key in nested_added {
                        replacements.remove(key);
                    }
                }
                rebuilt.push(other, span);
            }
        }
    }

    *block = rebuilt;
    (changed, added)
}

/// `Emit` statements covering every expression appended after `old_len`
/// that needs one, split around declarative expressions.
fn push_emit_ranges_for_new_expressions(
    block: &mut naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    old_len: usize,
    span: naga::Span,
) {
    if expressions.len() <= old_len {
        return;
    }

    let mut start = None;
    let mut end = None;

    for handle in expressions.range_from(old_len) {
        if expression_needs_emit(&expressions[handle]) {
            if start.is_none() {
                start = Some(handle);
            }
            end = Some(handle);
        } else if let (Some(first), Some(last)) = (start.take(), end.take()) {
            block.push(
                naga::Statement::Emit(naga::Range::new_from_bounds(first, last)),
                span,
            );
        }
    }

    if let (Some(first), Some(last)) = (start, end) {
        block.push(
            naga::Statement::Emit(naga::Range::new_from_bounds(first, last)),
            span,
        );
    }
}

/// Clone `handle` from the template into `caller_expressions`, substituting
/// `arguments` for `FunctionArgument`s; `memo` maps each template handle to
/// exactly one caller handle so shared sub-DAGs stay shared.  Every handle
/// reached comes from the template arena, so `handle.index() < memo.len()`
/// holds and both memo accesses index directly, panicking at the offending
/// site on a violation.
fn clone_inline_expression(
    handle: naga::Handle<naga::Expression>,
    template: &InlineTemplate,
    arguments: &[naga::Handle<naga::Expression>],
    caller_expressions: &mut naga::Arena<naga::Expression>,
    types: &naga::UniqueArena<naga::Type>,
    memo: &mut [Option<naga::Handle<naga::Expression>>],
) -> Option<naga::Handle<naga::Expression>> {
    if let Some(mapped) = memo[handle.index()] {
        return Some(mapped);
    }

    let expr = &template.expressions[handle];
    if is_disallowed_inline_expression(expr) {
        return None;
    }

    let mapped = match expr {
        naga::Expression::FunctionArgument(index) => arguments.get(*index as usize).copied()?,
        _ => {
            // Substitution can MANUFACTURE a statically out-of-bounds index
            // the template never had (`v[i]` called as `f(vec4(...), 6)`),
            // and naga rejects a known-OOB constant index into a fixed-size
            // composite, or a NEGATIVE one regardless of base type, rolling
            // the whole pass back.  Checked BEFORE cloning children so the
            // decline strands nothing: the index resolves through the
            // argument list, the base length through the callee's declared
            // parameter type or a structural template composite.
            if let naga::Expression::Access { base, index } = expr
                && let Some(i) =
                    substituted_index_value(*index, template, arguments, caller_expressions)
                && (i < 0
                    || substituted_base_len(*base, template, types)
                        .is_some_and(|len| i as u64 >= len))
            {
                return None;
            }
            let mut cloned = expr.clone();
            try_map_expression_handles_in_place(&mut cloned, &mut |child| {
                clone_inline_expression(child, template, arguments, caller_expressions, types, memo)
            })?;
            // Backstop for shapes the pre-clone gate cannot size (base or
            // index built from nested template expressions); the cloned
            // children stay behind as dead entries for the rebuild /
            // truncation to remove.
            if let naga::Expression::Access { base, index } = cloned
                && let Some(i) = const_index_value(index, caller_expressions)
                && (i < 0
                    || static_composite_len(base, caller_expressions, types)
                        .is_some_and(|len| i as u64 >= len))
            {
                return None;
            }
            caller_expressions.append(cloned, Default::default())
        }
    };

    memo[handle.index()] = Some(mapped);
    Some(mapped)
}

/// Static element count of `ty` (vector size, matrix columns, fixed array
/// length); `None` means no static bound and the OOB gates stand aside.
fn type_element_count(
    ty: naga::Handle<naga::Type>,
    types: &naga::UniqueArena<naga::Type>,
) -> Option<u64> {
    match &types[ty].inner {
        naga::TypeInner::Vector { size, .. } => Some(*size as u64),
        naga::TypeInner::Matrix { columns, .. } => Some(*columns as u64),
        naga::TypeInner::Array {
            size: naga::ArraySize::Constant(n),
            ..
        } => Some(n.get() as u64),
        _ => None,
    }
}

/// Static element count of the composite VALUE at `handle`, read
/// structurally (no typifier): `Compose` / `ZeroValue` carry their type,
/// `Splat` / `Swizzle` their size.
fn static_composite_len(
    handle: naga::Handle<naga::Expression>,
    arena: &naga::Arena<naga::Expression>,
    types: &naga::UniqueArena<naga::Type>,
) -> Option<u64> {
    match &arena[handle] {
        naga::Expression::Compose { ty, .. } | naga::Expression::ZeroValue(ty) => {
            type_element_count(*ty, types)
        }
        naga::Expression::Splat { size, .. } => Some(*size as u64),
        naga::Expression::Swizzle { size, .. } => Some(*size as u64),
        _ => None,
    }
}

/// Post-substitution constant value of a template `Access` index, resolved
/// WITHOUT cloning: a template literal, or the caller argument a
/// `FunctionArgument` maps to.  (`Constant` indices need module access and
/// stay unresolvable here.)
fn substituted_index_value(
    index: naga::Handle<naga::Expression>,
    template: &InlineTemplate,
    arguments: &[naga::Handle<naga::Expression>],
    caller_expressions: &naga::Arena<naga::Expression>,
) -> Option<i64> {
    match &template.expressions[index] {
        naga::Expression::FunctionArgument(k) => {
            const_index_value(*arguments.get(*k as usize)?, caller_expressions)
        }
        naga::Expression::Literal(_) => const_index_value(index, &template.expressions),
        _ => None,
    }
}

/// Static element count of a template `Access` base: the callee's declared
/// parameter type for an argument (whatever the caller passes), else a
/// structural template composite.
fn substituted_base_len(
    base: naga::Handle<naga::Expression>,
    template: &InlineTemplate,
    types: &naga::UniqueArena<naga::Type>,
) -> Option<u64> {
    match &template.expressions[base] {
        naga::Expression::FunctionArgument(m) => {
            type_element_count(*template.argument_types.get(*m as usize)?, types)
        }
        _ => static_composite_len(base, &template.expressions, types),
    }
}

/// Remap the handles `statement` references (its Emit'd expressions'
/// operands and its own fields); nested blocks are the caller's.
fn apply_replacements_to_statement(
    statement: &mut naga::Statement,
    expressions: &mut naga::Arena<naga::Expression>,
    replacements: &HandleMap<naga::Expression, naga::Handle<naga::Expression>>,
) {
    let mut remap =
        |handle: naga::Handle<naga::Expression>| resolve_replacement(handle, replacements);
    if let naga::Statement::Emit(range) = statement {
        for handle in range.clone() {
            let expression = expressions.get_mut(handle);
            let _ = try_map_expression_handles_in_place(expression, &mut |h| Some(remap(h)));
        }
        return;
    }
    remap_statement_handles(statement, &mut remap);
}

/// Terminal target of `handle` through `replacements`; self-loops stop, and
/// callers build acyclic maps, so it halts.
fn resolve_replacement(
    mut handle: naga::Handle<naga::Expression>,
    replacements: &HandleMap<naga::Expression, naga::Handle<naga::Expression>>,
) -> naga::Handle<naga::Expression> {
    while let Some(next) = replacements.get(handle).copied() {
        if next == handle {
            break;
        }
        handle = next;
    }
    handle
}

// MARK: Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn run_pass(source: &str) -> (bool, naga::Module) {
        let mut module = naga::front::wgsl::parse_str(source).expect("source should parse");
        let mut pass = InliningPass::default();
        let config = Config::default();
        let ctx = PassContext {
            config: &config,
            name_log: None,
        };

        let changed = pass
            .run(&mut module, &ctx)
            .expect("inlining pass should run");
        let _ = crate::io::validate_module(&module).expect("module should remain valid");
        (changed, module)
    }

    fn find_function_handle_by_name(
        module: &naga::Module,
        name: &str,
    ) -> naga::Handle<naga::Function> {
        module
            .functions
            .iter()
            .find_map(|(handle, function)| {
                (function.name.as_deref() == Some(name)).then_some(handle)
            })
            .expect("function should exist")
    }

    fn count_calls_to_function(block: &naga::Block, target: naga::Handle<naga::Function>) -> usize {
        let mut count = 0usize;
        super::super::expr_util::for_each_statement(block, &mut |statement| {
            if let naga::Statement::Call { function, .. } = statement
                && *function == target
            {
                count += 1;
            }
        });
        count
    }

    #[test]
    fn inlines_simple_return_expression_function() {
        let source = r#"
fn helper(x: f32) -> f32 {
    return x + 1.0;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    let y = helper(2.0);
    return vec4f(y, y, y, 1.0);
}
"#;

        let before = naga::front::wgsl::parse_str(source).expect("source should parse");
        let helper_before = find_function_handle_by_name(&before, "helper");
        let before_calls =
            count_calls_to_function(&before.entry_points[0].function.body, helper_before);
        assert_eq!(before_calls, 1, "expected one helper call before inlining");

        let (changed, after) = run_pass(source);
        assert!(changed, "inlining should report a change");

        let helper_after = find_function_handle_by_name(&after, "helper");
        let after_calls =
            count_calls_to_function(&after.entry_points[0].function.body, helper_after);
        assert_eq!(
            after_calls, 0,
            "helper call should be removed from entry point"
        );
    }

    #[test]
    fn library_module_keeps_call_sites_intact() {
        // No entry point: every function survives compaction, so inlining
        // would leave two copies of the body.
        let src = r#"
fn helper() -> vec4f { return vec4f(0.1, 0.2, 0.3, 1.0); }
fn main_color() -> vec4f { return helper(); }
"#;
        let (changed, after) = run_pass(src);
        assert!(!changed, "a library module must not inline");
        let helper = find_function_handle_by_name(&after, "helper");
        let caller = after
            .functions
            .iter()
            .find(|(_, f)| f.name.as_deref() == Some("main_color"))
            .map(|(_, f)| f)
            .expect("main_color survives");
        assert_eq!(count_calls_to_function(&caller.body, helper), 1);
    }

    /// A call-site literal substituted into a template's runtime `v[i]` can
    /// manufacture a statically out-of-bounds index naga rejects; the clone
    /// must decline (call kept) instead of rolling the whole pass back.
    /// `run_pass` validates, and the in-bounds call proves the gate is
    /// value-sensitive.
    #[test]
    fn declines_call_site_whose_literal_index_is_out_of_bounds() {
        let source = r#"
fn pick(v: vec4<f32>, i: i32) -> f32 {
    return v[i];
}

@compute @workgroup_size(1)
fn main() {
    let bad = pick(vec4<f32>(2.0, 3.0, 4.0, 5.0), 6);
    let good = pick(vec4<f32>(2.0, 3.0, 4.0, 5.0), 2);
    _ = bad + good;
}
"#;
        let (changed, after) = run_pass(source);
        assert!(changed, "the in-bounds call site must still inline");
        let pick = find_function_handle_by_name(&after, "pick");
        assert_eq!(
            count_calls_to_function(&after.entry_points[0].function.body, pick),
            1,
            "exactly the OOB call site must survive as a call"
        );
    }

    /// The pre-clone gate must size the base from the callee's DECLARED
    /// parameter type (a forwarded argument has no structural composite in
    /// the caller arena) and reject a negative index outright.
    /// Dawn on Metal flushes a negative-zero LITERAL to +0 but keeps the
    /// sign of a runtime negation, so `-(x*x)` must not be inlined with
    /// `x = -0.0` substituted; the other site still inlines.
    #[test]
    fn declines_call_site_passing_a_negative_zero_literal() {
        let source = "fn sq(x: f32) -> f32 { return -(x * x); }\n\
                      @group(0) @binding(0) var<storage, read_write> out: array<u32>;\n\
                      @compute @workgroup_size(1) fn main() {\n\
                        out[0] = bitcast<u32>(sq(-0.0));\n\
                        out[1] = bitcast<u32>(sq(2.0));\n\
                      }";
        let (changed, module) = run_pass(source);
        assert!(changed);
        let sq = find_function_handle_by_name(&module, "sq");
        assert_eq!(
            count_calls_to_function(&module.entry_points[0].function.body, sq),
            1
        );
    }

    #[test]
    fn declines_forwarded_argument_base_and_negative_index() {
        let source = r#"
fn pick(v: vec4<f32>, i: i32) -> f32 {
    return v[i];
}

fn outer(v: vec4<f32>) -> f32 {
    return pick(v, 6);
}

@compute @workgroup_size(1)
fn main() {
    var w = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    let neg = pick(w, -1);
    _ = outer(vec4<f32>(0.5, 0.5, 0.5, 0.5)) + neg;
}
"#;
        let (_, after) = run_pass(source);
        let pick = find_function_handle_by_name(&after, "pick");
        let mut surviving_calls = 0;
        for (_, func) in after.functions.iter() {
            surviving_calls += count_calls_to_function(&func.body, pick);
        }
        surviving_calls += count_calls_to_function(&after.entry_points[0].function.body, pick);
        assert_eq!(
            surviving_calls, 2,
            "both statically-invalid call sites must survive as calls"
        );
    }

    #[test]
    fn skips_function_with_local_variables() {
        let source = r#"
fn helper(x: f32) -> f32 {
    var t: f32;
    t = x + 1.0;
    return t;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    let y = helper(2.0);
    return vec4f(y, y, y, 1.0);
}
"#;

        let before = naga::front::wgsl::parse_str(source).expect("source should parse");
        let helper_before = find_function_handle_by_name(&before, "helper");
        let before_calls =
            count_calls_to_function(&before.entry_points[0].function.body, helper_before);
        assert_eq!(before_calls, 1, "expected one helper call before inlining");

        let (changed, after) = run_pass(source);
        assert!(!changed, "helper with locals should not be inlined");

        let helper_after = find_function_handle_by_name(&after, "helper");
        let after_calls =
            count_calls_to_function(&after.entry_points[0].function.body, helper_after);
        assert_eq!(
            after_calls, 1,
            "helper call should remain when inlining is skipped"
        );
    }

    #[test]
    fn skips_function_with_too_many_call_sites() {
        let source = r#"
fn helper(x: f32) -> f32 {
    return x + 1.0;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    let a = helper(1.0);
    let b = helper(2.0);
    let c = helper(3.0);
    let d = helper(4.0);
    return vec4f(a + b + c + d, 0.0, 0.0, 1.0);
}
"#;

        let before = naga::front::wgsl::parse_str(source).expect("source should parse");
        let helper_before = find_function_handle_by_name(&before, "helper");
        let before_calls =
            count_calls_to_function(&before.entry_points[0].function.body, helper_before);
        assert_eq!(
            before_calls, 4,
            "expected four helper calls before inlining"
        );

        let (changed, after) = run_pass(source);
        assert!(
            !changed,
            "helper should not be inlined when call-site count exceeds threshold"
        );

        let helper_after = find_function_handle_by_name(&after, "helper");
        let after_calls =
            count_calls_to_function(&after.entry_points[0].function.body, helper_after);
        assert_eq!(after_calls, 4, "all helper calls should remain");
    }

    #[test]
    fn inlining_with_local_init_preserves_valid_handles() {
        let source = r#"
fn helper(x: f32) -> f32 {
    return x + 1.0;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    var tmp: f32;
    let y = helper(2.0);
    tmp = y;
    return vec4f(tmp, 0.0, 0.0, 1.0);
}
"#;
        let mut module = naga::front::wgsl::parse_str(source).expect("source should parse");
        let ep_fn = &mut module.entry_points[0].function;

        let init_lit = ep_fn.expressions.append(
            naga::Expression::Literal(naga::Literal::F32(0.0)),
            naga::Span::UNDEFINED,
        );
        let local_handle = ep_fn
            .local_variables
            .iter()
            .next()
            .map(|(h, _)| h)
            .expect("expected a local variable");
        ep_fn.local_variables[local_handle].init = Some(init_lit);

        let mut pass = InliningPass::default();
        let config = Config::default();
        let ctx = PassContext {
            config: &config,
            name_log: None,
        };
        let changed = pass.run(&mut module, &ctx).expect("inlining should run");
        assert!(changed, "helper should be inlined");

        let ep_fn = &module.entry_points[0].function;
        let local = &ep_fn.local_variables[local_handle];
        assert!(local.init.is_some(), "local init should still be present");
        let init_handle = local.init.unwrap();
        assert!(
            init_handle.index() < ep_fn.expressions.len(),
            "local init handle ({}) should be within rebuilt expression arena (len={})",
            init_handle.index(),
            ep_fn.expressions.len(),
        );
        match ep_fn.expressions[init_handle] {
            naga::Expression::Literal(naga::Literal::F32(v)) => {
                assert!(
                    (v - 0.0).abs() < f32::EPSILON,
                    "init should still be 0.0, got {v}"
                );
            }
            ref other => panic!("init expression should be F32 literal, got {other:?}"),
        }
    }

    #[test]
    fn expansion_budget_rejects_large_multi_site_function() {
        // node_count 9, two sites: expansion 9 > 6.
        let source = r#"
fn helper(x: f32) -> f32 {
    return x + 1.0 + 2.0 + 3.0 + 4.0;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    let a = helper(1.0);
    let b = helper(2.0);
    return vec4f(a + b, 0.0, 0.0, 1.0);
}
"#;

        let before = naga::front::wgsl::parse_str(source).expect("source should parse");
        let helper_before = find_function_handle_by_name(&before, "helper");
        let before_calls =
            count_calls_to_function(&before.entry_points[0].function.body, helper_before);
        assert_eq!(before_calls, 2, "expected two helper calls before inlining");

        let (changed, after) = run_pass(source);
        assert!(
            !changed,
            "helper should NOT be inlined: expansion budget exceeded"
        );

        let helper_after = find_function_handle_by_name(&after, "helper");
        let after_calls =
            count_calls_to_function(&after.entry_points[0].function.body, helper_after);
        assert_eq!(after_calls, 2, "both helper calls should remain");
    }

    #[test]
    fn expansion_budget_allows_small_multi_site_function() {
        // node_count 3, two sites: expansion 3 <= 6.
        let source = r#"
fn helper(x: f32) -> f32 {
    return x + 1.0;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    let a = helper(1.0);
    let b = helper(2.0);
    return vec4f(a + b, 0.0, 0.0, 1.0);
}
"#;

        let before = naga::front::wgsl::parse_str(source).expect("source should parse");
        let helper_before = find_function_handle_by_name(&before, "helper");
        let before_calls =
            count_calls_to_function(&before.entry_points[0].function.body, helper_before);
        assert_eq!(before_calls, 2, "expected two helper calls before inlining");

        let (changed, after) = run_pass(source);
        assert!(changed, "helper should be inlined: expansion within budget");

        let helper_after = find_function_handle_by_name(&after, "helper");
        let after_calls =
            count_calls_to_function(&after.entry_points[0].function.body, helper_after);
        assert_eq!(
            after_calls, 0,
            "both helper calls should be replaced by inlined expressions"
        );
    }

    #[test]
    fn inlines_call_in_loop_body_used_in_continuing() {
        // naga lets `continuing` reference body-defined expressions; a stale
        // `CallResult` there fails `run_pass`'s validation.
        let source = r#"
fn helper(x: i32) -> i32 {
    return x + 1;
}
@compute @workgroup_size(1)
fn cs_main() {
    var i: i32 = 0;
    loop {
        let x = helper(i);
        if (i > 10) { break; }
        continuing {
            i = i + x;
        }
    }
}
"#;
        let (changed, module) = run_pass(source);
        assert!(changed, "helper call in the loop body should be inlined");
        let helper = find_function_handle_by_name(&module, "helper");
        let body = &module.entry_points[0].function.body;
        assert_eq!(
            count_calls_to_function(body, helper),
            0,
            "the body call to helper must be inlined (result threads into continuing)"
        );
    }

    #[test]
    fn inlines_call_in_loop_body_used_in_break_if() {
        // Same hazard via `break_if`.
        let source = r#"
fn limit(x: i32) -> i32 {
    return x + 5;
}
@compute @workgroup_size(1)
fn cs_main() {
    var i: i32 = 0;
    loop {
        let lim = limit(i);
        i = i + 1;
        continuing {
            break if i >= lim;
        }
    }
}
"#;
        let (changed, module) = run_pass(source);
        assert!(changed, "limit call in the loop body should be inlined");
        let limit = find_function_handle_by_name(&module, "limit");
        let body = &module.entry_points[0].function.body;
        assert_eq!(
            count_calls_to_function(body, limit),
            0,
            "the body call to limit must be inlined (result threads into break_if)"
        );
    }
}
