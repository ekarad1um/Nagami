//! Function inlining for pure, expression-only helpers.
//!
//! The pass only touches functions whose body is a clean
//! `[Emit*, Return { value }]` sequence with no locals, stores, calls,
//! or other side-effecting statements.  Bodies matching that shape are
//! captured as `InlineTemplate` expression DAGs and cloned into each
//! call site, where the call expression is replaced with the cloned
//! return expression.
//!
//! Two budgets keep output size from regressing: `max_node_count`
//! caps a single template's DAG size, and `max_call_sites` caps how
//! many times a template may be reused.  Above those thresholds, or
//! when duplicating a non-trivial body across multiple sites would
//! exceed `MAX_MULTI_SITE_EXPANSION` net added nodes, the function
//! is left alone.

use std::collections::{HashMap, HashSet};

use crate::error::Error;
use crate::pipeline::{Pass, PassContext};

use super::expr_util::{
    expression_needs_emit, is_disallowed_inline_expression, map_atomic_function_handles,
    map_cooperative_data_handles, map_gather_mode_handles, map_ray_pipeline_function_handles,
    map_ray_query_function_handles, remap_statement_handles, try_map_expression_handles_in_place,
};

/// Default inlining budgets (used by [`super::Profile::Aggressive`]).
pub const DEFAULT_MAX_INLINE_NODE_COUNT: usize = 24;
/// Default inlining call-site budget (used by [`super::Profile::Aggressive`]).
pub const DEFAULT_MAX_INLINE_CALL_SITES: usize = 3;
/// Widened node budget when running under [`super::Profile::Max`].
pub const MAX_PROFILE_MAX_INLINE_NODE_COUNT: usize = 48;
/// Widened call-site budget when running under [`super::Profile::Max`].
pub const MAX_PROFILE_MAX_INLINE_CALL_SITES: usize = 6;

/// Maximum additional expression nodes tolerated when duplicating a
/// template across several call sites, equal to
/// `node_count * (call_sites - 1)`.
///
/// After minification, call syntax is already 1-2 characters, so
/// cloning a non-trivial body over multiple sites is almost always a
/// net size regression.  Single-site inlining is unaffected (the
/// expansion term is zero).
const MAX_MULTI_SITE_EXPANSION: usize = 6;

/// Inlining pass parameterised on the per-run node and call-site
/// budgets.
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
    /// Construct a pass with explicit budgets; the defaults live in
    /// `DEFAULT_*` and `MAX_PROFILE_*` constants above and are
    /// selected by [`super::build_ir_passes`] based on the profile.
    pub fn new(max_node_count: usize, max_call_sites: usize) -> Self {
        Self {
            max_node_count,
            max_call_sites,
        }
    }
}

/// Snapshot of an inlinable function: the expression arena alongside
/// the handle that delivers the return value.  Cloned into each
/// caller's arena when the call is replaced.
#[derive(Clone)]
struct InlineTemplate {
    argument_count: usize,
    return_expr: naga::Handle<naga::Expression>,
    expressions: naga::Arena<naga::Expression>,
}

impl Pass for InliningPass {
    fn name(&self) -> &'static str {
        "function_inlining"
    }

    fn run(&mut self, module: &mut naga::Module, _ctx: &PassContext<'_>) -> Result<bool, Error> {
        let templates = collect_inline_templates(module, self.max_node_count, self.max_call_sites);
        if templates.is_empty() {
            return Ok(false);
        }

        let mut changed = 0usize;
        for (_, function) in module.functions.iter_mut() {
            changed += inline_in_function(function, &templates);
        }
        for entry in module.entry_points.iter_mut() {
            changed += inline_in_function(&mut entry.function, &templates);
        }

        Ok(changed > 0)
    }
}

// MARK: Template collection

/// Collect every function eligible for inlining and build its
/// [`InlineTemplate`].
///
/// Safety invariant: only pure, expression-only functions are eligible.
/// The body must be `[Emit*, Return { value }]` with no locals, no
/// stores, no calls, and no other side-effecting statements, which
/// guarantees the template's expression DAG can be cloned into the
/// caller's arena without invalidating any assumptions about stores or
/// control flow between statements.
fn collect_inline_templates(
    module: &naga::Module,
    max_node_count: usize,
    max_call_sites: usize,
) -> HashMap<naga::Handle<naga::Function>, InlineTemplate> {
    let call_counts = collect_call_counts(module);
    let mut templates = HashMap::new();

    for (function_handle, function) in module.functions.iter() {
        let call_sites = call_counts.get(&function_handle).copied().unwrap_or(0);
        if call_sites == 0 || call_sites > max_call_sites {
            continue;
        }
        if !function.local_variables.is_empty() || function.result.is_none() {
            continue;
        }

        let Some(return_expr) = extract_inline_return_expression(&function.body) else {
            continue;
        };

        let mut visited = HashSet::new();
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

        // For multi-site functions, bound the extra nodes introduced by
        // body duplication.  Each additional call site beyond the first
        // copies the entire expression tree, while removing only the
        // single function declaration.
        if call_sites > 1 {
            let expansion = node_count * (call_sites - 1);
            if expansion > MAX_MULTI_SITE_EXPANSION {
                continue;
            }
        }

        templates.insert(
            function_handle,
            InlineTemplate {
                argument_count: function.arguments.len(),
                return_expr,
                expressions: function.expressions.clone(),
            },
        );
    }

    templates
}

/// Count call-site references to each function across every function
/// body and entry point; used to gate the `max_call_sites` budget.
fn collect_call_counts(module: &naga::Module) -> HashMap<naga::Handle<naga::Function>, usize> {
    let mut counts = HashMap::new();

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
    counts: &mut HashMap<naga::Handle<naga::Function>, usize>,
) {
    for statement in block {
        match statement {
            naga::Statement::Call { function, .. } => {
                *counts.entry(*function).or_insert(0) += 1;
            }
            naga::Statement::Block(inner) => collect_call_counts_in_block(inner, counts),
            naga::Statement::If { accept, reject, .. } => {
                collect_call_counts_in_block(accept, counts);
                collect_call_counts_in_block(reject, counts);
            }
            naga::Statement::Switch { cases, .. } => {
                for case in cases {
                    collect_call_counts_in_block(&case.body, counts);
                }
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => {
                collect_call_counts_in_block(body, counts);
                collect_call_counts_in_block(continuing, counts);
            }
            _ => {}
        }
    }
}

/// Return the expression handle delivered by a body of the form
/// `[Emit*, Return { value }]`, or `None` if the block contains
/// anything else (loops, stores, multi-return, etc.).  Single gate
/// for the pass's purity requirement.
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

/// Count expression nodes reachable from `handle` and ensure every
/// `FunctionArgument` index fits inside the advertised argument list.
/// Returns `None` when the DAG contains an inline-disallowed
/// expression (statement-attached results, local variables, and so
/// on) or an out-of-range argument index.
fn analyze_inline_expression(
    handle: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
    argument_count: usize,
    visited: &mut HashSet<naga::Handle<naga::Expression>>,
) -> Option<usize> {
    if !visited.insert(handle) {
        return Some(0);
    }

    let expr = &expressions[handle];
    if is_disallowed_inline_expression(expr) {
        return None;
    }

    if let naga::Expression::FunctionArgument(index) = expr {
        return ((*index as usize) < argument_count).then_some(1);
    }

    let mut total = 1usize;
    let mut cloned = expr.clone();
    try_map_expression_handles_in_place(&mut cloned, &mut |child| {
        total += analyze_inline_expression(child, expressions, argument_count, visited)?;
        Some(child)
    })?;

    Some(total)
}

// MARK: Call-site rewriting

/// Walk `function`'s body, replacing every eligible call with the
/// template's cloned return expression.  Returns the number of call
/// sites rewritten.  The entry-point driver calls this with the same
/// `templates` map so every caller sees the same inlinable set.
fn inline_in_function(
    function: &mut naga::Function,
    templates: &HashMap<naga::Handle<naga::Function>, InlineTemplate>,
) -> usize {
    let changed = inline_in_block(
        &mut function.body,
        &mut function.expressions,
        templates,
        &HashMap::new(),
    );

    if changed > 0 {
        rebuild_function_expressions(function);
        function.named_expressions.clear();
    }

    changed
}

fn inline_in_block(
    block: &mut naga::Block,
    expressions: &mut naga::Arena<naga::Expression>,
    templates: &HashMap<naga::Handle<naga::Function>, InlineTemplate>,
    inherited_replacements: &HashMap<
        naga::Handle<naga::Expression>,
        naga::Handle<naga::Expression>,
    >,
) -> usize {
    let mut changed = 0usize;
    let mut replacements = inherited_replacements.clone();

    let original = std::mem::take(block);
    let mut rebuilt = naga::Block::with_capacity(original.len());

    for (mut statement, span) in original.span_into_iter() {
        apply_replacements_to_statement(&mut statement, expressions, &replacements);

        match statement {
            naga::Statement::Call {
                function,
                arguments,
                result: Some(result_handle),
            } => {
                if let Some(template) = templates.get(&function)
                    && template.argument_count == arguments.len()
                {
                    let old_len = expressions.len();
                    let mut memo = HashMap::new();
                    if let Some(root_handle) = clone_inline_expression(
                        template.return_expr,
                        template,
                        &arguments,
                        expressions,
                        &mut memo,
                    ) {
                        push_emit_ranges_for_new_expressions(
                            &mut rebuilt,
                            expressions,
                            old_len,
                            span,
                        );

                        replacements.insert(result_handle, root_handle);
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
            naga::Statement::Block(mut inner) => {
                changed += inline_in_block(&mut inner, expressions, templates, &replacements);
                rebuilt.push(naga::Statement::Block(inner), span);
            }
            naga::Statement::If {
                condition,
                mut accept,
                mut reject,
            } => {
                changed += inline_in_block(&mut accept, expressions, templates, &replacements);
                changed += inline_in_block(&mut reject, expressions, templates, &replacements);
                rebuilt.push(
                    naga::Statement::If {
                        condition,
                        accept,
                        reject,
                    },
                    span,
                );
            }
            naga::Statement::Switch {
                selector,
                mut cases,
            } => {
                for case in cases.iter_mut() {
                    changed +=
                        inline_in_block(&mut case.body, expressions, templates, &replacements);
                }
                rebuilt.push(naga::Statement::Switch { selector, cases }, span);
            }
            naga::Statement::Loop {
                mut body,
                mut continuing,
                break_if,
            } => {
                changed += inline_in_block(&mut body, expressions, templates, &replacements);
                changed += inline_in_block(&mut continuing, expressions, templates, &replacements);
                rebuilt.push(
                    naga::Statement::Loop {
                        body,
                        continuing,
                        break_if,
                    },
                    span,
                );
            }
            other => rebuilt.push(other, span),
        }
    }

    *block = rebuilt;
    changed
}

/// Emit `Emit` statements covering every expression appended to
/// `expressions` after `old_len` that requires an emit range, split
/// around declarative expressions so the ranges remain contiguous.
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

/// Recursively clone `handle` from `template.expressions` into
/// `caller_expressions`, replacing `FunctionArgument` references with
/// the caller-supplied `arguments`.  `memo` guarantees each template
/// handle produces exactly one caller handle so shared sub-DAGs stay
/// shared after cloning.
fn clone_inline_expression(
    handle: naga::Handle<naga::Expression>,
    template: &InlineTemplate,
    arguments: &[naga::Handle<naga::Expression>],
    caller_expressions: &mut naga::Arena<naga::Expression>,
    memo: &mut HashMap<naga::Handle<naga::Expression>, naga::Handle<naga::Expression>>,
) -> Option<naga::Handle<naga::Expression>> {
    if let Some(mapped) = memo.get(&handle).copied() {
        return Some(mapped);
    }

    let expr = &template.expressions[handle];
    if is_disallowed_inline_expression(expr) {
        return None;
    }

    let mapped = match expr {
        naga::Expression::FunctionArgument(index) => arguments.get(*index as usize).copied()?,
        _ => {
            let mut cloned = expr.clone();
            try_map_expression_handles_in_place(&mut cloned, &mut |child| {
                clone_inline_expression(child, template, arguments, caller_expressions, memo)
            })?;
            caller_expressions.append(cloned, Default::default())
        }
    };

    memo.insert(handle, mapped);
    Some(mapped)
}

/// Remap every expression handle referenced by `statement` through
/// the inlining `replacements` map.  Covers every naga statement
/// variant exhaustively so inlined call results are consistently
/// substituted across control flow.
fn apply_replacements_to_statement(
    statement: &mut naga::Statement,
    expressions: &mut naga::Arena<naga::Expression>,
    replacements: &HashMap<naga::Handle<naga::Expression>, naga::Handle<naga::Expression>>,
) {
    let mut remap =
        |handle: naga::Handle<naga::Expression>| resolve_replacement(handle, replacements);

    match statement {
        naga::Statement::Emit(range) => {
            for handle in range.clone() {
                let expression = expressions.get_mut(handle);
                let _ = try_map_expression_handles_in_place(expression, &mut |h| Some(remap(h)));
            }
        }
        naga::Statement::Block(_) => {}
        naga::Statement::If { condition, .. } => {
            *condition = remap(*condition);
        }
        naga::Statement::Switch { selector, .. } => {
            *selector = remap(*selector);
        }
        naga::Statement::Loop { break_if, .. } => {
            if let Some(handle) = break_if {
                *handle = remap(*handle);
            }
        }
        naga::Statement::Break | naga::Statement::Continue | naga::Statement::Kill => {}
        naga::Statement::Return { value } => {
            if let Some(handle) = value {
                *handle = remap(*handle);
            }
        }
        naga::Statement::ControlBarrier(_) | naga::Statement::MemoryBarrier(_) => {}
        naga::Statement::Store { pointer, value } => {
            *pointer = remap(*pointer);
            *value = remap(*value);
        }
        naga::Statement::ImageStore {
            image,
            coordinate,
            array_index,
            value,
        } => {
            *image = remap(*image);
            *coordinate = remap(*coordinate);
            if let Some(index) = array_index {
                *index = remap(*index);
            }
            *value = remap(*value);
        }
        naga::Statement::Atomic {
            pointer,
            fun,
            value,
            ..
        } => {
            *pointer = remap(*pointer);
            map_atomic_function_handles(fun, &mut remap);
            *value = remap(*value);
        }
        naga::Statement::ImageAtomic {
            image,
            coordinate,
            array_index,
            fun,
            value,
        } => {
            *image = remap(*image);
            *coordinate = remap(*coordinate);
            if let Some(index) = array_index {
                *index = remap(*index);
            }
            map_atomic_function_handles(fun, &mut remap);
            *value = remap(*value);
        }
        naga::Statement::WorkGroupUniformLoad { pointer, .. } => {
            *pointer = remap(*pointer);
        }
        naga::Statement::Call { arguments, .. } => {
            for argument in arguments {
                *argument = remap(*argument);
            }
        }
        naga::Statement::RayQuery { query, fun } => {
            *query = remap(*query);
            map_ray_query_function_handles(fun, &mut remap);
        }
        naga::Statement::RayPipelineFunction(fun) => {
            map_ray_pipeline_function_handles(fun, &mut remap);
        }
        naga::Statement::SubgroupBallot { predicate, .. } => {
            if let Some(handle) = predicate {
                *handle = remap(*handle);
            }
        }
        naga::Statement::SubgroupGather { mode, argument, .. } => {
            map_gather_mode_handles(mode, &mut remap);
            *argument = remap(*argument);
        }
        naga::Statement::SubgroupCollectiveOperation { argument, .. } => {
            *argument = remap(*argument);
        }
        naga::Statement::CooperativeStore { target, data } => {
            *target = remap(*target);
            map_cooperative_data_handles(data, &mut remap);
        }
    }
}

/// Follow `replacements` transitively and return the terminal target.
/// Self-loops terminate early; the callers guarantee the map is
/// acyclic by construction, so a well-formed map always halts.
fn resolve_replacement(
    mut handle: naga::Handle<naga::Expression>,
    replacements: &HashMap<naga::Handle<naga::Expression>, naga::Handle<naga::Expression>>,
) -> naga::Handle<naga::Expression> {
    while let Some(next) = replacements.get(&handle).copied() {
        if next == handle {
            break;
        }
        handle = next;
    }
    handle
}

// MARK: Arena compaction

/// Rebuild `function.expressions` after inlining so only reachable
/// expressions survive, remapping every handle referenced by the body,
/// locals, named expressions, and result.  The walk mirrors naga's
/// `compact` pass shape but is scoped to a single function.
fn rebuild_function_expressions(function: &mut naga::Function) {
    let old_expressions = std::mem::take(&mut function.expressions);
    let mut new_expressions = naga::Arena::new();
    let mut handle_map = HashMap::new();

    rebuild_block_expressions(
        &mut function.body,
        &old_expressions,
        &mut new_expressions,
        &mut handle_map,
    );

    // Remap local variable init handles into the new expression arena.
    for (_, local) in function.local_variables.iter_mut() {
        if let Some(init) = &mut local.init {
            *init = clone_expression_handle(
                *init,
                &old_expressions,
                &mut new_expressions,
                &mut handle_map,
            );
        }
    }

    function.expressions = new_expressions;
}

fn rebuild_block_expressions(
    block: &mut naga::Block,
    old_expressions: &naga::Arena<naga::Expression>,
    new_expressions: &mut naga::Arena<naga::Expression>,
    handle_map: &mut HashMap<naga::Handle<naga::Expression>, naga::Handle<naga::Expression>>,
) {
    let original = std::mem::take(block);
    let mut rebuilt = naga::Block::with_capacity(original.len());

    for (mut statement, span) in original.span_into_iter() {
        // Handle Emit specially: clone expressions and split into contiguous
        // sub-ranges to avoid including interleaved non-emittable dependencies.
        if let naga::Statement::Emit(ref range) = statement {
            let mut mapped_handles = Vec::new();
            for handle in range.clone() {
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

            if !mapped_handles.is_empty() {
                let mut start = mapped_handles[0];
                let mut end = mapped_handles[0];
                for &h in &mapped_handles[1..] {
                    if h.index() == end.index() + 1 {
                        end = h;
                    } else {
                        rebuilt.push(
                            naga::Statement::Emit(naga::Range::new_from_bounds(start, end)),
                            span,
                        );
                        start = h;
                        end = h;
                    }
                }
                rebuilt.push(
                    naga::Statement::Emit(naga::Range::new_from_bounds(start, end)),
                    span,
                );
            }
            continue;
        }

        remap_statement_handles(&mut statement, &mut |h| {
            clone_expression_handle(h, old_expressions, new_expressions, handle_map)
        });

        match &mut statement {
            naga::Statement::Block(inner) => {
                rebuild_block_expressions(inner, old_expressions, new_expressions, handle_map);
            }
            naga::Statement::If { accept, reject, .. } => {
                rebuild_block_expressions(accept, old_expressions, new_expressions, handle_map);
                rebuild_block_expressions(reject, old_expressions, new_expressions, handle_map);
            }
            naga::Statement::Switch { cases, .. } => {
                for case in cases.iter_mut() {
                    rebuild_block_expressions(
                        &mut case.body,
                        old_expressions,
                        new_expressions,
                        handle_map,
                    );
                }
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => {
                rebuild_block_expressions(body, old_expressions, new_expressions, handle_map);
                rebuild_block_expressions(continuing, old_expressions, new_expressions, handle_map);
            }
            _ => {}
        }

        rebuilt.push(statement, span);
    }

    *block = rebuilt;
}

fn clone_expression_handle(
    handle: naga::Handle<naga::Expression>,
    old_expressions: &naga::Arena<naga::Expression>,
    new_expressions: &mut naga::Arena<naga::Expression>,
    handle_map: &mut HashMap<naga::Handle<naga::Expression>, naga::Handle<naga::Expression>>,
) -> naga::Handle<naga::Expression> {
    if let Some(mapped) = handle_map.get(&handle).copied() {
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
    use crate::config::Config;

    fn run_pass(source: &str) -> (bool, naga::Module) {
        let mut module = naga::front::wgsl::parse_str(source).expect("source should parse");
        let mut pass = InliningPass::default();
        let config = Config::default();
        let ctx = PassContext {
            config: &config,
            trace_run_dir: None,
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
        for statement in block {
            match statement {
                naga::Statement::Call { function, .. } if *function == target => {
                    count += 1;
                }
                naga::Statement::Block(inner) => {
                    count += count_calls_to_function(inner, target);
                }
                naga::Statement::If { accept, reject, .. } => {
                    count += count_calls_to_function(accept, target);
                    count += count_calls_to_function(reject, target);
                }
                naga::Statement::Switch { cases, .. } => {
                    for case in cases {
                        count += count_calls_to_function(&case.body, target);
                    }
                }
                naga::Statement::Loop {
                    body, continuing, ..
                } => {
                    count += count_calls_to_function(body, target);
                    count += count_calls_to_function(continuing, target);
                }
                _ => {}
            }
        }
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
        // Construct a module where the caller has a local with init: Some(...)
        // and inline a simple helper into it.  After inlining, the local's init
        // handle must still be valid in the rebuilt expression arena.

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

        // Manually set init on the local variable to exercise the code path.
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

        // Run inlining pass
        let mut pass = InliningPass::default();
        let config = Config::default();
        let ctx = PassContext {
            config: &config,
            trace_run_dir: None,
        };
        let changed = pass.run(&mut module, &ctx).expect("inlining should run");
        assert!(changed, "helper should be inlined");

        // After inlining, the local's init handle must be valid.
        let ep_fn = &module.entry_points[0].function;
        let local = &ep_fn.local_variables[local_handle];
        assert!(local.init.is_some(), "local init should still be present");
        let init_handle = local.init.unwrap();
        // The init handle must be within range of the (rebuilt) expression arena.
        assert!(
            init_handle.index() < ep_fn.expressions.len(),
            "local init handle ({}) should be within rebuilt expression arena (len={})",
            init_handle.index(),
            ep_fn.expressions.len(),
        );
        // Verify it's still a literal 0.0
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
        // A helper with many expression nodes called from 2 sites should be
        // rejected: node_count=9, call_sites=2, expansion=9*(2-1)=9 > 6.
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
        // A tiny helper (node_count=3) called from 2 sites should be allowed:
        // expansion=3*(2-1)=3 <= 6.
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
}
