//! IR-level optimization passes assembled by [`build_ir_passes`].
//!
//! # Interaction matrix
//!
//! The driver sweeps the sequence to a fixed point, so a missed
//! opportunity usually costs one extra sweep, not the optimization; the
//! load-bearing exceptions are exactly these:
//!
//! * `compact` -> `dead_local`: only after statement-unreachable
//!   expressions are culled is "no `LocalVariable` expression" exactly
//!   "dead local".
//! * `dead_local` -> `inlining`: clearing dead locals lifts the
//!   multi-site template's no-locals veto (a single-call body splices
//!   with its locals).
//! * `dead_branch` -> `inlining`: a guard chain of pure early returns
//!   merged into one `select` return is what makes a multi-return
//!   single-call body spliceable in the same sweep.
//! * `const_fold` <-> `dead_branch`: folds expose constant branches,
//!   pruned branches shrink bodies toward the inlinable
//!   `[Emit*, Return]` shape; the pair therefore repeats after
//!   `load_dedup` (inlined bodies and forwarded values surface fresh
//!   constants; a third copy right after `inlining` changed nothing and
//!   only cost pass runs).
//! * `inlining` -> `load_dedup`: it must see the fully-materialised call
//!   bodies, or it forwards across a boundary the inliner is about to
//!   erase.
//! * `struct_build` -> `coalescing` / `rename`: member-wise struct
//!   builds collapse to one constructor store first.
//! * `dead_param` late: a parameter's uses must dissolve before its
//!   removal is visible, and it rewrites caller arity.
//! * `emit_merge` late: re-merged `Emit` ranges restore the generator's
//!   single-use inlining that pass-level range-splitting fragmented.
//! * `rename` last: its output is the stable identifier surface
//!   everything downstream reads.
//! * `const_hoist` runs once, after the fixed point ([`build_tail_passes`]),
//!   and only under `Max` with mangling on: it prices every hoist by
//!   rendering the module, so it wants the converged one, and a name of
//!   one or two characters that only rename's mangling delivers.  The
//!   `compact` behind it culls the lanes a hoist orphans (the alias planner
//!   counts every constructor in the arena, culled or not), and the
//!   `rename` names the hoisted constant before anything is emitted, or
//!   re-minified output names it differently (idempotence).  That rename
//!   ranks by the text (`RenamePass::by_render`): a value the emitter
//!   inlines at several uses spells its operands once per use, which the
//!   IR census counts once and a re-parse of the text counts per use.

use std::collections::HashSet;

use crate::config::{Config, Profile};
use crate::pipeline::{Pass, PassContext};

pub mod coalescing;
pub mod compact;
pub mod const_fold;
pub mod const_hoist;
pub mod dead_branch;
pub mod dead_local;
pub mod dead_param;
pub mod emit_merge;
pub mod expr_util;
pub mod inlining;
pub mod load_dedup;
pub mod rename;
pub mod scoped_map;
pub mod specialize_ptr_params;
pub mod struct_build;

/// Build the pass pipeline for `config.profile`; ordering rationale
/// lives in the module-level interaction matrix.
pub fn build_ir_passes(config: &Config) -> Vec<Box<dyn Pass>> {
    let rename = Box::new(rename::RenamePass::new(
        config.preserve_symbols.clone(),
        config.mangle(),
    )) as Box<dyn Pass>;

    match config.profile {
        Profile::Baseline => {
            // Keeps the IR recognisable for debugging: no inlining or load dedup.
            vec![
                Box::new(compact::CompactPass) as Box<dyn Pass>,
                Box::new(const_fold::ConstFoldPass),
                Box::new(dead_branch::DeadBranchPass),
                Box::new(dead_param::DeadParamPass),
                Box::new(emit_merge::EmitMergePass),
                rename,
            ]
        }
        Profile::Aggressive | Profile::Max => {
            let (default_nodes, default_sites) = match config.profile {
                Profile::Max => (
                    inlining::MAX_PROFILE_MAX_INLINE_NODE_COUNT,
                    inlining::MAX_PROFILE_MAX_INLINE_CALL_SITES,
                ),
                _ => (
                    inlining::DEFAULT_MAX_INLINE_NODE_COUNT,
                    inlining::DEFAULT_MAX_INLINE_CALL_SITES,
                ),
            };
            let inline_pass: Box<dyn Pass> = Box::new(inlining::InliningPass::new(
                config.max_inline_node_count.unwrap_or(default_nodes),
                config.max_inline_call_sites.unwrap_or(default_sites),
            ));

            let mut passes: Vec<Box<dyn Pass>> = vec![
                Box::new(compact::CompactPass) as Box<dyn Pass>,
                Box::new(dead_local::DeadLocalPass),
                Box::new(const_fold::ConstFoldPass),
                Box::new(dead_branch::DeadBranchPass),
                inline_pass,
            ];

            passes.extend([
                Box::new(load_dedup::LoadDedupPass) as Box<dyn Pass>,
                Box::new(const_fold::ConstFoldPass),
                Box::new(dead_branch::DeadBranchPass),
                Box::new(struct_build::StructBuildPass),
                Box::new(coalescing::CoalescingPass),
                Box::new(dead_param::DeadParamPass),
                Box::new(emit_merge::EmitMergePass),
            ]);

            passes.push(rename);

            passes
        }
    }
}

/// The passes run once after the fixed point: each reaches its own fixed
/// point in one application and prices against the converged module.
pub fn build_tail_passes(config: &Config) -> Vec<Box<dyn Pass>> {
    if config.profile == Profile::Max && config.mangle() {
        vec![
            Box::new(const_hoist::ConstHoistPass) as Box<dyn Pass>,
            Box::new(compact::CompactPass),
            Box::new(rename::RenamePass::by_render(
                config.preserve_symbols.clone(),
                config.mangle(),
            )),
        ]
    } else {
        Vec::new()
    }
}

/// The bytes `module` ships as, for a pass that confirms a priced rewrite
/// by rendering: what the tail does behind it - named expressions cleared
/// first (rename clears them, and a splice its caller's; naga's compaction
/// keeps a named value alive, so a stale source `let` would print a `var`
/// for a local nothing reads), compacted, renamed - then generated.  `None`
/// where naga or the emitter declines the module, which reads as "not a
/// win".  The rewrite is judged against this same rendering of the module
/// without it, so the two texts differ in the rewrite alone.
pub(crate) fn shipped_len(mut module: naga::Module, ctx: &PassContext<'_>) -> Option<usize> {
    let preserve: HashSet<String> = ctx.config.preserve_symbols.iter().cloned().collect();
    crate::ir::visit::for_each_function_mut(
        &mut module.functions,
        &mut module.entry_points,
        &mut |f| f.named_expressions.clear(),
    );
    compact::compact_module(&mut module, &|name| preserve.contains(name));
    rename::plan_names(&module, &preserve, ctx.config.mangle()).apply(&mut module, None);
    let info = crate::io::validate_module(&module).ok()?;
    let options = crate::generator::GenerateOptions::from_config(ctx.config);
    crate::generator::generate(&module, &info, options)
        .ok()
        .map(|emission| emission.source.len())
}
