//! IR-level optimization passes assembled by [`build_ir_passes`].
//!
//! # Canonical form
//!
//! The driver restores `normalize`'s form before the first pass and
//! after every accepted change: no declaration or expression the entry
//! points cannot reach, no local without an expression naming it.  A pass
//! therefore never reads the orphans another pass left, a dead local is
//! gone by the time the inliner meets the body it would have vetoed (the
//! multi-site template refuses bodies with locals), and no sweep is spent
//! culling.
//!
//! # Interaction matrix
//!
//! The driver sweeps the sequence to a fixed point, so a missed
//! opportunity usually costs one extra sweep, not the optimization; the
//! load-bearing exceptions are exactly these:
//!
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
//!   builds collapse to one constructor store first; coalescing itself
//!   runs only in the second stage, behind forwarding's fixed point
//!   ([`build_late_passes`]).
//! * `dead_param` late: a parameter's uses must dissolve before its
//!   removal is visible, and it rewrites caller arity.
//! * `emit_merge` late: re-merged `Emit` ranges restore the generator's
//!   single-use inlining that pass-level range-splitting fragmented.
//! * `rename` last: its output is the stable identifier surface
//!   everything downstream reads.
//! * `const_hoist` runs once, after the fixed point ([`build_tail_passes`]),
//!   and only under `Max` with mangling on: it prices every hoist by
//!   rendering the module, so it wants the converged one, and a name of
//!   one or two characters that only rename's mangling delivers.
//!   Normalization behind it culls the lanes a hoist orphans (the alias
//!   planner counts every constructor in the arena, culled or not), and the
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

/// The ring the driver sweeps to a fixed point for `config.profile`;
/// ordering rationale lives in the module-level interaction matrix.
/// Coalescing joins it only in the second stage ([`build_late_passes`]).
pub fn build_ir_passes(config: &Config) -> Vec<Box<dyn Pass>> {
    ring(config, false)
}

/// The second fixed point: the ring again, coalescing included, for the
/// profiles that coalesce.  A reused slot is a store between a load and
/// its uses that forwarding can no longer cross, so coalescing packs the
/// locals whose lifetimes forwarding has finished shaping, never ones it
/// is still about to shorten; the passes around it then settle what the
/// packing changed (a merged local's weight, its stores).  Empty for
/// `Baseline`, which never coalesces.
pub fn build_late_passes(config: &Config) -> Vec<Box<dyn Pass>> {
    match config.profile {
        Profile::Baseline => Vec::new(),
        Profile::Aggressive | Profile::Max => ring(config, true),
    }
}

fn ring(config: &Config, with_coalescing: bool) -> Vec<Box<dyn Pass>> {
    let rename = Box::new(rename::RenamePass::new(
        config.preserve_symbols.clone(),
        config.mangle(),
    )) as Box<dyn Pass>;

    match config.profile {
        Profile::Baseline => {
            // Keeps the IR recognisable for debugging: no inlining or load dedup.
            vec![
                Box::new(const_fold::ConstFoldPass) as Box<dyn Pass>,
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
                Box::new(const_fold::ConstFoldPass) as Box<dyn Pass>,
                Box::new(dead_branch::DeadBranchPass),
                inline_pass,
                Box::new(load_dedup::LoadDedupPass),
                Box::new(const_fold::ConstFoldPass),
                Box::new(dead_branch::DeadBranchPass),
                Box::new(struct_build::StructBuildPass),
            ];
            if with_coalescing {
                passes.push(Box::new(coalescing::CoalescingPass));
            }
            passes.extend([
                Box::new(dead_param::DeadParamPass) as Box<dyn Pass>,
                Box::new(emit_merge::EmitMergePass),
                rename,
            ]);
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
            Box::new(rename::RenamePass::by_render(
                config.preserve_symbols.clone(),
                config.mangle(),
            )),
        ]
    } else {
        Vec::new()
    }
}

/// The canonical form every pass reads, restored by the driver before the
/// first pass and after every accepted change: nothing the entry points
/// cannot reach in any arena, and no local without an expression naming
/// it.  Compaction and dead-local removal feed each other (a culled store
/// orphans its local, a removed local orphans its initialiser), so both
/// repeat until neither removes anything; every round shrinks an arena, so
/// the loop ends.  `true` when anything was removed.
pub(crate) fn normalize(module: &mut naga::Module, preserved: &dyn Fn(&str) -> bool) -> bool {
    let mut removed = false;
    while compact::compact_module(module, preserved) | dead_local::remove_dead_locals(module) {
        removed = true;
    }
    removed
}

/// The bytes `module` ships as, for a pass that confirms a priced rewrite
/// by rendering: what the tail does behind it - compacted, renamed - then
/// generated.  `None` where naga or the emitter declines the module, which
/// reads as "not a win".  The rewrite is judged against this same rendering
/// of the module without it, so the two texts differ in the rewrite alone.
pub(crate) fn shipped_len(mut module: naga::Module, ctx: &PassContext<'_>) -> Option<usize> {
    let preserve: HashSet<String> = ctx.config.preserve_symbols.iter().cloned().collect();
    compact::compact_module(&mut module, &|name| preserve.contains(name));
    rename::plan_names(&module, &preserve, ctx.config.mangle()).apply(&mut module, None);
    let info = crate::io::validate_module(&module).ok()?;
    let options = crate::generator::GenerateOptions::from_config(ctx.config);
    crate::generator::generate(&module, &info, options)
        .ok()
        .map(|emission| emission.source.len())
}
