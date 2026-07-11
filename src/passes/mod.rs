//! IR-level optimization passes assembled by [`build_ir_passes`].
//!
//! # Interaction matrix
//!
//! The driver sweeps the sequence to a fixed point, so a missed
//! opportunity usually costs one extra sweep, not the optimization.
//! The exceptions - ordering constraints that are load-bearing within a
//! sweep or across the rename boundary - are exactly these:
//!
//! * `compact` -> `dead_local`: only after statement-unreachable
//!   expressions are culled is "no `LocalVariable` expression" exactly
//!   "dead local".
//! * `dead_local` -> `inlining`: clearing dead locals lifts the
//!   inliner's no-locals veto.
//! * `const_fold` <-> `dead_branch`: folds expose constant branches,
//!   pruned branches shrink bodies toward the inlinable
//!   `[Emit*, Return]` shape; the pair therefore repeats after
//!   `inlining` and after `load_dedup` (forwarded values surface fresh
//!   constants).
//! * `inlining` -> `cse` / `load_dedup`: both must see the
//!   fully-materialised call bodies, or they dedup/forward across a
//!   boundary the inliner is about to erase.
//! * `struct_build` -> `coalescing` / `rename`: member-wise struct
//!   builds collapse to one constructor store first.
//! * `dead_param` late: a parameter's uses must dissolve before its
//!   removal is visible, and it rewrites caller arity.
//! * `emit_merge` late: re-merged `Emit` ranges restore the generator's
//!   single-use inlining that pass-level range-splitting fragmented.
//! * `cse` / `const_hoist` only under `Max` with mangling on: both
//!   price their savings at a 1-2 character bound name that only
//!   rename's mangling delivers.
//! * `const_hoist` -> `rename`: the hoisted constant must exist before
//!   frequency-based naming, or re-minified output names it differently
//!   (idempotence).
//! * `rename` last: its output is the stable identifier surface
//!   everything downstream reads.
//!
//! Individual pass modules document their own invariants.

use crate::config::{Config, Profile};
use crate::pipeline::Pass;

pub mod coalescing;
pub mod compact;
pub mod const_fold;
pub mod const_hoist;
pub mod cse;
pub mod dead_branch;
pub mod dead_local;
pub mod dead_param;
pub mod emit_merge;
pub mod expr_util;
pub mod inlining;
pub mod load_dedup;
pub mod rename;
pub mod scoped_map;
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
            // Quick sanity pipeline that keeps the IR recognisable for
            // debugging: no inlining, CSE, or load dedup.
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
                Box::new(const_fold::ConstFoldPass),
                Box::new(dead_branch::DeadBranchPass),
            ];

            if config.profile == Profile::Max && config.mangle() {
                passes.push(Box::new(cse::CSEPass));
            }

            passes.extend([
                Box::new(load_dedup::LoadDedupPass) as Box<dyn Pass>,
                Box::new(const_fold::ConstFoldPass),
                Box::new(dead_branch::DeadBranchPass),
                Box::new(struct_build::StructBuildPass),
                Box::new(coalescing::CoalescingPass),
                Box::new(dead_param::DeadParamPass),
                Box::new(emit_merge::EmitMergePass),
            ]);

            if config.profile == Profile::Max && config.mangle() {
                passes.push(Box::new(const_hoist::ConstHoistPass));
            }

            passes.push(rename);

            passes
        }
    }
}
