//! IR-level optimization passes assembled by [`build_ir_passes`].
//!
//! Pass ordering is load-bearing: const folding exposes dead branches,
//! which in turn unlock further folding and enable inlining; CSE and
//! load dedup run after inlining so they see the fully-materialised
//! call bodies.  Adjust the order only with the interaction matrix
//! in mind.  Individual pass modules document their own invariants.

use crate::config::{Config, Profile};
use crate::pipeline::Pass;

pub mod coalescing;
pub mod compact;
pub mod const_fold;
pub mod const_hoist;
pub mod cse;
pub mod dead_branch;
pub mod dead_param;
pub mod emit_merge;
pub mod expr_util;
pub mod inlining;
pub mod load_dedup;
pub mod rename;
pub mod scoped_map;
pub mod struct_build;

/// Build the pass pipeline for `config.profile`.
///
/// Each profile bundles a fixed sequence of boxed [`Pass`] trait
/// objects; the pipeline driver sweeps them to a fixed point.  The
/// `rename` pass is shared across profiles because its output is the
/// stable identifier surface everything downstream reads from.
pub fn build_ir_passes(config: &Config) -> Vec<Box<dyn Pass>> {
    let rename = Box::new(rename::RenamePass::new(
        config.preserve_symbols.clone(),
        config.mangle(),
    )) as Box<dyn Pass>;

    match config.profile {
        Profile::Baseline => {
            // Minimal chain: compact (DCE) -> const-fold -> dead-branch
            // -> dead-param -> emit merge -> rename.  No inlining, CSE,
            // or load dedup; intended as a quick sanity pipeline that
            // keeps the IR recognisable for debugging.
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
                Box::new(const_fold::ConstFoldPass),
                Box::new(dead_branch::DeadBranchPass),
                inline_pass,
                Box::new(const_fold::ConstFoldPass),
                Box::new(dead_branch::DeadBranchPass),
            ];

            // CSE introduces `let` bindings for shared sub-expressions.
            // Profitable only under [`Profile::Max`], where mangling
            // shrinks the binding name down to a single character and
            // amortises the added line.  Under Aggressive the longer
            // identifier defeats the saving, so CSE stays off.
            if config.profile == Profile::Max {
                passes.push(Box::new(cse::CSEPass));
            }

            passes.extend([
                Box::new(load_dedup::LoadDedupPass) as Box<dyn Pass>,
                Box::new(const_fold::ConstFoldPass),
                Box::new(dead_branch::DeadBranchPass),
                // Collapse field-wise struct builds (`var X; X.a=..; X.b=..;`)
                // into one whole-struct `Store(X, T(..))` before the final
                // coalescing + rename, so a fully-built struct local emits as a
                // single `T(..)` constructor rather than a run of member stores.
                Box::new(struct_build::StructBuildPass),
                Box::new(coalescing::CoalescingPass),
                Box::new(dead_param::DeadParamPass),
                Box::new(emit_merge::EmitMergePass),
                // Hoist repeated all-literal vector constants into shared module
                // consts BEFORE rename, so rename names them by frequency (this
                // is what makes the optimization idempotent).
                Box::new(const_hoist::ConstHoistPass),
                rename,
            ]);

            passes
        }
    }
}
