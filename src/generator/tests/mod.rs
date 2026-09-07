//! Generator unit-test suite: shared fixtures in [`helpers`], one sub-module
//! per emitter feature, assertions on the emitted text.

mod helpers;

mod compound_assign;
mod const_hazard;
mod dead_code;
mod expressions;
mod miscompile_regressions;
mod names_and_output;
mod pipeline;
mod precedence;
mod roundtrip;
mod statements;
mod textures;
mod type_alias;
mod type_elision;
