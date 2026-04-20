//! Generator unit-test suite.
//!
//! Shared fixtures live in [`helpers`]; every other sub-module groups
//! tests by the emitter feature under test (type elision, compound
//! assignment folding, splat collapsing, and so on).  The suite is
//! oriented around output-text assertions: most tests parse a WGSL
//! snippet, run it through [`super::generate`], and compare the
//! emitted source to the expected shape.

mod helpers;

mod compound_assign;
mod dead_code;
mod expressions;
mod names_and_output;
mod pipeline;
mod precedence;
mod roundtrip;
mod statements;
mod textures;
mod type_alias;
mod type_elision;
