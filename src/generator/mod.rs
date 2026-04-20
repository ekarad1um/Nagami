//! Custom WGSL emitter.
//!
//! Sits behind [`generate`] as an alternative to naga's built-in
//! WGSL backend.  Where naga emphasises round-tripping, this emitter
//! aggressively minifies: short identifiers, literal extraction,
//! splat elision, and single-use expression inlining.  The pipeline
//! ([`crate::run`]) falls back to naga's emitter when this one errors
//! or produces output that fails validation, so the custom path only
//! has to optimise for the happy case.
//!
//! Sub-modules split by responsibility:
//!
//! * `syntax` - grammar constants (operator precedence, symbol sets).
//! * `core` - the `Generator` type, output buffer, and the options
//!   that flow in from [`crate::run`].
//! * `literal_extract` - per-run extraction of repeated literals
//!   into named `const` declarations.
//! * `expr_emit` / `stmt_emit` / `module_emit` - one file per IR
//!   scope (expression, statement, module), each driving the
//!   generator buffer.

mod core;
mod expr_emit;
mod literal_extract;
mod module_emit;
mod stmt_emit;
mod syntax;

use crate::error::Error;
use core::Generator;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

pub use core::GenerateOptions;

/// Emitter output bundle: the final WGSL source plus the wall-clock
/// cost of producing it.  `duration_us` is zero on wasm where no
/// high-resolution clock is available.
#[derive(Debug)]
pub struct Emission {
    /// Minified WGSL source produced by the generator.
    pub source: String,
    /// Wall-clock cost in microseconds; zero on wasm.
    pub duration_us: u64,
}

/// Internal entry point shared by [`generate`] and the test harness.
/// Holds the pattern that threads options through a fresh
/// [`Generator`] and drains its output buffer.
fn generate_wgsl(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    options: GenerateOptions,
) -> Result<String, Error> {
    let mut generator = Generator::new(module, info, options);
    generator.generate_module()?;
    Ok(generator.into_output())
}

/// Emit minified WGSL for `module` using the custom generator.
///
/// Wraps the internal generator entry point with timing
/// instrumentation and packages the result in [`Emission`].  Caller
/// is [`crate::run`]; if this function errors, the pipeline silently
/// falls back to naga's emitter except when a preamble is active.
///
/// # Errors
///
/// Returns [`Error::Emit`] when the backing generator cannot render
/// a construct (for example an unsupported `ImageClass` or naga IR
/// that survived validation but fails emission).
pub fn generate(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    options: GenerateOptions,
) -> Result<Emission, Error> {
    #[cfg(not(target_arch = "wasm32"))]
    let start = Instant::now();
    let source = generate_wgsl(module, info, options)?;
    #[cfg(not(target_arch = "wasm32"))]
    let duration_us = start.elapsed().as_micros() as u64;
    #[cfg(target_arch = "wasm32")]
    let duration_us = 0u64;
    Ok(Emission {
        source,
        duration_us,
    })
}

#[cfg(test)]
mod tests;
