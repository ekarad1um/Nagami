//! Custom WGSL emitter behind [`generate`], an alternative to naga's WGSL
//! backend that minifies aggressively (short identifiers, literal extraction,
//! splat elision, single-use inlining).  The pipeline ([`crate::run`]) falls
//! back to naga's emitter when this one errors or its output fails validation,
//! so the custom path only optimises the happy case.  `syntax` holds grammar
//! constants, `core` the `Generator` state and options, `literal_extract` the
//! repeated-literal `const` extraction, `const_hazard` the tint
//! const-expression guard, `expr_emit` / `stmt_emit` / `module_emit` one
//! emitter per IR scope, and `price` the same emitter as a pass's cost model.

mod const_hazard;
mod core;
mod expr_emit;
mod literal_extract;
mod module_emit;
pub(crate) mod price;
mod stmt_emit;
pub(crate) mod syntax;

use crate::error::Error;
use core::Generator;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

pub use core::{GenerateOptions, TypeUses};

/// Emitter output: the WGSL source plus what the name map needs.
#[derive(Debug)]
pub struct Emission {
    /// Minified WGSL source.
    pub source: String,
    /// Wall-clock cost in microseconds; zero on wasm (no high-resolution clock).
    pub duration_us: u64,
    /// Emitted struct / member names keyed by original struct name, for
    /// host-addressable structs only; struct renaming lives here, the IR keeps
    /// source names.
    pub structs: std::collections::BTreeMap<String, crate::name_map::StructRename>,
    /// Constants this emission declared; a constant can survive the IR with
    /// every use folded away, so the name map filters through this.
    pub live_const_names: std::collections::HashSet<String>,
    /// How often the text spells each renameable name, for the rename
    /// that ranks by the text.
    pub(crate) name_weights: crate::passes::rename::Weights,
    /// The type spellings the text holds, for the alias plan of the
    /// render that ships (`GenerateOptions::type_uses`).
    pub(crate) type_uses: TypeUses,
    /// The options the text was rendered under and what the spellings
    /// decided, for a render that may ship this text again.
    pub(crate) options: GenerateOptions,
    pub(crate) type_plan: core::TypePlan,
    /// What the render computed from the arenas, for a render of the
    /// same arenas (`generate_after`).
    pub(crate) analyses: core::Analyses,
}

/// Wall clock of one emission; zero on wasm (no high-resolution clock).
struct Timer(#[cfg(not(target_arch = "wasm32"))] Instant);

impl Timer {
    fn start() -> Self {
        Self(
            #[cfg(not(target_arch = "wasm32"))]
            Instant::now(),
        )
    }

    fn elapsed_us(&self) -> u64 {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.0.elapsed().as_micros() as u64
        }
        #[cfg(target_arch = "wasm32")]
        {
            0
        }
    }
}

/// Emitted struct / member names keyed by originals; unnamed types are
/// skipped, members missing from the rename table are identity.
fn struct_name_table(
    module: &naga::Module,
    generator: &Generator,
) -> std::collections::BTreeMap<String, crate::name_map::StructRename> {
    let mut structs = std::collections::BTreeMap::new();
    for (ty_h, ty) in module.types.iter() {
        let Some(original) = &ty.name else { continue };
        let naga::TypeInner::Struct { members, .. } = &ty.inner else {
            continue;
        };
        // Dead structs get no declaration; naga-predeclared ones are not WGSL.
        if !generator.map_visible_structs.contains(ty_h) {
            continue;
        }
        let Some(emitted) = generator.type_names.get(ty_h) else {
            continue;
        };
        let mut member_map = std::collections::BTreeMap::new();
        for (idx, member) in members.iter().enumerate() {
            if let Some(member_original) = &member.name {
                let member_emitted = generator
                    .member_names
                    .get(ty_h)
                    .and_then(|names| names.get(idx))
                    .cloned()
                    .unwrap_or_else(|| member_original.clone());
                member_map.insert(member_original.clone(), member_emitted);
            }
        }
        structs.insert(
            original.clone(),
            crate::name_map::StructRename {
                name: emitted.clone(),
                members: member_map,
            },
        );
    }
    structs
}

/// Emit minified WGSL for `module` with the custom generator.  On error the
/// pipeline ([`crate::run`]) falls back to naga's emitter, except when a
/// preamble is active.
///
/// # Errors
///
/// [`Error::Emit`] when a construct cannot be rendered (an unsupported
/// `ImageClass`, or IR that validates but fails emission).
pub fn generate(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    options: GenerateOptions,
) -> Result<Emission, Error> {
    let timer = Timer::start();
    finish(Generator::new(module, info, options), timer)
}

/// [`generate`], given `prior`: an emission of a module with these arenas
/// (the names may differ) under these options but for `type_uses`.  What
/// a render computes from the arenas alone is the same, so this render
/// takes it over instead of computing it again.
pub fn generate_after(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    options: GenerateOptions,
    prior: Emission,
) -> Result<Emission, Error> {
    let timer = Timer::start();
    let (generator, _) = generator_after(module, info, options, prior);
    finish(generator, timer)
}

/// [`generate_after`] when `prior` is an emission of this module, names
/// included, whose spellings `options` now carries.  A render is a
/// function of the module, the info and the options, and the spellings
/// decide the type plan alone, so `prior` is the text this render would
/// produce whenever the plans agree, and ships again.
pub fn generate_reusing(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    options: GenerateOptions,
    prior: Emission,
) -> Result<Emission, Error> {
    let timer = Timer::start();
    let (generator, prior) = generator_after(module, info, options, prior);
    if let Some(prior) = prior
        && generator.plans_types_as(&prior.type_plan)
    {
        return Ok(Emission {
            duration_us: timer.elapsed_us(),
            ..prior
        });
    }
    finish(generator, timer)
}

/// The generator of a render after `prior`: over `prior`'s analyses when
/// the options agree but for `type_uses`, fresh otherwise; and `prior`
/// itself when its text may still ship.
fn generator_after<'a>(
    module: &'a naga::Module,
    info: &'a naga::valid::ModuleInfo,
    options: GenerateOptions,
    mut prior: Emission,
) -> (Generator<'a>, Option<Emission>) {
    let same_options = prior.options
        == GenerateOptions {
            type_uses: None,
            ..options.clone()
        };
    if !same_options {
        return (Generator::new(module, info, options), None);
    }
    let analyses = std::mem::take(&mut prior.analyses);
    (
        Generator::with_analyses(module, info, options, Some(analyses)),
        Some(prior),
    )
}

/// Render the module `generator` was built for.
fn finish(mut generator: Generator<'_>, timer: Timer) -> Result<Emission, Error> {
    generator.generate_module()?;
    let module = generator.module;
    let structs = struct_name_table(module, &generator);
    let live_const_names = module
        .constants
        .iter()
        .filter(|(h, c)| c.name.is_some() && generator.live_constants.contains(h))
        .filter_map(|(_, c)| c.name.clone())
        .collect();
    Ok(Emission {
        duration_us: timer.elapsed_us(),
        structs,
        live_const_names,
        name_weights: std::mem::take(&mut generator.name_weights),
        type_uses: std::mem::take(&mut generator.type_uses),
        options: std::mem::take(&mut generator.options),
        type_plan: generator.take_type_plan(),
        analyses: generator.take_analyses(),
        source: generator.into_output(),
    })
}

#[cfg(test)]
mod tests;
