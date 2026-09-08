//! Custom WGSL emitter behind [`generate`], an alternative to naga's WGSL
//! backend that minifies aggressively (short identifiers, literal extraction,
//! splat elision, single-use inlining).  The pipeline ([`crate::run`]) falls
//! back to naga's emitter when this one errors or its output fails validation,
//! so the custom path only optimises the happy case.  `syntax` holds grammar
//! constants, `core` the `Generator` state and options, `literal_extract` the
//! repeated-literal `const` extraction, `const_hazard` the tint
//! const-expression guard, and `expr_emit` / `stmt_emit` / `module_emit` one
//! emitter per IR scope.

mod const_hazard;
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
                    .get(&(ty_h, idx as u32))
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

/// (source, struct table, kept-constant names); see [`Emission`].
type GeneratedWgsl = (
    String,
    std::collections::BTreeMap<String, crate::name_map::StructRename>,
    std::collections::HashSet<String>,
);

/// Entry point shared by [`generate`] and the test harness.
fn generate_wgsl(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    options: GenerateOptions,
) -> Result<GeneratedWgsl, Error> {
    let mut generator = Generator::new(module, info, options);
    generator.generate_module()?;
    let structs = struct_name_table(module, &generator);
    let live_const_names = module
        .constants
        .iter()
        .filter(|(h, c)| c.name.is_some() && generator.live_constants.contains(h))
        .filter_map(|(_, c)| c.name.clone())
        .collect();
    Ok((generator.into_output(), structs, live_const_names))
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
    #[cfg(not(target_arch = "wasm32"))]
    let start = Instant::now();
    let (source, structs, live_const_names) = generate_wgsl(module, info, options)?;
    #[cfg(not(target_arch = "wasm32"))]
    let duration_us = start.elapsed().as_micros() as u64;
    #[cfg(target_arch = "wasm32")]
    let duration_us = 0u64;
    Ok(Emission {
        source,
        duration_us,
        structs,
        live_const_names,
    })
}

#[cfg(test)]
mod tests;
