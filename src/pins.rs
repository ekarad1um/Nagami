//! Static access, carried in the IR.  WGSL's resource interface is
//! syntactic: an entry point statically accesses a binding whose name
//! appears anywhere in its call graph - dead code, a phony `_ = x;` - and
//! an automatic pipeline layout lists exactly those.  The passes see
//! values, so a dead read dies, and with it the access.  naga keeps
//! `_ = x;` as a NAMED expression (`phony`), which its compactor roots and
//! its analyzer counts as a use; a pin is that form - a named bare
//! `GlobalVariable` of a host-visible global - planted in every function
//! for every global its call graph references ([`plant`]), so the access
//! outlives whatever the passes delete: the driver's clear of the front
//! end's names keeps pins ([`retain`]), the arena rebuild roots them,
//! compaction roots them and their globals.  The emitter prints `_=N;` for
//! a pin whose function's text mentions the global nowhere else and whose
//! callees' call graphs do not reach it, decided from what the text
//! rendered, never from a model of it; the interface check reads the
//! arena, so a shipped text has every access its input had.
//!
//! An immediate admits `READ` only, so its bare named reference (a `QUERY`
//! use) is invalid IR; its pin is the named `Load` of it, what `_ = pc;`
//! lowers to, in no `Emit` range, which naga accepts of an expression
//! nothing consumes.

use crate::handle_set::HandleSet;
use crate::interface::{StaticUses, host_visible};

/// A pin: the expression and the global it keeps accessed.
pub(crate) type Pin = (
    naga::Handle<naga::Expression>,
    naga::Handle<naga::GlobalVariable>,
);

/// The global `handle` pins, if it is a pin of `function`.
pub(crate) fn pinned_global(
    globals: &naga::Arena<naga::GlobalVariable>,
    function: &naga::Function,
    handle: naga::Handle<naga::Expression>,
) -> Option<naga::Handle<naga::GlobalVariable>> {
    if !function.named_expressions.contains_key(&handle) {
        return None;
    }
    pin_shape(globals, &function.expressions, handle)
}

/// The global a pin-shaped expression refers to, whatever its space: a
/// bare `GlobalVariable`, or the `Load` of one.
pub(crate) fn shape(
    expressions: &naga::Arena<naga::Expression>,
    handle: naga::Handle<naga::Expression>,
) -> Option<naga::Handle<naga::GlobalVariable>> {
    match expressions[handle] {
        naga::Expression::GlobalVariable(global) => Some(global),
        naga::Expression::Load { pointer } => match expressions[pointer] {
            naga::Expression::GlobalVariable(global) => Some(global),
            _ => None,
        },
        _ => None,
    }
}

/// The global a pin refers to: a bare `GlobalVariable` of a host-visible
/// global, or the `Load` of an immediate.
fn pin_shape(
    globals: &naga::Arena<naga::GlobalVariable>,
    expressions: &naga::Arena<naga::Expression>,
    handle: naga::Handle<naga::Expression>,
) -> Option<naga::Handle<naga::GlobalVariable>> {
    let global = shape(expressions, handle)?;
    let admitted = match expressions[handle] {
        naga::Expression::Load { .. } => globals[global].space == naga::AddressSpace::Immediate,
        _ => host_visible(&globals[global]),
    };
    admitted.then_some(global)
}

/// Per global, its first pin in `function`.
fn first_pins(
    globals: &naga::Arena<naga::GlobalVariable>,
    function: &naga::Function,
) -> Vec<Option<naga::Handle<naga::Expression>>> {
    let mut first = vec![None; globals.len()];
    for &handle in function.named_expressions.keys() {
        if let Some(global) = pinned_global(globals, function, handle) {
            first[global.index()].get_or_insert(handle);
        }
    }
    first
}

/// The pins of `function`, one per global (its first pin) in the globals'
/// order.
pub(crate) fn of(
    globals: &naga::Arena<naga::GlobalVariable>,
    function: &naga::Function,
) -> Vec<Pin> {
    let first = first_pins(globals, function);
    globals
        .iter()
        .filter_map(|(global, _)| first[global.index()].map(|handle| (handle, global)))
        .collect()
}

/// Keep the pins among `function`'s named expressions, drop the rest (the
/// front end's `let` names, which would root dead values).
pub(crate) fn retain(globals: &naga::Arena<naga::GlobalVariable>, function: &mut naga::Function) {
    let expressions = &function.expressions;
    function
        .named_expressions
        .retain(|&handle, _| pin_shape(globals, expressions, handle).is_some());
}

/// Plant the pins of every function: its call graph's host-visible static
/// uses, as the interface check will read them.  Idempotent.  The arenas
/// grow, so the module's `ModuleInfo` is computed after this.
pub(crate) fn plant(module: &mut naga::Module) {
    let mut planned = Vec::with_capacity(module.functions.len() + module.entry_points.len());
    {
        let mut static_uses = StaticUses::new(module);
        for (handle, function) in module.functions.iter() {
            let uses = static_uses.of_function(handle);
            planned.push(missing_pins(&module.global_variables, function, uses));
        }
        for ep in &module.entry_points {
            let uses = static_uses.of(&ep.function);
            planned.push(missing_pins(&module.global_variables, &ep.function, &uses));
        }
    }
    let mut planned = planned.into_iter();
    let module_globals = &module.global_variables;
    crate::ir::visit::for_each_function_mut(
        &mut module.functions,
        &mut module.entry_points,
        &mut |function| {
            let span = naga::Span::default();
            for global in planned.next().expect("one plan per function") {
                let mut handle = function
                    .expressions
                    .append(naga::Expression::GlobalVariable(global), span);
                if module_globals[global].space == naga::AddressSpace::Immediate {
                    handle = function
                        .expressions
                        .append(naga::Expression::Load { pointer: handle }, span);
                }
                function
                    .named_expressions
                    .insert(handle, "phony".to_string());
            }
        },
    );
}

/// Of `uses`, the host-visible globals `function` has no pin for, in the
/// globals' order.
fn missing_pins(
    globals: &naga::Arena<naga::GlobalVariable>,
    function: &naga::Function,
    uses: &HandleSet<naga::GlobalVariable>,
) -> Vec<naga::Handle<naga::GlobalVariable>> {
    let pinned = first_pins(globals, function);
    globals
        .iter()
        .filter(|&(handle, global)| {
            uses.contains(handle) && host_visible(global) && pinned[handle.index()].is_none()
        })
        .map(|(handle, _)| handle)
        .collect()
}
