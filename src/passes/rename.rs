//! Identifier rename pass.  Replaces every user-chosen identifier the
//! pipeline is free to touch with a short generated name, keeping the
//! output deterministic and collision-free against both WGSL reserved
//! words and any symbols the caller asked to preserve.
//!
//! Struct type and member names are renamed at the generator layer
//! (naga's `UniqueArena<Type>` is immutable mid-pipeline), so this
//! pass only covers the mutable arenas: globals, constants, overrides,
//! functions, arguments, and locals.

use std::collections::HashSet;

use crate::error::Error;
use crate::name_gen;
use crate::pipeline::{Pass, PassContext};

/// Rename pass state.  `preserve` lists names that must survive
/// verbatim; `mangle` extends the rename scope to include constants
/// and overrides (whose names otherwise leak into the final output).
#[derive(Debug)]
pub struct RenamePass {
    preserve: HashSet<String>,
    mangle: bool,
}

impl RenamePass {
    /// Construct a new pass from the user-facing `preserve_symbols`
    /// vector and the resolved `mangle` flag.
    pub fn new(preserve_symbols: Vec<String>, mangle: bool) -> Self {
        Self {
            preserve: preserve_symbols.into_iter().collect(),
            mangle,
        }
    }
}

impl Pass for RenamePass {
    fn name(&self) -> &'static str {
        "rename_identifiers"
    }

    fn run(&mut self, module: &mut naga::Module, _ctx: &PassContext<'_>) -> Result<bool, Error> {
        let mut counter = 0usize;
        let mut changed = false;
        let mut used_names = collect_reserved_names(module, &self.preserve, self.mangle);

        // Mangling extends renaming to module-scope constants and
        // overrides for minimal output size.  Struct types and
        // members are still handled at the generator layer because
        // naga's `UniqueArena<Type>` is immutable after lowering.
        if self.mangle {
            for (_, c) in module.constants.iter_mut() {
                if let Some(name) = c.name.as_deref()
                    && !self.preserve.contains(name)
                {
                    let new_name = next_available_name(&mut counter, &mut used_names);
                    changed |= c.name.as_deref() != Some(new_name.as_str());
                    c.name = Some(new_name);
                }
            }

            for (_, ov) in module.overrides.iter_mut() {
                if let Some(name) = ov.name.as_deref()
                    && !self.preserve.contains(name)
                {
                    let new_name = next_available_name(&mut counter, &mut used_names);
                    changed |= ov.name.as_deref() != Some(new_name.as_str());
                    ov.name = Some(new_name);
                }
            }
        }

        for (_, global) in module.global_variables.iter_mut() {
            if let Some(name) = global.name.as_deref()
                && self.preserve.contains(name)
            {
                continue;
            }
            let new_name = next_available_name(&mut counter, &mut used_names);
            changed |= global.name.as_deref() != Some(new_name.as_str());
            global.name = Some(new_name);
        }

        for (_, function) in module.functions.iter_mut() {
            changed |= rename_function(
                function,
                &self.preserve,
                &mut counter,
                &mut used_names,
                true,
            );
        }

        for entry in module.entry_points.iter_mut() {
            changed |= rename_function(
                &mut entry.function,
                &self.preserve,
                &mut counter,
                &mut used_names,
                false,
            );
        }

        Ok(changed)
    }
}

/// Rename every argument and local inside `function`, optionally
/// renaming the function itself.  Entry points pass
/// `rename_function_name = false` so their pipeline-bound name stays
/// intact; regular functions pass `true`.  `named_expressions` is
/// cleared unconditionally because downstream passes would otherwise
/// see stale bindings that reference long names the generator is
/// about to replace.
fn rename_function(
    function: &mut naga::Function,
    preserve: &HashSet<String>,
    counter: &mut usize,
    used_names: &mut HashSet<String>,
    rename_function_name: bool,
) -> bool {
    let mut changed = false;

    if rename_function_name {
        if let Some(name) = function.name.as_deref() {
            if !preserve.contains(name) {
                let next = next_available_name(counter, used_names);
                changed |= function.name.as_deref() != Some(next.as_str());
                function.name = Some(next);
            }
        } else {
            function.name = Some(next_available_name(counter, used_names));
            changed = true;
        }
    }

    for argument in function.arguments.iter_mut() {
        if let Some(name) = argument.name.as_deref()
            && preserve.contains(name)
        {
            continue;
        }
        let next = next_available_name(counter, used_names);
        changed |= argument.name.as_deref() != Some(next.as_str());
        argument.name = Some(next);
    }

    for (_, local) in function.local_variables.iter_mut() {
        if let Some(name) = local.name.as_deref()
            && preserve.contains(name)
        {
            continue;
        }
        let next = next_available_name(counter, used_names);
        changed |= local.name.as_deref() != Some(next.as_str());
        local.name = Some(next);
    }

    // Clear `named_expressions` whenever the function carries any -
    // even if no identifier was renamed in this invocation - and
    // report that as a change.  The "report as change" piece looks
    // like a perf wart (it costs one extra convergence sweep on
    // shaders that name expressions but rename nothing else), but
    // experimentally it is load-bearing: in real-world shaders the
    // extra sweep gives downstream passes a chance to observe IR that
    // was settled only in this sweep's earlier passes (e.g. DCE
    // catches an orphaned global that became unreachable once an
    // upstream `_ = expr` phony-assignment-style load was eliminated).
    if !function.named_expressions.is_empty() {
        function.named_expressions.clear();
        changed = true;
    }

    changed
}

/// Build the starting `used_names` set for one rename sweep.
///
/// The policy differs by `mangle` because idempotence hinges on it:
///
/// - With `mangle = false`, constant and override names are kept
///   verbatim and must be reserved so generated names do not collide
///   with them.
/// - With `mangle = true`, those names are themselves rewritten, so
///   reserving the previous sweep's assignments would pollute the used
///   set and shift subsequent assignments one slot, producing a
///   two-sweep oscillation that keeps the pipeline from converging.
///
/// Preserve-listed entries in every arena are always reserved
/// regardless of `mangle` so the user-visible names survive.
fn collect_reserved_names(
    module: &naga::Module,
    preserve: &HashSet<String>,
    mangle: bool,
) -> HashSet<String> {
    let mut reserved = HashSet::new();

    if !mangle {
        for (_, c) in module.constants.iter() {
            if let Some(name) = c.name.as_deref() {
                reserved.insert(name.to_string());
            }
        }

        for (_, ov) in module.overrides.iter() {
            if let Some(name) = ov.name.as_deref() {
                reserved.insert(name.to_string());
            }
        }
    } else {
        // Even when mangling, preserve-listed constants and overrides
        // retain their names and must be reserved.
        for (_, c) in module.constants.iter() {
            if let Some(name) = c.name.as_deref()
                && preserve.contains(name)
            {
                reserved.insert(name.to_string());
            }
        }
        for (_, ov) in module.overrides.iter() {
            if let Some(name) = ov.name.as_deref()
                && preserve.contains(name)
            {
                reserved.insert(name.to_string());
            }
        }
    }

    for (_, global) in module.global_variables.iter() {
        if let Some(name) = global.name.as_deref()
            && preserve.contains(name)
        {
            reserved.insert(name.to_string());
        }
    }

    for (_, function) in module.functions.iter() {
        if let Some(name) = function.name.as_deref()
            && preserve.contains(name)
        {
            reserved.insert(name.to_string());
        }
        collect_preserved_function_names(function, preserve, &mut reserved);
    }

    for entry in module.entry_points.iter() {
        reserved.insert(entry.name.clone());
        collect_preserved_function_names(&entry.function, preserve, &mut reserved);
    }

    // Always reserve every source struct type name and struct member
    // name, regardless of `mangle`.  Two reasons:
    //
    // * Under `mangle = false` the generator emits source struct type
    //   names verbatim (see `core.rs::type_names`).  If the rename
    //   counter ever mints one of those names for a global / function
    //   / local, the WGSL output contains two same-named symbols and
    //   the user's struct becomes unreachable.  Round-trip validation
    //   catches the bad output (see lib.rs fallback), but the bug
    //   silently halves compaction quality.
    //
    // * Under `mangle = true` the generator independently re-mangles
    //   struct type / member names via its own `used_names` set in
    //   `core.rs`, so the source names are *not* the final WGSL names.
    //   Reserving them is still harmless: the rename counter just
    //   skips a few short names, and the generator's mangling decides
    //   what each struct is ultimately called.  The cost of reserving
    //   here is bounded by the number of source-named structs.
    //
    // Preserve-listed names in the type / member arenas are an extra
    // case the generator never mangles regardless of `mangle`, so
    // reserving unconditionally also covers them.
    for (_, ty) in module.types.iter() {
        if let Some(name) = ty.name.as_deref() {
            reserved.insert(name.to_string());
        }
        if let naga::TypeInner::Struct { members, .. } = &ty.inner {
            for m in members {
                if let Some(name) = m.name.as_deref() {
                    reserved.insert(name.to_string());
                }
            }
        }
    }

    reserved
}

/// Reserve every preserve-listed argument and local inside `function`
/// so the rename sweep cannot repurpose those names for unrelated
/// declarations elsewhere in the module.
fn collect_preserved_function_names(
    function: &naga::Function,
    preserve: &HashSet<String>,
    reserved: &mut HashSet<String>,
) {
    for argument in function.arguments.iter() {
        if let Some(name) = argument.name.as_deref()
            && preserve.contains(name)
        {
            reserved.insert(name.to_string());
        }
    }

    for (_, local) in function.local_variables.iter() {
        if let Some(name) = local.name.as_deref()
            && preserve.contains(name)
        {
            reserved.insert(name.to_string());
        }
    }
}

/// Thin wrapper over [`name_gen::next_name_insert`] so the pass body
/// is readable without the longer function name spelled out.
fn next_available_name(counter: &mut usize, used_names: &mut HashSet<String>) -> String {
    name_gen::next_name_insert(counter, used_names)
}

// MARK: Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use std::collections::HashSet;

    fn follows_expected_name_pattern(name: &str) -> bool {
        let mut chars = name.chars();
        let Some(first) = chars.next() else {
            return false;
        };

        if !first.is_ascii_alphabetic() {
            return false;
        }

        chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
    }

    fn run_pass(source: &str, preserve: &[&str]) -> (bool, naga::Module) {
        run_pass_with_mangle(source, preserve, false)
    }

    fn run_pass_with_mangle(source: &str, preserve: &[&str], mangle: bool) -> (bool, naga::Module) {
        let mut module = naga::front::wgsl::parse_str(source).expect("source should parse");
        let mut pass = RenamePass::new(preserve.iter().map(|s| s.to_string()).collect(), mangle);
        let config = Config::default();
        let ctx = PassContext {
            config: &config,
            trace_run_dir: None,
        };

        let changed = pass.run(&mut module, &ctx).expect("rename pass should run");
        let _ = crate::io::validate_module(&module).expect("module should remain valid");
        (changed, module)
    }

    fn collect_declaration_names(module: &naga::Module) -> Vec<String> {
        let mut out = Vec::new();

        for (_, global) in module.global_variables.iter() {
            if let Some(name) = global.name.as_deref() {
                out.push(name.to_string());
            }
        }

        for (_, function) in module.functions.iter() {
            if let Some(name) = function.name.as_deref() {
                out.push(name.to_string());
            }
            for argument in function.arguments.iter() {
                if let Some(name) = argument.name.as_deref() {
                    out.push(name.to_string());
                }
            }
            for (_, local) in function.local_variables.iter() {
                if let Some(name) = local.name.as_deref() {
                    out.push(name.to_string());
                }
            }
        }

        for entry in module.entry_points.iter() {
            for argument in entry.function.arguments.iter() {
                if let Some(name) = argument.name.as_deref() {
                    out.push(name.to_string());
                }
            }
            for (_, local) in entry.function.local_variables.iter() {
                if let Some(name) = local.name.as_deref() {
                    out.push(name.to_string());
                }
            }
        }

        out
    }

    fn count_declaration_name(module: &naga::Module, target: &str) -> usize {
        collect_declaration_names(module)
            .into_iter()
            .filter(|name| name == target)
            .count()
    }

    #[test]
    fn renames_non_preserved_identifiers_and_keeps_names_unique() {
        let source = r#"
var<private> global_long_name: f32 = 2.0;

fn helper(input_value: f32) -> f32 {
    var local_value: f32;
    local_value = input_value + global_long_name;
    return local_value;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    let y = helper(1.0);
    return vec4f(y, 0.0, 0.0, 1.0);
}
"#;

        let (changed, module) = run_pass(source, &[]);
        assert!(changed, "rename pass should report changes");

        let decl_names = collect_declaration_names(&module);
        assert!(
            !decl_names.iter().any(|n| n == "global_long_name"),
            "global should be renamed"
        );
        assert!(
            !decl_names.iter().any(|n| n == "helper"),
            "helper function should be renamed"
        );
        assert!(
            !decl_names.iter().any(|n| n == "input_value"),
            "helper argument should be renamed"
        );
        assert!(
            !decl_names.iter().any(|n| n == "local_value"),
            "helper local should be renamed"
        );

        let unique_count = decl_names.iter().collect::<HashSet<_>>().len();
        assert_eq!(
            unique_count,
            decl_names.len(),
            "renamed declarations should be unique"
        );

        assert!(
            decl_names.iter().all(|n| follows_expected_name_pattern(n)),
            "all generated declaration names should match expected character pattern"
        );

        assert_eq!(
            module.entry_points[0].name, "fs_main",
            "entry point name should not change"
        );
    }

    #[test]
    fn preserves_requested_symbols_without_reusing_them() {
        let source = r#"
var<private> keep_global: f32 = 1.0;
var<private> rename_global: f32 = 2.0;

fn helper(keep_arg: f32, rename_arg: f32) -> f32 {
    var rename_local: f32;
    rename_local = keep_arg + rename_arg + keep_global + rename_global;
    return rename_local;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    let y = helper(1.0, 2.0);
    return vec4f(y, 0.0, 0.0, 1.0);
}
"#;

        let (changed, module) = run_pass(source, &["keep_global", "keep_arg"]);
        assert!(
            changed,
            "rename pass should still rename non-preserved symbols"
        );

        assert_eq!(
            count_declaration_name(&module, "keep_global"),
            1,
            "preserved global name should remain and not be reused"
        );
        assert_eq!(
            count_declaration_name(&module, "keep_arg"),
            1,
            "preserved argument name should remain and not be reused"
        );

        let decl_names = collect_declaration_names(&module);
        assert!(decl_names.iter().any(|n| n == "keep_global"));
        assert!(decl_names.iter().any(|n| n == "keep_arg"));
        assert!(!decl_names.iter().any(|n| n == "rename_global"));
        assert!(!decl_names.iter().any(|n| n == "rename_arg"));
        assert!(!decl_names.iter().any(|n| n == "rename_local"));
    }

    #[test]
    fn name_generator_sequence_and_pattern_are_expected() {
        let mut counter = 0usize;
        let generated = (0..120)
            .map(|_| name_gen::next_name(&mut counter))
            .collect::<Vec<_>>();

        assert_eq!(generated[0], "A");
        assert_eq!(generated[1], "a");
        assert_eq!(generated[2], "B");

        assert!(
            generated.iter().all(|n| follows_expected_name_pattern(n)),
            "all generated names should follow FIRST/NEXT character-table pattern"
        );
    }

    #[test]
    fn mangle_renames_constants() {
        let source = r#"
const MY_CONSTANT: f32 = 3.14;

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(MY_CONSTANT, 0.0, 0.0, 1.0);
}
"#;

        let (changed, module) = run_pass_with_mangle(source, &[], true);
        assert!(changed, "mangle should rename constants");

        let has_original = module
            .constants
            .iter()
            .any(|(_, c)| c.name.as_deref() == Some("MY_CONSTANT"));
        assert!(!has_original, "original constant name should be replaced");

        // All constant names should follow the short-name pattern.
        for (_, c) in module.constants.iter() {
            if let Some(name) = c.name.as_deref() {
                assert!(
                    follows_expected_name_pattern(name),
                    "mangled constant name '{}' should follow pattern",
                    name
                );
            }
        }
    }

    #[test]
    fn mangle_preserves_specified_constants() {
        let source = r#"
const KEEP_ME: f32 = 1.0;
const RENAME_ME: f32 = 2.0;

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(KEEP_ME, RENAME_ME, 0.0, 1.0);
}
"#;

        let (_, module) = run_pass_with_mangle(source, &["KEEP_ME"], true);
        let has_keep = module
            .constants
            .iter()
            .any(|(_, c)| c.name.as_deref() == Some("KEEP_ME"));
        assert!(has_keep, "preserved constant should keep its name");

        let has_rename = module
            .constants
            .iter()
            .any(|(_, c)| c.name.as_deref() == Some("RENAME_ME"));
        assert!(!has_rename, "non-preserved constant should be renamed");
    }

    #[test]
    fn no_collision_with_unrenamed_constant_names() {
        // With mangle off, constants keep their original names.  The
        // generator must not reassign those same names to other
        // declarations.  Here the constant `A` is exactly the first
        // generated name, so the sweep must skip past it.
        let source = r#"
const A: f32 = 1.0;
var<private> long_global_name: f32 = 2.0;

fn helper(x: f32) -> f32 {
    return x + A + long_global_name;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    let y = helper(1.0);
    return vec4f(y, 0.0, 0.0, 1.0);
}
"#;

        let (_, module) = run_pass(source, &[]);

        // Constant "A" should still exist.
        let has_const_a = module
            .constants
            .iter()
            .any(|(_, c)| c.name.as_deref() == Some("A"));
        assert!(
            has_const_a,
            "constant A should keep its name when mangle is off"
        );

        // No renamed declaration should collide with "A".
        let decl_names = collect_declaration_names(&module);
        let a_count = decl_names.iter().filter(|n| n.as_str() == "A").count();
        assert_eq!(
            a_count, 0,
            "no generated declaration name should collide with unrenamed constant 'A'"
        );
    }

    #[test]
    fn mangle_rename_is_idempotent() {
        // Running the rename pass twice with `mangle = true` on the
        // same module must produce identical results; the second run
        // has to report `changed = false`.  Without this the pipeline's
        // convergence loop oscillates between two name assignments.
        let source = r#"
const LONG_CONST_A: f32 = 1.0;
const LONG_CONST_B: f32 = 2.0;
var<private> long_global: f32 = 3.0;

fn helper(x: f32) -> f32 {
    var tmp: f32;
    tmp = x + LONG_CONST_A + LONG_CONST_B + long_global;
    return tmp;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    let y = helper(1.0);
    return vec4f(y, 0.0, 0.0, 1.0);
}
"#;

        let (changed1, module1) = run_pass_with_mangle(source, &[], true);
        assert!(changed1, "first rename should change names");

        // Run a second rename on the already-renamed module.
        let mut module2 = module1.clone();
        let mut pass = RenamePass::new(Vec::new(), true);
        let config = Config::default();
        let ctx = PassContext {
            config: &config,
            trace_run_dir: None,
        };
        let changed2 = pass
            .run(&mut module2, &ctx)
            .expect("second rename should work");
        assert!(
            !changed2,
            "second rename with mangle should be idempotent (no changes)"
        );

        // Names should be identical.
        let names1 = collect_declaration_names(&module1);
        let names2 = collect_declaration_names(&module2);
        assert_eq!(names1, names2, "names must be identical across runs");
    }

    #[test]
    fn clears_named_expressions_and_reports_change_even_when_nothing_renamed() {
        // INVERSE regression: a tempting "perf" fix gated this clear
        // on `changed > 0` so a preserve-all pass would not report a
        // change.  Real-world test corpus
        // (data/extra-test4/bug/{tint/1737, chromium/1449474}.wgsl,
        // data/extra-test3/glsl-fma.frag.wgsl) showed the convergence
        // loop exits one sweep too early in that mode and downstream
        // DCE never gets a chance to remove orphaned globals.  This
        // test pins the "always clear and report changed" behaviour
        // so a future maintainer who notices the apparent redundancy
        // does not silently re-introduce the regression.
        let source = r#"
@fragment
fn fs_main() -> @location(0) vec4f {
    let y = 1.0;
    return vec4f(y, 0.0, 0.0, 1.0);
}
"#;
        let preserve = ["fs_main", "y"];
        let (changed, module) = run_pass_with_mangle(source, &preserve, true);
        assert!(
            changed,
            "rename must report `changed = true` when it clears `named_expressions`, \
             even if no identifier was renamed - downstream convergence depends on it"
        );
        let entry = module
            .entry_points
            .first()
            .expect("entry point should exist");
        assert!(
            entry.function.named_expressions.is_empty(),
            "named_expressions must be cleared so the WGSL emitter does not pick up \
             stale bindings on the next sweep"
        );
    }

    // MARK: Struct-name reservation regression

    /// Source struct type names must be reserved as the rename pass
    /// mints fresh short identifiers - otherwise (with `mangle = false`,
    /// where the generator keeps source struct names verbatim) the
    /// pass can mint a local / global / function name that collides
    /// with an existing struct name and the emitted WGSL ends up with
    /// two same-named symbols.
    #[test]
    fn reserves_source_struct_type_names_without_mangle() {
        // The struct is named single-character `A`; with enough
        // unrenamed globals and a multi-arg function the rename
        // counter would normally reach `A` quickly and clobber the
        // type name.  We assert `A` appears in the reserved set.
        let src = r#"
struct A { x: f32, y: f32 }
@group(0) @binding(0) var<uniform> g: A;
fn h(p: f32, q: f32, r: f32) -> A {
    var out: A;
    out.x = p + q;
    out.y = r + g.x;
    return out;
}
@fragment fn m() -> @location(0) vec4f {
    let v = h(1.0, 2.0, 3.0);
    return vec4f(v.x, v.y, 0.0, 1.0);
}
"#;
        let module = naga::front::wgsl::parse_str(src).expect("parses");
        let preserve = HashSet::new();
        let reserved = collect_reserved_names(&module, &preserve, /*mangle=*/ false);
        assert!(
            reserved.contains("A"),
            "source struct type name must be in the reserved set so rename does not \
             collide with it"
        );
        assert!(
            reserved.contains("x") && reserved.contains("y"),
            "source struct member names must be in the reserved set"
        );
    }

    /// Same reservation must apply under `mangle = true`: even though
    /// the generator independently re-mangles type / member names,
    /// reserving them in the rename pass is safe and prevents short-
    /// lived collisions that a future refactor could expose.
    #[test]
    fn reserves_source_struct_type_names_with_mangle() {
        let src = r#"
struct A { x: f32 }
@group(0) @binding(0) var<uniform> g: A;
@fragment fn m() -> @location(0) vec4f {
    return vec4f(g.x);
}
"#;
        let module = naga::front::wgsl::parse_str(src).expect("parses");
        let preserve = HashSet::new();
        let reserved = collect_reserved_names(&module, &preserve, /*mangle=*/ true);
        assert!(
            reserved.contains("A"),
            "source struct type name must be reserved even under mangle"
        );
        assert!(
            reserved.contains("x"),
            "source struct member name must be reserved even under mangle"
        );
    }
}
