//! Original-to-final identifier map for hosts that address shaders by
//! source names (bindings, struct members, entry points, overrides) and
//! would otherwise have to forfeit mangling.  Renames happen in two
//! places - the rename pass mutates module-scope names in the IR across
//! sweeps, the generator names struct types / members at emission without
//! touching the IR - so [`NameLog`] accumulates the former, `Emission`
//! reports the latter, and [`crate::run`] assembles the [`NameMap`].
//! Keys are ORIGINAL names, values final; every surviving module-scope
//! declaration has an entry (identity included), so an absent key means
//! eliminated, never unchanged.

use naga::proc::NameKey;
use std::collections::{BTreeMap, HashMap};

/// Module-scope renames composed across sweeps (`a -> b` then `b -> c`
/// yields `a -> c`).  Module scope is one namespace, so a current name
/// identifies one declaration across all four renameable arenas.
#[derive(Debug, Default, Clone)]
pub struct NameLog {
    current_to_original: HashMap<String, String>,
}

impl NameLog {
    /// One sweep's renames, applied simultaneously: an old name may equal
    /// ANOTHER pair's new name (`x -> a` while `a -> b`), so originals
    /// resolve against the pre-batch state before any insert.
    pub fn record_batch(&mut self, renames: &[(String, String)]) {
        let renames: Vec<&(String, String)> =
            renames.iter().filter(|(old, new)| old != new).collect();
        let originals: Vec<String> = renames
            .iter()
            .map(|(old, _)| {
                self.current_to_original
                    .remove(old)
                    .unwrap_or_else(|| old.clone())
            })
            .collect();
        for ((_, new), original) in renames.iter().zip(originals) {
            self.current_to_original.insert(new.clone(), original);
        }
    }

    /// `None`: never renamed, or synthetic.
    pub fn original_of(&self, current: &str) -> Option<&str> {
        self.current_to_original.get(current).map(String::as_str)
    }
}

/// One emitted struct: final type name plus member map.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructRename {
    /// Emitted struct type name.
    pub name: String,
    /// Original member name -> emitted member name, identity included.
    pub members: BTreeMap<String, String>,
}

/// Original-to-final map for every surviving module-scope symbol; `None`
/// when the shipped text carries no rename (a bailout, the verbatim guard).
/// The naga-emitter fallback gets one spelled as that emitter prints.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NameMap {
    /// Entry-point names, always identity (pipeline-bound, never renamed).
    pub entry_points: BTreeMap<String, String>,
    /// Module-scope `var` declarations, resource bindings included.
    pub globals: BTreeMap<String, String>,
    /// Function declarations; a specialization's first clone keeps the
    /// original's name, later clones have no original and are keyed by
    /// their synthesized `<name>_sp<n>`.
    pub functions: BTreeMap<String, String>,
    /// Module-scope `const` declarations.
    pub constants: BTreeMap<String, String>,
    /// `override` declarations.
    pub overrides: BTreeMap<String, String>,
    /// Emitted struct types, keyed by original struct name.
    pub structs: BTreeMap<String, StructRename>,
}

impl NameMap {
    /// From the FINAL module: names resolved back through `log` and printed
    /// by `spell` (identity for the generator, naga's namer for its fallback);
    /// `live_const_names` drops constants whose every use folded away and
    /// left no declaration (`None`: all declared).  Preamble-owned symbols
    /// appear as identity.
    pub(crate) fn assemble(
        module: &naga::Module,
        log: &NameLog,
        structs: BTreeMap<String, StructRename>,
        live_const_names: Option<&std::collections::HashSet<String>>,
        spell: &dyn Fn(&str, NameKey) -> String,
    ) -> Self {
        let mut map = NameMap {
            structs,
            ..Default::default()
        };
        let insert = |bucket: &mut BTreeMap<String, String>, ir: &Option<String>, key: NameKey| {
            if let Some(ir) = ir {
                let original = log.original_of(ir).unwrap_or(ir);
                bucket.insert(original.to_owned(), spell(ir, key));
            }
        };
        for (h, gv) in module.global_variables.iter() {
            insert(&mut map.globals, &gv.name, NameKey::GlobalVariable(h));
        }
        for (h, f) in module.functions.iter() {
            insert(&mut map.functions, &f.name, NameKey::Function(h));
        }
        for (h, c) in module.constants.iter() {
            if live_const_names
                .is_none_or(|live| c.name.as_deref().is_some_and(|n| live.contains(n)))
            {
                insert(&mut map.constants, &c.name, NameKey::Constant(h));
            }
        }
        for (h, o) in module.overrides.iter() {
            insert(&mut map.overrides, &o.name, NameKey::Override(h));
        }
        for (i, entry) in module.entry_points.iter().enumerate() {
            map.entry_points.insert(
                entry.name.clone(),
                spell(&entry.name, NameKey::EntryPoint(i as u16)),
            );
        }
        map
    }

    /// The map for naga's emitter output: [`NameMap::assemble`] printed as
    /// [`naga_spelling`] spells it, structs and members included (that
    /// emitter keeps the IR's struct names and declares every constant).
    pub(crate) fn assemble_naga_spelled(module: &naga::Module, log: &NameLog) -> Self {
        let names = naga_spelling(module);
        let spelled =
            |ir: &str, key: NameKey| names.get(&key).cloned().unwrap_or_else(|| ir.to_owned());
        let mut structs = BTreeMap::new();
        for (h, ty) in module.types.iter() {
            if let (Some(name), naga::TypeInner::Struct { members, .. }) =
                (ty.name.as_deref(), &ty.inner)
            {
                // Inserts, not `collect`: a `BTreeMap` built from an iterator
                // sorts first, a code path nothing else instantiates.
                let mut member_map = BTreeMap::new();
                for (i, m) in members.iter().enumerate() {
                    if let Some(n) = m.name.as_deref() {
                        member_map
                            .insert(n.to_owned(), spelled(n, NameKey::StructMember(h, i as u32)));
                    }
                }
                structs.insert(
                    name.to_owned(),
                    StructRename {
                        name: spelled(name, NameKey::Type(h)),
                        members: member_map,
                    },
                );
            }
        }
        Self::assemble(module, log, structs, None, &spelled)
    }
}

/// The names naga's WGSL back-end prints, keyed like the IR: its `Namer`
/// suffixes reserved words and trailing digits (`fs1` -> `fs1_`) and numbers
/// repeats.  The seeding must mirror `naga::back::wgsl::Writer`'s exactly.
fn naga_spelling(module: &naga::Module) -> naga::FastHashMap<NameKey, String> {
    let mut names = naga::FastHashMap::default();
    naga::proc::Namer::default().reset(
        module,
        &naga::keywords::wgsl::RESERVED_SET,
        &naga::keywords::wgsl::BUILTIN_IDENTIFIER_SET,
        naga::proc::CaseInsensitiveKeywordSet::empty(),
        &["__", "_naga"],
        &mut names,
    );
    names
}

/// The first host-addressed name (entry point, named override, preserved
/// global / function / constant / struct / member) naga's emitter would print
/// differently from the IR, as `(ir, printed)`: such a fallback breaks the
/// host's contract and never ships.  Under an identity log the spelled map's
/// keys are the IR names (nagami never renames these kinds).
pub(crate) fn naga_respelled_interface_name(
    module: &naga::Module,
    preserve: &[String],
) -> Option<(String, String)> {
    let map = NameMap::assemble_naga_spelled(module, &NameLog::default());
    let differs = |(k, v): (&String, &String)| (k != v).then(|| (k.clone(), v.clone()));
    let preserved = |(k, _): &(&String, &String)| preserve.contains(k);
    map.entry_points
        .iter()
        .chain(map.overrides.iter())
        .find_map(differs)
        .or_else(|| {
            map.globals
                .iter()
                .chain(map.functions.iter())
                .chain(map.constants.iter())
                .filter(preserved)
                .find_map(differs)
        })
        .or_else(|| {
            map.structs.iter().find_map(|(k, s)| {
                (preserve.contains(k) && &s.name != k)
                    .then(|| (k.clone(), s.name.clone()))
                    .or_else(|| s.members.iter().filter(preserved).find_map(differs))
            })
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chains_compose_across_batches() {
        let mut log = NameLog::default();
        log.record_batch(&[("longName".into(), "a".into())]);
        log.record_batch(&[("a".into(), "b".into())]);
        assert_eq!(log.original_of("b"), Some("longName"));
        assert_eq!(log.original_of("a"), None, "intermediate names vanish");
    }

    #[test]
    fn same_batch_swap_does_not_cross_link() {
        // `x -> a` while the declaration currently named `a` moves to `b`:
        // sequential composition would thread x through a's chain.
        let mut log = NameLog::default();
        log.record_batch(&[("orig".into(), "a".into())]);
        log.record_batch(&[("a".into(), "b".into()), ("x".into(), "a".into())]);
        assert_eq!(log.original_of("b"), Some("orig"));
        assert_eq!(log.original_of("a"), Some("x"));
    }

    /// The fallback's map says what naga's namer printed (`A1` -> `A1_`), and
    /// a respelled host-visible name blocks the fallback.
    #[test]
    fn naga_spelled_map_reads_the_emitter_suffixes() {
        let module = naga::front::wgsl::parse_str(
            "struct S1 { m1: f32 }\n\
             @group(0) @binding(0) var<uniform> A1: S1;\n\
             fn f1() -> f32 { return A1.m1; }\n\
             @fragment fn fs() -> @location(0) vec4f { return vec4f(f1()); }",
        )
        .expect("parses");
        let mut log = NameLog::default();
        log.record_batch(&[("tint".into(), "A1".into())]);
        let map = NameMap::assemble_naga_spelled(&module, &log);
        assert_eq!(map.globals["tint"], "A1_");
        assert_eq!(map.functions["f1"], "f1_");
        assert_eq!(map.structs["S1"].name, "S1_");
        assert_eq!(map.structs["S1"].members["m1"], "m1_");
        assert_eq!(map.entry_points["fs"], "fs");
        let info = crate::io::validate_module(&module).expect("valid");
        let text =
            naga::back::wgsl::write_string(&module, &info, naga::back::wgsl::WriterFlags::empty())
                .expect("naga prints it");
        for printed in ["A1_", "fn f1_", "struct S1_", "m1_"] {
            assert!(text.contains(printed), "{printed} in {text}");
        }
        assert_eq!(
            naga_respelled_interface_name(&module, &["A1".to_owned()]),
            Some(("A1".to_owned(), "A1_".to_owned()))
        );
        assert_eq!(naga_respelled_interface_name(&module, &[]), None);
    }

    #[test]
    fn identity_renames_are_not_recorded() {
        let mut log = NameLog::default();
        log.record_batch(&[("a".into(), "a".into())]);
        assert_eq!(log.original_of("a"), None);
    }
}
