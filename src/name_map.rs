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

/// Original-to-final map for every surviving module-scope symbol; produced
/// only when the custom generator's output shipped (the naga-emitter
/// fallback and the verbatim guard ship names this map would misreport).
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
    /// From the FINAL module: module-scope names resolved back through
    /// `log`, struct names from `structs`, constants filtered by
    /// `live_const_names` (a constant can survive the IR with every use
    /// folded away, leaving no declaration).  Preamble-owned symbols appear
    /// as identity: their declarations live in the consumer's preamble.
    pub(crate) fn assemble(
        module: &naga::Module,
        log: &NameLog,
        structs: BTreeMap<String, StructRename>,
        live_const_names: &std::collections::HashSet<String>,
    ) -> Self {
        let mut map = NameMap {
            structs,
            ..Default::default()
        };
        let insert = |bucket: &mut BTreeMap<String, String>, current: &Option<String>| {
            if let Some(current) = current {
                let original = log.original_of(current).unwrap_or(current);
                bucket.insert(original.to_string(), current.clone());
            }
        };
        for (_, gv) in module.global_variables.iter() {
            insert(&mut map.globals, &gv.name);
        }
        for (_, f) in module.functions.iter() {
            insert(&mut map.functions, &f.name);
        }
        for (_, c) in module.constants.iter() {
            if c.name
                .as_deref()
                .is_some_and(|n| live_const_names.contains(n))
            {
                insert(&mut map.constants, &c.name);
            }
        }
        for (_, o) in module.overrides.iter() {
            insert(&mut map.overrides, &o.name);
        }
        for entry in module.entry_points.iter() {
            map.entry_points
                .insert(entry.name.clone(), entry.name.clone());
        }
        map
    }
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

    #[test]
    fn identity_renames_are_not_recorded() {
        let mut log = NameLog::default();
        log.record_batch(&[("a".into(), "a".into())]);
        assert_eq!(log.original_of("a"), None);
    }
}
