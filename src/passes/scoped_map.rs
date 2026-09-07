//! Scope-aware hash map with `O(in-scope writes)` restoration.
//!
//! Dominance-scoped passes (CSE canonicals, known local values, cached
//! loads) would otherwise clone the whole map at every control-flow
//! boundary: `O(map_size * branches)`.  Here every write appends its
//! prior value to an undo log; a scope records the log length on entry
//! and pops back to it on exit.
//!
//! Invariant: rolling back to a checkpoint yields a map observationally
//! equal to the one present when the checkpoint was taken.

use rustc_hash::FxHashMap;
use std::hash::Hash;

#[derive(Debug)]
pub(crate) struct ScopedMap<K: Eq + Hash + Clone, V: Clone> {
    map: FxHashMap<K, V>,
    /// Prior value per write (`None` = key was absent); popped in
    /// reverse on rollback.
    undo: Vec<(K, Option<V>)>,
}

impl<K: Eq + Hash + Clone, V: Clone> ScopedMap<K, V> {
    pub(crate) fn new() -> Self {
        Self {
            map: FxHashMap::default(),
            undo: Vec::new(),
        }
    }

    pub(crate) fn get(&self, key: &K) -> Option<&V> {
        self.map.get(key)
    }

    pub(crate) fn insert(&mut self, key: K, value: V) {
        let prev = self.map.insert(key.clone(), value);
        self.undo.push((key, prev));
    }

    /// A miss is unlogged: nothing to restore, and a tombstone would
    /// only slow rollback.
    pub(crate) fn remove(&mut self, key: &K) {
        if let Some(prev) = self.map.remove(key) {
            self.undo.push((key.clone(), Some(prev)));
        }
    }

    pub(crate) fn checkpoint(&self) -> usize {
        self.undo.len()
    }

    /// Restore the map to its state at `checkpoint`.
    ///
    /// Asserts in release too: a checkpoint from another instance or one
    /// already popped past would otherwise silently leave dataflow acting
    /// on a stale map.
    pub(crate) fn rollback_to(&mut self, checkpoint: usize) {
        assert!(
            checkpoint <= self.undo.len(),
            "rollback_to(checkpoint={checkpoint}) exceeds undo log length {}",
            self.undo.len(),
        );
        while self.undo.len() > checkpoint {
            let (key, prev) = self.undo.pop().expect("len > checkpoint");
            match prev {
                Some(v) => {
                    self.map.insert(key, v);
                }
                None => {
                    self.map.remove(&key);
                }
            }
        }
    }

    pub(crate) fn as_map(&self) -> &FxHashMap<K, V> {
        &self.map
    }

    /// Logged clear: rollback reinstates everything.  Built for loop
    /// handlers, where body and continuing must start from an empty map
    /// (iteration count unknown) yet pre-loop entries return afterwards;
    /// per-key `remove` would cost a second lookup and a key clone each.
    pub(crate) fn drain_logged(&mut self) {
        self.undo.reserve(self.map.len());
        for (k, v) in self.map.drain() {
            self.undo.push((k, Some(v)));
        }
    }

    /// Logged `HashMap::retain`: removed pairs come back on rollback.
    /// Collect-then-`remove` would cost a `Vec<K>` plus a second lookup
    /// per removal on the per-store cache-invalidation hot path.
    pub(crate) fn retain_logged<F>(&mut self, mut predicate: F)
    where
        F: FnMut(&K, &V) -> bool,
    {
        let undo = &mut self.undo;
        self.map.retain(|k, v| {
            if predicate(k, v) {
                true
            } else {
                undo.push((k.clone(), Some(v.clone())));
                false
            }
        });
    }
}

// MARK: Tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rollback_restores_prior_value() {
        let mut m = ScopedMap::<u32, &'static str>::new();
        m.insert(1, "a");
        let cp = m.checkpoint();
        m.insert(1, "b");
        assert_eq!(m.get(&1), Some(&"b"));
        m.rollback_to(cp);
        assert_eq!(m.get(&1), Some(&"a"));
    }

    #[test]
    fn rollback_removes_newly_inserted_keys() {
        let mut m = ScopedMap::<u32, u32>::new();
        let cp = m.checkpoint();
        m.insert(7, 42);
        assert_eq!(m.get(&7), Some(&42));
        m.rollback_to(cp);
        assert_eq!(m.get(&7), None);
    }

    #[test]
    fn rollback_undoes_remove() {
        let mut m = ScopedMap::<u32, u32>::new();
        m.insert(1, 10);
        let cp = m.checkpoint();
        m.remove(&1);
        assert_eq!(m.get(&1), None);
        m.rollback_to(cp);
        assert_eq!(m.get(&1), Some(&10));
    }

    #[test]
    fn remove_absent_key_is_unlogged() {
        let mut m = ScopedMap::<u32, u32>::new();
        let cp = m.checkpoint();
        m.remove(&99);
        assert_eq!(m.checkpoint(), cp);
    }

    #[test]
    fn multiple_overwrites_on_same_key_restore_to_original() {
        let mut m = ScopedMap::<u32, u32>::new();
        m.insert(1, 10);
        let cp = m.checkpoint();
        m.insert(1, 20);
        m.insert(1, 30);
        m.insert(1, 40);
        assert_eq!(m.get(&1), Some(&40));
        m.rollback_to(cp);
        assert_eq!(m.get(&1), Some(&10));
    }

    #[test]
    fn nested_checkpoints_restore_correctly() {
        let mut m = ScopedMap::<u32, u32>::new();
        m.insert(1, 100);
        let outer = m.checkpoint();
        m.insert(2, 200);
        let inner = m.checkpoint();
        m.insert(3, 300);
        m.insert(2, 999);
        m.rollback_to(inner);
        assert_eq!(m.get(&3), None);
        assert_eq!(m.get(&2), Some(&200));
        m.rollback_to(outer);
        assert_eq!(m.get(&2), None);
        assert_eq!(m.get(&1), Some(&100));
    }

    #[test]
    fn drain_logged_empties_map_and_rolls_back_to_prior_state() {
        let mut m = ScopedMap::<u32, u32>::new();
        m.insert(1, 10);
        m.insert(2, 20);
        m.insert(3, 30);
        let cp = m.checkpoint();
        m.drain_logged();
        assert!(m.as_map().is_empty());
        m.rollback_to(cp);
        assert_eq!(m.get(&1), Some(&10));
        assert_eq!(m.get(&2), Some(&20));
        assert_eq!(m.get(&3), Some(&30));
    }

    #[test]
    fn drain_logged_on_empty_map_is_noop() {
        let mut m = ScopedMap::<u32, u32>::new();
        let cp_before = m.checkpoint();
        m.drain_logged();
        assert!(m.as_map().is_empty());
        assert_eq!(m.checkpoint(), cp_before);
    }

    #[test]
    fn drain_logged_composes_with_subsequent_writes() {
        // Loop-handler shape: body writes roll back independently of the drain.
        let mut m = ScopedMap::<u32, u32>::new();
        m.insert(1, 10);
        m.insert(2, 20);
        let cp_pre = m.checkpoint();
        m.drain_logged();
        let cp_empty = m.checkpoint();
        m.insert(3, 30);
        m.rollback_to(cp_empty);
        assert!(m.as_map().is_empty());
        m.rollback_to(cp_pre);
        assert_eq!(m.get(&1), Some(&10));
        assert_eq!(m.get(&2), Some(&20));
        assert_eq!(m.get(&3), None);
    }

    #[test]
    fn retain_logged_drops_failing_entries_and_rolls_back() {
        let mut m = ScopedMap::<u32, u32>::new();
        m.insert(1, 10);
        m.insert(2, 20);
        m.insert(3, 30);
        let cp = m.checkpoint();
        m.retain_logged(|k, _| *k != 2);
        assert_eq!(m.get(&1), Some(&10));
        assert_eq!(m.get(&2), None);
        assert_eq!(m.get(&3), Some(&30));
        m.rollback_to(cp);
        assert_eq!(m.get(&1), Some(&10));
        assert_eq!(m.get(&2), Some(&20));
        assert_eq!(m.get(&3), Some(&30));
    }

    #[test]
    fn retain_logged_keep_all_does_not_advance_undo() {
        let mut m = ScopedMap::<u32, u32>::new();
        m.insert(1, 10);
        m.insert(2, 20);
        let cp = m.checkpoint();
        m.retain_logged(|_, _| true);
        assert_eq!(m.checkpoint(), cp);
        assert_eq!(m.get(&1), Some(&10));
        assert_eq!(m.get(&2), Some(&20));
    }

    #[test]
    fn retain_logged_remove_all_restores_via_rollback() {
        let mut m = ScopedMap::<u32, u32>::new();
        m.insert(1, 10);
        m.insert(2, 20);
        m.insert(3, 30);
        let cp = m.checkpoint();
        m.retain_logged(|_, _| false);
        assert!(m.as_map().is_empty());
        m.rollback_to(cp);
        assert_eq!(m.get(&1), Some(&10));
        assert_eq!(m.get(&2), Some(&20));
        assert_eq!(m.get(&3), Some(&30));
    }
}
