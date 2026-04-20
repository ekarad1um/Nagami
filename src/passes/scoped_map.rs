//! Scope-aware hash map with `O(in-scope writes)` restoration.
//!
//! Optimization passes that track per-scope facts (CSE canonical
//! handles, known local-variable values, and so on) traditionally
//! model dominance by snapshotting the whole map at each control-flow
//! boundary and overwriting it on exit.  For large maps and deep
//! nesting this is `O(map_size * branches)`.
//!
//! `ScopedMap` keeps a single backing `HashMap` and appends every
//! write to an undo log.  A scope captures the log length via
//! `ScopedMap::checkpoint` on entry and replays the tail in reverse
//! via `ScopedMap::rollback_to` on exit, so cost becomes
//! `O(writes_in_scope)` per scope, strictly bounded by the clone cost
//! and typically far smaller.
//!
//! The semantics match snapshot-clone-restore exactly: rolling back to
//! a checkpoint yields a map that is observationally equal to the one
//! present when the checkpoint was taken.

use std::collections::HashMap;
use std::hash::Hash;

/// A `HashMap<K, V>` augmented with an append-only undo log that enables
/// O(in-scope writes) restoration via checkpoints.
#[derive(Debug)]
pub(crate) struct ScopedMap<K: Eq + Hash + Clone, V: Clone> {
    map: HashMap<K, V>,
    /// Each entry records a key plus the value previously held under
    /// that key (or `None` if the key was absent).  Appended on every
    /// write; popped in reverse on rollback.
    undo: Vec<(K, Option<V>)>,
}

impl<K: Eq + Hash + Clone, V: Clone> ScopedMap<K, V> {
    pub(crate) fn new() -> Self {
        Self {
            map: HashMap::new(),
            undo: Vec::new(),
        }
    }

    pub(crate) fn get(&self, key: &K) -> Option<&V> {
        self.map.get(key)
    }

    /// Insert `value` under `key`, logging the prior value for undo.
    pub(crate) fn insert(&mut self, key: K, value: V) {
        let prev = self.map.insert(key.clone(), value);
        self.undo.push((key, prev));
    }

    /// Remove the entry for `key`, logging the prior value for undo.
    ///
    /// Missing keys skip the log entry entirely.  This mirrors
    /// `HashMap::remove`'s idempotent semantics: a rollback of a no-op
    /// is a no-op, so there is nothing to restore and a tombstone
    /// would only slow later rollbacks.  Locked in by
    /// `remove_absent_key_is_unlogged`.
    pub(crate) fn remove(&mut self, key: &K) {
        if let Some(prev) = self.map.remove(key) {
            self.undo.push((key.clone(), Some(prev)));
        }
    }

    /// Capture the current undo-log length for later restoration.
    pub(crate) fn checkpoint(&self) -> usize {
        self.undo.len()
    }

    /// Replay undo entries in reverse until the log length equals
    /// `checkpoint`, restoring the map to its state at that point.
    pub(crate) fn rollback_to(&mut self, checkpoint: usize) {
        debug_assert!(checkpoint <= self.undo.len());
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

    /// Read-only view of the underlying map; iterators should prefer
    /// this accessor over direct field access and never mutate through it.
    pub(crate) fn as_map(&self) -> &HashMap<K, V> {
        &self.map
    }

    /// Clear the map while logging every removal so rollback restores
    /// the original contents.
    ///
    /// Equivalent to calling [`ScopedMap::remove`] on every key but
    /// avoids the double hash lookup (backing-map lookup plus cloned-key
    /// re-insert) and the intermediate `Vec<K>` a naive caller needs.
    /// The canonical use is a loop handler: the body and continuing
    /// arm must see a fresh cache because the iteration runs an
    /// unknown number of times, yet pre-loop entries must still be
    /// reinstated after the loop ends.
    pub(crate) fn drain_logged(&mut self) {
        self.undo.reserve(self.map.len());
        // `drain()` walks the buckets once in `O(N)` and yields `(K, V)`
        // pairs; each pair is moved straight into the undo log.
        for (k, v) in self.map.drain() {
            self.undo.push((k, Some(v)));
        }
    }

    /// In-place [`HashMap::retain`] that logs every removed pair so
    /// `rollback_to` can restore them.  `predicate` returns `true` for
    /// entries that should remain.
    ///
    /// Equivalent to iterating and calling [`ScopedMap::remove`] on
    /// every failing entry, but without the intermediate `Vec<K>` and
    /// without the second hash lookup that pattern requires.  The hot
    /// caller is `load_dedup`'s per-Store and per-Call cache
    /// invalidation, which saves a key allocation per write.
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
        m.insert(2, 999); // overwrite inside inner
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
        // No entries to log, so the undo cursor must not advance.
        assert_eq!(m.checkpoint(), cp_before);
    }

    #[test]
    fn drain_logged_composes_with_subsequent_writes() {
        // Simulates the load_dedup loop-handler pattern: drain before
        // body+continuing, then later mutations are rolled back
        // independently of the drain.
        let mut m = ScopedMap::<u32, u32>::new();
        m.insert(1, 10);
        m.insert(2, 20);
        let cp_pre = m.checkpoint();
        m.drain_logged();
        let cp_empty = m.checkpoint();
        m.insert(3, 30); // body writes
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
        // No removals, so no undo entries should be appended.
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
