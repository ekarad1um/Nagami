//! Dense sets and maps keyed by arena handles.  One `Vec` slot per handle
//! index replaces a hash table: membership is a bounds check, and each
//! key/value pair costs a few inlined methods instead of a hashbrown
//! instantiation (the largest removable block of the wasm build).  Slots
//! grow on demand, so a handle appended to the arena after construction
//! reads as absent until inserted.  Iteration follows FIRST insertion: a
//! removed and re-inserted handle keeps its original position.  No caller
//! depends on the order, only on its determinism.

use std::borrow::Borrow;
use std::marker::PhantomData;

use naga::Handle;

const PRESENT: u8 = 1;
const LISTED: u8 = 2;

pub struct HandleSet<T> {
    flags: Vec<u8>,
    order: Vec<Handle<T>>,
    len: usize,
    marker: PhantomData<T>,
}

impl<T> Default for HandleSet<T> {
    fn default() -> Self {
        Self {
            flags: Vec::new(),
            order: Vec::new(),
            len: 0,
            marker: PhantomData,
        }
    }
}

impl<T> Clone for HandleSet<T> {
    fn clone(&self) -> Self {
        Self {
            flags: self.flags.clone(),
            order: self.order.clone(),
            len: self.len,
            marker: PhantomData,
        }
    }
}

impl<T> HandleSet<T> {
    /// `true` when `handle` was not a member.
    pub fn insert(&mut self, handle: Handle<T>) -> bool {
        let i = handle.index();
        if i >= self.flags.len() {
            self.flags.resize(i + 1, 0);
        }
        let flags = &mut self.flags[i];
        if *flags & PRESENT != 0 {
            return false;
        }
        if *flags & LISTED == 0 {
            self.order.push(handle);
        }
        *flags = PRESENT | LISTED;
        self.len += 1;
        true
    }

    pub fn contains(&self, handle: impl Borrow<Handle<T>>) -> bool {
        self.flags
            .get(handle.borrow().index())
            .is_some_and(|f| f & PRESENT != 0)
    }

    /// `true` when `handle` was a member.
    pub fn remove(&mut self, handle: impl Borrow<Handle<T>>) -> bool {
        match self.flags.get_mut(handle.borrow().index()) {
            Some(f) if *f & PRESENT != 0 => {
                *f &= !PRESENT;
                self.len -= 1;
                true
            }
            _ => false,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Members in first-insertion order.
    pub fn iter(&self) -> SetIter<'_, T> {
        SetIter {
            order: self.order.iter(),
            flags: &self.flags,
        }
    }

    pub fn intersection<'a>(&'a self, other: &'a Self) -> impl Iterator<Item = &'a Handle<T>> + 'a {
        self.iter().filter(move |h| other.contains(*h))
    }

    /// Members of either set, each once.
    pub fn union<'a>(&'a self, other: &'a Self) -> impl Iterator<Item = &'a Handle<T>> + 'a {
        self.iter()
            .chain(other.iter().filter(move |h| !self.contains(*h)))
    }

    /// Empties the set while keeping the slot array: every nonzero flag was
    /// listed, so clearing costs the members, not the arena.  Reusing one set
    /// across a walk then pays no per-clear memset.
    pub fn clear(&mut self) {
        for h in self.order.drain(..) {
            self.flags[h.index()] = 0;
        }
        self.len = 0;
    }
}

/// [`HandleSet::iter`].  A named type rather than `impl Iterator` so
/// `for h in &set` needs no boxed trait object: the erased form cost a heap
/// allocation per loop on walks that run per statement.
pub struct SetIter<'a, T> {
    order: std::slice::Iter<'a, Handle<T>>,
    flags: &'a [u8],
}

impl<'a, T> Iterator for SetIter<'a, T> {
    type Item = &'a Handle<T>;
    fn next(&mut self) -> Option<Self::Item> {
        let flags = self.flags;
        self.order
            .find(|h| flags.get(h.index()).is_some_and(|f| f & PRESENT != 0))
    }
}

impl<T> Extend<Handle<T>> for HandleSet<T> {
    fn extend<I: IntoIterator<Item = Handle<T>>>(&mut self, iter: I) {
        for h in iter {
            self.insert(h);
        }
    }
}

impl<T> IntoIterator for HandleSet<T> {
    type Item = Handle<T>;
    type IntoIter = std::vec::IntoIter<Handle<T>>;
    fn into_iter(self) -> Self::IntoIter {
        let live: Vec<_> = self.iter().copied().collect();
        live.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a HandleSet<T> {
    type Item = &'a Handle<T>;
    type IntoIter = SetIter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T> FromIterator<Handle<T>> for HandleSet<T> {
    fn from_iter<I: IntoIterator<Item = Handle<T>>>(iter: I) -> Self {
        let mut set = Self::default();
        set.extend(iter);
        set
    }
}

pub struct HandleMap<T, V> {
    slots: Vec<Option<V>>,
    listed: Vec<bool>,
    order: Vec<Handle<T>>,
    len: usize,
}

impl<T, V> Default for HandleMap<T, V> {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            listed: Vec::new(),
            order: Vec::new(),
            len: 0,
        }
    }
}

impl<T, V: Clone> Clone for HandleMap<T, V> {
    fn clone(&self) -> Self {
        Self {
            slots: self.slots.clone(),
            listed: self.listed.clone(),
            order: self.order.clone(),
            len: self.len,
        }
    }
}

impl<T, V: std::fmt::Debug> std::fmt::Debug for HandleMap<T, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map()
            .entries(self.iter().map(|(h, v)| (h.index(), v)))
            .finish()
    }
}

/// A slot of a [`HandleMap`], for `entry(h).or_insert(..)` call sites.
pub struct Entry<'a, T, V> {
    map: &'a mut HandleMap<T, V>,
    handle: Handle<T>,
}

impl<'a, T, V> Entry<'a, T, V> {
    pub fn or_insert(self, default: V) -> &'a mut V {
        self.map.get_or_insert_with(self.handle, || default)
    }

    pub fn or_insert_with(self, default: impl FnOnce() -> V) -> &'a mut V {
        self.map.get_or_insert_with(self.handle, default)
    }

    pub fn or_default(self) -> &'a mut V
    where
        V: Default,
    {
        self.map.get_or_insert_with(self.handle, V::default)
    }

    pub fn and_modify(self, f: impl FnOnce(&mut V)) -> Self {
        if let Some(v) = self.map.get_mut(self.handle) {
            f(v);
        }
        self
    }
}

impl<T, V> HandleMap<T, V> {
    pub fn entry(&mut self, handle: Handle<T>) -> Entry<'_, T, V> {
        Entry { map: self, handle }
    }

    pub fn insert(&mut self, handle: Handle<T>, value: V) -> Option<V> {
        let i = handle.index();
        if i >= self.slots.len() {
            self.slots.resize_with(i + 1, || None);
            self.listed.resize(i + 1, false);
        }
        if !self.listed[i] {
            self.listed[i] = true;
            self.order.push(handle);
        }
        let prev = self.slots[i].replace(value);
        if prev.is_none() {
            self.len += 1;
        }
        prev
    }

    pub fn get(&self, handle: impl Borrow<Handle<T>>) -> Option<&V> {
        self.slots
            .get(handle.borrow().index())
            .and_then(Option::as_ref)
    }

    pub fn get_mut(&mut self, handle: impl Borrow<Handle<T>>) -> Option<&mut V> {
        self.slots
            .get_mut(handle.borrow().index())
            .and_then(Option::as_mut)
    }

    pub fn contains_key(&self, handle: impl Borrow<Handle<T>>) -> bool {
        self.get(handle).is_some()
    }

    pub fn remove(&mut self, handle: impl Borrow<Handle<T>>) -> Option<V> {
        let prev = self.slots.get_mut(handle.borrow().index())?.take();
        if prev.is_some() {
            self.len -= 1;
        }
        prev
    }

    pub fn get_or_insert_with(&mut self, handle: Handle<T>, default: impl FnOnce() -> V) -> &mut V {
        if !self.contains_key(handle) {
            self.insert(handle, default());
        }
        self.slots[handle.index()].as_mut().expect("just inserted")
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Entries in first-insertion order.
    pub fn iter(&self) -> MapIter<'_, T, V> {
        MapIter {
            order: self.order.iter(),
            slots: &self.slots,
        }
    }

    pub fn keys(&self) -> impl Iterator<Item = &Handle<T>> + '_ {
        self.iter().map(|(h, _)| h)
    }

    pub fn values(&self) -> impl Iterator<Item = &V> + '_ {
        self.iter().map(|(_, v)| v)
    }

    pub fn retain(&mut self, mut keep: impl FnMut(&Handle<T>, &mut V) -> bool) {
        for i in 0..self.order.len() {
            let h = self.order[i];
            if let Some(v) = self.slots[h.index()].as_mut()
                && !keep(&h, v)
            {
                self.remove(h);
            }
        }
    }

    /// Empties the map while keeping the slot arrays; see [`HandleSet::clear`].
    pub fn clear(&mut self) {
        for h in self.order.drain(..) {
            self.slots[h.index()] = None;
            self.listed[h.index()] = false;
        }
        self.len = 0;
    }

    /// Every entry, in first-insertion order, leaving the map empty.  The slots move
    /// into the iterator rather than staying behind half-emptied, so dropping
    /// it early is as consistent as running it out.
    pub fn drain(&mut self) -> impl Iterator<Item = (Handle<T>, V)> {
        let order = std::mem::take(&mut self.order);
        let mut slots = std::mem::take(&mut self.slots);
        self.listed.clear();
        self.len = 0;
        order
            .into_iter()
            .filter_map(move |h| slots[h.index()].take().map(|v| (h, v)))
    }
}

/// [`HandleMap::iter`]; named for the same reason as [`SetIter`].
pub struct MapIter<'a, T, V> {
    order: std::slice::Iter<'a, Handle<T>>,
    slots: &'a [Option<V>],
}

impl<'a, T, V> Iterator for MapIter<'a, T, V> {
    type Item = (&'a Handle<T>, &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        let slots = self.slots;
        self.order
            .find_map(|h| slots.get(h.index())?.as_ref().map(|v| (h, v)))
    }
}

impl<T, V> std::ops::Index<Handle<T>> for HandleMap<T, V> {
    type Output = V;
    fn index(&self, handle: Handle<T>) -> &V {
        self.get(handle).expect("handle absent from HandleMap")
    }
}

impl<T, V> std::ops::Index<&Handle<T>> for HandleMap<T, V> {
    type Output = V;
    fn index(&self, handle: &Handle<T>) -> &V {
        self.get(handle).expect("handle absent from HandleMap")
    }
}

impl<T, V> Extend<(Handle<T>, V)> for HandleMap<T, V> {
    fn extend<I: IntoIterator<Item = (Handle<T>, V)>>(&mut self, iter: I) {
        for (h, v) in iter {
            self.insert(h, v);
        }
    }
}

impl<T, V> IntoIterator for HandleMap<T, V> {
    type Item = (Handle<T>, V);
    type IntoIter = std::vec::IntoIter<(Handle<T>, V)>;
    fn into_iter(mut self) -> Self::IntoIter {
        let order = std::mem::take(&mut self.order);
        let live: Vec<_> = order
            .into_iter()
            .filter_map(|h| self.slots[h.index()].take().map(|v| (h, v)))
            .collect();
        live.into_iter()
    }
}

impl<'a, T, V> IntoIterator for &'a HandleMap<T, V> {
    type Item = (&'a Handle<T>, &'a V);
    type IntoIter = MapIter<'a, T, V>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T, V> FromIterator<(Handle<T>, V)> for HandleMap<T, V> {
    fn from_iter<I: IntoIterator<Item = (Handle<T>, V)>>(iter: I) -> Self {
        let mut map = Self::default();
        map.extend(iter);
        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn handles(n: usize) -> Vec<Handle<naga::Expression>> {
        let mut arena = naga::Arena::new();
        (0..n)
            .map(|i| {
                arena.append(
                    naga::Expression::Literal(naga::Literal::U32(i as u32)),
                    naga::Span::UNDEFINED,
                )
            })
            .collect()
    }

    #[test]
    fn set_tracks_membership_and_insertion_order_once() {
        let h = handles(6);
        let mut set = HandleSet::default();
        assert!(set.insert(h[4]));
        assert!(set.insert(h[1]));
        assert!(!set.insert(h[4]));
        assert!(set.contains(h[1]) && !set.contains(h[5]));
        assert!(set.remove(h[4]) && !set.remove(h[4]));
        assert!(set.insert(h[4]), "re-insert after removal");
        assert_eq!(set.iter().copied().collect::<Vec<_>>(), [h[4], h[1]]);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn map_is_reusable_after_drain_and_clear() {
        let h = handles(4);
        let mut map: HandleMap<naga::Expression, u8> = HandleMap::default();
        map.insert(h[2], 5);
        map.insert(h[0], 6);
        assert_eq!(map.drain().collect::<Vec<_>>(), [(h[2], 5), (h[0], 6)]);
        assert!(map.is_empty() && map.get(h[2]).is_none());
        map.insert(h[0], 7);
        assert_eq!(map.iter().collect::<Vec<_>>(), [(&h[0], &7)]);
        map.clear();
        assert!(map.is_empty() && map.get(h[0]).is_none());
        map.insert(h[3], 8);
        assert_eq!(map.iter().collect::<Vec<_>>(), [(&h[3], &8)]);
    }

    #[test]
    fn set_is_reusable_after_clear() {
        let h = handles(4);
        let mut set = HandleSet::default();
        set.insert(h[3]);
        set.insert(h[1]);
        set.clear();
        assert!(set.is_empty() && !set.contains(h[3]));
        assert!(set.insert(h[1]));
        assert_eq!(set.iter().copied().collect::<Vec<_>>(), [h[1]]);
    }

    #[test]
    fn map_replaces_removes_and_iterates_in_insertion_order() {
        let h = handles(4);
        let mut map: HandleMap<naga::Expression, u8> = HandleMap::default();
        assert_eq!(map.insert(h[3], 1), None);
        assert_eq!(map.insert(h[0], 2), None);
        assert_eq!(map.insert(h[3], 3), Some(1));
        assert_eq!(map.remove(h[0]), Some(2));
        assert_eq!(map.remove(h[0]), None);
        *map.get_or_insert_with(h[2], || 7) += 1;
        assert_eq!(map.iter().collect::<Vec<_>>(), [(&h[3], &3), (&h[2], &8)]);
        assert_eq!(map.len(), 2);
        assert!(map.get(h[1]).is_none() && !map.contains_key(h[0]));
    }
}
