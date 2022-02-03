use super::Cached;
use crate::lru_list::LRUList;
use hashbrown::raw::RawTable;
use std::cmp::Eq;
use std::collections::hash_map::RandomState;
use std::fmt;
use std::hash::{BuildHasher, Hash, Hasher};

#[cfg(feature = "async")]
use {super::CachedAsync, async_trait::async_trait, futures::Future};

/// Low Inter-reference Recency Set Cache
///
/// Stores a limited number of values, evicting
/// keys with the highest distance between reuses
///
/// Note: This cache is in-memory only
#[derive(Clone)]
pub struct LirsCache<K, V> {
    pub(super) lir_store: RawTable<usize>,
    pub(super) hir_store: RawTable<usize>,
    pub(super) hash_builder: RandomState,
    pub(super) lir_order: LRUList<(K, V)>,
    pub(super) hir_order: LRUList<(K, V)>,
    pub(super) lir_capacity: usize,
    pub(super) hir_capacity: usize,
    pub(super) hits: u64,
    pub(super) misses: u64,
}

impl<K, V> fmt::Debug for Lirs<K, V>
where
    K: fmt::Debug,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LirsCache")
            .field("lir_order", &self.order)
            .field("hir_order", &self.order)
            .field("lir_capacity", &self.capacity)
            .field("hir_capacity", &self.capacity)
            .field("hits", &self.hits)
            .field("misses", &self.misses)
            .finish()
    }
}

impl<K, V> PartialEq for LirsCache<K, V>
where
    K: Eq + Hash + Clone,
    V: PartialEq,
{
    fn eq(&self, other: &SizedCache<K, V>) -> bool {
        self.lir_store.len() == other.store.len() &&
        self.hir_store.len() == other.hir_store.len() && {
            self.lir_order
                .iter()
                .all(|(key, value)| match other.get_index(other.hash(key), key) {
                    Some(i) => value == &other.lir_order.get(i).1,
                    None => false,
                })
        } && {
            self.hir_order
                .iter()
                .all(|(key, value)| match other.get_index(other.hash(key), key) {
                    Some(i) => value == &other.hir_order.get(i).1,
                    None => false,
                })
        }
    }
}

impl<K, V> Eq for LirsCache<K, V>
where
    K: Eq + Hash + Clone,
    V: PartialEq,
{
}

impl<K: Hash + Eq + Clone, V> SizedCache<K, V> {
    #[deprecated(since = "0.5.1", note = "method renamed to `with_size`")]
    pub fn with_capacity(size: usize) -> SizedCache<K, V> {
        Self::with_size(size)
    }

    /// Creates a new `SizedCache` with a given size limit and pre-allocated backing data
    pub fn with_size(size: usize) -> SizedCache<K, V> {
        if size == 0 {
            panic!("`size` of `SizedCache` must be greater than zero.")
        }
        SizedCache {
            store: RawTable::with_capacity(size),
            hash_builder: RandomState::new(),
            order: LRUList::<(K, V)>::with_capacity(size),
            capacity: size,
            hits: 0,
            misses: 0,
        }
    }

    /// Creates a new `SizedCache` with a given size limit and pre-allocated backing data
    pub fn try_with_size(size: usize) -> std::io::Result<SizedCache<K, V>> {
        if size == 0 {
            // EINVAL
            return Err(std::io::Error::from_raw_os_error(22));
        }

        let store = match RawTable::try_with_capacity(size) {
            Ok(store) => store,
            Err(e) => {
                let errcode = match e {
                    // ENOMEM
                    hashbrown::TryReserveError::AllocError { .. } => 12,
                    // EINVAL
                    hashbrown::TryReserveError::CapacityOverflow => 22,
                };
                return Err(std::io::Error::from_raw_os_error(errcode));
            }
        };

        Ok(SizedCache {
            store,
            hash_builder: RandomState::new(),
            order: LRUList::<(K, V)>::with_capacity(size),
            capacity: size,
            hits: 0,
            misses: 0,
        })
    }

    pub(super) fn iter_order(&self) -> impl Iterator<Item = &(K, V)> {
        self.order.iter()
    }

    /// Return an iterator of keys in the current order from most
    /// to least recently used.
    pub fn key_order(&self) -> impl Iterator<Item = &K> {
        self.order.iter().map(|(k, _v)| k)
    }

    /// Return an iterator of values in the current order from most
    /// to least recently used.
    pub fn value_order(&self) -> impl Iterator<Item = &V> {
        self.order.iter().map(|(_k, v)| v)
    }

    fn hash(&self, key: &K) -> u64 {
        let hasher = &mut self.hash_builder.build_hasher();
        key.hash(hasher);
        hasher.finish()
    }

    fn insert_index(&mut self, hash: u64, index: usize) {
        let Self {
            ref mut store,
            ref order,
            ref hash_builder,
            ..
        } = *self;
        // insert the value `index` at `hash`, the closure provided
        // is used to rehash values if a resize is necessary.
        store.insert(hash, index, move |&i| {
            // rehash the "key" value stored at index `i` - requires looking
            // up the original "key" value in the `order` list.
            let hasher = &mut hash_builder.build_hasher();
            order.get(i).0.hash(hasher);
            hasher.finish()
        });
    }

    fn get_index(&self, hash: u64, key: &K) -> Option<usize> {
        let Self { store, order, .. } = self;
        // Get the `order` index store under `hash`, the closure provided
        // is used to compare against matching hashes - we lookup the original
        // `key` value from the `order` list.
        // This pattern is repeated in other lookup situations.
        store.get(hash, |&i| *key == order.get(i).0).copied()
    }

    fn remove_index(&mut self, hash: u64, key: &K) -> Option<usize> {
        let Self { store, order, .. } = self;
        store.remove_entry(hash, |&i| *key == order.get(i).0)
    }

    fn check_capacity(&mut self) {
        if self.store.len() >= self.capacity {
            // store has reached capacity, evict the oldest item.
            // store capacity cannot be zero, so there must be content in `self.order`.
            let index = self.order.back();
            let key = &self.order.get(index).0;
            let hash = self.hash(key);

            let order = &self.order;
            let erased = self.store.erase_entry(hash, |&i| *key == order.get(i).0);
            assert!(erased, "SizedCache::cache_set failed evicting cache key");
            self.order.remove(index);
        }
    }

    pub(super) fn get_if<F: FnOnce(&V) -> bool>(&mut self, key: &K, is_valid: F) -> Option<&V> {
        if let Some(index) = self.get_index(self.hash(key), key) {
            if is_valid(&self.order.get(index).1) {
                self.order.move_to_front(index);
                self.hits += 1;
                return Some(&self.order.get(index).1);
            }
        }
        self.misses += 1;
        None
    }

    pub(super) fn get_mut_if<F: FnOnce(&V) -> bool>(
        &mut self,
        key: &K,
        is_valid: F,
    ) -> Option<&mut V> {
        if let Some(index) = self.get_index(self.hash(key), key) {
            if is_valid(&self.order.get(index).1) {
                self.order.move_to_front(index);
                self.hits += 1;
                return Some(&mut self.order.get_mut(index).1);
            }
        }
        self.misses += 1;
        None
    }

    /// Get the cached value, or set it using `f` if the value
    /// is either not-set or if `is_valid` returns `false` for
    /// the set value.
    ///
    /// Returns (was_present, was_valid, mut ref to set value)
    /// `was_valid` will be false when `was_present` is false
    pub(super) fn get_or_set_with_if<F: FnOnce() -> V, FC: FnOnce(&V) -> bool>(
        &mut self,
        key: K,
        f: F,
        is_valid: FC,
    ) -> (bool, bool, &mut V) {
        let hash = self.hash(&key);
        let index = self.get_index(hash, &key);
        if let Some(index) = index {
            self.hits += 1;
            let replace_existing = {
                let v = &self.order.get(index).1;
                !is_valid(v)
            };
            if replace_existing {
                self.order.set(index, (key, f()));
            }
            self.order.move_to_front(index);
            (true, !replace_existing, &mut self.order.get_mut(index).1)
        } else {
            self.check_capacity();
            self.misses += 1;
            let index = self.order.push_front((key, f()));
            self.insert_index(hash, index);
            (false, false, &mut self.order.get_mut(index).1)
        }
    }

    #[allow(dead_code)]
    fn try_get_or_set_with_if<E, F: FnOnce() -> Result<V, E>, FC: FnOnce(&V) -> bool>(
        &mut self,
        key: K,
        f: F,
        is_valid: FC,
    ) -> Result<(bool, bool, &mut V), E> {
        let hash = self.hash(&key);
        let index = self.get_index(hash, &key);
        if let Some(index) = index {
            self.hits += 1;
            let replace_existing = {
                let v = &self.order.get(index).1;
                !is_valid(v)
            };
            if replace_existing {
                self.order.set(index, (key, f()?));
            }
            self.order.move_to_front(index);
            Ok((true, !replace_existing, &mut self.order.get_mut(index).1))
        } else {
            self.check_capacity();
            self.misses += 1;
            let index = self.order.push_front((key, f()?));
            self.insert_index(hash, index);
            Ok((false, false, &mut self.order.get_mut(index).1))
        }
    }

    /// Returns a reference to the cache's `order`
    pub fn get_order(&self) -> &LRUList<(K, V)> {
        &self.order
    }

    pub fn retain<F: Fn(&K, &V) -> bool>(&mut self, keep: F) {
        let remove_keys = self
            .iter_order()
            .filter_map(|(k, v)| if !keep(k, v) { Some(k.clone()) } else { None })
            .collect::<Vec<_>>();
        for k in remove_keys {
            self.cache_remove(&k);
        }
    }
}

#[cfg(feature = "async")]
impl<K, V> SizedCache<K, V>
where
    K: Hash + Eq + Clone + Send,
{
    /// Get the cached value, or set it using `f` if the value
    /// is either not-set or if `is_valid` returns `false` for
    /// the set value.
    ///
    /// Returns (was_present, was_valid, mut ref to set value)
    /// `was_valid` will be false when `was_present` is false
    pub(super) async fn get_or_set_with_if_async<F, Fut, FC>(
        &mut self,
        key: K,
        f: F,
        is_valid: FC,
    ) -> (bool, bool, &mut V)
    where
        V: Send,
        F: FnOnce() -> Fut + Send,
        Fut: Future<Output = V> + Send,
        FC: FnOnce(&V) -> bool,
    {
        let hash = self.hash(&key);
        let index = self.get_index(hash, &key);
        if let Some(index) = index {
            self.hits += 1;
            let replace_existing = {
                let v = &self.order.get(index).1;
                !is_valid(v)
            };
            if replace_existing {
                self.order.set(index, (key, f().await));
            }
            self.order.move_to_front(index);
            (true, !replace_existing, &mut self.order.get_mut(index).1)
        } else {
            self.check_capacity();
            self.misses += 1;
            let index = self.order.push_front((key, f().await));
            self.insert_index(hash, index);
            (false, false, &mut self.order.get_mut(index).1)
        }
    }

    pub(super) async fn try_get_or_set_with_if_async<E, F, Fut, FC>(
        &mut self,
        key: K,
        f: F,
        is_valid: FC,
    ) -> Result<(bool, bool, &mut V), E>
    where
        V: Send,
        F: FnOnce() -> Fut + Send,
        Fut: Future<Output = Result<V, E>> + Send,
        FC: FnOnce(&V) -> bool,
    {
        let hash = self.hash(&key);
        let index = self.get_index(hash, &key);
        if let Some(index) = index {
            self.hits += 1;
            let replace_existing = {
                let v = &self.order.get(index).1;
                !is_valid(v)
            };
            if replace_existing {
                self.order.set(index, (key, f().await?));
            }
            self.order.move_to_front(index);
            Ok((true, !replace_existing, &mut self.order.get_mut(index).1))
        } else {
            self.check_capacity();
            self.misses += 1;
            let index = self.order.push_front((key, f().await?));
            self.insert_index(hash, index);
            Ok((false, false, &mut self.order.get_mut(index).1))
        }
    }
}

impl<K: Hash + Eq + Clone, V> Cached<K, V> for SizedCache<K, V> {
    fn cache_get(&mut self, key: &K) -> Option<&V> {
        self.get_if(key, |_| true)
    }

    fn cache_get_mut(&mut self, key: &K) -> std::option::Option<&mut V> {
        self.get_mut_if(key, |_| true)
    }

    fn cache_set(&mut self, key: K, val: V) -> Option<V> {
        self.check_capacity();
        let hash = self.hash(&key);
        if let Some(index) = self.get_index(hash, &key) {
            self.order.set(index, (key, val)).map(|(_, v)| v)
        } else {
            let index = self.order.push_front((key, val));
            self.insert_index(hash, index);
            None
        }
    }

    fn cache_get_or_set_with<F: FnOnce() -> V>(&mut self, key: K, f: F) -> &mut V {
        let (_, _, v) = self.get_or_set_with_if(key, f, |_| true);
        v
    }

    fn cache_remove(&mut self, k: &K) -> Option<V> {
        // try and remove item from mapping, and then from order list if it was in mapping
        let hash = self.hash(k);
        if let Some(index) = self.remove_index(hash, k) {
            // need to remove the key in the order list
            let (_key, value) = self.order.remove(index);
            Some(value)
        } else {
            None
        }
    }
    fn cache_clear(&mut self) {
        // clear both the store and the order list
        self.store.clear();
        self.order.clear();
    }
    fn cache_reset(&mut self) {
        // SizedCache uses cache_clear because capacity is fixed.
        self.cache_clear();
    }
    fn cache_reset_metrics(&mut self) {
        self.misses = 0;
        self.hits = 0;
    }
    fn cache_size(&self) -> usize {
        self.store.len()
    }
    fn cache_hits(&self) -> Option<u64> {
        Some(self.hits)
    }
    fn cache_misses(&self) -> Option<u64> {
        Some(self.misses)
    }
    fn cache_capacity(&self) -> Option<usize> {
        Some(self.capacity)
    }
}

#[cfg(feature = "async")]
#[async_trait]
impl<K, V> CachedAsync<K, V> for SizedCache<K, V>
where
    K: Hash + Eq + Clone + Send,
{
    async fn get_or_set_with<F, Fut>(&mut self, k: K, f: F) -> &mut V
    where
        V: Send,
        F: FnOnce() -> Fut + Send,
        Fut: Future<Output = V> + Send,
    {
        let (_, _, v) = self.get_or_set_with_if_async(k, f, |_| true).await;
        v
    }

    async fn try_get_or_set_with<F, Fut, E>(&mut self, k: K, f: F) -> Result<&mut V, E>
    where
        V: Send,
        F: FnOnce() -> Fut + Send,
        Fut: Future<Output = Result<V, E>> + Send,
    {
        let (_, _, v) = self.try_get_or_set_with_if_async(k, f, |_| true).await?;
        Ok(v)
    }
}
