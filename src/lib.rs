use dashmap::DashMap;
use once_cell::sync::OnceCell;
use std::any::{Any, TypeId};
use std::hash::Hash;
use std::{
    collections::{btree_map::Entry, BTreeMap},
    sync::{
        atomic::{AtomicIsize, Ordering},
        Arc, Mutex,
    },
};

struct HashContainer<T: ?Sized> {
    hashed: DashMap<Arc<T>, ()>,
    h_count: AtomicIsize,
}

impl<T: Eq + Hash + ?Sized> HashContainer<T> {
    pub fn new() -> Self {
        HashContainer {
            hashed: DashMap::new(),
            h_count: AtomicIsize::new(1),
        }
    }
}

struct TreeContainer<T: ?Sized> {
    tree: Mutex<BTreeMap<Arc<T>, ()>>,
    t_count: AtomicIsize,
}

impl<T: Ord + ?Sized> TreeContainer<T> {
    pub fn new() -> Self {
        TreeContainer {
            tree: Mutex::new(BTreeMap::new()),
            t_count: AtomicIsize::new(1),
        }
    }
}

static CONTAINER_HASH: OnceCell<DashMap<TypeId, Box<dyn Any + Send + Sync>>> = OnceCell::new();
static CONTAINER_TREE: OnceCell<DashMap<TypeId, Box<dyn Any + Send + Sync>>> = OnceCell::new();

pub fn intern_hash_arc<T>(val: Arc<T>) -> Arc<T>
where
    T: Eq + Hash + Send + Sync + ?Sized + 'static,
{
    let type_map = CONTAINER_HASH.get_or_init(DashMap::new);

    // Prefer taking the read lock to reduce contention, only use entry api if necessary.
    let boxed = if let Some(boxed) = type_map.get(&TypeId::of::<T>()) {
        boxed
    } else {
        type_map
            .entry(TypeId::of::<T>())
            .or_insert_with(|| Box::new(HashContainer::<T>::new()))
            .downgrade()
    };

    let m: &HashContainer<T> = boxed.value().downcast_ref::<HashContainer<T>>().unwrap();
    let b = m.hashed.entry(val).or_insert(());
    let ret = b.key().clone();
    drop(b);

    // maintenance
    if m.h_count.fetch_sub(1, Ordering::Relaxed) == 0 {
        janitor_h(m);
    }

    ret
}

pub fn intern_hash<T>(val: T) -> Arc<T>
where
    T: Eq + Hash + Send + Sync + 'static,
{
    intern_hash_arc(Arc::new(val))
}

/// ```
/// use arc_interner::intern_hash_unsized;
/// use std::sync::Arc;
///
/// let data = &[1, 2, 3, 4, 5];
/// let interned: Arc<[u8]> = intern_hash_unsized(data);
/// ```
pub fn intern_hash_unsized<T>(val: &T) -> Arc<T>
where
    T: Eq + Hash + Send + Sync + ?Sized + 'static,
    Arc<T>: for<'a> From<&'a T>,
{
    intern_hash_arc(Arc::from(val))
}

/// ```
/// use arc_interner::intern_hash_boxed;
/// use std::sync::Arc;
///
/// let data: Vec<u8> = vec![1, 2, 3, 4, 5];
/// let interned: Arc<[u8]> = intern_hash_boxed(Box::from(data));
/// ```
pub fn intern_hash_boxed<T>(val: Box<T>) -> Arc<T>
where
    T: Eq + Hash + Send + Sync + ?Sized + 'static,
{
    intern_hash_arc(Arc::from(val))
}

pub fn intern_tree_arc<T>(val: Arc<T>) -> Arc<T>
where
    T: Ord + Send + Sync + ?Sized + 'static,
{
    let type_map = CONTAINER_TREE.get_or_init(DashMap::new);

    // Prefer taking the read lock to reduce contention, only use entry api if necessary.
    let boxed = if let Some(boxed) = type_map.get(&TypeId::of::<T>()) {
        boxed
    } else {
        type_map
            .entry(TypeId::of::<T>())
            .or_insert_with(|| Box::new(TreeContainer::<T>::new()))
            .downgrade()
    };

    let m: &TreeContainer<T> = boxed.value().downcast_ref::<TreeContainer<T>>().unwrap();
    let mut set = m.tree.lock().unwrap();
    let ret = match set.entry(val) {
        Entry::Vacant(x) => {
            let ret = x.key().clone();
            x.insert(());
            ret
        }
        Entry::Occupied(x) => x.key().clone(),
    };

    // maintenance
    if m.t_count.fetch_sub(1, Ordering::Relaxed) == 0 {
        janitor_t(&mut *set, &m.t_count);
    }

    ret
}

pub fn intern_tree<T>(val: T) -> Arc<T>
where
    T: Ord + Send + Sync + 'static,
{
    intern_tree_arc(Arc::new(val))
}

pub fn intern_tree_unsized<T>(val: &T) -> Arc<T>
where
    T: Ord + Send + Sync + ?Sized + 'static,
    Arc<T>: for<'a> From<&'a T>,
{
    intern_tree_arc(Arc::from(val))
}

pub fn intern_tree_boxed<T>(val: Box<T>) -> Arc<T>
where
    T: Ord + Send + Sync + ?Sized + 'static,
{
    intern_tree_arc(Arc::from(val))
}

pub fn intern<T>(val: T) -> Arc<T>
where
    T: Eq + Hash + Ord + Send + Sync + 'static,
{
    if std::mem::size_of::<T>() > 1000 {
        intern_tree(val)
    } else {
        intern_hash(val)
    }
}

pub fn intern_unsized<T: ?Sized>(val: &T) -> Arc<T>
where
    T: Eq + Hash + Ord + Send + Sync + 'static,
    Arc<T>: for<'a> From<&'a T>,
{
    if std::mem::size_of_val(val) > 1000 {
        intern_tree_unsized(val)
    } else {
        intern_hash_unsized(val)
    }
}

pub fn intern_boxed<T: ?Sized>(val: Box<T>) -> Arc<T>
where
    T: Eq + Hash + Ord + Send + Sync + 'static,
{
    if std::mem::size_of_val(val.as_ref()) > 1000 {
        intern_tree_boxed(val)
    } else {
        intern_hash_boxed(val)
    }
}

pub fn intern_arc<T: ?Sized>(val: Arc<T>) -> Arc<T>
where
    T: Eq + Hash + Ord + Send + Sync + 'static,
{
    if std::mem::size_of_val(val.as_ref()) > 1000 {
        intern_tree_arc(val)
    } else {
        intern_hash_arc(val)
    }
}

/// Perform internal maintenance (removing otherwise unreferenced elements) and return count of elements
pub fn num_objects_interned_hash<T: Eq + Hash + ?Sized + 'static>() -> usize {
    if let Some(m) = CONTAINER_HASH
        .get()
        .and_then(|type_map| type_map.get(&TypeId::of::<T>()))
    {
        println!("num_objects_hash {:?}", m.key());
        let m = m.downcast_ref::<HashContainer<T>>().unwrap();
        janitor_h(m);
        m.hashed.len()
    } else {
        0
    }
}

/// Perform internal maintenance (removing otherwise unreferenced elements) and return count of elements
pub fn num_objects_interned_tree<T: Ord + ?Sized + 'static>() -> usize {
    if let Some(m) = CONTAINER_TREE
        .get()
        .and_then(|type_map| type_map.get(&TypeId::of::<T>()))
    {
        println!("num_objects_tree {:?}", m.key());
        let m = m.downcast_ref::<TreeContainer<T>>().unwrap();
        let mut s = m.tree.lock().unwrap();
        janitor_t(&mut *s, &m.t_count);
        s.len()
    } else {
        0
    }
}

fn janitor_h<T: Eq + Hash + ?Sized + 'static>(m: &HashContainer<T>) {
    let before = m.hashed.len();
    m.hashed.retain(|k, _v| Arc::strong_count(k) > 1);
    let after = m.hashed.len();
    let removed = (before as isize - after as isize).max(1) as usize;
    // assume removals are always possible
    // the interval is tuned such that it is very short for high churn and very long for low churn
    // this is done such that the amortized cost is one retain check per insert
    m.h_count
        .store((before / removed) as isize, Ordering::Relaxed);
}

fn janitor_t<T: Ord + ?Sized + 'static>(set: &mut BTreeMap<Arc<T>, ()>, count: &AtomicIsize) {
    let before = set.len();
    let to_remove = set
        .iter()
        .filter_map(|(k, _v)| {
            if Arc::strong_count(k) == 1 {
                Some(k.clone())
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    for a in to_remove {
        set.remove(&a);
    }
    let after = set.len();
    let removed = (before - after).max(1);
    // assume removals are always possible
    // the interval is tuned such that it is very short for high churn and very long for low churn
    // this is done such that the amortized cost is one retain check per insert
    count.store((before / removed) as isize, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use crate::*;
    use std::thread;

    // Test basic functionality.
    #[test]
    fn basic_hash() {
        assert_eq!(intern_hash("foo"), intern_hash("foo"));
        assert_ne!(intern_hash("foo"), intern_hash("bar"));
        // The above refs should be deallocate by now.
        assert_eq!(num_objects_interned_hash::<&str>(), 0);

        let _interned1 = intern_hash("foo".to_string());
        {
            let interned2 = intern_hash("foo".to_string());
            let interned3 = intern_hash("bar".to_string());

            assert_eq!(Arc::strong_count(&interned2), 3);
            assert_eq!(Arc::strong_count(&interned3), 2);
            // We now have two unique interned strings: "foo" and "bar".
            assert_eq!(num_objects_interned_hash::<String>(), 2);
        }

        // "bar" is now gone.
        assert_eq!(num_objects_interned_hash::<String>(), 1);
    }

    // Test basic functionality.
    #[test]
    fn basic_hash_unsized() {
        assert_eq!(intern_hash_unsized("foo"), intern_hash_unsized("foo"));
        assert_ne!(intern_hash_unsized("foo"), intern_hash_unsized("bar"));
        // The above refs should be deallocate by now.
        assert_eq!(num_objects_interned_hash::<str>(), 0);

        let _interned1 = intern_hash_unsized("foo");
        {
            let interned2 = intern_hash_unsized("foo");
            let interned3 = intern_hash_unsized("bar");

            assert_eq!(Arc::strong_count(&interned2), 3);
            assert_eq!(Arc::strong_count(&interned3), 2);
            // We now have two unique interned strings: "foo" and "bar".
            assert_eq!(num_objects_interned_hash::<str>(), 2);
        }

        // "bar" is now gone.
        assert_eq!(num_objects_interned_hash::<str>(), 1);
    }

    // Test basic functionality.
    #[test]
    fn basic_tree() {
        assert_eq!(intern_tree("foo"), intern_tree("foo"));
        assert_ne!(intern_tree("foo"), intern_tree("bar"));
        // The above refs should be deallocate by now.
        assert_eq!(num_objects_interned_tree::<&str>(), 0);

        let _interned1 = intern_tree("foo".to_string());
        {
            let interned2 = intern_tree("foo".to_string());
            let interned3 = intern_tree("bar".to_string());

            assert_eq!(Arc::strong_count(&interned2), 3);
            assert_eq!(Arc::strong_count(&interned3), 2);
            // We now have two unique interned strings: "foo" and "bar".
            assert_eq!(num_objects_interned_tree::<String>(), 2);
        }

        // "bar" is now gone.
        assert_eq!(num_objects_interned_tree::<String>(), 1);
    }

    // Test basic functionality.
    #[test]
    fn basic_tree_unsized() {
        assert_eq!(intern_tree_unsized("foo"), intern_tree_unsized("foo"));
        assert_ne!(intern_tree_unsized("foo"), intern_tree_unsized("bar"));
        // The above refs should be deallocate by now.
        assert_eq!(num_objects_interned_tree::<str>(), 0);

        let _interned1 = intern_tree_unsized("foo");
        {
            let interned2 = intern_tree_unsized("foo");
            let interned3 = intern_tree_unsized("bar");

            assert_eq!(Arc::strong_count(&interned2), 3);
            assert_eq!(Arc::strong_count(&interned3), 2);
            // We now have two unique interned strings: "foo" and "bar".
            assert_eq!(num_objects_interned_tree::<str>(), 2);
        }

        // "bar" is now gone.
        assert_eq!(num_objects_interned_tree::<str>(), 1);
    }

    // Ordering should be based on values, not pointers.
    // Also tests `Display` implementation.
    #[test]
    fn sorting() {
        let mut interned_vals = vec![
            intern_hash(4),
            intern_hash(2),
            intern_hash(5),
            intern_hash(0),
            intern_hash(1),
            intern_hash(3),
        ];
        interned_vals.sort();
        let sorted: Vec<String> = interned_vals.iter().map(|v| format!("{}", v)).collect();
        assert_eq!(&sorted.join(","), "0,1,2,3,4,5");
    }

    #[derive(Eq, PartialEq, Ord, PartialOrd, Hash)]
    pub struct TestStruct2(String, u64);

    #[test]
    fn sequential() {
        for _i in 0..10_000 {
            let mut interned = Vec::with_capacity(100);
            for j in 0..100 {
                interned.push(intern_hash(TestStruct2("foo".to_string(), j)));
            }
        }

        assert_eq!(num_objects_interned_hash::<TestStruct2>(), 0);
    }

    #[derive(Eq, PartialEq, Ord, PartialOrd, Hash)]
    pub struct TestStruct(String, u64);

    // Quickly create and destroy a small number of interned objects from
    // multiple threads.
    #[test]
    fn multithreading1() {
        let mut thandles = vec![];
        for _i in 0..10 {
            thandles.push(thread::spawn(|| {
                for _i in 0..100_000 {
                    let _interned1 = intern_hash(TestStruct("foo".to_string(), 5));
                    let _interned2 = intern_hash(TestStruct("bar".to_string(), 10));
                }
            }));
        }
        for h in thandles.into_iter() {
            h.join().unwrap()
        }

        assert_eq!(num_objects_interned_hash::<TestStruct>(), 0);
    }
}
