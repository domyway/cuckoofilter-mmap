//! A disk-based Cuckoo Filter implementation using memory-mapped files.
//!
//! This library provides a Cuckoo Filter that stores its data on disk, making it suitable for
//! large datasets that may not fit into RAM. It uses `memmap2` for efficient file I/O.
//!
//! # Features
//!
//! - `insert`: Add an item to the filter.
//! - `contains`: Check if an item is in the filter.
//! - `delete`: Remove an item from the filter.
//! - `builder`: Create a new, empty filter backed by a file with custom parameters.
//! - `open`: Open an existing filter from a file.
//!
//! # Example
//!
//! ```rust,no_run
//! use cuckoofilter_mmap::{CuckooFilter, FlushMode};
//! use std::fs;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let path = "my_filter.db";
//!     let capacity = 1_000_000;
//!
//!     // Create a new filter using the builder
//!     let mut filter = CuckooFilter::<str>::builder(capacity)
//!         .fingerprint_size(2)
//!         .bucket_size(4)
//!         .flush_mode(FlushMode::None)
//!         .build(path)?;
//!
//!     // Insert an item
//!     let item = "hello world";
//!     if filter.insert(item)? {
//!         println!("Item inserted successfully.");
//!     }
//!
//!     // Check for the item
//!     assert!(filter.contains(item));
//!
//!     // Delete the item
//!     assert!(filter.delete(item)?);
//!     assert!(!filter.contains(item));
//!
//!     // Clean up the file
//!     fs::remove_file(path)?;
//!
//!     Ok(())
//! }
//! ```

use log::{debug, trace};
use memmap2::{MmapMut, MmapOptions};
use rand::Rng;
use std::fs::OpenOptions;
use std::hash::{Hash, Hasher};
use std::io::{self, Cursor};
use std::marker::PhantomData;
use std::path::Path;
use std::str::FromStr;
use thiserror::Error;

// Metadata header at the start of the file.
const METADATA_SIZE: usize = 4; // 2 bytes for fingerprint_size, 2 for bucket_size

/// Custom error types for the Cuckoo Filter.
#[derive(Error, Debug)]
pub enum CuckooError {
    #[error("I/O error")]
    Io(#[from] io::Error),
    #[error("Filter is full")]
    Full,
    #[error("Item not found")]
    NotFound,
    #[error("Invalid file format or metadata")]
    InvalidFileFormat,
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
}

/// A hashable type that can be used with the Cuckoo Filter.
pub trait Item: Hash {}
impl<T: Hash + ?Sized> Item for T {}

/// The main Cuckoo Filter structure.
pub struct CuckooFilter<T: ?Sized> {
    mmap: MmapMut,
    num_buckets: usize,
    fingerprint_size: usize,
    bucket_size: usize,
    max_kicks: usize,
    flush_mode: FlushMode,
    op_count: usize,
    _phantom: PhantomData<T>,
}

/// Defines the flushing strategy for the Cuckoo Filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlushMode {
    None,
    Always,
    AfterNOperations(usize),
}

impl Default for FlushMode {
    fn default() -> Self {
        FlushMode::None
    }
}

impl FromStr for FlushMode {
    type Err = CuckooError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "none" => Ok(FlushMode::None),
            "always" => Ok(FlushMode::Always),
            s if s.starts_with("after:") => {
                let num_str = s.trim_start_matches("after:");
                if let Ok(n) = num_str.parse::<usize>() {
                    Ok(FlushMode::AfterNOperations(n))
                } else {
                    Err(CuckooError::InvalidParameter("Invalid number for AfterNOperations".to_string()))
                }
            }
            _ => Err(CuckooError::InvalidParameter("Invalid FlushMode string".to_string())),
        }
    }
}

/// Represents the location and fingerprint of an item.
struct Location {
    bucket_index: usize,
    fingerprint: Vec<u8>,
}

/// A builder for creating `CuckooFilter` instances with custom parameters.
pub struct CuckooFilterBuilder {
    capacity: usize,
    fingerprint_size: usize,
    bucket_size: usize,
    max_kicks: usize,
    flush_mode: FlushMode,
}

impl CuckooFilterBuilder {
    /// Creates a new builder with a given capacity and default parameters.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            fingerprint_size: 2,
            bucket_size: 4,
            max_kicks: 500,
            flush_mode: FlushMode::None,
        }
    }

    /// Sets the size of the fingerprint in bytes.
    pub fn fingerprint_size(mut self, size: usize) -> Self {
        self.fingerprint_size = size;
        self
    }

    /// Sets the number of fingerprints per bucket.
    pub fn bucket_size(mut self, size: usize) -> Self {
        self.bucket_size = size;
        self
    }

    /// Sets the maximum number of displacements before giving up on an insert.
    pub fn max_kicks(mut self, kicks: usize) -> Self {
        self.max_kicks = kicks;
        self
    }

    /// Sets the flush mode for the filter.
    pub fn flush_mode(mut self, mode: FlushMode) -> Self {
        self.flush_mode = mode;
        self
    }

    /// Builds the Cuckoo Filter, creating the backing file.
    pub fn build<P: AsRef<Path>, T: Item + ?Sized>(self, path: P) -> Result<CuckooFilter<T>, CuckooError> {
        if self.bucket_size == 0 {
            return Err(CuckooError::InvalidParameter("Bucket size cannot be zero".to_string()));
        }
        if self.fingerprint_size == 0 {
            return Err(CuckooError::InvalidParameter("Fingerprint size cannot be zero".to_string()));
        }
        if self.fingerprint_size > 4 {
            return Err(CuckooError::InvalidParameter("Fingerprint size cannot be greater than 4".to_string()));
        }

        let mut num_buckets = (self.capacity as f64 / self.bucket_size as f64).ceil() as usize;
        if num_buckets == 0 {
            num_buckets = 1;
        }
        let num_buckets = num_buckets.next_power_of_two();
        let data_size = num_buckets * self.bucket_size * self.fingerprint_size;
        let file_size = METADATA_SIZE + data_size;

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        file.set_len(file_size as u64)?;

        let mut mmap = unsafe { MmapOptions::new().populate().map_mut(&file)? };

        // Write metadata
        mmap[0..2].copy_from_slice(&(self.fingerprint_size as u16).to_be_bytes());
        mmap[2..4].copy_from_slice(&(self.bucket_size as u16).to_be_bytes());

        Ok(CuckooFilter {
            mmap,
            num_buckets,
            fingerprint_size: self.fingerprint_size,
            bucket_size: self.bucket_size,
            max_kicks: self.max_kicks,
            flush_mode: self.flush_mode,
            op_count: 0,
            _phantom: PhantomData,
        })
    }
}

impl<T: ?Sized> CuckooFilter<T> {
    /// Returns a builder for creating a `CuckooFilter`.
    pub fn builder(capacity: usize) -> CuckooFilterBuilder {
        CuckooFilterBuilder::new(capacity)
    }
}

impl<T: Item + ?Sized> CuckooFilter<T> {
    /// Opens an existing Cuckoo Filter from a file.
    pub fn open<P: AsRef<Path>>(path: P, flush_mode: FlushMode, max_kicks: usize) -> Result<Self, CuckooError> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let metadata = file.metadata()?;
        let file_size = metadata.len() as usize;

        if file_size < METADATA_SIZE {
            return Err(CuckooError::InvalidFileFormat);
        }

        let mmap = unsafe { MmapOptions::new().populate().map_mut(&file)? };

        let fingerprint_size = u16::from_be_bytes(mmap[0..2].try_into().unwrap()) as usize;
        let bucket_size = u16::from_be_bytes(mmap[2..4].try_into().unwrap()) as usize;

        if bucket_size == 0 || fingerprint_size == 0 {
            return Err(CuckooError::InvalidFileFormat);
        }

        let data_size = file_size - METADATA_SIZE;
        if data_size % (bucket_size * fingerprint_size) != 0 {
            return Err(CuckooError::InvalidFileFormat);
        }

        let num_buckets = data_size / (bucket_size * fingerprint_size);

        Ok(Self {
            mmap,
            num_buckets,
            fingerprint_size,
            bucket_size,
            max_kicks,
            flush_mode,
            op_count: 0,
            _phantom: PhantomData,
        })
    }

    pub fn insert(&mut self, item: &T) -> Result<bool, CuckooError> {
        let mut loc = self.location_for(item);

        if self.contains_fingerprint(loc.bucket_index, &loc.fingerprint) {
            trace!("Item already exists, not inserting.");
            return Ok(false);
        }

        let inserted = if self.write_to_bucket(loc.bucket_index, &loc.fingerprint) {
            trace!("Wrote fingerprint to initial bucket.");
            true
        } else {
            let alt_index = self.alt_index(loc.bucket_index, &loc.fingerprint);
            if self.write_to_bucket(alt_index, &loc.fingerprint) {
                trace!("Wrote fingerprint to alternate bucket.");
                true
            } else {
                let mut rng = rand::thread_rng();
                let mut current_index = if rng.gen() { alt_index } else { loc.bucket_index };

                for _ in 0..self.max_kicks {
                    let evicted_fp = self.swap_fingerprint(current_index, &loc.fingerprint);
                    loc.fingerprint = evicted_fp;
                    current_index = self.alt_index(current_index, &loc.fingerprint);

                    if self.write_to_bucket(current_index, &loc.fingerprint) {
                        debug!("Successfully inserted item after kicking.");
                        return Ok(true);
                    }
                }
                false
            }
        };

        if inserted {
            self.handle_flush_on_op()?;
            Ok(true)
        } else {
            Err(CuckooError::Full)
        }
    }

    pub fn contains(&self, item: &T) -> bool {
        let loc = self.location_for(item);
        let alt_index = self.alt_index(loc.bucket_index, &loc.fingerprint);

        self.contains_fingerprint(loc.bucket_index, &loc.fingerprint)
            || self.contains_fingerprint(alt_index, &loc.fingerprint)
    }

    pub fn delete(&mut self, item: &T) -> Result<bool, CuckooError> {
        let loc = self.location_for(item);
        let alt_index = self.alt_index(loc.bucket_index, &loc.fingerprint);

        let deleted = if self.delete_from_bucket(loc.bucket_index, &loc.fingerprint) {
            true
        } else {
            self.delete_from_bucket(alt_index, &loc.fingerprint)
        };

        if deleted {
            self.handle_flush_on_op()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub fn flush(&self) -> io::Result<()> {
        self.mmap.flush()
    }

    pub fn num_buckets(&self) -> usize {
        self.num_buckets
    }

    pub fn capacity(&self) -> usize {
        self.num_buckets * self.bucket_size
    }

    fn location_for(&self, item: &T) -> Location {
        let mut hasher = Murmur3Hasher::new();
        item.hash(&mut hasher);
        let hash_value = hasher.finish();

        let bucket_index = (hash_value % self.num_buckets as u64) as usize;
        let fingerprint_bytes = hash_value.to_be_bytes();
        let mut fingerprint = vec![0u8; self.fingerprint_size];
        fingerprint.copy_from_slice(&fingerprint_bytes[0..self.fingerprint_size]);

        if fingerprint.iter().all(|&b| b == 0) {
            fingerprint[self.fingerprint_size - 1] = 1;
        }

        Location {
            bucket_index,
            fingerprint,
        }
    }

    fn alt_index(&self, bucket_index: usize, fingerprint: &[u8]) -> usize {
        let mut hasher = Murmur3Hasher::new();
        hasher.write(fingerprint);
        let fp_hash = hasher.finish();
        (bucket_index ^ (fp_hash % self.num_buckets as u64) as usize) % self.num_buckets
    }

    fn get_bucket_offset(&self, bucket_index: usize) -> usize {
        METADATA_SIZE + (bucket_index * self.bucket_size * self.fingerprint_size)
    }

    fn get_fingerprint_offset(&self, bucket_index: usize, slot_index: usize) -> usize {
        self.get_bucket_offset(bucket_index) + (slot_index * self.fingerprint_size)
    }

    fn get_fingerprint(&self, bucket_index: usize, slot_index: usize) -> &[u8] {
        let offset = self.get_fingerprint_offset(bucket_index, slot_index);
        &self.mmap[offset..offset + self.fingerprint_size]
    }

    fn set_fingerprint(&mut self, bucket_index: usize, slot_index: usize, fingerprint: &[u8]) {
        let offset = self.get_fingerprint_offset(bucket_index, slot_index);
        self.mmap[offset..offset + self.fingerprint_size].copy_from_slice(fingerprint);
    }

    fn contains_fingerprint(&self, bucket_index: usize, fingerprint: &[u8]) -> bool {
        for i in 0..self.bucket_size {
            if self.get_fingerprint(bucket_index, i) == fingerprint {
                return true;
            }
        }
        false
    }

    fn write_to_bucket(&mut self, bucket_index: usize, fingerprint: &[u8]) -> bool {
        for i in 0..self.bucket_size {
            if self.get_fingerprint(bucket_index, i).iter().all(|&b| b == 0) {
                self.set_fingerprint(bucket_index, i, fingerprint);
                return true;
            }
        }
        false
    }

    fn delete_from_bucket(&mut self, bucket_index: usize, fingerprint: &[u8]) -> bool {
        for i in 0..self.bucket_size {
            if self.get_fingerprint(bucket_index, i) == fingerprint {
                let zeros = vec![0u8; self.fingerprint_size];
                self.set_fingerprint(bucket_index, i, &zeros);
                return true;
            }
        }
        false
    }

    fn swap_fingerprint(&mut self, bucket_index: usize, new_fingerprint: &[u8]) -> Vec<u8> {
        let mut rng = rand::thread_rng();
        let slot_to_evict = rng.gen_range(0..self.bucket_size);
        let evicted_fp = self.get_fingerprint(bucket_index, slot_to_evict).to_vec();
        self.set_fingerprint(bucket_index, slot_to_evict, new_fingerprint);
        evicted_fp
    }

    fn handle_flush_on_op(&mut self) -> Result<(), CuckooError> {
        match self.flush_mode {
            FlushMode::None => Ok(()),
            FlushMode::Always => {
                self.mmap.flush()?;
                Ok(())
            }
            FlushMode::AfterNOperations(n) => {
                self.op_count += 1;
                if self.op_count >= n {
                    self.mmap.flush()?;
                    self.op_count = 0;
                }
                Ok(())
            }
        }
    }
}

struct Murmur3Hasher(u128);

impl Murmur3Hasher {
    fn new() -> Self {
        Murmur3Hasher(0)
    }
}

impl Hasher for Murmur3Hasher {
    fn finish(&self) -> u64 {
        (self.0 >> 64) as u64 ^ (self.0 & 0xFFFFFFFFFFFFFFFF) as u64
    }

    fn write(&mut self, bytes: &[u8]) {
        let mut cursor = Cursor::new(bytes);
        self.0 = murmur3::murmur3_x64_128(&mut cursor, self.0 as u32).unwrap();
    }
}

use std::sync::{Arc, RwLock};

/// A thread-safe, concurrent version of the Cuckoo Filter.
pub struct ConcurrentCuckooFilter<T: ?Sized> {
    inner: Arc<RwLock<CuckooFilter<T>>>,
}

impl<T: Item + ?Sized> ConcurrentCuckooFilter<T> {
    /// Creates a new thread-safe Cuckoo Filter from an existing filter.
    pub fn new(filter: CuckooFilter<T>) -> Self {
        Self {
            inner: Arc::new(RwLock::new(filter)),
        }
    }

    /// Opens an existing thread-safe Cuckoo Filter from a file.
    pub fn open<P: AsRef<Path>>(path: P, flush_mode: FlushMode, max_kicks: usize) -> Result<Self, CuckooError> {
        let filter = CuckooFilter::open(path, flush_mode, max_kicks)?;
        Ok(Self {
            inner: Arc::new(RwLock::new(filter)),
        })
    }

    pub fn insert(&self, item: &T) -> Result<bool, CuckooError> {
        self.inner.write().unwrap().insert(item)
    }

    pub fn contains(&self, item: &T) -> bool {
        self.inner.read().unwrap().contains(item)
    }

    pub fn delete(&self, item: &T) -> Result<bool, CuckooError> {
        self.inner.write().unwrap().delete(item)
    }

    pub fn flush(&self) -> io::Result<()> {
        self.inner.read().unwrap().flush()
    }
}

impl<T: ?Sized> Clone for ConcurrentCuckooFilter<T> {
    fn clone(&self) -> Self {
        Self { inner: self.inner.clone() }
    }
}

#[cfg(test)]
mod concurrent_tests {
    use super::*;
    use std::fs;
    use std::thread;

    fn get_test_file_path(name: &str) -> String {
        format!("/tmp/cuckoo_concurrent_{}.db", name)
    }
    
    #[test]
    fn test_capacity_file_size() {
        let path = get_test_file_path("insert_contains");
        let _ = fs::remove_file(&path);
        let filter: CuckooFilter<String> = CuckooFilter::<String>::builder(50000000)
            .build(&path)
            .unwrap();
    }
    #[test]
    fn test_concurrent_insert_and_contains() {
        let path = get_test_file_path("insert_contains");
        let _ = fs::remove_file(&path);
        let filter: CuckooFilter<String> = CuckooFilter::<String>::builder(100_000)
            .build(&path)
            .unwrap();
        let filter = ConcurrentCuckooFilter {
            inner: Arc::new(RwLock::new(filter)),
        };

        let items_to_insert: Vec<String> = (0..1000).map(|i| format!("item-{}", i)).collect();

        let handles: Vec<_> = items_to_insert
            .chunks(100)
            .map(|chunk| {
                let filter_clone = filter.clone();
                let chunk_clone: Vec<String> = chunk.iter().map(|s| s.to_string()).collect();
                thread::spawn(move || {
                    for item in &chunk_clone {
                        filter_clone.insert(item).unwrap();
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        for item in &items_to_insert {
            assert!(filter.contains(item));
        }

        assert!(!filter.contains(&"not-an-item".to_string()));

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_concurrent_read_and_write() {
        let path = get_test_file_path("read_write");
        let _ = fs::remove_file(&path);
        let filter: CuckooFilter<String> = CuckooFilter::<String>::builder(50_000)
            .build(&path)
            .unwrap();
        let filter = ConcurrentCuckooFilter {
            inner: Arc::new(RwLock::new(filter)),
        };

        for i in 0..500 {
            filter.insert(&format!("pre-item-{}", i)).unwrap();
        }

        let mut handles = vec![];

        for i in 0..4 {
            let filter_clone = filter.clone();
            let handle = thread::spawn(move || {
                for j in 0..200 {
                    let item = format!("writer-{}-item-{}", i, j);
                    filter_clone.insert(&item).unwrap();
                }
            });
            handles.push(handle);
        }

        for _ in 0..4 {
            let filter_clone = filter.clone();
            let handle = thread::spawn(move || {
                for i in 0..500 {
                    assert!(filter_clone.contains(&format!("pre-item-{}", i)));
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        for i in 0..4 {
            for j in 0..200 {
                assert!(filter.contains(&format!("writer-{}-item-{}", i, j)));
            }
        }

        fs::remove_file(path).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::Once;

    static INIT: Once = Once::new();

    fn setup() {
        INIT.call_once(|| {
            let _ = env_logger::builder().is_test(true).try_init();
        });
    }

    fn get_test_file_path(name: &str) -> String {
        format!("/tmp/cuckoo_{}.db", name)
    }

    #[test]
    fn test_new_and_open() {
        setup();
        let path = get_test_file_path("new_and_open");
        let _ = fs::remove_file(&path);

        let capacity = 1000;
        {
            let filter: CuckooFilter<str> = CuckooFilter::<str>::builder(capacity).build(&path).unwrap();
            assert_eq!(filter.capacity(), 1024);
            assert_eq!(filter.num_buckets(), 256);
        }

        {
            let filter = CuckooFilter::<str>::open(&path, FlushMode::None, 500).unwrap();
            assert_eq!(filter.capacity(), 1024);
            assert_eq!(filter.num_buckets(), 256);
        }

        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_insert_and_contains() {
        setup();
        let path = get_test_file_path("insert_and_contains");
        let _ = fs::remove_file(&path);

        let mut filter = CuckooFilter::<str>::builder(100).build(&path).unwrap();

        let item1 = "hello";
        let item2 = "world";
        let item3 = "rust";

        assert!(filter.insert(item1).unwrap());
        assert!(filter.insert(item2).unwrap());

        assert!(filter.contains(item1));
        assert!(filter.contains(item2));
        assert!(!filter.contains(item3));

        assert!(!filter.insert(item1).unwrap());

        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_delete() {
        setup();
        let path = get_test_file_path("delete");
        let _ = fs::remove_file(&path);

        let mut filter = CuckooFilter::<str>::builder(100).build(&path).unwrap();

        let item = "test_item";
        filter.insert(item).unwrap();
        assert!(filter.contains(item));

        assert!(filter.delete(item).unwrap());
        assert!(!filter.contains(item));

        assert!(!filter.delete("non_existent").unwrap());
    }

    #[test]
    fn test_persistence() {
        setup();
        let path = get_test_file_path("persistence");
        let _ = fs::remove_file(&path);

        let item1 = "persistent_data";
        let item2 = "another_one";

        {
            let mut filter = CuckooFilter::<str>::builder(100).build(&path).unwrap();
            filter.insert(item1).unwrap();
            filter.insert(item2).unwrap();
            assert!(filter.contains(item1));
            filter.flush().unwrap();
        }

        {
            let mut filter = CuckooFilter::<str>::open(&path, FlushMode::None, 500).unwrap();
            assert!(filter.contains(item1));
            assert!(filter.contains(item2));
            assert!(!filter.contains("not_present"));

            filter.delete(item1).unwrap();
            filter.flush().unwrap();
        }

        {
            let filter = CuckooFilter::<str>::open(&path, FlushMode::None, 500).unwrap();
            assert!(!filter.contains(item1));
            assert!(filter.contains(item2));
        }

        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_full_filter() {
        setup();
        let path = get_test_file_path("full_filter");
        let _ = fs::remove_file(&path);

        let capacity = 16;
        let mut filter = CuckooFilter::<String>::builder(capacity).build(&path).unwrap();

        let mut inserted_count = 0;
        for i in 0..100 {
            let item = format!("item-{}", i);
            if let Ok(true) = filter.insert(&item) {
                inserted_count += 1;
            } else {
                break;
            }
        }

        let result = filter.insert(&"one_more_item".to_string());
        assert!(matches!(result, Err(CuckooError::Full)));

        println!("Filter filled up after {} insertions.", inserted_count);

        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_zero_capacity() {
        setup();
        let path = get_test_file_path("zero_capacity");
        let _ = fs::remove_file(&path);

        let mut filter = CuckooFilter::<str>::builder(0).build(&path).unwrap();
        assert_eq!(filter.capacity(), 4);
        assert_eq!(filter.num_buckets(), 1);

        assert!(filter.insert("a").unwrap());
        assert!(filter.insert("b").unwrap());
        assert!(filter.insert("c").unwrap());
        assert!(filter.insert("d").unwrap());

        assert!(filter.contains("a"));
        assert!(filter.contains("d"));

        assert!(matches!(filter.insert("e"), Err(CuckooError::Full)));

        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_flush_after_n_operations() {
        setup();
        let path = get_test_file_path("flush_after_n");
        let _ = fs::remove_file(&path);

        let n = 5;
        let mut filter = CuckooFilter::<String>::builder(100)
            .flush_mode(FlushMode::AfterNOperations(n))
            .build(&path)
            .unwrap();

        for i in 0..n - 1 {
            filter.insert(&format!("item-{}", i)).unwrap();
        }
        assert_eq!(filter.op_count, n - 1);

        filter.insert(&format!("item-{}", n - 1)).unwrap();
        assert_eq!(filter.op_count, 0);

        let reopened_filter = CuckooFilter::<String>::open(&path, FlushMode::None, 500).unwrap();
        for i in 0..n {
            assert!(reopened_filter.contains(&format!("item-{}", i)));
        }

        fs::remove_file(&path).unwrap();
    }
}