use criterion::{criterion_group, criterion_main, Criterion};
use cuckoofilter_mmap::{ConcurrentCuckooFilter, CuckooFilter, FlushMode};
use std::env;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::thread;

const INSERT_BENCH_ITEMS: usize = 1_000; // Number of items to insert for the insert benchmark
const CONTAINS_BENCH_ITEMS: usize = 10_000; // Number of items to check for the contains benchmark

const THREAD_COUNTS: &[usize] = &[1, 2, 4, 8]; // Different thread counts to test
const CONCURRENT_ITEMS_PER_THREAD: usize = 250; // Items per thread for concurrent insert/contains

fn setup_filter(capacity: usize) -> (CuckooFilter<String>, String) {
    let path = Path::new("/tmp").join(format!("cuckoo_bench_{}.db", capacity));
    let path_str = path.to_str().unwrap().to_string();
    let _ = fs::remove_file(&path);
    let filter = CuckooFilter::<String>::builder(capacity)
        .build(&path)
        .unwrap();
    (filter, path_str)
}

fn setup_concurrent_filter(capacity: usize) -> (ConcurrentCuckooFilter<String>, String) {
    let path = Path::new("/tmp").join(format!("cuckoo_concurrent_bench_{}.db", capacity));
    let path_str = path.to_str().unwrap().to_string();
    let _ = fs::remove_file(&path);
    let filter = CuckooFilter::<String>::builder(capacity)
        .build(&path)
        .unwrap();
    let concurrent_filter = ConcurrentCuckooFilter::new(filter);
    (concurrent_filter, path_str)
}

fn cleanup_filter(path_str: &str) {
    let _ = fs::remove_file(path_str);
}

fn run_benchmark(c: &mut Criterion, capacity: usize) {
    let mut group = c.benchmark_group(format!("CuckooFilter_Capacity_{}", capacity));

    // Benchmark insert operation
    group.bench_function("insert", |b| {
        let (mut filter, path_str) = setup_filter(capacity);
        let items: Vec<String> = (0..INSERT_BENCH_ITEMS)
            .map(|i| format!("insert-item-{}", i))
            .collect();
        let mut i = 0;
        b.iter(|| {
            let item = &items[i % INSERT_BENCH_ITEMS];
            filter.insert(item).unwrap();
            i += 1;
        });
        cleanup_filter(&path_str);
    });

    // Benchmark contains operation
    group.bench_function("contains", |b| {
        let (mut filter, path_str) = setup_filter(capacity);
        let items: Vec<String> = (0..CONTAINS_BENCH_ITEMS)
            .map(|i| format!("contains-item-{}", i))
            .collect();
        // Pre-fill the filter with items for contains benchmark
        for item in &items {
            filter.insert(item).unwrap();
        }
        let mut i = 0;
        b.iter(|| {
            let item = &items[i % CONTAINS_BENCH_ITEMS];
            filter.contains(item);
            i += 1;
        });
        cleanup_filter(&path_str);
    });

    // Benchmark concurrent insert and contains for different thread counts
    for &num_threads in THREAD_COUNTS.iter() {
        // Concurrent Insert
        group.bench_function(format!("concurrent_insert_{}_threads", num_threads), |b| {
            let (filter, path_str) = setup_concurrent_filter(capacity);
            b.iter(|| {
                let mut handles = vec![];
                for t_id in 0..num_threads {
                    let filter_clone = filter.clone();
                    let items_for_thread: Vec<String> = (0..CONCURRENT_ITEMS_PER_THREAD)
                        .map(|i| format!("concurrent-insert-item-{}-thread-{}", i, t_id))
                        .collect();
                    handles.push(thread::spawn(move || {
                        for item in items_for_thread {
                            filter_clone.insert(&item).unwrap();
                        }
                    }));
                }
                for handle in handles {
                    handle.join().unwrap();
                }
            });
            cleanup_filter(&path_str);
        });

        // Concurrent Contains
        group.bench_function(
            format!("concurrent_contains_{}_threads", num_threads),
            |b| {
                let (filter, path_str) = setup_concurrent_filter(capacity);
                let total_items_to_insert = num_threads * CONCURRENT_ITEMS_PER_THREAD;
                let items_to_check: Vec<String> = (0..total_items_to_insert)
                    .map(|i| format!("concurrent-contains-item-{}", i))
                    .collect();

                // Pre-fill the filter with items for concurrent contains benchmark
                for item in &items_to_check {
                    filter.insert(item).unwrap();
                }

                b.iter(|| {
                    let mut handles = vec![];
                    for t_id in 0..num_threads {
                        let filter_clone = filter.clone();
                        let items_for_thread: Vec<String> = items_to_check
                            .iter()
                            .skip(t_id * CONCURRENT_ITEMS_PER_THREAD)
                            .take(CONCURRENT_ITEMS_PER_THREAD)
                            .map(|s| s.to_string())
                            .collect();

                        handles.push(thread::spawn(move || {
                            for item in items_for_thread {
                                filter_clone.contains(&item);
                            }
                        }));
                    }
                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
                cleanup_filter(&path_str);
            },
        );
    }

    group.finish();
}

use rand::Rng;

fn setup_filter_bytes(capacity: usize) -> (CuckooFilter<[u8; 16]>, String) {
    let path = Path::new("/tmp").join(format!("cuckoo_bench_{}.db", capacity));
    let path_str = path.to_str().unwrap().to_string();
    let _ = fs::remove_file(&path);
    let filter = CuckooFilter::<[u8; 16]>::builder(capacity)
        .build(&path)
        .unwrap();
    (filter, path_str)
}

fn setup_concurrent_filter_bytes(capacity: usize) -> (ConcurrentCuckooFilter<[u8; 16]>, String) {
    let path = Path::new("/tmp").join(format!("cuckoo_concurrent_bench_{}.db", capacity));
    let path_str = path.to_str().unwrap().to_string();
    let _ = fs::remove_file(&path);
    let filter = CuckooFilter::<[u8; 16]>::builder(capacity)
        .build(&path)
        .unwrap();
    let concurrent_filter = ConcurrentCuckooFilter::new(filter);
    (concurrent_filter, path_str)
}

fn cuckoo_benchmarks(c: &mut Criterion) {
    let capacities = [
        50_000,
        100_000,
        1_000_000,
        10_000_000,
        100_000_000,
        1_000_000_000,
        // 10_000_000_000, // This capacity might be too large for practical benchmarking
    ];

    for &capacity in capacities.iter() {
        run_benchmark(c, capacity);
    }
}

fn traceid_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("traceid_benches");
    group.sample_size(10);

    group.bench_function("insert_100m_traceids", |b| {
        let capacity = 100_000_000;
        let (mut filter, path_str) = setup_filter_bytes(capacity);
        let mut rng = rand::thread_rng();

        b.iter(|| {
            let mut item = [0u8; 16];
            rng.fill(&mut item);
            filter.insert(&item).unwrap();
        });

        cleanup_filter(&path_str);
    });

    group.bench_function("concurrent_insert_100m_traceids_8_threads", |b| {
        let capacity = 100_000_000;
        let num_threads = 8;
        let items_per_thread = capacity / num_threads;

        b.iter_batched(
            || setup_concurrent_filter_bytes(capacity),
            |(filter, path_str)| {
                let mut handles = vec![];
                for _ in 0..num_threads {
                    let filter_clone = filter.clone();
                    handles.push(thread::spawn(move || {
                        let mut rng = rand::thread_rng();
                        for _ in 0..items_per_thread {
                            let mut item = [0u8; 16];
                            rng.fill(&mut item);
                            filter_clone.insert(&item).unwrap();
                        }
                    }));
                }
                for handle in handles {
                    handle.join().unwrap();
                }
                cleanup_filter(&path_str);
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(benches, cuckoo_benchmarks, traceid_benches);
criterion_main!(benches);
