# CuckooFilter-mmap Size Benchmark

This table shows the estimated file size for a `cuckoofilter-mmap` instance based on its desired capacity (number of trace ID elements) and the corresponding number of buckets.

| Capacity (Trace IDs) | Number of Buckets | File Size |
| :------------------- | :---------------- | :-------- |
| 50,000               | 16,384            | 128 KB    |
| 100,000              | 32,768            | 256 KB    |
| 1,000,000            | 262,144           | 2 MB      |
| 10,000,000           | 4,194,304         | 32 MB     |
| 100,000,000          | 33,554,432        | 256 MB    |
| 1,000,000,000        | 268,435,456       | 2 GB      |
| 10,000,000,000       | 4,294,967,296     | 32 GB     |

# Performance Benchmark

This table shows the average time taken for `insert`, `contains`, and concurrent operations for different filter capacities and thread counts. The times are in nanoseconds per operation for `insert` and `contains`, and microseconds per operation for concurrent operations.

| Capacity (Trace IDs) | Insert (ns/op) | Contains (ns/op) | Concurrent Insert (µs/op) - 1 Thread | Concurrent Contains (µs/op) - 1 Thread | Concurrent Insert (µs/op) - 2 Threads | Concurrent Contains (µs/op) - 2 Threads | Concurrent Insert (µs/op) - 4 Threads | Concurrent Contains (µs/op) - 4 Threads | Concurrent Insert (µs/op) - 8 Threads | Concurrent Contains (µs/op) - 8 Threads |
| :------------------- | :------------- | :--------------- | :----------------------------------- | :------------------------------------- | :----------------------------------- | :------------------------------------- | :----------------------------------- | :------------------------------------- | :----------------------------------- | :------------------------------------- |
| 50,000               | 31.60          | 41.72            | 38.16                                | 46.70                                  | 62.67                                | 72.83                                  | 161.10                               | 134.69                                 | 338.24                               | 396.26                                 |
| 100,000              | 30.29          | 41.33            | 41.63                                | 46.83                                  | 67.45                                | 53.84                                  | 122.45                               | 133.77                                 | 218.08                               | 393.61                                 |
| 1,000,000            | 29.98          | 39.77            | 42.03                                | 32.15                                  | 58.23                                | 55.39                                  | 104.93                               | 142.42                                 | 244.77                               | 453.56                                 |
| 10,000,000           | 30.83          | 38.85            | 40.41                                | 40.73                                  | 64.54                                | 64.02                                  | 118.59                               | 139.53                                 | 299.28                               | 394.89                                 |
| 100,000,000          | 32.21          | 39.62            | 38.01                                | 32.65                                  | 68.75                                | 60.58                                  | 122.94                               | 146.20                                 | 264.77                               | 404.99                                 |
| 1,000,000,000        | 32.98          | 41.09            | 41.55                                | 33.28                                  | 63.91                                | 64.43                                  | 127.79                               | 171.44                                 | 268.39                               | 411.39                                 |


# Cuckoo Filter Benchmark Results

This document presents the benchmark results for inserting 100 million trace IDs into the Cuckoo Filter under different threading scenarios.

## Test Environment

- **Operating System:** macOS
- **CPU:** Apple M-series (ARM64)
- **Rust Version:** (Specify if known, otherwise omit)

## Benchmark: Inserting 100 Million Trace IDs

| Scenario                | Time Taken (seconds) |
| ----------------------- | -------------------- |
| Single-threaded         | 16.15                |
| Multi-threaded (8 threads) | 40.13                |

## Analysis

The results indicate that for this specific workload, single-threaded insertion is significantly faster than multi-threaded insertion.

The performance degradation in the multi-threaded scenario is likely due to heavy lock contention on the underlying data structure. The `ConcurrentCuckooFilter` uses a `RwLock` to ensure thread safety. When multiple threads are attempting to write to the filter simultaneously, they must acquire a write lock, which serializes their access and introduces overhead, negating the benefits of parallelization for this insert-heavy workload.
