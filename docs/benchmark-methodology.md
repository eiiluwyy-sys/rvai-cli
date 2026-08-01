# Benchmark Methodology

The first `builtin-gemm-int8` benchmark uses a fixed and reproducible
measurement convention.

## Warmup and correctness

Before collecting latency samples, `rvai-bench` runs one untimed GEMM. That
result is compared element by element with an independent reference
implementation that accumulates into INT64 values. The untimed run warms the
same data and code paths, but it is not included in the reported mean or P95.

## P95 latency

The benchmark sorts all measured latency samples in ascending order and uses
the nearest-rank definition:

```text
index = ceil(0.95 * sample_count) - 1
```

The index is zero-based. With the default 20 iterations, P95 is therefore the
19th sample in ascending order (index 18).

## Throughput

One INT8 multiplication and one accumulation are counted as two operations.
Throughput uses the mean measured latency:

```text
operations = 2 * M * N * K
GOPS = operations / mean_seconds / 1e9
```

## Memory

`memory_bytes` reports the allocated input and output matrix buffers. It is not
a measurement of process peak RSS.
