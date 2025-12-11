//! Comprehensive benchmarks for SIMD optimization candidates.
//!
//! This module provides benchmarks for all computational hot paths that are
//! candidates for SIMD optimization. Each benchmark establishes a baseline
//! that can be compared against optimized implementations.
//!
//! Run with: `cargo bench --bench simd_benchmarks`
//!
//! Generate HTML reports: `cargo bench --bench simd_benchmarks -- --plotting-backend plotters`

use criterion::{
    black_box, criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup,
    BenchmarkId, Criterion, Throughput,
};
use mokosh::prelude::*;
use mokosh::utils::Random;
use std::collections::HashSet;

// =============================================================================
// SDR OPERATIONS BENCHMARKS
// =============================================================================

/// Benchmark SDR overlap computation (two-pointer merge on sorted vectors).
/// This is a critical hot path called multiple times per SP/TM compute.
fn bench_sdr_overlap(c: &mut Criterion) {
    let mut group = c.benchmark_group("sdr_overlap");
    group.sample_size(1000);

    let mut rng = Random::new(42);

    // Test different SDR sizes and sparsities
    for (size, sparsity) in &[(2048, 0.02), (2048, 0.05), (2048, 0.10), (65536, 0.02)] {
        let mut sdr1 = Sdr::new(&[*size as u32]);
        let mut sdr2 = Sdr::new(&[*size as u32]);

        sdr1.randomize(*sparsity, &mut rng);
        sdr2.randomize(*sparsity, &mut rng);

        let active_bits = ((*size as f32) * sparsity) as usize;

        group.throughput(Throughput::Elements(active_bits as u64 * 2));
        group.bench_with_input(
            BenchmarkId::new(format!("size_{}_sparsity_{}", size, sparsity), active_bits),
            &(&sdr1, &sdr2),
            |b, (s1, s2)| {
                b.iter(|| black_box(s1.get_overlap(s2)));
            },
        );
    }

    group.finish();
}

/// Benchmark SDR intersection (set intersection of sorted vectors).
fn bench_sdr_intersection(c: &mut Criterion) {
    let mut group = c.benchmark_group("sdr_intersection");
    group.sample_size(500);

    let mut rng = Random::new(42);

    for (size, sparsity) in &[(2048, 0.05), (65536, 0.02)] {
        let mut sdr1 = Sdr::new(&[*size as u32]);
        let mut sdr2 = Sdr::new(&[*size as u32]);

        sdr1.randomize(*sparsity, &mut rng);
        sdr2.randomize(*sparsity, &mut rng);

        let active_bits = ((*size as f32) * sparsity) as usize;
        let size_copy = *size;

        group.throughput(Throughput::Elements(active_bits as u64 * 2));
        group.bench_function(
            BenchmarkId::new(format!("size_{}_sparsity_{}", size, sparsity), active_bits),
            |b| {
                let mut result = Sdr::new(&[size_copy as u32]);
                b.iter(|| {
                    result.intersection(&sdr1, &sdr2).unwrap();
                    black_box(result.get_sum())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark SDR union (set union of sorted vectors).
fn bench_sdr_union(c: &mut Criterion) {
    let mut group = c.benchmark_group("sdr_union");
    group.sample_size(500);

    let mut rng = Random::new(42);

    for (size, sparsity) in &[(2048, 0.05), (65536, 0.02)] {
        let mut sdr1 = Sdr::new(&[*size as u32]);
        let mut sdr2 = Sdr::new(&[*size as u32]);

        sdr1.randomize(*sparsity, &mut rng);
        sdr2.randomize(*sparsity, &mut rng);

        let active_bits = ((*size as f32) * sparsity) as usize;
        let size_copy = *size;

        group.throughput(Throughput::Elements(active_bits as u64 * 2));
        group.bench_function(
            BenchmarkId::new(format!("size_{}_sparsity_{}", size, sparsity), active_bits),
            |b| {
                let mut result = Sdr::new(&[size_copy as u32]);
                b.iter(|| {
                    result.set_union(&sdr1, &sdr2).unwrap();
                    black_box(result.get_sum())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark dense-to-sparse conversion.
fn bench_sdr_dense_to_sparse(c: &mut Criterion) {
    let mut group = c.benchmark_group("sdr_dense_to_sparse");
    group.sample_size(500);

    for (size, sparsity) in &[(2048, 0.05), (65536, 0.02)] {
        let mut dense = vec![0u8; *size];
        let active_count = ((*size as f32) * sparsity) as usize;

        // Set some bits active
        let mut rng = Random::new(42);
        let indices = rng.sample((0..*size).collect(), active_count);
        for idx in indices {
            dense[idx] = 1;
        }

        let mut sdr = Sdr::new(&[*size as u32]);

        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("size_{}_sparsity_{}", size, sparsity), size),
            &dense,
            |b, d| {
                b.iter(|| {
                    sdr.set_dense(d).unwrap();
                    black_box(sdr.get_sparse())
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// SPATIAL POOLER BENCHMARKS
// =============================================================================

/// Benchmark the full SP compute cycle.
fn bench_spatial_pooler_compute(c: &mut Criterion) {
    let mut group = c.benchmark_group("sp_compute");
    group.sample_size(100);

    for (input_size, column_count) in &[(100, 2048), (1024, 2048), (2048, 4096)] {
        let sp = SpatialPooler::new(SpatialPoolerParams {
            input_dimensions: vec![*input_size as u32],
            column_dimensions: vec![*column_count as u32],
            potential_radius: (*input_size / 2) as u32,
            global_inhibition: true,
            local_area_density: 0.02,
            ..Default::default()
        })
        .unwrap();

        let mut input = Sdr::new(&[*input_size as u32]);
        let mut rng = Random::new(42);
        input.randomize(0.05, &mut rng);

        group.throughput(Throughput::Elements(*column_count as u64));
        group.bench_function(
            BenchmarkId::new(
                format!("in_{}_cols_{}", input_size, column_count),
                column_count,
            ),
            |b| {
                let mut sp = sp.clone();
                let mut output = Sdr::new(&[*column_count as u32]);
                b.iter(|| {
                    sp.compute(&input, false, &mut output);
                    black_box(output.get_sum())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark SP overlap calculation (hot path).
fn bench_sp_calculate_overlaps(c: &mut Criterion) {
    let mut group = c.benchmark_group("sp_calculate_overlaps");
    group.sample_size(200);

    for column_count in &[2048, 4096] {
        let sp = SpatialPooler::new(SpatialPoolerParams {
            input_dimensions: vec![1024],
            column_dimensions: vec![*column_count as u32],
            potential_radius: 512,
            global_inhibition: true,
            local_area_density: 0.02,
            ..Default::default()
        })
        .unwrap();

        let mut input = Sdr::new(&[1024]);
        let mut rng = Random::new(42);
        input.randomize(0.05, &mut rng);

        group.throughput(Throughput::Elements(*column_count as u64));
        group.bench_function(
            BenchmarkId::new("columns", column_count),
            |b| {
                let mut sp = sp.clone();
                let mut output = Sdr::new(&[*column_count as u32]);
                b.iter(|| {
                    // compute internally calls calculate_overlaps
                    sp.compute(&input, false, &mut output);
                    black_box(output.get_sum())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark SP global inhibition (sorting hot path).
fn bench_sp_inhibition(c: &mut Criterion) {
    let mut group = c.benchmark_group("sp_inhibition_sort");
    group.sample_size(500);

    // Simulate the inhibition step by sorting overlaps
    for size in &[2048, 4096, 8192] {
        let mut rng = Random::new(42);
        let overlaps: Vec<f32> = (0..*size)
            .map(|_| rng.get_real32() * 100.0)
            .collect();

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("columns", size),
            &overlaps,
            |b, overlaps| {
                b.iter(|| {
                    let mut indexed: Vec<(usize, f32)> = overlaps
                        .iter()
                        .enumerate()
                        .map(|(i, &o)| (i, o))
                        .collect();
                    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    black_box(indexed)
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// TEMPORAL MEMORY BENCHMARKS
// =============================================================================

/// Benchmark TM compute cycle.
fn bench_temporal_memory_compute(c: &mut Criterion) {
    let mut group = c.benchmark_group("tm_compute");
    group.sample_size(100);

    for (columns, cells_per_col) in &[(2048, 32), (4096, 16)] {
        let mut tm = TemporalMemory::new(TemporalMemoryParams {
            column_dimensions: vec![*columns as u32],
            cells_per_column: *cells_per_col,
            activation_threshold: 13,
            min_threshold: 10,
            ..Default::default()
        })
        .unwrap();

        let mut active_columns = Sdr::new(&[*columns as u32]);
        let mut rng = Random::new(42);
        active_columns.randomize(0.02, &mut rng);

        // Prime the TM with some learning
        for _ in 0..10 {
            tm.compute(&active_columns, true);
            active_columns.randomize(0.02, &mut rng);
        }

        let tm_clone = tm.clone();
        let cols_clone = active_columns.clone();

        group.throughput(Throughput::Elements((*columns * (*cells_per_col as usize)) as u64));
        group.bench_function(
            BenchmarkId::new(
                format!("cols_{}_cells_{}", columns, cells_per_col),
                columns * (*cells_per_col as usize),
            ),
            |b| {
                let mut tm = tm_clone.clone();
                b.iter(|| {
                    tm.compute(&cols_clone, false);
                    black_box(tm.active_cells().len())
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// CONNECTIONS BENCHMARKS
// =============================================================================

/// Benchmark Connections activity computation.
fn bench_connections_activity(c: &mut Criterion) {
    let mut group = c.benchmark_group("connections_activity");
    group.sample_size(200);

    for num_cells in &[10000, 50000, 100000] {
        let mut conn = Connections::new(ConnectionsParams {
            num_cells: *num_cells as u32,
            connected_threshold: 0.5,
            timeseries: false,
        });

        let mut rng = Random::new(42);

        // Create segments and synapses
        let num_segments = num_cells / 10;
        for i in 0..num_segments {
            let cell = (i % *num_cells) as u32;
            let segment = conn.create_segment(cell, None);

            // Add 20-40 synapses per segment
            let num_synapses = 20 + (i % 20);
            for j in 0..num_synapses {
                let presynaptic = ((i * 31 + j * 17) % *num_cells) as u32;
                let permanence = 0.3 + (j as f32 * 0.02).min(0.4);
                conn.create_synapse(segment, presynaptic, permanence);
            }
        }

        // Create active cells
        let num_active = (*num_cells as f32 * 0.02) as usize;
        let active_cells: Vec<u32> = rng.sample((0..*num_cells as u32).collect(), num_active);

        let conn_clone = conn.clone();
        let active_clone = active_cells.clone();

        group.throughput(Throughput::Elements(*num_cells as u64));
        group.bench_function(
            BenchmarkId::new("cells", num_cells),
            |b| {
                let mut conn = conn_clone.clone();
                b.iter(|| {
                    let activity = conn.compute_activity(&active_clone, false);
                    black_box(activity)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark adapt_segment (permanence updates with clamping).
fn bench_connections_adapt_segment(c: &mut Criterion) {
    let mut group = c.benchmark_group("connections_adapt_segment");
    group.sample_size(500);

    for num_synapses in &[32, 64, 128, 256] {
        let mut conn = Connections::new(ConnectionsParams {
            num_cells: 10000,
            connected_threshold: 0.5,
            timeseries: false,
        });

        let segment = conn.create_segment(0, None);

        // Add synapses
        for i in 0..*num_synapses {
            conn.create_synapse(segment, (i + 1) as u32, 0.5);
        }

        // Create input SDR
        let mut input = Sdr::new(&[10000]);
        let active: Vec<u32> = (0..*num_synapses / 2).map(|i| (i + 1) as u32).collect();
        input.set_sparse(&active).unwrap();

        let conn_clone = conn.clone();
        let input_clone = input.clone();

        group.throughput(Throughput::Elements(*num_synapses as u64));
        group.bench_function(
            BenchmarkId::new("synapses", num_synapses),
            |b| {
                let mut conn = conn_clone.clone();
                b.iter(|| {
                    conn.adapt_segment(segment, &input_clone, 0.1, 0.1, false, 0);
                    black_box(conn.segment_flat_list_length())
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// VECTORIZED MATH OPERATIONS BENCHMARKS
// =============================================================================

/// Benchmark boost factor computation (exp() calls).
fn bench_boost_factors(c: &mut Criterion) {
    let mut group = c.benchmark_group("boost_factors_exp");
    group.sample_size(500);

    for size in &[2048, 4096, 8192] {
        let duty_cycles: Vec<f32> = (0..*size).map(|i| 0.02 + (i as f32 * 0.0001)).collect();
        let target_density = 0.02f32;
        let boost_strength = 3.0f32;

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_function(
            BenchmarkId::new("columns", size),
            |b| {
                b.iter(|| {
                    let boosts: Vec<f32> = duty_cycles
                        .iter()
                        .map(|&duty| (boost_strength * (target_density - duty)).exp())
                        .collect();
                    black_box(boosts)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark duty cycle updates (division per element).
fn bench_duty_cycle_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("duty_cycle_update");
    group.sample_size(500);

    for size in &[2048, 4096, 8192] {
        let duty_cycles: Vec<f32> = (0..*size).map(|i| 0.01 + (i as f32 * 0.00001)).collect();
        let overlaps: Vec<u16> = (0..*size).map(|i| ((i * 7) % 50) as u16).collect();
        let period = 1000.0f32;

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_function(
            BenchmarkId::new("columns", size),
            |b| {
                b.iter(|| {
                    let duties: Vec<f32> = duty_cycles
                        .iter()
                        .zip(overlaps.iter())
                        .map(|(&duty, &overlap)| {
                            let value = if overlap > 0 { 1.0 } else { 0.0 };
                            ((period - 1.0) * duty + value) / period
                        })
                        .collect();
                    black_box(duties)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark permanence clamping operations.
fn bench_permanence_clamp(c: &mut Criterion) {
    let mut group = c.benchmark_group("permanence_clamp");
    group.sample_size(1000);

    for size in &[256, 1024, 4096] {
        let permanences: Vec<f32> = (0..*size)
            .map(|i| -0.1 + (i as f32 * 0.001))
            .collect();
        let deltas: Vec<f32> = (0..*size)
            .map(|i| if i % 2 == 0 { 0.1 } else { -0.1 })
            .collect();

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("synapses", size),
            &(&permanences, &deltas),
            |b, (perms, delts)| {
                b.iter(|| {
                    let result: Vec<f32> = perms
                        .iter()
                        .zip(delts.iter())
                        .map(|(&p, &d)| (p + d).clamp(0.0, 1.0))
                        .collect();
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// HASH SET MEMBERSHIP BENCHMARKS
// =============================================================================

/// Benchmark HashSet membership checks (used in learning).
fn bench_hashset_membership(c: &mut Criterion) {
    let mut group = c.benchmark_group("hashset_membership");
    group.sample_size(500);

    for (set_size, query_count) in &[(100, 1000), (1000, 5000), (5000, 10000)] {
        let set: HashSet<u32> = (0..*set_size as u32).collect();
        let queries: Vec<u32> = (0..*query_count)
            .map(|i| (i * 7 % (*set_size * 2)) as u32)
            .collect();

        group.throughput(Throughput::Elements(*query_count as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("set_{}_queries_{}", set_size, query_count), query_count),
            &(&set, &queries),
            |b, (set, queries)| {
                b.iter(|| {
                    let count: usize = queries.iter().filter(|q| set.contains(q)).count();
                    black_box(count)
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// SORTED VECTOR BINARY SEARCH BENCHMARKS
// =============================================================================

/// Benchmark binary search in sorted vectors (alternative to HashSet).
fn bench_sorted_vector_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("sorted_vector_search");
    group.sample_size(500);

    for (vec_size, query_count) in &[(100, 1000), (1000, 5000), (5000, 10000)] {
        let sorted: Vec<u32> = (0..*vec_size as u32).collect();
        let queries: Vec<u32> = (0..*query_count)
            .map(|i| (i * 7 % (*vec_size * 2)) as u32)
            .collect();

        group.throughput(Throughput::Elements(*query_count as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("vec_{}_queries_{}", vec_size, query_count), query_count),
            &(&sorted, &queries),
            |b, (sorted, queries)| {
                b.iter(|| {
                    let count: usize = queries
                        .iter()
                        .filter(|q| sorted.binary_search(q).is_ok())
                        .count();
                    black_box(count)
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// CRITERION CONFIGURATION
// =============================================================================

criterion_group!(
    sdr_benches,
    bench_sdr_overlap,
    bench_sdr_intersection,
    bench_sdr_union,
    bench_sdr_dense_to_sparse,
);

criterion_group!(
    sp_benches,
    bench_spatial_pooler_compute,
    bench_sp_calculate_overlaps,
    bench_sp_inhibition,
);

criterion_group!(tm_benches, bench_temporal_memory_compute,);

criterion_group!(
    connections_benches,
    bench_connections_activity,
    bench_connections_adapt_segment,
);

criterion_group!(
    math_benches,
    bench_boost_factors,
    bench_duty_cycle_update,
    bench_permanence_clamp,
);

criterion_group!(
    membership_benches,
    bench_hashset_membership,
    bench_sorted_vector_search,
);

criterion_main!(
    sdr_benches,
    sp_benches,
    tm_benches,
    connections_benches,
    math_benches,
    membership_benches
);
