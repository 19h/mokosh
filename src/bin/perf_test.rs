//! Performance Testing and Baseline Comparison Tool
//!
//! This binary provides a comprehensive performance testing framework that:
//! - Runs microbenchmarks for SIMD optimization candidates
//! - Saves baseline results to JSON files
//! - Compares current performance against saved baselines
//! - Generates detailed reports showing speedups/regressions
//!
//! Usage:
//!   cargo run --release --bin perf_test -- [OPTIONS]
//!
//! Options:
//!   --save-baseline <name>   Save results as a named baseline
//!   --compare <name>         Compare against a saved baseline
//!   --list-baselines         List all saved baselines
//!   --quick                  Run quick benchmarks (fewer iterations)
//!   --verbose                Show detailed timing information

use mokosh::prelude::*;
use mokosh::utils::Random;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Number of warmup iterations before measurement.
const WARMUP_ITERATIONS: usize = 10;

/// Number of measurement iterations.
const MEASURE_ITERATIONS: usize = 100;

/// Quick mode measurement iterations.
const QUICK_ITERATIONS: usize = 20;

/// Baseline storage directory.
const BASELINE_DIR: &str = "target/perf_baselines";

/// Result of a single benchmark.
#[derive(Debug, Clone)]
struct BenchResult {
    name: String,
    mean_ns: f64,
    std_dev_ns: f64,
    min_ns: f64,
    max_ns: f64,
    iterations: usize,
    throughput: Option<f64>, // ops/sec or elements/sec
}

impl BenchResult {
    fn new(name: &str, times_ns: &[f64], throughput: Option<f64>) -> Self {
        let n = times_ns.len() as f64;
        let mean = times_ns.iter().sum::<f64>() / n;
        let variance = times_ns.iter().map(|&t| (t - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();
        let min = times_ns.iter().copied().fold(f64::INFINITY, f64::min);
        let max = times_ns.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        Self {
            name: name.to_string(),
            mean_ns: mean,
            std_dev_ns: std_dev,
            min_ns: min,
            max_ns: max,
            iterations: times_ns.len(),
            throughput,
        }
    }

    fn to_json(&self) -> String {
        format!(
            r#"{{"name":"{}","mean_ns":{},"std_dev_ns":{},"min_ns":{},"max_ns":{},"iterations":{},"throughput":{}}}"#,
            self.name,
            self.mean_ns,
            self.std_dev_ns,
            self.min_ns,
            self.max_ns,
            self.iterations,
            self.throughput.map_or("null".to_string(), |t| t.to_string())
        )
    }

    fn from_json(json: &str) -> Option<Self> {
        // Simple JSON parsing (production code would use serde_json)
        let name = extract_string(json, "name")?;
        let mean_ns = extract_f64(json, "mean_ns")?;
        let std_dev_ns = extract_f64(json, "std_dev_ns")?;
        let min_ns = extract_f64(json, "min_ns")?;
        let max_ns = extract_f64(json, "max_ns")?;
        let iterations = extract_usize(json, "iterations")?;
        let throughput = extract_f64(json, "throughput");

        Some(Self {
            name,
            mean_ns,
            std_dev_ns,
            min_ns,
            max_ns,
            iterations,
            throughput,
        })
    }
}

fn extract_string(json: &str, key: &str) -> Option<String> {
    let pattern = format!(r#""{}":""#, key);
    let start = json.find(&pattern)? + pattern.len();
    let end = json[start..].find('"')? + start;
    Some(json[start..end].to_string())
}

fn extract_f64(json: &str, key: &str) -> Option<f64> {
    let pattern = format!(r#""{}":"#, key);
    let start = json.find(&pattern)? + pattern.len();
    let end_comma = json[start..].find(',').unwrap_or(json.len() - start);
    let end_brace = json[start..].find('}').unwrap_or(json.len() - start);
    let end = start + end_comma.min(end_brace);
    let value_str = json[start..end].trim();
    if value_str == "null" {
        return None;
    }
    value_str.parse().ok()
}

fn extract_usize(json: &str, key: &str) -> Option<usize> {
    extract_f64(json, key).map(|f| f as usize)
}

/// Benchmark runner that handles warmup and measurement.
struct BenchRunner {
    iterations: usize,
    verbose: bool,
}

impl BenchRunner {
    fn new(quick: bool, verbose: bool) -> Self {
        Self {
            iterations: if quick { QUICK_ITERATIONS } else { MEASURE_ITERATIONS },
            verbose,
        }
    }

    fn run<F>(&self, name: &str, throughput_base: Option<u64>, mut f: F) -> BenchResult
    where
        F: FnMut(),
    {
        if self.verbose {
            print!("  Running {}... ", name);
        }

        // Warmup
        for _ in 0..WARMUP_ITERATIONS {
            f();
        }

        // Measure
        let mut times_ns = Vec::with_capacity(self.iterations);
        for _ in 0..self.iterations {
            let start = Instant::now();
            f();
            let elapsed = start.elapsed();
            times_ns.push(elapsed.as_nanos() as f64);
        }

        let throughput = throughput_base.map(|base| {
            let mean_s = times_ns.iter().sum::<f64>() / times_ns.len() as f64 / 1e9;
            base as f64 / mean_s
        });

        let result = BenchResult::new(name, &times_ns, throughput);

        if self.verbose {
            println!(
                "{:.2} ns/op (±{:.2})",
                result.mean_ns, result.std_dev_ns
            );
        }

        result
    }
}

/// Run all benchmarks and return results.
fn run_all_benchmarks(runner: &BenchRunner) -> Vec<BenchResult> {
    let mut results = Vec::new();

    println!("\n=== SDR Operations ===\n");
    results.extend(bench_sdr_operations(runner));

    println!("\n=== Spatial Pooler ===\n");
    results.extend(bench_spatial_pooler(runner));

    println!("\n=== Temporal Memory ===\n");
    results.extend(bench_temporal_memory(runner));

    println!("\n=== Connections ===\n");
    results.extend(bench_connections(runner));

    println!("\n=== Math Operations ===\n");
    results.extend(bench_math_operations(runner));

    results
}

fn bench_sdr_operations(runner: &BenchRunner) -> Vec<BenchResult> {
    let mut results = Vec::new();
    let mut rng = Random::new(42);

    // SDR overlap - small
    {
        let mut sdr1 = Sdr::new(&[2048]);
        let mut sdr2 = Sdr::new(&[2048]);
        sdr1.randomize(0.02, &mut rng);
        sdr2.randomize(0.02, &mut rng);

        results.push(runner.run("sdr_overlap_2048_2pct", Some(41 * 2), || {
            std::hint::black_box(sdr1.get_overlap(&sdr2));
        }));
    }

    // SDR overlap - medium
    {
        let mut sdr1 = Sdr::new(&[2048]);
        let mut sdr2 = Sdr::new(&[2048]);
        sdr1.randomize(0.05, &mut rng);
        sdr2.randomize(0.05, &mut rng);

        results.push(runner.run("sdr_overlap_2048_5pct", Some(102 * 2), || {
            std::hint::black_box(sdr1.get_overlap(&sdr2));
        }));
    }

    // SDR overlap - large
    {
        let mut sdr1 = Sdr::new(&[65536]);
        let mut sdr2 = Sdr::new(&[65536]);
        sdr1.randomize(0.02, &mut rng);
        sdr2.randomize(0.02, &mut rng);

        results.push(runner.run("sdr_overlap_65536_2pct", Some(1311 * 2), || {
            std::hint::black_box(sdr1.get_overlap(&sdr2));
        }));
    }

    // SDR intersection
    {
        let mut sdr1 = Sdr::new(&[2048]);
        let mut sdr2 = Sdr::new(&[2048]);
        let mut result = Sdr::new(&[2048]);
        sdr1.randomize(0.05, &mut rng);
        sdr2.randomize(0.05, &mut rng);

        results.push(runner.run("sdr_intersection_2048_5pct", Some(102 * 2), || {
            result.intersection(&sdr1, &sdr2).unwrap();
            std::hint::black_box(&result);
        }));
    }

    // SDR union
    {
        let mut sdr1 = Sdr::new(&[2048]);
        let mut sdr2 = Sdr::new(&[2048]);
        let mut result = Sdr::new(&[2048]);
        sdr1.randomize(0.05, &mut rng);
        sdr2.randomize(0.05, &mut rng);

        results.push(runner.run("sdr_union_2048_5pct", Some(102 * 2), || {
            result.set_union(&sdr1, &sdr2).unwrap();
            std::hint::black_box(&result);
        }));
    }

    results
}

fn bench_spatial_pooler(runner: &BenchRunner) -> Vec<BenchResult> {
    let mut results = Vec::new();

    // SP compute - small
    {
        let mut sp = SpatialPooler::new(SpatialPoolerParams {
            input_dimensions: vec![100],
            column_dimensions: vec![2048],
            potential_radius: 50,
            global_inhibition: true,
            local_area_density: 0.02,
            ..Default::default()
        })
        .unwrap();

        let mut input = Sdr::new(&[100]);
        let mut output = Sdr::new(&[2048]);
        let mut rng = Random::new(42);
        input.randomize(0.05, &mut rng);

        results.push(runner.run("sp_compute_100_2048", Some(2048), || {
            sp.compute(&input, false, &mut output);
            std::hint::black_box(&output);
        }));
    }

    // SP compute - medium
    {
        let mut sp = SpatialPooler::new(SpatialPoolerParams {
            input_dimensions: vec![1024],
            column_dimensions: vec![2048],
            potential_radius: 512,
            global_inhibition: true,
            local_area_density: 0.02,
            ..Default::default()
        })
        .unwrap();

        let mut input = Sdr::new(&[1024]);
        let mut output = Sdr::new(&[2048]);
        let mut rng = Random::new(42);
        input.randomize(0.05, &mut rng);

        results.push(runner.run("sp_compute_1024_2048", Some(2048), || {
            sp.compute(&input, false, &mut output);
            std::hint::black_box(&output);
        }));
    }

    // SP compute with learning
    {
        let mut sp = SpatialPooler::new(SpatialPoolerParams {
            input_dimensions: vec![100],
            column_dimensions: vec![2048],
            potential_radius: 50,
            global_inhibition: true,
            local_area_density: 0.02,
            ..Default::default()
        })
        .unwrap();

        let mut input = Sdr::new(&[100]);
        let mut output = Sdr::new(&[2048]);
        let mut rng = Random::new(42);
        input.randomize(0.05, &mut rng);

        results.push(runner.run("sp_compute_learn_100_2048", Some(2048), || {
            sp.compute(&input, true, &mut output);
            std::hint::black_box(&output);
        }));
    }

    results
}

fn bench_temporal_memory(runner: &BenchRunner) -> Vec<BenchResult> {
    let mut results = Vec::new();

    // TM compute - small
    {
        let mut tm = TemporalMemory::new(TemporalMemoryParams {
            column_dimensions: vec![2048],
            cells_per_column: 32,
            activation_threshold: 13,
            min_threshold: 10,
            ..Default::default()
        })
        .unwrap();

        let mut active_columns = Sdr::new(&[2048]);
        let mut rng = Random::new(42);
        active_columns.randomize(0.02, &mut rng);

        // Prime with some learning
        for _ in 0..5 {
            tm.compute(&active_columns, true);
            active_columns.randomize(0.02, &mut rng);
        }

        results.push(runner.run("tm_compute_2048x32", Some(2048 * 32), || {
            tm.compute(&active_columns, false);
            std::hint::black_box(tm.active_cells());
        }));
    }

    // TM compute with learning
    {
        let mut tm = TemporalMemory::new(TemporalMemoryParams {
            column_dimensions: vec![2048],
            cells_per_column: 32,
            activation_threshold: 13,
            min_threshold: 10,
            ..Default::default()
        })
        .unwrap();

        let mut active_columns = Sdr::new(&[2048]);
        let mut rng = Random::new(42);
        active_columns.randomize(0.02, &mut rng);

        // Prime with some learning
        for _ in 0..5 {
            tm.compute(&active_columns, true);
            active_columns.randomize(0.02, &mut rng);
        }

        results.push(runner.run("tm_compute_learn_2048x32", Some(2048 * 32), || {
            tm.compute(&active_columns, true);
            std::hint::black_box(tm.active_cells());
        }));
    }

    results
}

fn bench_connections(runner: &BenchRunner) -> Vec<BenchResult> {
    let mut results = Vec::new();

    // Connections activity computation
    {
        let mut conn = Connections::new(ConnectionsParams {
            num_cells: 65536,
            connected_threshold: 0.5,
            timeseries: false,
        });

        let mut rng = Random::new(42);

        // Create segments and synapses
        for i in 0..6554 {
            let cell = (i % 65536) as u32;
            let segment = conn.create_segment(cell, None);

            for j in 0..30 {
                let presynaptic = ((i * 31 + j * 17) % 65536) as u32;
                let permanence = 0.3 + (j as f32 * 0.02).min(0.4);
                conn.create_synapse(segment, presynaptic, permanence);
            }
        }

        let active_cells: Vec<u32> = rng.sample((0..65536u32).collect(), 1311);

        results.push(runner.run("conn_activity_65536", Some(65536), || {
            let activity = conn.compute_activity(&active_cells, false);
            std::hint::black_box(activity);
        }));
    }

    // Connections adapt_segment
    {
        let mut conn = Connections::new(ConnectionsParams {
            num_cells: 10000,
            connected_threshold: 0.5,
            timeseries: false,
        });

        let segment = conn.create_segment(0, None);
        for i in 0..128 {
            conn.create_synapse(segment, (i + 1) as u32, 0.5);
        }

        let mut input = Sdr::new(&[10000]);
        let active: Vec<u32> = (1..65).collect();
        input.set_sparse(&active).unwrap();

        results.push(runner.run("conn_adapt_128syn", Some(128), || {
            conn.adapt_segment(segment, &input, 0.1, 0.1, false, 0);
            std::hint::black_box(&conn);
        }));
    }

    results
}

fn bench_math_operations(runner: &BenchRunner) -> Vec<BenchResult> {
    let mut results = Vec::new();

    // Boost factor computation (exp)
    {
        let duty_cycles: Vec<f32> = (0..2048).map(|i| 0.02 + (i as f32 * 0.0001)).collect();
        let target_density = 0.02f32;
        let boost_strength = 3.0f32;
        let mut boost_factors = vec![1.0f32; 2048];

        results.push(runner.run("boost_exp_2048", Some(2048), || {
            for (i, &duty) in duty_cycles.iter().enumerate() {
                boost_factors[i] = (boost_strength * (target_density - duty)).exp();
            }
            std::hint::black_box(&boost_factors);
        }));
    }

    // Duty cycle update (division)
    {
        let mut duty_cycles: Vec<f32> = (0..2048).map(|i| 0.01 + (i as f32 * 0.00001)).collect();
        let overlaps: Vec<u16> = (0..2048).map(|i| ((i * 7) % 50) as u16).collect();
        let period = 1000.0f32;

        results.push(runner.run("duty_cycle_2048", Some(2048), || {
            for (i, &overlap) in overlaps.iter().enumerate() {
                let value = if overlap > 0 { 1.0 } else { 0.0 };
                duty_cycles[i] = ((period - 1.0) * duty_cycles[i] + value) / period;
            }
            std::hint::black_box(&duty_cycles);
        }));
    }

    // Permanence clamping
    {
        let permanences: Vec<f32> = (0..4096)
            .map(|i| -0.1 + (i as f32 * 0.001))
            .collect();
        let deltas: Vec<f32> = (0..4096)
            .map(|i| if i % 2 == 0 { 0.1 } else { -0.1 })
            .collect();

        results.push(runner.run("perm_clamp_4096", Some(4096), || {
            let result: Vec<f32> = permanences
                .iter()
                .zip(deltas.iter())
                .map(|(&p, &d)| (p + d).clamp(0.0, 1.0))
                .collect();
            std::hint::black_box(result);
        }));
    }

    // Sorting for inhibition
    {
        let mut rng = Random::new(42);
        let overlaps: Vec<f32> = (0..2048)
            .map(|_| rng.get_real32() * 100.0)
            .collect();

        results.push(runner.run("inhibition_sort_2048", Some(2048), || {
            let mut indexed: Vec<(usize, f32)> = overlaps
                .iter()
                .enumerate()
                .map(|(i, &o)| (i, o))
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            std::hint::black_box(indexed);
        }));
    }

    results
}

/// Save results to a baseline file.
fn save_baseline(name: &str, results: &[BenchResult]) {
    fs::create_dir_all(BASELINE_DIR).expect("Failed to create baseline directory");

    let path = PathBuf::from(BASELINE_DIR).join(format!("{}.json", name));

    let json_entries: Vec<String> = results.iter().map(|r| r.to_json()).collect();
    let json = format!("[{}]", json_entries.join(","));

    fs::write(&path, &json).expect("Failed to write baseline file");
    println!("\nBaseline saved to: {}", path.display());
}

/// Load results from a baseline file.
fn load_baseline(name: &str) -> Option<Vec<BenchResult>> {
    let path = PathBuf::from(BASELINE_DIR).join(format!("{}.json", name));
    let json = fs::read_to_string(&path).ok()?;

    // Simple JSON array parsing
    let json = json.trim();
    if !json.starts_with('[') || !json.ends_with(']') {
        return None;
    }

    let inner = &json[1..json.len() - 1];
    let mut results = Vec::new();
    let mut depth = 0;
    let mut start = 0;

    for (i, c) in inner.char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    if let Some(r) = BenchResult::from_json(&inner[start..=i]) {
                        results.push(r);
                    }
                    start = i + 2; // Skip comma and space
                }
            }
            _ => {}
        }
    }

    Some(results)
}

/// List available baselines.
fn list_baselines() {
    let path = PathBuf::from(BASELINE_DIR);
    if !path.exists() {
        println!("No baselines found.");
        return;
    }

    println!("Available baselines:");
    if let Ok(entries) = fs::read_dir(&path) {
        for entry in entries.flatten() {
            if let Some(name) = entry.path().file_stem() {
                println!("  - {}", name.to_string_lossy());
            }
        }
    }
}

/// Compare results against a baseline.
fn compare_results(current: &[BenchResult], baseline: &[BenchResult]) {
    let baseline_map: HashMap<&str, &BenchResult> = baseline
        .iter()
        .map(|r| (r.name.as_str(), r))
        .collect();

    println!("\n{:=^80}", " Performance Comparison ");
    println!(
        "{:<35} {:>12} {:>12} {:>12} {:>8}",
        "Benchmark", "Baseline", "Current", "Change", "Status"
    );
    println!("{:-<80}", "");

    let mut total_speedup = 0.0f64;
    let mut count = 0;

    for result in current {
        if let Some(baseline) = baseline_map.get(result.name.as_str()) {
            let change = (result.mean_ns - baseline.mean_ns) / baseline.mean_ns * 100.0;
            let speedup = baseline.mean_ns / result.mean_ns;

            let status = if change < -5.0 {
                "FASTER"
            } else if change > 5.0 {
                "SLOWER"
            } else {
                "SAME"
            };

            let status_color = match status {
                "FASTER" => "\x1b[32m", // Green
                "SLOWER" => "\x1b[31m", // Red
                _ => "\x1b[33m",        // Yellow
            };

            println!(
                "{:<35} {:>10.0} ns {:>10.0} ns {:>+10.1}% {}{:>8}\x1b[0m",
                result.name,
                baseline.mean_ns,
                result.mean_ns,
                change,
                status_color,
                status
            );

            total_speedup += speedup;
            count += 1;
        } else {
            println!(
                "{:<35} {:>12} {:>10.0} ns {:>12} {:>8}",
                result.name, "N/A", result.mean_ns, "N/A", "NEW"
            );
        }
    }

    if count > 0 {
        println!("{:-<80}", "");
        println!(
            "Average speedup: {:.2}x (geometric mean would be better for production)",
            total_speedup / count as f64
        );
    }
}

/// Print a summary table of results.
fn print_summary(results: &[BenchResult]) {
    println!("\n{:=^80}", " Performance Summary ");
    println!(
        "{:<35} {:>12} {:>12} {:>15}",
        "Benchmark", "Mean", "Std Dev", "Throughput"
    );
    println!("{:-<80}", "");

    for result in results {
        let throughput_str = result
            .throughput
            .map(|t| format!("{:.0} ops/s", t))
            .unwrap_or_else(|| "N/A".to_string());

        println!(
            "{:<35} {:>10.0} ns {:>10.0} ns {:>15}",
            result.name, result.mean_ns, result.std_dev_ns, throughput_str
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut save_baseline_name: Option<String> = None;
    let mut compare_baseline_name: Option<String> = None;
    let mut list_mode = false;
    let mut quick = false;
    let mut verbose = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--save-baseline" => {
                i += 1;
                if i < args.len() {
                    save_baseline_name = Some(args[i].clone());
                }
            }
            "--compare" => {
                i += 1;
                if i < args.len() {
                    compare_baseline_name = Some(args[i].clone());
                }
            }
            "--list-baselines" => {
                list_mode = true;
            }
            "--quick" => {
                quick = true;
            }
            "--verbose" => {
                verbose = true;
            }
            "--help" | "-h" => {
                println!("Usage: perf_test [OPTIONS]");
                println!();
                println!("Options:");
                println!("  --save-baseline <name>   Save results as a named baseline");
                println!("  --compare <name>         Compare against a saved baseline");
                println!("  --list-baselines         List all saved baselines");
                println!("  --quick                  Run quick benchmarks (fewer iterations)");
                println!("  --verbose                Show detailed timing information");
                println!("  --help, -h               Show this help message");
                return;
            }
            _ => {
                eprintln!("Unknown option: {}", args[i]);
            }
        }
        i += 1;
    }

    if list_mode {
        list_baselines();
        return;
    }

    println!("Mokosh Performance Testing Framework");
    println!("=====================================");
    println!();
    println!(
        "Mode: {} ({} iterations)",
        if quick { "Quick" } else { "Full" },
        if quick { QUICK_ITERATIONS } else { MEASURE_ITERATIONS }
    );

    let runner = BenchRunner::new(quick, verbose);
    let results = run_all_benchmarks(&runner);

    print_summary(&results);

    if let Some(name) = save_baseline_name {
        save_baseline(&name, &results);
    }

    if let Some(name) = compare_baseline_name {
        if let Some(baseline) = load_baseline(&name) {
            compare_results(&results, &baseline);
        } else {
            eprintln!("Baseline '{}' not found.", name);
        }
    }
}
