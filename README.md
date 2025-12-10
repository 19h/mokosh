# Mokosh

A high-performance Rust implementation of Hierarchical Temporal Memory (HTM) algorithms, ported from the [htm.core](https://github.com/htm-community/htm.core) C++ library.

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org)

## Overview

Mokosh provides a complete implementation of the core HTM algorithms for sequence learning, anomaly detection, and pattern recognition. HTM is a machine learning technology that aims to capture the structural and algorithmic properties of the neocortex.

### Key Features

- **Sparse Distributed Representations (SDR)** - The fundamental data structure for HTM
- **Spatial Pooler** - Creates sparse, distributed representations of input patterns
- **Temporal Memory** - Learns sequences and makes predictions
- **Anomaly Detection** - Identifies unusual patterns in data streams
- **Multiple Encoders** - Convert raw data into SDR format
- **Serialization** - Save and load models in binary or JSON format
- **Zero unsafe code** - Safe Rust implementation
- **Well tested** - 130+ unit tests with comprehensive coverage

## Installation

Add mokosh to your `Cargo.toml`:

```toml
[dependencies]
mokosh = "0.1"
```

### Feature Flags

- `std` (default) - Standard library support
- `serde` - Serialization/deserialization support (adds `serde`, `serde_json`, `bincode`)
- `rayon` - Parallel processing support
- `simd` - SIMD optimizations (experimental)

```toml
[dependencies]
mokosh = { version = "0.1", features = ["serde"] }
```

## Quick Start

```rust
use mokosh::prelude::*;

// Create a Spatial Pooler
let mut sp = SpatialPooler::new(SpatialPoolerParams {
    input_dimensions: vec![1000],
    column_dimensions: vec![2048],
    potential_radius: 16,
    num_active_columns_per_inh_area: 40,
    ..Default::default()
}).unwrap();

// Create a Temporal Memory
let mut tm = TemporalMemory::new(TemporalMemoryParams {
    column_dimensions: vec![2048],
    cells_per_column: 32,
    ..Default::default()
}).unwrap();

// Encode input data
let encoder = ScalarEncoder::new(ScalarEncoderParams {
    minimum: 0.0,
    maximum: 100.0,
    size: 1000,
    active_bits: 21,
    ..Default::default()
}).unwrap();

// Process a value
let input_sdr = encoder.encode_to_sdr(42.0).unwrap();

// Run through Spatial Pooler
let mut active_columns = Sdr::new(&[2048]);
sp.compute(&input_sdr, true, &mut active_columns);

// Run through Temporal Memory
tm.compute(&active_columns, true);

// Get predictions and anomaly score
let predictions = tm.get_predictive_cells();
let anomaly = tm.anomaly();
println!("Anomaly score: {}", anomaly);
```

## Core Components

### Sparse Distributed Representations (SDR)

SDRs are the fundamental data structure in HTM - binary vectors where only a small percentage of bits are active.

```rust
use mokosh::types::Sdr;

// Create a 100-bit SDR
let mut sdr = Sdr::new(&[100]);

// Set active bits
sdr.set_sparse(&[5, 12, 23, 45, 67]).unwrap();

// Access in different formats
let sparse = sdr.get_sparse();      // Active indices: [5, 12, 23, 45, 67]
let dense = sdr.get_dense();        // Full binary vector
let sum = sdr.get_sum();            // Number of active bits: 5
let sparsity = sdr.get_sparsity();  // Fraction active: 0.05

// Compute overlap between SDRs
let other = Sdr::new(&[100]);
let overlap = sdr.get_overlap(&other);
```

### Spatial Pooler

The Spatial Pooler creates sparse, distributed representations that maintain semantic similarity.

```rust
use mokosh::algorithms::{SpatialPooler, SpatialPoolerParams};
use mokosh::types::Sdr;

let mut sp = SpatialPooler::new(SpatialPoolerParams {
    input_dimensions: vec![1000],
    column_dimensions: vec![2048],
    potential_pct: 0.85,
    global_inhibition: true,
    local_area_density: 0.02,
    syn_perm_connected: 0.1,
    syn_perm_active_inc: 0.05,
    syn_perm_inactive_dec: 0.008,
    boost_strength: 3.0,
    ..Default::default()
}).unwrap();

let input = Sdr::new(&[1000]);
let mut output = Sdr::new(&[2048]);

// learn=true enables learning
sp.compute(&input, true, &mut output);
```

### Temporal Memory

The Temporal Memory learns sequences and makes predictions about future inputs.

```rust
use mokosh::algorithms::{TemporalMemory, TemporalMemoryParams, AnomalyMode};
use mokosh::types::Sdr;

let mut tm = TemporalMemory::new(TemporalMemoryParams {
    column_dimensions: vec![2048],
    cells_per_column: 32,
    activation_threshold: 13,
    initial_permanence: 0.21,
    connected_permanence: 0.5,
    min_threshold: 10,
    max_new_synapse_count: 20,
    permanence_increment: 0.1,
    permanence_decrement: 0.1,
    predicted_segment_decrement: 0.0,
    max_segments_per_cell: 255,
    max_synapses_per_segment: 255,
    anomaly_mode: AnomalyMode::Raw,
    ..Default::default()
}).unwrap();

let active_columns = Sdr::new(&[2048]);

// Process input (learn=true)
tm.compute(&active_columns, true);

// Get results
let active_cells = tm.get_active_cells();
let predictive_cells = tm.get_predictive_cells();
let winner_cells = tm.get_winner_cells();
let anomaly_score = tm.anomaly();

// Reset for new sequence
tm.reset();
```

### Encoders

Encoders convert raw data into SDR format.

#### Scalar Encoder

Encodes numerical values with semantic similarity preserved.

```rust
use mokosh::encoders::{ScalarEncoder, ScalarEncoderParams, Encoder};

let encoder = ScalarEncoder::new(ScalarEncoderParams {
    minimum: 0.0,
    maximum: 100.0,
    size: 400,
    active_bits: 21,
    periodic: false,    // true for circular values like angles
    clip_input: true,   // clip out-of-range values
    ..Default::default()
}).unwrap();

let sdr = encoder.encode_to_sdr(42.0).unwrap();

// Similar values have overlapping representations
let sdr_43 = encoder.encode_to_sdr(43.0).unwrap();
let overlap = sdr.get_overlap(&sdr_43);  // High overlap
```

#### Random Distributed Scalar Encoder (RDSE)

Hash-based encoder that doesn't require knowing the input range in advance.

```rust
use mokosh::encoders::{RandomDistributedScalarEncoder, RdseParams, Encoder};

let encoder = RandomDistributedScalarEncoder::new(RdseParams {
    size: 1000,
    active_bits: 40,
    resolution: 1.0,  // Values differing by < resolution have same encoding
    ..Default::default()
}).unwrap();

let sdr = encoder.encode_to_sdr(42.0).unwrap();
```

#### Date Encoder

Encodes date/time values with multiple components.

```rust
use mokosh::encoders::{DateEncoder, DateEncoderParams, DateTime, Encoder};

let encoder = DateEncoder::new(DateEncoderParams {
    season_width: 3,
    season_radius: 91.5,
    day_of_week_width: 1,
    day_of_week_radius: 1.0,
    weekend_width: 3,
    time_of_day_width: 5,
    time_of_day_radius: 4.0,
    ..Default::default()
}).unwrap();

let dt = DateTime::new(2024, 6, 15, 14, 30, 0);
let sdr = encoder.encode_to_sdr(dt).unwrap();
```

#### SimHash Document Encoder

Encodes text documents using locality-sensitive hashing.

```rust
use mokosh::encoders::{SimHashDocumentEncoder, SimHashDocumentEncoderParams, Encoder};

let encoder = SimHashDocumentEncoder::new(SimHashDocumentEncoderParams {
    size: 400,
    active_bits: 21,
    case_sensitivity: false,
    token_similarity: false,
    ..Default::default()
}).unwrap();

// Encode from string
let sdr = encoder.encode_to_sdr("The quick brown fox").unwrap();

// Similar documents have overlapping SDRs
let sdr2 = encoder.encode_to_sdr("The fast brown fox").unwrap();
let overlap = sdr.get_overlap(&sdr2);  // High overlap
```

### Anomaly Detection

Detect anomalies in data streams.

```rust
use mokosh::algorithms::{Anomaly, AnomalyLikelihood};
use mokosh::types::Sdr;

// Raw anomaly score
let anomaly = Anomaly::new();
let active = Sdr::new(&[100]);
let predicted = Sdr::new(&[100]);
let score = anomaly.compute(&active, &predicted);

// Anomaly likelihood (statistical)
let mut likelihood = AnomalyLikelihood::new(288, 100, 100);

// Feed anomaly scores and get likelihood
for raw_score in scores.iter() {
    let prob = likelihood.anomaly_probability(*raw_score);
    if prob > 0.9999 {
        println!("Highly anomalous!");
    }
}
```

### SDR Classifier

Classify SDR patterns and make predictions.

```rust
use mokosh::algorithms::{SdrClassifier, SdrClassifierParams};
use mokosh::types::Sdr;

let mut classifier = SdrClassifier::new(SdrClassifierParams {
    steps: vec![1],  // Predict 1 step ahead
    alpha: 0.1,
    ..Default::default()
});

let pattern = Sdr::new(&[2048]);
let bucket_idx = 5u32;

// Learn association
classifier.learn(&pattern, bucket_idx);

// Infer probabilities
let probabilities = classifier.infer(&pattern, 1);  // 1-step prediction
```

### Serialization

Save and load models (requires `serde` feature).

```rust
use mokosh::prelude::*;
use mokosh::serialization::{Serializable, SerializableFormat};

// Save to file
sp.save_to_file("spatial_pooler.bin", SerializableFormat::Binary)?;
sp.save_to_file("spatial_pooler.json", SerializableFormat::Json)?;

// Load from file
let sp2 = SpatialPooler::load_from_file("spatial_pooler.bin", SerializableFormat::Binary)?;

// Auto-detect format from extension
sp.save_to_file_auto("model.json")?;  // Uses JSON
let sp3 = SpatialPooler::load_from_file_auto("model.json")?;

// Serialize to bytes/string
let bytes = sp.to_bytes(SerializableFormat::Binary)?;
let json = sp.to_json()?;

// Deserialize
let sp4 = SpatialPooler::from_bytes(&bytes, SerializableFormat::Binary)?;
let sp5 = SpatialPooler::from_json(&json)?;
```

## Examples

### Anomaly Detection in Time Series

```rust
use mokosh::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Setup encoder for values 0-100
    let encoder = ScalarEncoder::new(ScalarEncoderParams {
        minimum: 0.0,
        maximum: 100.0,
        size: 2048,
        active_bits: 41,
        ..Default::default()
    })?;

    // Setup Spatial Pooler
    let mut sp = SpatialPooler::new(SpatialPoolerParams {
        input_dimensions: vec![2048],
        column_dimensions: vec![2048],
        ..Default::default()
    })?;

    // Setup Temporal Memory with anomaly detection
    let mut tm = TemporalMemory::new(TemporalMemoryParams {
        column_dimensions: vec![2048],
        cells_per_column: 32,
        anomaly_mode: AnomalyMode::Raw,
        ..Default::default()
    })?;

    // Process data stream
    let data = vec![10.0, 11.0, 12.0, 13.0, 14.0, 50.0, 15.0, 16.0];

    for (i, &value) in data.iter().enumerate() {
        let input = encoder.encode_to_sdr(value)?;

        let mut columns = Sdr::new(&[2048]);
        sp.compute(&input, true, &mut columns);
        tm.compute(&columns, true);

        let anomaly = tm.anomaly();
        println!("Step {}: value={:.1}, anomaly={:.3}", i, value, anomaly);
    }

    Ok(())
}
```

### Sequence Learning

```rust
use mokosh::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut tm = TemporalMemory::new(TemporalMemoryParams {
        column_dimensions: vec![100],
        cells_per_column: 4,
        ..Default::default()
    })?;

    // Define a simple sequence: A -> B -> C -> D
    let sequence = vec![
        vec![0, 1, 2, 3, 4],      // A
        vec![20, 21, 22, 23, 24], // B
        vec![40, 41, 42, 43, 44], // C
        vec![60, 61, 62, 63, 64], // D
    ];

    // Train on the sequence multiple times
    for _ in 0..10 {
        tm.reset();
        for pattern in &sequence {
            let mut sdr = Sdr::new(&[100]);
            sdr.set_sparse(pattern)?;
            tm.compute(&sdr, true);
        }
    }

    // Test prediction
    tm.reset();
    let mut sdr = Sdr::new(&[100]);
    sdr.set_sparse(&sequence[0])?;  // Present A
    tm.compute(&sdr, false);

    let predictions = tm.get_predictive_cells();
    println!("After A, predicting {} cells", predictions.get_sum());

    Ok(())
}
```

## Architecture

```
mokosh/
├── src/
│   ├── lib.rs              # Library root, prelude, error types
│   ├── types/
│   │   ├── mod.rs
│   │   ├── primitives.rs   # Real, UInt, Permanence, etc.
│   │   └── sdr.rs          # Sparse Distributed Representation
│   ├── algorithms/
│   │   ├── mod.rs
│   │   ├── connections.rs  # Synaptic connections management
│   │   ├── spatial_pooler.rs
│   │   ├── temporal_memory.rs
│   │   ├── anomaly.rs      # Anomaly & AnomalyLikelihood
│   │   └── sdr_classifier.rs
│   ├── encoders/
│   │   ├── mod.rs
│   │   ├── base.rs         # Encoder trait
│   │   ├── scalar.rs       # ScalarEncoder
│   │   ├── rdse.rs         # RandomDistributedScalarEncoder
│   │   ├── date.rs         # DateEncoder
│   │   └── simhash.rs      # SimHashDocumentEncoder
│   ├── utils/
│   │   ├── mod.rs
│   │   ├── random.rs       # Deterministic RNG
│   │   ├── topology.rs     # Spatial topology utilities
│   │   └── sdr_metrics.rs  # SDR analysis metrics
│   └── serialization.rs    # Serde-based serialization
```

## Comparison with htm.core

Mokosh is a faithful port of the core HTM algorithms from htm.core. Here's what's included:

| Component | Status | Notes |
|-----------|--------|-------|
| SDR | ✅ Complete | Full sparse/dense/coordinate support |
| Spatial Pooler | ✅ Complete | Global and local inhibition |
| Temporal Memory | ✅ Complete | All learning rules |
| Connections | ✅ Complete | Segment and synapse management |
| Anomaly Detection | ✅ Complete | Raw and likelihood modes |
| SDR Classifier | ✅ Complete | Multi-step prediction |
| Scalar Encoder | ✅ Complete | Periodic and category modes |
| RDSE | ✅ Complete | Hash-based encoding |
| Date Encoder | ✅ Complete | All time components |
| SimHash Encoder | ✅ Complete | Document encoding |
| Serialization | ✅ Complete | Binary and JSON formats |

### Not Included

The following htm.core features are **not ported** as they are advanced orchestration features rarely needed for typical HTM applications:

- **Network Engine** - Multi-region network orchestration with YAML/JSON configuration
- **Region Framework** - Plugin-based region implementations
- **REST API** - HTTP interface to HTM networks

These features may be added in future versions if there is demand.

## Performance

Mokosh is designed for performance:

- **Efficient SDR operations** - O(k) for sparse operations where k is active bits
- **Cache-friendly** - Data structures optimized for CPU cache utilization
- **Zero-copy where possible** - Minimizes memory allocations
- **Optional parallelism** - Enable `rayon` feature for parallel processing

## Testing

Run the test suite:

```bash
# Run all tests
cargo test

# Run with serde feature
cargo test --features serde

# Run specific test module
cargo test spatial_pooler

# Run with output
cargo test -- --nocapture
```

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0), the same license as htm.core.

## Acknowledgments

- [Numenta](https://numenta.com/) for creating the HTM theory and algorithms
- [htm.core](https://github.com/htm-community/htm.core) contributors for the reference C++ implementation
- The HTM community for ongoing research and development

## References

- [Hierarchical Temporal Memory (HTM) Whitepaper](https://numenta.com/neuroscience-research/research-publications/papers/hierarchical-temporal-memory-white-paper/)
- [HTM School (Video Series)](https://www.youtube.com/playlist?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9)
- [Biological and Machine Intelligence (BAMI)](https://numenta.com/resources/biological-and-machine-intelligence/)
