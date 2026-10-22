//! Utility modules for the HTM library.
//!
//! This module contains utility functions and data structures used throughout
//! the library, including random number generation, topology utilities, and metrics.

mod random;
mod sdr_metrics;
pub mod simd;
mod topology;

pub use random::Random;
pub use sdr_metrics::SdrMetrics;
pub use topology::{Neighborhood, Topology, WrappingMode};

mod metrics;
pub use metrics::{ActivationFrequency, AttachedMetrics, Metrics, Overlap, Sparsity};
