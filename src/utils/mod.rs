//! Utility modules for the HTM library.
//!
//! This module contains utility functions and data structures used throughout
//! the library, including random number generation, topology utilities, and metrics.

mod random;
mod topology;
mod sdr_metrics;
pub mod simd;

pub use random::Random;
pub use topology::{Topology, WrappingMode, Neighborhood};
pub use sdr_metrics::SdrMetrics;
