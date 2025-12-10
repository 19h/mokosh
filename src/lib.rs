//! # Mokosh - Hierarchical Temporal Memory in Rust
//!
//! Mokosh is a high-performance, idiomatic Rust implementation of Hierarchical Temporal Memory (HTM)
//! algorithms, ported from the htm.core C++ library.
//!
//! ## Overview
//!
//! HTM is a machine learning technology that aims to capture the structural and algorithmic
//! properties of the neocortex. The main components include:
//!
//! - **Sparse Distributed Representations (SDR)**: The fundamental data structure
//! - **Encoders**: Convert various data types into SDRs
//! - **Spatial Pooler**: Creates sparse representations of input patterns
//! - **Temporal Memory**: Learns sequences and makes predictions
//! - **Anomaly Detection**: Identifies unusual patterns in data streams
//!
//! ## Quick Start
//!
//! ```rust
//! use mokosh::prelude::*;
//!
//! // Create an SDR with dimensions 10x10
//! let mut sdr = Sdr::new(&[10, 10]);
//!
//! // Set some active bits
//! sdr.set_sparse(&[1, 4, 8, 15, 42]);
//!
//! // Create a Spatial Pooler
//! let sp = SpatialPooler::new(SpatialPoolerParams {
//!     input_dimensions: vec![100],
//!     column_dimensions: vec![2048],
//!     ..Default::default()
//! });
//!
//! // Create a Temporal Memory
//! let tm = TemporalMemory::new(TemporalMemoryParams {
//!     column_dimensions: vec![2048],
//!     cells_per_column: 32,
//!     ..Default::default()
//! });
//! ```
//!
//! ## Feature Flags
//!
//! - `std` (default): Enable standard library features
//! - `serde`: Enable serialization/deserialization support
//! - `rayon`: Enable parallel processing
//! - `simd`: Enable SIMD optimizations

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::similar_names)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod types;
pub mod algorithms;
pub mod encoders;
pub mod utils;

#[cfg(feature = "serde")]
pub mod serialization;

/// Re-export of commonly used types and traits for convenience.
pub mod prelude {
    pub use crate::types::{
        Sdr, SdrDense, SdrSparse, SdrCoordinate,
        CellIdx, SegmentIdx, SynapseIdx, Segment, Synapse, Permanence,
        Real, Real32, Real64, UInt, UInt16, UInt32, UInt64, Int, Int32, Int64,
    };
    pub use crate::algorithms::{
        SpatialPooler, SpatialPoolerParams,
        TemporalMemory, TemporalMemoryParams, AnomalyMode,
        Connections, ConnectionsParams,
        Anomaly, AnomalyLikelihood,
        SdrClassifier, SdrClassifierParams,
    };
    pub use crate::encoders::{
        Encoder,
        ScalarEncoder, ScalarEncoderParams,
        RandomDistributedScalarEncoder, RdseParams,
        DateEncoder, DateEncoderParams,
    };
    pub use crate::utils::{
        Random,
        Topology, WrappingMode,
    };

    #[cfg(feature = "serde")]
    pub use crate::serialization::{Serializable, SerializableFormat};
}

/// Library version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Error types for the library.
pub mod error {
    use thiserror::Error;

    /// Main error type for mokosh operations.
    #[derive(Error, Debug)]
    pub enum MokoshError {
        /// Invalid dimensions provided.
        #[error("Invalid dimensions: {0}")]
        InvalidDimensions(String),

        /// Invalid parameter value.
        #[error("Invalid parameter '{name}': {message}")]
        InvalidParameter {
            /// Name of the invalid parameter.
            name: &'static str,
            /// Description of the error.
            message: String,
        },

        /// Index out of bounds.
        #[error("Index {index} out of bounds (size: {size})")]
        IndexOutOfBounds {
            /// The invalid index.
            index: usize,
            /// The valid size.
            size: usize,
        },

        /// Dimension mismatch between SDRs or other structures.
        #[error("Dimension mismatch: expected {expected:?}, got {actual:?}")]
        DimensionMismatch {
            /// Expected dimensions.
            expected: Vec<u32>,
            /// Actual dimensions.
            actual: Vec<u32>,
        },

        /// SDR data is invalid (e.g., unsorted sparse indices).
        #[error("Invalid SDR data: {0}")]
        InvalidSdrData(String),

        /// Serialization error.
        #[cfg(feature = "serde")]
        #[error("Serialization error: {message}")]
        SerializationError {
            /// Description of the serialization error.
            message: String,
        },

        /// I/O error.
        #[error("I/O error: {message}")]
        IoError {
            /// Description of the I/O error.
            message: String,
        },

        /// Internal error that should not occur.
        #[error("Internal error: {0}")]
        InternalError(String),
    }

    /// Result type alias using MokoshError.
    pub type Result<T> = std::result::Result<T, MokoshError>;
}

pub use error::{MokoshError, Result};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
