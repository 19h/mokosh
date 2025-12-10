//! Encoders for converting data into SDR representations.
//!
//! Encoders transform various data types (scalars, dates, categories, text, coordinates)
//! into Sparse Distributed Representations suitable for HTM processing.
//!
//! # Available Encoders
//!
//! ## Scalar Encoders
//! - [`ScalarEncoder`]: Encodes numeric values as contiguous blocks of active bits
//! - [`RandomDistributedScalarEncoder`]: Encodes numeric values using hash-based random bit placement
//! - [`LogEncoder`]: Encodes positive values on a logarithmic scale
//! - [`DeltaEncoder`]: Encodes rate of change between consecutive values
//!
//! ## Categorical Encoders
//! - [`CategoryEncoder`]: Encodes discrete categories into non-overlapping patterns
//! - [`BooleanEncoder`]: Encodes true/false values
//!
//! ## Temporal Encoders
//! - [`DateEncoder`]: Encodes date/time values with multiple configurable attributes
//!
//! ## Spatial Encoders
//! - [`CoordinateEncoder`]: Encodes N-dimensional coordinates with locality sensitivity
//! - [`GeospatialEncoder`]: Encodes GPS coordinates (latitude/longitude)
//! - [`GridCellEncoder`]: Biologically-inspired 2D position encoder
//!
//! ## Text Encoders
//! - [`SimHashDocumentEncoder`]: Encodes documents/text with similarity preservation
//!
//! ## Composite Encoders
//! - [`MultiEncoder`]: Combines multiple encoders for the same input type
//! - [`VecMultiEncoder`]: Combines multiple encoders for different input values
//! - [`PassThroughEncoder`]: Passes pre-encoded sparse indices through
//!
//! # Example
//!
//! ```rust
//! use mokosh::encoders::{ScalarEncoder, ScalarEncoderParams, Encoder};
//!
//! let encoder = ScalarEncoder::new(ScalarEncoderParams {
//!     minimum: 0.0,
//!     maximum: 100.0,
//!     size: 100,
//!     active_bits: 10,
//!     ..Default::default()
//! }).unwrap();
//!
//! let sdr = encoder.encode_to_sdr(50.0).unwrap();
//! assert_eq!(sdr.get_sum(), 10);
//! ```

mod base;
mod boolean;
mod category;
mod coordinate;
pub mod date;
mod delta;
mod geospatial;
mod grid;
mod log;
mod multi;
mod passthrough;
mod rdse;
mod scalar;
mod simhash;

pub use base::Encoder;
pub use boolean::{BooleanEncoder, BooleanEncoderParams};
pub use category::{CategoryEncoder, CategoryEncoderParams};
pub use coordinate::{Coord2D, Coord3D, CoordinateEncoder, CoordinateEncoderParams};
pub use date::{DateEncoder, DateEncoderParams, DateTime, DayOfWeek, Holiday};
pub use delta::{DeltaEncoder, DeltaEncoderParams};
pub use geospatial::{GeospatialEncoder, GeospatialEncoderParams, GpsCoordinate};
pub use grid::{GridCellEncoder, GridCellEncoderParams};
pub use log::{LogEncoder, LogEncoderParams};
pub use multi::{EncoderField, MultiEncoder, MultiEncoderBuilder, VecMultiEncoder, VecMultiEncoderBuilder};
pub use passthrough::{PassThroughEncoder, PassThroughEncoderParams};
pub use rdse::{RandomDistributedScalarEncoder, Rdse, RdseParams};
pub use scalar::{ScalarEncoder, ScalarEncoderParams};
pub use simhash::{SimHashDocumentEncoder, SimHashDocumentEncoderParams};
