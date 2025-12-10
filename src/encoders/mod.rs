//! Encoders for converting data into SDR representations.
//!
//! Encoders transform various data types (scalars, dates, categories, text)
//! into Sparse Distributed Representations suitable for HTM processing.
//!
//! # Available Encoders
//!
//! - [`ScalarEncoder`]: Encodes numeric values as contiguous blocks of active bits
//! - [`RandomDistributedScalarEncoder`]: Encodes numeric values using hash-based random bit placement
//! - [`DateEncoder`]: Encodes date/time values with multiple configurable attributes
//! - [`SimHashDocumentEncoder`]: Encodes documents/text with similarity preservation
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
pub mod date;
mod rdse;
mod scalar;
mod simhash;

pub use base::Encoder;
pub use date::{DateEncoder, DateEncoderParams, DateTime, DayOfWeek, Holiday};
pub use rdse::{RandomDistributedScalarEncoder, Rdse, RdseParams};
pub use scalar::{ScalarEncoder, ScalarEncoderParams};
pub use simhash::{SimHashDocumentEncoder, SimHashDocumentEncoderParams};
