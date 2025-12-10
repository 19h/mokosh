//! Core types for the HTM library.
//!
//! This module contains fundamental type definitions and the SDR (Sparse Distributed Representation)
//! data structure that forms the foundation of all HTM algorithms.

mod primitives;
mod sdr;

pub use primitives::*;
pub use sdr::*;
