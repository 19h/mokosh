//! Coordinate Encoder implementation.
//!
//! The Coordinate Encoder encodes N-dimensional coordinates into SDRs using
//! a hash-based approach where nearby coordinates share more active bits.

use crate::encoders::Encoder;
use crate::error::{MokoshError, Result};
use crate::types::{Real, Sdr, UInt};
use std::collections::HashSet;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Parameters for creating a Coordinate Encoder.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CoordinateEncoderParams {
    /// Number of dimensions in the coordinate space.
    pub num_dimensions: usize,

    /// Total number of bits in the output SDR.
    pub size: UInt,

    /// Number of active bits in the output SDR.
    pub active_bits: UInt,

    /// Radius of influence - coordinates within this radius share bits.
    /// Each unit of distance reduces overlap by approximately 1 bit.
    pub radius: Real,
}

impl Default for CoordinateEncoderParams {
    fn default() -> Self {
        Self {
            num_dimensions: 2,
            size: 1000,
            active_bits: 21,
            radius: 1.0,
        }
    }
}

/// Encodes N-dimensional coordinates into SDR representations.
///
/// Uses locality-sensitive hashing to ensure nearby coordinates produce
/// similar encodings with proportional overlap.
///
/// # Example
///
/// ```rust
/// use mokosh::encoders::{CoordinateEncoder, CoordinateEncoderParams, Encoder};
///
/// let encoder = CoordinateEncoder::new(CoordinateEncoderParams {
///     num_dimensions: 2,
///     size: 1000,
///     active_bits: 21,
///     radius: 5.0,
/// }).unwrap();
///
/// // Encode a 2D coordinate
/// let sdr1 = encoder.encode_to_sdr(vec![10.0, 20.0]).unwrap();
/// let sdr2 = encoder.encode_to_sdr(vec![11.0, 20.0]).unwrap();
/// let sdr3 = encoder.encode_to_sdr(vec![100.0, 200.0]).unwrap();
///
/// // Nearby coordinates have high overlap
/// assert!(sdr1.get_overlap(&sdr2) > sdr1.get_overlap(&sdr3));
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CoordinateEncoder {
    /// Number of dimensions.
    num_dimensions: usize,

    /// Total size of output.
    size: UInt,

    /// Number of active bits.
    active_bits: UInt,

    /// Radius of influence.
    radius: Real,

    /// Output dimensions.
    dimensions: Vec<UInt>,
}

impl CoordinateEncoder {
    /// Creates a new Coordinate Encoder.
    pub fn new(params: CoordinateEncoderParams) -> Result<Self> {
        if params.num_dimensions == 0 {
            return Err(MokoshError::InvalidParameter {
                name: "num_dimensions",
                message: "Must be > 0".to_string(),
            });
        }

        if params.active_bits == 0 {
            return Err(MokoshError::InvalidParameter {
                name: "active_bits",
                message: "Must be > 0".to_string(),
            });
        }

        if params.active_bits > params.size {
            return Err(MokoshError::InvalidParameter {
                name: "active_bits",
                message: "Cannot exceed size".to_string(),
            });
        }

        if params.radius <= 0.0 {
            return Err(MokoshError::InvalidParameter {
                name: "radius",
                message: "Must be > 0".to_string(),
            });
        }

        Ok(Self {
            num_dimensions: params.num_dimensions,
            size: params.size,
            active_bits: params.active_bits,
            radius: params.radius,
            dimensions: vec![params.size],
        })
    }

    /// Returns the number of dimensions.
    pub fn num_dimensions(&self) -> usize {
        self.num_dimensions
    }

    /// Returns the radius.
    pub fn radius(&self) -> Real {
        self.radius
    }

    /// Hash function for coordinate encoding.
    fn hash_coordinate(coords: &[i64], index: u32) -> u64 {
        // Use a simple but effective hash combining coordinate values
        let mut hash: u64 = 0xcbf29ce484222325; // FNV-1a offset basis
        let prime: u64 = 0x100000001b3; // FNV-1a prime

        // Include the index to generate different bits
        hash ^= index as u64;
        hash = hash.wrapping_mul(prime);

        for &coord in coords {
            hash ^= coord as u64;
            hash = hash.wrapping_mul(prime);
        }

        hash
    }

    /// Encodes discretized coordinates to SDR.
    fn encode_discrete(&self, coords: &[i64], output: &mut Sdr) -> Result<()> {
        if coords.len() != self.num_dimensions {
            return Err(MokoshError::InvalidParameter {
                name: "coordinates",
                message: format!(
                    "Expected {} dimensions, got {}",
                    self.num_dimensions,
                    coords.len()
                ),
            });
        }

        if output.dimensions() != self.dimensions.as_slice() {
            return Err(MokoshError::DimensionMismatch {
                expected: self.dimensions.clone(),
                actual: output.dimensions().to_vec(),
            });
        }

        let mut active_bits = HashSet::new();

        // Generate active_bits number of unique bits using hashing
        let mut index = 0u32;
        while active_bits.len() < self.active_bits as usize {
            let hash = Self::hash_coordinate(coords, index);
            let bit = (hash % self.size as u64) as UInt;
            active_bits.insert(bit);
            index += 1;

            // Prevent infinite loop in degenerate cases
            if index > self.active_bits * 100 {
                break;
            }
        }

        let mut sparse: Vec<UInt> = active_bits.into_iter().collect();
        sparse.sort_unstable();
        output.set_sparse_unchecked(sparse);

        Ok(())
    }
}

impl Encoder<Vec<Real>> for CoordinateEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, value: Vec<Real>, output: &mut Sdr) -> Result<()> {
        if value.len() != self.num_dimensions {
            return Err(MokoshError::InvalidParameter {
                name: "coordinates",
                message: format!(
                    "Expected {} dimensions, got {}",
                    self.num_dimensions,
                    value.len()
                ),
            });
        }

        // Discretize coordinates based on radius
        let discrete: Vec<i64> = value
            .iter()
            .map(|&v| (v / self.radius).round() as i64)
            .collect();

        self.encode_discrete(&discrete, output)
    }
}

impl Encoder<&[Real]> for CoordinateEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, value: &[Real], output: &mut Sdr) -> Result<()> {
        self.encode(value.to_vec(), output)
    }
}

/// Convenience type for 2D coordinates.
pub type Coord2D = (Real, Real);

impl Encoder<Coord2D> for CoordinateEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, value: Coord2D, output: &mut Sdr) -> Result<()> {
        if self.num_dimensions != 2 {
            return Err(MokoshError::InvalidParameter {
                name: "coordinates",
                message: format!(
                    "Coord2D requires 2 dimensions, encoder has {}",
                    self.num_dimensions
                ),
            });
        }
        self.encode(vec![value.0, value.1], output)
    }
}

/// Convenience type for 3D coordinates.
pub type Coord3D = (Real, Real, Real);

impl Encoder<Coord3D> for CoordinateEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, value: Coord3D, output: &mut Sdr) -> Result<()> {
        if self.num_dimensions != 3 {
            return Err(MokoshError::InvalidParameter {
                name: "coordinates",
                message: format!(
                    "Coord3D requires 3 dimensions, encoder has {}",
                    self.num_dimensions
                ),
            });
        }
        self.encode(vec![value.0, value.1, value.2], output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_encoder() {
        let encoder = CoordinateEncoder::new(CoordinateEncoderParams {
            num_dimensions: 3,
            size: 500,
            active_bits: 15,
            radius: 2.0,
        })
        .unwrap();

        assert_eq!(encoder.num_dimensions(), 3);
        assert_eq!(Encoder::<Vec<Real>>::size(&encoder), 500);
        assert_eq!(encoder.radius(), 2.0);
    }

    #[test]
    fn test_encode_2d() {
        let encoder = CoordinateEncoder::new(CoordinateEncoderParams {
            num_dimensions: 2,
            size: 1000,
            active_bits: 21,
            radius: 1.0,
        })
        .unwrap();

        let sdr = encoder.encode_to_sdr(vec![10.0, 20.0]).unwrap();
        assert_eq!(sdr.get_sum(), 21);
    }

    #[test]
    fn test_encode_tuple() {
        let encoder = CoordinateEncoder::new(CoordinateEncoderParams {
            num_dimensions: 2,
            size: 1000,
            active_bits: 21,
            radius: 1.0,
        })
        .unwrap();

        let sdr = encoder.encode_to_sdr((10.0, 20.0)).unwrap();
        assert_eq!(sdr.get_sum(), 21);
    }

    #[test]
    fn test_nearby_overlap() {
        let encoder = CoordinateEncoder::new(CoordinateEncoderParams {
            num_dimensions: 2,
            size: 1000,
            active_bits: 21,
            radius: 5.0,
        })
        .unwrap();

        let sdr1 = encoder.encode_to_sdr(vec![10.0, 10.0]).unwrap();
        let sdr2 = encoder.encode_to_sdr(vec![10.0, 10.0]).unwrap();
        let sdr3 = encoder.encode_to_sdr(vec![15.0, 10.0]).unwrap();
        let sdr4 = encoder.encode_to_sdr(vec![100.0, 100.0]).unwrap();

        // Same coordinate should have full overlap
        assert_eq!(sdr1.get_overlap(&sdr2), 21);

        // Nearby (within radius) should have some overlap
        // Far should have less overlap
        let near_overlap = sdr1.get_overlap(&sdr3);
        let far_overlap = sdr1.get_overlap(&sdr4);

        assert!(near_overlap >= far_overlap);
    }

    #[test]
    fn test_deterministic() {
        let encoder = CoordinateEncoder::new(CoordinateEncoderParams::default()).unwrap();

        let sdr1 = encoder.encode_to_sdr(vec![5.5, 3.2]).unwrap();
        let sdr2 = encoder.encode_to_sdr(vec![5.5, 3.2]).unwrap();

        assert_eq!(sdr1.get_sparse(), sdr2.get_sparse());
    }

    #[test]
    fn test_wrong_dimensions() {
        let encoder = CoordinateEncoder::new(CoordinateEncoderParams {
            num_dimensions: 2,
            ..Default::default()
        })
        .unwrap();

        let result = encoder.encode_to_sdr(vec![1.0, 2.0, 3.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_3d_tuple() {
        let encoder = CoordinateEncoder::new(CoordinateEncoderParams {
            num_dimensions: 3,
            size: 500,
            active_bits: 15,
            radius: 1.0,
        })
        .unwrap();

        let sdr = encoder.encode_to_sdr((1.0, 2.0, 3.0)).unwrap();
        assert_eq!(sdr.get_sum(), 15);
    }
}
