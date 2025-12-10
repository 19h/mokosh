//! Random Distributed Scalar Encoder (RDSE) implementation.
//!
//! The RDSE encodes a numeric scalar value into an SDR using random hashing.
//! Unlike the ScalarEncoder, it does not need to know the minimum and maximum
//! of the input range at construction time.

use crate::encoders::Encoder;
use crate::error::{MokoshError, Result};
use crate::types::{Real, Sdr, UInt};
use crate::utils::Random;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Parameters for creating an RDSE.
///
/// Members "active_bits" & "sparsity" are mutually exclusive, specify exactly one.
/// Members "radius", "resolution", & "category" are mutually exclusive, specify exactly one.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RdseParams {
    /// Total number of bits in the encoded output SDR.
    pub size: UInt,

    /// Number of true bits in the encoded output SDR.
    /// Mutually exclusive with `sparsity`.
    pub active_bits: UInt,

    /// Fraction of bits in the encoded output which this encoder will activate.
    /// This is an alternative way to specify `active_bits`.
    /// Mutually exclusive with `active_bits`.
    pub sparsity: Real,

    /// Two inputs separated by more than the radius have non-overlapping representations.
    /// Mutually exclusive with `resolution` and `category`.
    pub radius: Real,

    /// Two inputs separated by greater than or equal to the resolution will have
    /// different representations.
    /// Mutually exclusive with `radius` and `category`.
    pub resolution: Real,

    /// If true, inputs are enumerated categories and all inputs will have
    /// unique / non-overlapping representations.
    /// Mutually exclusive with `radius` and `resolution`.
    pub category: bool,

    /// Forces different encoders to produce different outputs, even if the inputs
    /// and all other parameters are the same.
    /// Seed 0 is replaced with a random number.
    pub seed: u32,
}

impl Default for RdseParams {
    fn default() -> Self {
        Self {
            size: 400,
            active_bits: 0,
            sparsity: 0.0,
            radius: 0.0,
            resolution: 0.0,
            category: false,
            seed: 0,
        }
    }
}

/// Random Distributed Scalar Encoder.
///
/// The RDSE encodes numeric scalar (floating point) values into SDRs.
/// It is more flexible than the ScalarEncoder as it does not need to know
/// the minimum and maximum of the input range at construction time.
///
/// This implementation uses MurmurHash3-style hashing to determine active bit positions.
/// It does not save associations between inputs and active bits, relying instead on
/// the random & distributed nature of SDRs to prevent conflicts.
///
/// # Example
///
/// ```rust
/// use mokosh::encoders::{RandomDistributedScalarEncoder, RdseParams, Encoder};
/// use mokosh::types::Sdr;
///
/// let encoder = RandomDistributedScalarEncoder::new(RdseParams {
///     size: 1000,
///     sparsity: 0.05,
///     resolution: 1.0,
///     ..Default::default()
/// }).unwrap();
///
/// let sdr = encoder.encode_to_sdr(50.0).unwrap();
/// // Hash-based encoding may produce slightly fewer bits due to collisions
/// assert!(sdr.get_sum() > 0 && sdr.get_sum() <= 50);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RandomDistributedScalarEncoder {
    size: UInt,
    active_bits: UInt,
    sparsity: Real,
    radius: Real,
    resolution: Real,
    category: bool,
    seed: u32,

    /// Output dimensions.
    dimensions: Vec<UInt>,
}

/// Type alias for convenience.
pub type Rdse = RandomDistributedScalarEncoder;

impl RandomDistributedScalarEncoder {
    /// Creates a new RDSE.
    pub fn new(params: RdseParams) -> Result<Self> {
        // Validate size
        if params.size == 0 {
            return Err(MokoshError::InvalidParameter {
                name: "size",
                message: "Size must be > 0".to_string(),
            });
        }

        // Validate active_bits/sparsity (mutually exclusive)
        let num_active_args = (params.active_bits > 0) as u8 + (params.sparsity > 0.0) as u8;
        if num_active_args == 0 {
            return Err(MokoshError::InvalidParameter {
                name: "active_bits/sparsity",
                message: "Need one of: 'active_bits' or 'sparsity'".to_string(),
            });
        }
        if num_active_args > 1 {
            return Err(MokoshError::InvalidParameter {
                name: "active_bits/sparsity",
                message: "Specify only one of: 'active_bits' or 'sparsity'".to_string(),
            });
        }

        // Validate radius/resolution/category (mutually exclusive)
        let num_resolution_args =
            (params.radius > 0.0) as u8 + (params.resolution > 0.0) as u8 + params.category as u8;
        if num_resolution_args == 0 {
            return Err(MokoshError::InvalidParameter {
                name: "radius/resolution/category",
                message: "Need one of: 'radius', 'resolution', 'category'".to_string(),
            });
        }
        if num_resolution_args > 1 {
            return Err(MokoshError::InvalidParameter {
                name: "radius/resolution/category",
                message: "Specify only one of: 'radius', 'resolution', 'category'".to_string(),
            });
        }

        // Calculate active_bits from sparsity if needed
        let mut active_bits = params.active_bits;
        if params.sparsity > 0.0 {
            if !(0.0..=1.0).contains(&params.sparsity) {
                return Err(MokoshError::InvalidParameter {
                    name: "sparsity",
                    message: "Sparsity must be between 0.0 and 1.0".to_string(),
                });
            }
            active_bits = (params.size as Real * params.sparsity).round() as UInt;
            if active_bits == 0 {
                return Err(MokoshError::InvalidParameter {
                    name: "active_bits",
                    message: "Computed active_bits must be > 0".to_string(),
                });
            }
        }

        // Calculate sparsity for consistency
        let sparsity = active_bits as Real / params.size as Real;

        // Calculate resolution/radius
        let mut radius = params.radius;
        let mut resolution = params.resolution;

        if params.category {
            radius = 1.0;
            resolution = radius / active_bits as Real;
        } else if params.radius > 0.0 {
            resolution = radius / active_bits as Real;
        } else if params.resolution > 0.0 {
            radius = active_bits as Real * resolution;
        }

        // Generate seed if 0
        let seed = if params.seed == 0 {
            Random::new(rand::random::<i64>().abs()).get_uint32()
        } else {
            params.seed
        };

        let encoder = Self {
            size: params.size,
            active_bits,
            sparsity,
            radius,
            resolution,
            category: params.category,
            seed,
            dimensions: vec![params.size],
        };

        // Check parameters for hash collision resistance
        if !encoder.check_parameters() {
            return Err(MokoshError::InvalidParameter {
                name: "size/sparsity/active_bits",
                message: "Failed hash collision resistance check, please increase size, sparsity, and/or active_bits".to_string(),
            });
        }

        Ok(encoder)
    }

    /// Check that this RDSE is resistant to hash collisions.
    fn check_parameters(&self) -> bool {
        // Fast path for known-good parameters
        if self.size >= 1000 && self.active_bits >= 10 {
            return true;
        }

        const MAXIMUM_OVERLAP: Real = 0.33;
        const NUMBER_OF_TRIALS: usize = 1000;

        let mut rng = Random::new(42);
        let mut a = Sdr::new(&[self.size]);
        let mut b = Sdr::new(&[self.size]);
        a.randomize(self.sparsity, &mut rng);

        for _ in 0..NUMBER_OF_TRIALS {
            b.randomize(self.sparsity, &mut rng);
            let overlap = a.get_overlap(&b) as Real / self.active_bits as Real;
            if overlap > MAXIMUM_OVERLAP {
                return false;
            }
        }
        true
    }

    /// Returns the size (total number of bits).
    pub fn size_param(&self) -> UInt {
        self.size
    }

    /// Returns the number of active bits.
    pub fn active_bits(&self) -> UInt {
        self.active_bits
    }

    /// Returns the sparsity.
    pub fn sparsity(&self) -> Real {
        self.sparsity
    }

    /// Returns the resolution.
    pub fn resolution(&self) -> Real {
        self.resolution
    }

    /// Returns the radius.
    pub fn radius(&self) -> Real {
        self.radius
    }

    /// Returns whether this is a category encoder.
    pub fn category(&self) -> bool {
        self.category
    }

    /// Returns the seed.
    pub fn seed(&self) -> u32 {
        self.seed
    }

    /// MurmurHash3-style 32-bit hash function.
    #[inline]
    fn murmur_hash3_32(value: u32, seed: u32) -> u32 {
        let mut h = seed;
        let c1: u32 = 0xcc9e2d51;
        let c2: u32 = 0x1b873593;

        let mut k = value;
        k = k.wrapping_mul(c1);
        k = k.rotate_left(15);
        k = k.wrapping_mul(c2);

        h ^= k;
        h = h.rotate_left(13);
        h = h.wrapping_mul(5).wrapping_add(0xe6546b64);

        // Finalization mix
        h ^= 4; // length
        h ^= h >> 16;
        h = h.wrapping_mul(0x85ebca6b);
        h ^= h >> 13;
        h = h.wrapping_mul(0xc2b2ae35);
        h ^= h >> 16;

        h
    }
}

impl Encoder<Real> for RandomDistributedScalarEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, value: Real, output: &mut Sdr) -> Result<()> {
        if output.dimensions() != self.dimensions.as_slice() {
            return Err(MokoshError::DimensionMismatch {
                expected: self.dimensions.clone(),
                actual: output.dimensions().to_vec(),
            });
        }

        // Handle NaN
        if value.is_nan() {
            output.set_sparse(&[])?;
            return Ok(());
        }

        // Validate category input
        if self.category && value != (value as u64) as Real {
            return Err(MokoshError::InvalidParameter {
                name: "value",
                message: "Input to category encoder must be an unsigned integer".to_string(),
            });
        }

        // Calculate bucket index
        let index = (value / self.resolution) as u32;

        // Generate active bits using hashing
        let mut sparse = Vec::with_capacity(self.active_bits as usize);

        for offset in 0..self.active_bits {
            let hash_buffer = index.wrapping_add(offset);
            let bucket = Self::murmur_hash3_32(hash_buffer, self.seed);
            let bit = bucket % self.size;
            sparse.push(bit);
        }

        // Note: We don't deduplicate here as the C++ implementation doesn't either.
        // Hash collisions cause small deviations in sparsity.
        sparse.sort_unstable();
        sparse.dedup();
        output.set_sparse_unchecked(sparse);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_rdse() {
        let encoder = RandomDistributedScalarEncoder::new(RdseParams {
            size: 1000,
            sparsity: 0.05,
            resolution: 1.23,
            ..Default::default()
        })
        .unwrap();

        assert_eq!(encoder.size(), 1000);
        assert_eq!(encoder.active_bits(), 50); // 0.05 * 1000
    }

    #[test]
    fn test_encode_basic() {
        let encoder = RandomDistributedScalarEncoder::new(RdseParams {
            size: 1000,
            sparsity: 0.05,
            resolution: 1.0,
            ..Default::default()
        })
        .unwrap();

        let sdr = encoder.encode_to_sdr(3.0).unwrap();
        // Due to hash collisions, may not be exactly 50 bits
        assert!(sdr.get_sum() > 0);
        assert!(sdr.get_sum() <= 50);
    }

    #[test]
    fn test_encode_nan() {
        let encoder = RandomDistributedScalarEncoder::new(RdseParams {
            size: 1000,
            sparsity: 0.05,
            resolution: 1.0,
            ..Default::default()
        })
        .unwrap();

        let sdr = encoder.encode_to_sdr(Real::NAN).unwrap();
        assert_eq!(sdr.get_sum(), 0);
    }

    #[test]
    fn test_deterministic_encoding() {
        let encoder1 = RandomDistributedScalarEncoder::new(RdseParams {
            size: 1000,
            sparsity: 0.05,
            resolution: 1.0,
            seed: 42,
            ..Default::default()
        })
        .unwrap();

        let encoder2 = RandomDistributedScalarEncoder::new(RdseParams {
            size: 1000,
            sparsity: 0.05,
            resolution: 1.0,
            seed: 42,
            ..Default::default()
        })
        .unwrap();

        let sdr1 = encoder1.encode_to_sdr(44.4).unwrap();
        let sdr2 = encoder2.encode_to_sdr(44.4).unwrap();

        assert_eq!(sdr1.get_sparse(), sdr2.get_sparse());
    }

    #[test]
    fn test_different_seeds() {
        let encoder1 = RandomDistributedScalarEncoder::new(RdseParams {
            size: 1000,
            sparsity: 0.05,
            resolution: 1.0,
            seed: 42,
            ..Default::default()
        })
        .unwrap();

        let encoder2 = RandomDistributedScalarEncoder::new(RdseParams {
            size: 1000,
            sparsity: 0.05,
            resolution: 1.0,
            seed: 123,
            ..Default::default()
        })
        .unwrap();

        let sdr1 = encoder1.encode_to_sdr(44.4).unwrap();
        let sdr2 = encoder2.encode_to_sdr(44.4).unwrap();

        // Different seeds should produce different encodings
        assert_ne!(sdr1.get_sparse(), sdr2.get_sparse());
    }

    #[test]
    fn test_semantic_similarity() {
        let encoder = RandomDistributedScalarEncoder::new(RdseParams {
            size: 1000,
            active_bits: 50,
            resolution: 1.0,
            ..Default::default()
        })
        .unwrap();

        let sdr_10 = encoder.encode_to_sdr(10.0).unwrap();
        let sdr_11 = encoder.encode_to_sdr(11.0).unwrap();
        let sdr_100 = encoder.encode_to_sdr(100.0).unwrap();

        // Close values should have higher overlap than distant values
        let overlap_close = sdr_10.get_overlap(&sdr_11);
        let overlap_far = sdr_10.get_overlap(&sdr_100);

        assert!(overlap_close > overlap_far);
    }

    #[test]
    fn test_category_encoder() {
        let encoder = RandomDistributedScalarEncoder::new(RdseParams {
            size: 1000,
            active_bits: 50,
            category: true,
            ..Default::default()
        })
        .unwrap();

        let sdr_0 = encoder.encode_to_sdr(0.0).unwrap();
        let sdr_1 = encoder.encode_to_sdr(1.0).unwrap();
        let sdr_2 = encoder.encode_to_sdr(2.0).unwrap();

        // Categories should have distinct (non-overlapping) representations
        // Due to hash-based encoding, they may have some overlap
        assert!(sdr_0.get_overlap(&sdr_1) < 25); // Less than 50% overlap
        assert!(sdr_1.get_overlap(&sdr_2) < 25);
    }

    #[test]
    fn test_invalid_params_no_active() {
        let result = RandomDistributedScalarEncoder::new(RdseParams {
            size: 1000,
            active_bits: 0,
            sparsity: 0.0,
            resolution: 1.0,
            ..Default::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_params_both_active() {
        let result = RandomDistributedScalarEncoder::new(RdseParams {
            size: 1000,
            active_bits: 50,
            sparsity: 0.05,
            resolution: 1.0,
            ..Default::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_params_no_resolution() {
        let result = RandomDistributedScalarEncoder::new(RdseParams {
            size: 1000,
            sparsity: 0.05,
            radius: 0.0,
            resolution: 0.0,
            category: false,
            ..Default::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_with_active_bits() {
        let encoder = RandomDistributedScalarEncoder::new(RdseParams {
            size: 1000,
            active_bits: 100,
            resolution: 2.0,
            ..Default::default()
        })
        .unwrap();

        assert_eq!(encoder.active_bits(), 100);
        assert!((encoder.sparsity() - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_with_radius() {
        let encoder = RandomDistributedScalarEncoder::new(RdseParams {
            size: 1000,
            active_bits: 50,
            radius: 10.0,
            ..Default::default()
        })
        .unwrap();

        assert!((encoder.radius() - 10.0).abs() < 0.001);
        assert!((encoder.resolution() - 0.2).abs() < 0.001); // 10.0 / 50
    }
}
