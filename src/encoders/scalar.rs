//! Scalar Encoder implementation.
//!
//! The Scalar Encoder converts numerical values into SDR representations
//! where semantically similar values have overlapping active bits.

use crate::encoders::Encoder;
use crate::error::{MokoshError, Result};
use crate::types::{Real, Sdr, UInt};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Parameters for creating a Scalar Encoder.
///
/// The size can be specified directly, or computed automatically from radius.
/// Only specify one of: size, radius, or category.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ScalarEncoderParams {
    /// Minimum value of the input range.
    pub minimum: Real,

    /// Maximum value of the input range.
    pub maximum: Real,

    /// Total number of bits in the output.
    /// If 0, it will be computed from radius or resolution.
    pub size: UInt,

    /// Number of active bits for each encoding.
    pub active_bits: UInt,

    /// Two inputs separated by more than the radius have non-overlapping representations.
    /// Used to compute size if size is 0.
    pub radius: Real,

    /// Whether to clip values outside the range (vs. wrapping).
    pub clip_input: bool,

    /// Whether the input range is periodic (e.g., angles).
    pub periodic: bool,

    /// Whether inputs are enumerated categories.
    /// If true, all inputs will have unique, non-overlapping representations.
    pub category: bool,
}

impl Default for ScalarEncoderParams {
    fn default() -> Self {
        Self {
            minimum: 0.0,
            maximum: 100.0,
            size: 400,
            active_bits: 21,
            radius: 0.0,
            clip_input: true,
            periodic: false,
            category: false,
        }
    }
}

/// Encodes scalar values into SDR representations.
///
/// Similar values produce overlapping bit patterns, preserving
/// semantic similarity in the encoded representation.
///
/// # Example
///
/// ```rust
/// use mokosh::encoders::{ScalarEncoder, ScalarEncoderParams, Encoder};
/// use mokosh::types::Sdr;
///
/// let encoder = ScalarEncoder::new(ScalarEncoderParams {
///     minimum: 0.0,
///     maximum: 100.0,
///     size: 100,
///     active_bits: 10,
///     ..Default::default()
/// }).unwrap();
///
/// let sdr = encoder.encode_to_sdr(50.0).unwrap();
/// assert_eq!(sdr.get_sum(), 10);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ScalarEncoder {
    minimum: Real,
    maximum: Real,
    size: UInt,
    active_bits: UInt,
    clip_input: bool,
    periodic: bool,
    category: bool,

    /// Precomputed: range of input values.
    range: Real,

    /// Precomputed: resolution (input units per bit).
    resolution: Real,

    /// Precomputed: half width of the active region.
    half_width: UInt,

    /// Precomputed: number of positions for center bit.
    num_buckets: UInt,

    /// Output dimensions.
    dimensions: Vec<UInt>,
}

impl ScalarEncoder {
    /// Creates a new Scalar Encoder.
    pub fn new(params: ScalarEncoderParams) -> Result<Self> {
        if params.maximum <= params.minimum {
            return Err(MokoshError::InvalidParameter {
                name: "maximum",
                message: "Maximum must be greater than minimum".to_string(),
            });
        }

        if params.active_bits == 0 {
            return Err(MokoshError::InvalidParameter {
                name: "active_bits",
                message: "Must be > 0".to_string(),
            });
        }

        let range = params.maximum - params.minimum;
        let half_width = params.active_bits / 2;

        // Calculate resolution and size
        let (size, resolution) = if params.category {
            // For categories, each value is a distinct bucket
            let num_categories = (range + 1.0) as UInt;
            let size = num_categories + params.active_bits - 1;
            let resolution = 1.0;
            (size, resolution)
        } else if params.radius > 0.0 {
            // Compute from radius
            let resolution = params.radius / params.active_bits as Real;
            let num_buckets = (range / resolution).ceil() as UInt;
            let size = if params.periodic {
                num_buckets
            } else {
                num_buckets + params.active_bits - 1
            };
            (size, resolution)
        } else if params.size > 0 {
            // Check size vs active_bits before computing num_buckets to avoid overflow
            if params.size < params.active_bits {
                return Err(MokoshError::InvalidParameter {
                    name: "size",
                    message: "Size must be >= active_bits".to_string(),
                });
            }
            // Use provided size
            let num_buckets = if params.periodic {
                params.size
            } else {
                params.size - params.active_bits + 1
            };
            let resolution = range / num_buckets as Real;
            (params.size, resolution)
        } else {
            return Err(MokoshError::InvalidParameter {
                name: "size/radius",
                message: "Must specify either size or radius".to_string(),
            });
        };

        if size < params.active_bits {
            return Err(MokoshError::InvalidParameter {
                name: "size",
                message: "Size must be >= active_bits".to_string(),
            });
        }

        let num_buckets = if params.periodic {
            size
        } else {
            size - params.active_bits + 1
        };

        Ok(Self {
            minimum: params.minimum,
            maximum: params.maximum,
            size,
            active_bits: params.active_bits,
            clip_input: params.clip_input,
            periodic: params.periodic,
            category: params.category,
            range,
            resolution,
            half_width,
            num_buckets,
            dimensions: vec![size],
        })
    }

    /// Creates a Scalar Encoder from resolution and other parameters.
    pub fn from_resolution(
        minimum: Real,
        maximum: Real,
        resolution: Real,
        active_bits: UInt,
    ) -> Result<Self> {
        let range = maximum - minimum;
        let num_buckets = (range / resolution).ceil() as UInt;
        let size = num_buckets + active_bits - 1;

        Self::new(ScalarEncoderParams {
            minimum,
            maximum,
            size,
            active_bits,
            ..Default::default()
        })
    }

    /// Returns the bucket index for a value.
    pub fn bucket_index(&self, value: Real) -> UInt {
        let mut value = value;

        // Handle out-of-range values
        if value < self.minimum {
            if self.periodic {
                // Wrap below minimum
                let span = value - self.minimum;
                value = self.maximum + (span % self.range);
            } else if self.clip_input {
                value = self.minimum;
            }
        } else if self.periodic && value >= self.maximum {
            // For periodic, max wraps to min (360 degrees == 0 degrees)
            let span = value - self.minimum;
            value = self.minimum + (span % self.range);
        } else if value > self.maximum {
            if self.clip_input {
                value = self.maximum;
            }
        }

        // Normalize to [0, 1)
        let normalized = (value - self.minimum) / self.range;

        // Map to bucket - clamp to valid range
        let bucket = (normalized * self.num_buckets as Real).floor() as UInt;
        bucket.min(self.num_buckets - 1)
    }

    /// Returns the minimum value.
    pub fn minimum(&self) -> Real {
        self.minimum
    }

    /// Returns the maximum value.
    pub fn maximum(&self) -> Real {
        self.maximum
    }

    /// Returns the resolution.
    pub fn resolution(&self) -> Real {
        self.resolution
    }

    /// Returns the number of active bits.
    pub fn active_bits(&self) -> UInt {
        self.active_bits
    }

    /// Returns whether this is a periodic encoder.
    pub fn periodic(&self) -> bool {
        self.periodic
    }
}

impl Encoder<Real> for ScalarEncoder {
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

        let bucket = self.bucket_index(value);

        // Calculate start position
        let start = if self.periodic {
            bucket
        } else {
            bucket
        };

        // Generate active bits
        let mut sparse = Vec::with_capacity(self.active_bits as usize);

        for i in 0..self.active_bits {
            let bit = if self.periodic {
                (start + i) % self.size
            } else {
                start + i
            };
            sparse.push(bit);
        }

        sparse.sort_unstable();
        output.set_sparse_unchecked(sparse);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_encoder() {
        let encoder = ScalarEncoder::new(ScalarEncoderParams {
            minimum: 0.0,
            maximum: 100.0,
            size: 100,
            active_bits: 10,
            ..Default::default()
        })
        .unwrap();

        assert_eq!(encoder.size(), 100);
        assert_eq!(encoder.active_bits(), 10);
    }

    #[test]
    fn test_encode_basic() {
        let encoder = ScalarEncoder::new(ScalarEncoderParams {
            minimum: 0.0,
            maximum: 100.0,
            size: 100,
            active_bits: 10,
            ..Default::default()
        })
        .unwrap();

        let sdr = encoder.encode_to_sdr(50.0).unwrap();
        assert_eq!(sdr.get_sum(), 10);
    }

    #[test]
    fn test_encode_boundaries() {
        let encoder = ScalarEncoder::new(ScalarEncoderParams {
            minimum: 0.0,
            maximum: 100.0,
            size: 100,
            active_bits: 10,
            ..Default::default()
        })
        .unwrap();

        // Minimum value
        let sdr_min = encoder.encode_to_sdr(0.0).unwrap();
        let min_bits = sdr_min.get_sparse();
        assert!(min_bits.iter().all(|&b| b < 10));

        // Maximum value
        let sdr_max = encoder.encode_to_sdr(100.0).unwrap();
        let max_bits = sdr_max.get_sparse();
        assert!(max_bits.iter().all(|&b| b >= 90));
    }

    #[test]
    fn test_overlap_similarity() {
        let encoder = ScalarEncoder::new(ScalarEncoderParams {
            minimum: 0.0,
            maximum: 100.0,
            size: 100,
            active_bits: 21,
            ..Default::default()
        })
        .unwrap();

        let sdr_50 = encoder.encode_to_sdr(50.0).unwrap();
        let sdr_51 = encoder.encode_to_sdr(51.0).unwrap();
        let sdr_100 = encoder.encode_to_sdr(100.0).unwrap();

        // Close values should have high overlap
        let overlap_close = sdr_50.get_overlap(&sdr_51);

        // Distant values should have low overlap
        let overlap_far = sdr_50.get_overlap(&sdr_100);

        assert!(overlap_close > overlap_far);
    }

    #[test]
    fn test_clip_input() {
        let encoder = ScalarEncoder::new(ScalarEncoderParams {
            minimum: 0.0,
            maximum: 100.0,
            size: 100,
            active_bits: 10,
            clip_input: true,
            ..Default::default()
        })
        .unwrap();

        // Values outside range should be clipped
        let sdr_below = encoder.encode_to_sdr(-10.0).unwrap();
        let sdr_min = encoder.encode_to_sdr(0.0).unwrap();
        assert_eq!(sdr_below.get_sparse(), sdr_min.get_sparse());

        let sdr_above = encoder.encode_to_sdr(110.0).unwrap();
        let sdr_max = encoder.encode_to_sdr(100.0).unwrap();
        assert_eq!(sdr_above.get_sparse(), sdr_max.get_sparse());
    }

    #[test]
    fn test_periodic() {
        let encoder = ScalarEncoder::new(ScalarEncoderParams {
            minimum: 0.0,
            maximum: 360.0,
            size: 360,
            active_bits: 21,
            periodic: true,
            ..Default::default()
        })
        .unwrap();

        // 0 and 360 should produce same encoding
        let sdr_0 = encoder.encode_to_sdr(0.0).unwrap();
        let sdr_360 = encoder.encode_to_sdr(360.0).unwrap();
        assert_eq!(sdr_0.get_sparse(), sdr_360.get_sparse());

        // 355 and 5 should have high overlap (wrap around)
        // With 360 buckets, 21 active bits:
        // 355 -> bits [355..359, 0..15] (wraps around)
        // 5 -> bits [5..25]
        // Overlap should be bits 5..15 = 11 bits
        let sdr_355 = encoder.encode_to_sdr(355.0).unwrap();
        let sdr_5 = encoder.encode_to_sdr(5.0).unwrap();
        let overlap = sdr_355.get_overlap(&sdr_5);
        assert!(overlap > 10); // Significant overlap due to wrap-around
    }

    #[test]
    fn test_bucket_index() {
        let encoder = ScalarEncoder::new(ScalarEncoderParams {
            minimum: 0.0,
            maximum: 100.0,
            size: 100,
            active_bits: 10,
            ..Default::default()
        })
        .unwrap();

        assert_eq!(encoder.bucket_index(0.0), 0);
        assert!(encoder.bucket_index(50.0) > 0);
        assert!(encoder.bucket_index(100.0) > encoder.bucket_index(50.0));
    }

    #[test]
    fn test_invalid_params() {
        let result = ScalarEncoder::new(ScalarEncoderParams {
            minimum: 100.0,
            maximum: 0.0, // Invalid: max < min
            ..Default::default()
        });
        assert!(result.is_err());

        let result = ScalarEncoder::new(ScalarEncoderParams {
            size: 10,
            active_bits: 20, // Invalid: active_bits > size
            ..Default::default()
        });
        assert!(result.is_err());
    }
}
