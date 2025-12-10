//! Pass-Through Encoder implementation.
//!
//! The Pass-Through Encoder allows pre-encoded SDR data to be used in a pipeline
//! that expects an encoder interface.

use crate::encoders::Encoder;
use crate::error::{MokoshError, Result};
use crate::types::{Sdr, UInt};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Parameters for creating a Pass-Through Encoder.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PassThroughEncoderParams {
    /// Size of the SDR.
    pub size: UInt,

    /// Expected number of active bits (optional, for validation).
    pub active_bits: Option<UInt>,
}

impl Default for PassThroughEncoderParams {
    fn default() -> Self {
        Self {
            size: 2048,
            active_bits: None,
        }
    }
}

/// A pass-through encoder that accepts sparse indices directly.
///
/// This encoder is useful when you already have SDR data and want to
/// integrate it into a pipeline that uses the Encoder interface.
///
/// # Example
///
/// ```rust
/// use mokosh::encoders::{PassThroughEncoder, PassThroughEncoderParams, Encoder};
///
/// let encoder = PassThroughEncoder::new(PassThroughEncoderParams {
///     size: 100,
///     active_bits: Some(10),
/// }).unwrap();
///
/// // Pass pre-computed sparse indices
/// let sparse_indices = vec![5, 15, 25, 35, 45, 55, 65, 75, 85, 95];
/// let sdr = encoder.encode_to_sdr(sparse_indices).unwrap();
///
/// assert_eq!(sdr.get_sum(), 10);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PassThroughEncoder {
    /// Size of the SDR.
    size: UInt,

    /// Expected number of active bits (for validation).
    active_bits: Option<UInt>,

    /// Output dimensions.
    dimensions: Vec<UInt>,
}

impl PassThroughEncoder {
    /// Creates a new Pass-Through Encoder.
    pub fn new(params: PassThroughEncoderParams) -> Result<Self> {
        if params.size == 0 {
            return Err(MokoshError::InvalidParameter {
                name: "size",
                message: "Must be > 0".to_string(),
            });
        }

        if let Some(active_bits) = params.active_bits {
            if active_bits > params.size {
                return Err(MokoshError::InvalidParameter {
                    name: "active_bits",
                    message: "Cannot exceed size".to_string(),
                });
            }
        }

        Ok(Self {
            size: params.size,
            active_bits: params.active_bits,
            dimensions: vec![params.size],
        })
    }

    /// Returns the expected number of active bits, if set.
    pub fn expected_active_bits(&self) -> Option<UInt> {
        self.active_bits
    }
}

impl Encoder<Vec<UInt>> for PassThroughEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, value: Vec<UInt>, output: &mut Sdr) -> Result<()> {
        if output.dimensions() != self.dimensions.as_slice() {
            return Err(MokoshError::DimensionMismatch {
                expected: self.dimensions.clone(),
                actual: output.dimensions().to_vec(),
            });
        }

        // Validate indices are in range
        for &idx in &value {
            if idx >= self.size {
                return Err(MokoshError::IndexOutOfBounds {
                    index: idx as usize,
                    size: self.size as usize,
                });
            }
        }

        // Validate active bits count if specified
        if let Some(expected) = self.active_bits {
            if value.len() != expected as usize {
                return Err(MokoshError::InvalidParameter {
                    name: "active_bits",
                    message: format!(
                        "Expected {} active bits, got {}",
                        expected,
                        value.len()
                    ),
                });
            }
        }

        let mut sparse = value;
        sparse.sort_unstable();
        sparse.dedup();
        output.set_sparse_unchecked(sparse);

        Ok(())
    }
}

impl Encoder<&[UInt]> for PassThroughEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, value: &[UInt], output: &mut Sdr) -> Result<()> {
        self.encode(value.to_vec(), output)
    }
}

impl Encoder<Sdr> for PassThroughEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, value: Sdr, output: &mut Sdr) -> Result<()> {
        if value.dimensions() != self.dimensions.as_slice() {
            return Err(MokoshError::DimensionMismatch {
                expected: self.dimensions.clone(),
                actual: value.dimensions().to_vec(),
            });
        }

        // Validate active bits count if specified
        if let Some(expected) = self.active_bits {
            if value.get_sum() != expected as usize {
                return Err(MokoshError::InvalidParameter {
                    name: "active_bits",
                    message: format!(
                        "Expected {} active bits, got {}",
                        expected,
                        value.get_sum()
                    ),
                });
            }
        }

        output.set_sparse_unchecked(value.get_sparse().to_vec());

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_encoder() {
        let encoder = PassThroughEncoder::new(PassThroughEncoderParams {
            size: 500,
            active_bits: Some(25),
        })
        .unwrap();

        assert_eq!(Encoder::<Vec<UInt>>::size(&encoder), 500);
        assert_eq!(encoder.expected_active_bits(), Some(25));
    }

    #[test]
    fn test_encode_vec() {
        let encoder = PassThroughEncoder::new(PassThroughEncoderParams {
            size: 100,
            active_bits: None,
        })
        .unwrap();

        let sparse = vec![10, 20, 30, 40, 50];
        let sdr = encoder.encode_to_sdr(sparse).unwrap();

        assert_eq!(sdr.get_sum(), 5);
        assert_eq!(sdr.get_sparse(), &[10, 20, 30, 40, 50]);
    }

    #[test]
    fn test_encode_slice() {
        let encoder = PassThroughEncoder::new(PassThroughEncoderParams {
            size: 100,
            active_bits: None,
        })
        .unwrap();

        let sparse: &[UInt] = &[5, 15, 25];
        let sdr = encoder.encode_to_sdr(sparse).unwrap();

        assert_eq!(sdr.get_sum(), 3);
    }

    #[test]
    fn test_encode_sdr() {
        let encoder = PassThroughEncoder::new(PassThroughEncoderParams {
            size: 100,
            active_bits: None,
        })
        .unwrap();

        let mut input_sdr = Sdr::new(&[100]);
        input_sdr.set_sparse_unchecked(vec![1, 2, 3, 4, 5]);

        let sdr = encoder.encode_to_sdr(input_sdr).unwrap();
        assert_eq!(sdr.get_sum(), 5);
    }

    #[test]
    fn test_validates_active_bits() {
        let encoder = PassThroughEncoder::new(PassThroughEncoderParams {
            size: 100,
            active_bits: Some(5),
        })
        .unwrap();

        // Correct number of bits
        let result = encoder.encode_to_sdr(vec![1, 2, 3, 4, 5]);
        assert!(result.is_ok());

        // Wrong number of bits
        let result = encoder.encode_to_sdr(vec![1, 2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_validates_index_range() {
        let encoder = PassThroughEncoder::new(PassThroughEncoderParams {
            size: 100,
            active_bits: None,
        })
        .unwrap();

        let result = encoder.encode_to_sdr(vec![50, 150]); // 150 is out of range
        assert!(result.is_err());
    }

    #[test]
    fn test_deduplicates() {
        let encoder = PassThroughEncoder::new(PassThroughEncoderParams {
            size: 100,
            active_bits: None,
        })
        .unwrap();

        let sparse = vec![10, 20, 10, 30, 20]; // Duplicates
        let sdr = encoder.encode_to_sdr(sparse).unwrap();

        assert_eq!(sdr.get_sum(), 3);
        assert_eq!(sdr.get_sparse(), &[10, 20, 30]);
    }

    #[test]
    fn test_sorts_indices() {
        let encoder = PassThroughEncoder::new(PassThroughEncoderParams {
            size: 100,
            active_bits: None,
        })
        .unwrap();

        let sparse = vec![30, 10, 50, 20, 40]; // Unsorted
        let sdr = encoder.encode_to_sdr(sparse).unwrap();

        assert_eq!(sdr.get_sparse(), &[10, 20, 30, 40, 50]);
    }

    #[test]
    fn test_invalid_size() {
        let result = PassThroughEncoder::new(PassThroughEncoderParams {
            size: 0,
            active_bits: None,
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_active_bits_exceeds_size() {
        let result = PassThroughEncoder::new(PassThroughEncoderParams {
            size: 50,
            active_bits: Some(100),
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_no_validation_when_none() {
        let encoder = PassThroughEncoder::new(PassThroughEncoderParams {
            size: 100,
            active_bits: None,
        })
        .unwrap();

        // Any number of bits should be accepted
        assert!(encoder.encode_to_sdr(vec![1]).is_ok());
        assert!(encoder.encode_to_sdr(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).is_ok());
        assert!(encoder.encode_to_sdr(Vec::new()).is_ok());
    }
}
