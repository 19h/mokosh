//! Boolean Encoder implementation.
//!
//! The Boolean Encoder encodes true/false values into distinct SDR representations.

use crate::encoders::Encoder;
use crate::error::{MokoshError, Result};
use crate::types::{Sdr, UInt};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Parameters for creating a Boolean Encoder.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BooleanEncoderParams {
    /// Number of active bits for each boolean state.
    pub active_bits: UInt,
}

impl Default for BooleanEncoderParams {
    fn default() -> Self {
        Self { active_bits: 21 }
    }
}

/// Encodes boolean values into SDR representations.
///
/// True and false are encoded into non-overlapping bit patterns.
///
/// # Example
///
/// ```rust
/// use mokosh::encoders::{BooleanEncoder, BooleanEncoderParams, Encoder};
///
/// let encoder = BooleanEncoder::new(BooleanEncoderParams {
///     active_bits: 10,
/// }).unwrap();
///
/// let sdr_true = encoder.encode_to_sdr(true).unwrap();
/// let sdr_false = encoder.encode_to_sdr(false).unwrap();
///
/// // No overlap between true and false
/// assert_eq!(sdr_true.get_overlap(&sdr_false), 0);
/// assert_eq!(sdr_true.get_sum(), 10);
/// assert_eq!(sdr_false.get_sum(), 10);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BooleanEncoder {
    /// Number of active bits.
    active_bits: UInt,

    /// Total size (2 * active_bits).
    size: UInt,

    /// Output dimensions.
    dimensions: Vec<UInt>,
}

impl BooleanEncoder {
    /// Creates a new Boolean Encoder.
    pub fn new(params: BooleanEncoderParams) -> Result<Self> {
        if params.active_bits == 0 {
            return Err(MokoshError::InvalidParameter {
                name: "active_bits",
                message: "Must be > 0".to_string(),
            });
        }

        let size = params.active_bits * 2;

        Ok(Self {
            active_bits: params.active_bits,
            size,
            dimensions: vec![size],
        })
    }

    /// Returns the number of active bits.
    pub fn active_bits(&self) -> UInt {
        self.active_bits
    }
}

impl Encoder<bool> for BooleanEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, value: bool, output: &mut Sdr) -> Result<()> {
        if output.dimensions() != self.dimensions.as_slice() {
            return Err(MokoshError::DimensionMismatch {
                expected: self.dimensions.clone(),
                actual: output.dimensions().to_vec(),
            });
        }

        let sparse: Vec<UInt> = if value {
            // True: first half of bits
            (0..self.active_bits).collect()
        } else {
            // False: second half of bits
            (self.active_bits..self.size).collect()
        };

        output.set_sparse_unchecked(sparse);

        Ok(())
    }
}

impl Encoder<u8> for BooleanEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, value: u8, output: &mut Sdr) -> Result<()> {
        self.encode(value != 0, output)
    }
}

impl Encoder<i32> for BooleanEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, value: i32, output: &mut Sdr) -> Result<()> {
        self.encode(value != 0, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_encoder() {
        let encoder = BooleanEncoder::new(BooleanEncoderParams { active_bits: 15 }).unwrap();

        assert_eq!(encoder.active_bits(), 15);
        assert_eq!(Encoder::<bool>::size(&encoder), 30);
    }

    #[test]
    fn test_encode_true() {
        let encoder = BooleanEncoder::new(BooleanEncoderParams { active_bits: 10 }).unwrap();

        let sdr = encoder.encode_to_sdr(true).unwrap();

        assert_eq!(sdr.get_sum(), 10);

        // True should activate first 10 bits
        let sparse = sdr.get_sparse();
        assert_eq!(sparse.len(), 10);
        for (i, &bit) in sparse.iter().enumerate() {
            assert_eq!(bit, i as UInt);
        }
    }

    #[test]
    fn test_encode_false() {
        let encoder = BooleanEncoder::new(BooleanEncoderParams { active_bits: 10 }).unwrap();

        let sdr = encoder.encode_to_sdr(false).unwrap();

        assert_eq!(sdr.get_sum(), 10);

        // False should activate last 10 bits
        let sparse = sdr.get_sparse();
        assert_eq!(sparse.len(), 10);
        for (i, &bit) in sparse.iter().enumerate() {
            assert_eq!(bit, 10 + i as UInt);
        }
    }

    #[test]
    fn test_no_overlap() {
        let encoder = BooleanEncoder::new(BooleanEncoderParams { active_bits: 21 }).unwrap();

        let sdr_true = encoder.encode_to_sdr(true).unwrap();
        let sdr_false = encoder.encode_to_sdr(false).unwrap();

        assert_eq!(sdr_true.get_overlap(&sdr_false), 0);
    }

    #[test]
    fn test_deterministic() {
        let encoder = BooleanEncoder::new(BooleanEncoderParams::default()).unwrap();

        let sdr1 = encoder.encode_to_sdr(true).unwrap();
        let sdr2 = encoder.encode_to_sdr(true).unwrap();

        assert_eq!(sdr1.get_sparse(), sdr2.get_sparse());
    }

    #[test]
    fn test_encode_u8() {
        let encoder = BooleanEncoder::new(BooleanEncoderParams { active_bits: 5 }).unwrap();

        let sdr_zero = encoder.encode_to_sdr(0u8).unwrap();
        let sdr_one = encoder.encode_to_sdr(1u8).unwrap();
        let sdr_many = encoder.encode_to_sdr(42u8).unwrap();

        let sdr_false = encoder.encode_to_sdr(false).unwrap();
        let sdr_true = encoder.encode_to_sdr(true).unwrap();

        assert_eq!(sdr_zero.get_sparse(), sdr_false.get_sparse());
        assert_eq!(sdr_one.get_sparse(), sdr_true.get_sparse());
        assert_eq!(sdr_many.get_sparse(), sdr_true.get_sparse());
    }

    #[test]
    fn test_encode_i32() {
        let encoder = BooleanEncoder::new(BooleanEncoderParams { active_bits: 5 }).unwrap();

        let sdr_zero = Encoder::<i32>::encode_to_sdr(&encoder, 0).unwrap();
        let sdr_neg = Encoder::<i32>::encode_to_sdr(&encoder, -1).unwrap();
        let sdr_pos = Encoder::<i32>::encode_to_sdr(&encoder, 100).unwrap();

        let sdr_false = encoder.encode_to_sdr(false).unwrap();
        let sdr_true = encoder.encode_to_sdr(true).unwrap();

        assert_eq!(sdr_zero.get_sparse(), sdr_false.get_sparse());
        assert_eq!(sdr_neg.get_sparse(), sdr_true.get_sparse());
        assert_eq!(sdr_pos.get_sparse(), sdr_true.get_sparse());
    }

    #[test]
    fn test_invalid_active_bits() {
        let result = BooleanEncoder::new(BooleanEncoderParams { active_bits: 0 });
        assert!(result.is_err());
    }
}
