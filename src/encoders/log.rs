//! Log Encoder implementation.
//!
//! The Log Encoder encodes positive numeric values using a logarithmic scale,
//! providing better resolution for values near zero and wider ranges at higher values.

use crate::encoders::Encoder;
use crate::error::{MokoshError, Result};
use crate::types::{Real, Sdr, UInt};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Parameters for creating a Log Encoder.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LogEncoderParams {
    /// Minimum value (must be > 0).
    pub minimum: Real,

    /// Maximum value.
    pub maximum: Real,

    /// Total number of bits in the output SDR.
    pub size: UInt,

    /// Number of active bits.
    pub active_bits: UInt,

    /// If true, clips out-of-range values; otherwise returns error.
    pub clip_input: bool,
}

impl Default for LogEncoderParams {
    fn default() -> Self {
        Self {
            minimum: 1.0,
            maximum: 10000.0,
            size: 100,
            active_bits: 10,
            clip_input: true,
        }
    }
}

/// Encodes positive values on a logarithmic scale.
///
/// This encoder is useful for values that span several orders of magnitude,
/// where equal ratios should produce equal encoding differences.
///
/// # Example
///
/// ```rust
/// use mokosh::encoders::{LogEncoder, LogEncoderParams, Encoder};
///
/// let encoder = LogEncoder::new(LogEncoderParams {
///     minimum: 1.0,
///     maximum: 1000.0,
///     size: 100,
///     active_bits: 10,
///     ..Default::default()
/// }).unwrap();
///
/// let sdr_10 = encoder.encode_to_sdr(10.0).unwrap();
/// let sdr_100 = encoder.encode_to_sdr(100.0).unwrap();
///
/// // Values differ by same ratio, so should have similar overlap patterns
/// assert_eq!(sdr_10.get_sum(), 10);
/// assert_eq!(sdr_100.get_sum(), 10);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LogEncoder {
    /// Minimum value.
    minimum: Real,

    /// Maximum value.
    maximum: Real,

    /// Total size of output.
    size: UInt,

    /// Number of active bits.
    active_bits: UInt,

    /// Whether to clip out-of-range values.
    clip_input: bool,

    /// Log of minimum (precomputed).
    log_min: Real,

    /// Log of maximum (precomputed).
    log_max: Real,

    /// Log range (precomputed).
    log_range: Real,

    /// Output dimensions.
    dimensions: Vec<UInt>,
}

impl LogEncoder {
    /// Creates a new Log Encoder.
    pub fn new(params: LogEncoderParams) -> Result<Self> {
        if params.minimum <= 0.0 {
            return Err(MokoshError::InvalidParameter {
                name: "minimum",
                message: "Must be > 0 for log encoding".to_string(),
            });
        }

        if params.maximum <= params.minimum {
            return Err(MokoshError::InvalidParameter {
                name: "maximum",
                message: "Must be greater than minimum".to_string(),
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

        let log_min = params.minimum.ln();
        let log_max = params.maximum.ln();
        let log_range = log_max - log_min;

        Ok(Self {
            minimum: params.minimum,
            maximum: params.maximum,
            size: params.size,
            active_bits: params.active_bits,
            clip_input: params.clip_input,
            log_min,
            log_max,
            log_range,
            dimensions: vec![params.size],
        })
    }

    /// Returns the minimum value.
    pub fn minimum(&self) -> Real {
        self.minimum
    }

    /// Returns the maximum value.
    pub fn maximum(&self) -> Real {
        self.maximum
    }

    /// Converts a value to its position in the encoding (0.0 to 1.0).
    fn value_to_position(&self, value: Real) -> Real {
        let log_value = value.ln();
        (log_value - self.log_min) / self.log_range
    }
}

impl Encoder<Real> for LogEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, mut value: Real, output: &mut Sdr) -> Result<()> {
        if output.dimensions() != self.dimensions.as_slice() {
            return Err(MokoshError::DimensionMismatch {
                expected: self.dimensions.clone(),
                actual: output.dimensions().to_vec(),
            });
        }

        // Handle out-of-range values
        if value <= 0.0 {
            return Err(MokoshError::InvalidParameter {
                name: "value",
                message: "Log encoder requires positive values".to_string(),
            });
        }

        if value < self.minimum || value > self.maximum {
            if self.clip_input {
                value = value.clamp(self.minimum, self.maximum);
            } else {
                return Err(MokoshError::InvalidParameter {
                    name: "value",
                    message: format!(
                        "Value {} outside range [{}, {}]",
                        value, self.minimum, self.maximum
                    ),
                });
            }
        }

        // Calculate position in log space
        let position = self.value_to_position(value);

        // Calculate the center bucket
        let num_buckets = self.size - self.active_bits + 1;
        let center_bucket = (position * (num_buckets - 1) as Real).round() as UInt;

        // Generate active bits as a contiguous block
        let start = center_bucket;
        let sparse: Vec<UInt> = (start..start + self.active_bits).collect();
        output.set_sparse_unchecked(sparse);

        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_encoder() {
        let encoder = LogEncoder::new(LogEncoderParams {
            minimum: 1.0,
            maximum: 1000.0,
            size: 100,
            active_bits: 10,
            ..Default::default()
        })
        .unwrap();

        assert_eq!(encoder.minimum(), 1.0);
        assert_eq!(encoder.maximum(), 1000.0);
        assert_eq!(encoder.size(), 100);
    }

    #[test]
    fn test_encode_values() {
        let encoder = LogEncoder::new(LogEncoderParams {
            minimum: 1.0,
            maximum: 1000.0,
            size: 100,
            active_bits: 10,
            ..Default::default()
        })
        .unwrap();

        let sdr_1 = encoder.encode_to_sdr(1.0).unwrap();
        let sdr_10 = encoder.encode_to_sdr(10.0).unwrap();
        let sdr_100 = encoder.encode_to_sdr(100.0).unwrap();
        let sdr_1000 = encoder.encode_to_sdr(1000.0).unwrap();

        assert_eq!(sdr_1.get_sum(), 10);
        assert_eq!(sdr_10.get_sum(), 10);
        assert_eq!(sdr_100.get_sum(), 10);
        assert_eq!(sdr_1000.get_sum(), 10);

        // Min should be at the start
        assert!(sdr_1.get_sparse()[0] < 10);

        // Max should be at the end
        assert!(sdr_1000.get_sparse()[0] > 80);
    }

    #[test]
    fn test_log_spacing() {
        let encoder = LogEncoder::new(LogEncoderParams {
            minimum: 1.0,
            maximum: 10000.0,
            size: 200,
            active_bits: 10,
            ..Default::default()
        })
        .unwrap();

        // Values that differ by the same ratio should have similar overlap
        let sdr_1 = encoder.encode_to_sdr(1.0).unwrap();
        let sdr_10 = encoder.encode_to_sdr(10.0).unwrap();
        let sdr_100 = encoder.encode_to_sdr(100.0).unwrap();
        let sdr_1000 = encoder.encode_to_sdr(1000.0).unwrap();

        // Equal log spacing means equal representation spacing
        let overlap_1_10 = sdr_1.get_overlap(&sdr_10);
        let overlap_10_100 = sdr_10.get_overlap(&sdr_100);
        let overlap_100_1000 = sdr_100.get_overlap(&sdr_1000);

        // Overlaps should be approximately equal
        assert!((overlap_1_10 as i32 - overlap_10_100 as i32).abs() <= 2);
        assert!((overlap_10_100 as i32 - overlap_100_1000 as i32).abs() <= 2);
    }

    #[test]
    fn test_clip_input() {
        let encoder = LogEncoder::new(LogEncoderParams {
            minimum: 1.0,
            maximum: 100.0,
            size: 50,
            active_bits: 5,
            clip_input: true,
        })
        .unwrap();

        // Should clip to minimum
        let sdr_low = encoder.encode_to_sdr(0.1).unwrap();
        let sdr_min = encoder.encode_to_sdr(1.0).unwrap();
        assert_eq!(sdr_low.get_sparse(), sdr_min.get_sparse());

        // Should clip to maximum
        let sdr_high = encoder.encode_to_sdr(1000.0).unwrap();
        let sdr_max = encoder.encode_to_sdr(100.0).unwrap();
        assert_eq!(sdr_high.get_sparse(), sdr_max.get_sparse());
    }

    #[test]
    fn test_no_clip() {
        let encoder = LogEncoder::new(LogEncoderParams {
            minimum: 1.0,
            maximum: 100.0,
            size: 50,
            active_bits: 5,
            clip_input: false,
        })
        .unwrap();

        let result = encoder.encode_to_sdr(0.1);
        assert!(result.is_err());
    }

    #[test]
    fn test_negative_value() {
        let encoder = LogEncoder::new(LogEncoderParams::default()).unwrap();

        let result = encoder.encode_to_sdr(-5.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_value() {
        let encoder = LogEncoder::new(LogEncoderParams::default()).unwrap();

        let result = encoder.encode_to_sdr(0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_minimum() {
        let result = LogEncoder::new(LogEncoderParams {
            minimum: 0.0,
            ..Default::default()
        });
        assert!(result.is_err());

        let result = LogEncoder::new(LogEncoderParams {
            minimum: -1.0,
            ..Default::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_deterministic() {
        let encoder = LogEncoder::new(LogEncoderParams::default()).unwrap();

        let sdr1 = encoder.encode_to_sdr(50.0).unwrap();
        let sdr2 = encoder.encode_to_sdr(50.0).unwrap();

        assert_eq!(sdr1.get_sparse(), sdr2.get_sparse());
    }
}
