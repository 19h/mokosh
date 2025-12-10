//! Delta Encoder implementation.
//!
//! The Delta Encoder encodes the change (delta) between consecutive values,
//! useful for capturing rate-of-change patterns in time series data.

use crate::encoders::{Encoder, ScalarEncoder, ScalarEncoderParams};
use crate::error::{MokoshError, Result};
use crate::types::{Real, Sdr, UInt};
use std::cell::RefCell;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Parameters for creating a Delta Encoder.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DeltaEncoderParams {
    /// Maximum expected positive delta.
    pub max_delta: Real,

    /// Total number of bits in the output SDR.
    pub size: UInt,

    /// Number of active bits.
    pub active_bits: UInt,

    /// If true, wraps around for large deltas (periodic).
    pub periodic: bool,

    /// If true, clips out-of-range deltas; otherwise returns error.
    pub clip_input: bool,
}

impl Default for DeltaEncoderParams {
    fn default() -> Self {
        Self {
            max_delta: 100.0,
            size: 100,
            active_bits: 10,
            periodic: false,
            clip_input: true,
        }
    }
}

/// Encodes the change between consecutive values.
///
/// This encoder maintains state and encodes the difference between
/// the current and previous value. Useful for detecting rate of change.
///
/// # Example
///
/// ```rust
/// use mokosh::encoders::{DeltaEncoder, DeltaEncoderParams, Encoder};
///
/// let encoder = DeltaEncoder::new(DeltaEncoderParams {
///     max_delta: 10.0,
///     size: 100,
///     active_bits: 10,
///     ..Default::default()
/// }).unwrap();
///
/// // First value establishes baseline (delta = 0)
/// let sdr1 = encoder.encode_to_sdr(50.0).unwrap();
///
/// // Second value encodes the delta (+5)
/// let sdr2 = encoder.encode_to_sdr(55.0).unwrap();
///
/// // Delta of 0 vs delta of 5 should have some difference
/// assert!(sdr1.get_overlap(&sdr2) < 10);
///
/// // Reset to clear state
/// encoder.reset();
/// ```
#[derive(Debug)]
pub struct DeltaEncoder {
    /// Internal scalar encoder for the delta values.
    encoder: ScalarEncoder,

    /// Previous value (stored in RefCell for interior mutability).
    previous: RefCell<Option<Real>>,

    /// Maximum delta magnitude.
    max_delta: Real,

    /// Whether to clip out-of-range values.
    clip_input: bool,

    /// Output dimensions.
    dimensions: Vec<UInt>,
}

impl DeltaEncoder {
    /// Creates a new Delta Encoder.
    pub fn new(params: DeltaEncoderParams) -> Result<Self> {
        if params.max_delta <= 0.0 {
            return Err(MokoshError::InvalidParameter {
                name: "max_delta",
                message: "Must be > 0".to_string(),
            });
        }

        // Create internal scalar encoder for delta range [-max_delta, max_delta]
        let encoder = ScalarEncoder::new(ScalarEncoderParams {
            minimum: -params.max_delta,
            maximum: params.max_delta,
            size: params.size,
            active_bits: params.active_bits,
            periodic: params.periodic,
            clip_input: params.clip_input,
            radius: 0.0,
            category: false,
        })?;

        Ok(Self {
            encoder,
            previous: RefCell::new(None),
            max_delta: params.max_delta,
            clip_input: params.clip_input,
            dimensions: vec![params.size],
        })
    }

    /// Resets the encoder state, clearing the previous value.
    pub fn reset(&self) {
        *self.previous.borrow_mut() = None;
    }

    /// Returns whether the encoder has a previous value.
    pub fn has_previous(&self) -> bool {
        self.previous.borrow().is_some()
    }

    /// Returns the maximum delta.
    pub fn max_delta(&self) -> Real {
        self.max_delta
    }

    /// Encodes a delta value directly (without updating state).
    pub fn encode_delta(&self, delta: Real, output: &mut Sdr) -> Result<()> {
        self.encoder.encode(delta, output)
    }
}

impl Encoder<Real> for DeltaEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.encoder.size()
    }

    fn encode(&self, value: Real, output: &mut Sdr) -> Result<()> {
        let delta = {
            let mut prev = self.previous.borrow_mut();
            let delta = match *prev {
                Some(prev_value) => value - prev_value,
                None => 0.0, // First value, no delta
            };
            *prev = Some(value);
            delta
        };

        // Clip delta if needed
        let delta = if self.clip_input {
            delta.clamp(-self.max_delta, self.max_delta)
        } else {
            delta
        };

        self.encoder.encode(delta, output)
    }
}


impl Clone for DeltaEncoder {
    fn clone(&self) -> Self {
        Self {
            encoder: self.encoder.clone(),
            previous: RefCell::new(*self.previous.borrow()),
            max_delta: self.max_delta,
            clip_input: self.clip_input,
            dimensions: self.dimensions.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_encoder() {
        let encoder = DeltaEncoder::new(DeltaEncoderParams {
            max_delta: 50.0,
            size: 100,
            active_bits: 10,
            ..Default::default()
        })
        .unwrap();

        assert_eq!(encoder.max_delta(), 50.0);
        assert_eq!(encoder.size(), 100);
        assert!(!encoder.has_previous());
    }

    #[test]
    fn test_first_value() {
        let encoder = DeltaEncoder::new(DeltaEncoderParams::default()).unwrap();

        // First value should encode delta of 0
        let sdr = encoder.encode_to_sdr(100.0).unwrap();
        assert_eq!(sdr.get_sum(), 10);

        assert!(encoder.has_previous());
    }

    #[test]
    fn test_positive_delta() {
        let encoder = DeltaEncoder::new(DeltaEncoderParams {
            max_delta: 100.0,
            size: 100,
            active_bits: 10,
            ..Default::default()
        })
        .unwrap();

        // First value (delta = 0)
        let sdr0 = encoder.encode_to_sdr(50.0).unwrap();

        // Second value (delta = +10)
        let sdr1 = encoder.encode_to_sdr(60.0).unwrap();

        // Delta 0 should be in middle, delta +10 should be to the right
        let pos0 = sdr0.get_sparse()[0];
        let pos1 = sdr1.get_sparse()[0];

        assert!(pos1 > pos0, "Positive delta should shift right");
    }

    #[test]
    fn test_negative_delta() {
        let encoder = DeltaEncoder::new(DeltaEncoderParams {
            max_delta: 100.0,
            size: 100,
            active_bits: 10,
            ..Default::default()
        })
        .unwrap();

        // First value (delta = 0)
        let sdr0 = encoder.encode_to_sdr(50.0).unwrap();

        // Second value (delta = -10)
        let sdr1 = encoder.encode_to_sdr(40.0).unwrap();

        // Delta 0 should be in middle, delta -10 should be to the left
        let pos0 = sdr0.get_sparse()[0];
        let pos1 = sdr1.get_sparse()[0];

        assert!(pos1 < pos0, "Negative delta should shift left");
    }

    #[test]
    fn test_reset() {
        let encoder = DeltaEncoder::new(DeltaEncoderParams::default()).unwrap();

        let _ = encoder.encode_to_sdr(100.0).unwrap();
        assert!(encoder.has_previous());

        encoder.reset();
        assert!(!encoder.has_previous());

        // After reset, first value should encode as delta 0
        let sdr = encoder.encode_to_sdr(200.0).unwrap();
        assert_eq!(sdr.get_sum(), 10);
    }

    #[test]
    fn test_clipping() {
        let encoder = DeltaEncoder::new(DeltaEncoderParams {
            max_delta: 10.0,
            size: 100,
            active_bits: 10,
            clip_input: true,
            ..Default::default()
        })
        .unwrap();

        // First value
        let _ = encoder.encode_to_sdr(0.0).unwrap();

        // Large delta should be clipped
        let sdr = encoder.encode_to_sdr(1000.0).unwrap();
        assert_eq!(sdr.get_sum(), 10);
    }

    #[test]
    fn test_encode_delta_directly() {
        let encoder = DeltaEncoder::new(DeltaEncoderParams::default()).unwrap();

        let mut sdr = Sdr::new(&[100]);
        encoder.encode_delta(5.0, &mut sdr).unwrap();

        assert_eq!(sdr.get_sum(), 10);
    }

    #[test]
    fn test_clone() {
        let encoder = DeltaEncoder::new(DeltaEncoderParams::default()).unwrap();

        let _ = encoder.encode_to_sdr(100.0).unwrap();

        let cloned = encoder.clone();
        assert!(cloned.has_previous());

        // Cloned encoder should have same state
        let sdr1 = encoder.encode_to_sdr(110.0).unwrap();
        encoder.reset();
        let _ = encoder.encode_to_sdr(100.0).unwrap();

        let sdr2 = cloned.encode_to_sdr(110.0).unwrap();

        assert_eq!(sdr1.get_sparse(), sdr2.get_sparse());
    }

    #[test]
    fn test_invalid_max_delta() {
        let result = DeltaEncoder::new(DeltaEncoderParams {
            max_delta: 0.0,
            ..Default::default()
        });
        assert!(result.is_err());

        let result = DeltaEncoder::new(DeltaEncoderParams {
            max_delta: -10.0,
            ..Default::default()
        });
        assert!(result.is_err());
    }
}
