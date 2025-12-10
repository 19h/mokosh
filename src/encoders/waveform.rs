//! Waveform Encoder implementation.
//!
//! Encodes raw audio waveform samples into SDRs, preserving
//! temporal structure and amplitude information.

use crate::encoders::Encoder;
use crate::error::{MokoshError, Result};
use crate::types::{Real, Sdr, UInt};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Parameters for creating a Waveform Encoder.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct WaveformEncoderParams {
    /// Number of samples in the input window.
    pub window_size: usize,

    /// Bits allocated per sample.
    pub bits_per_sample: UInt,

    /// Number of active bits per sample.
    pub active_bits_per_sample: UInt,

    /// Minimum sample value (typically -1.0 for normalized audio).
    pub min_value: Real,

    /// Maximum sample value (typically 1.0 for normalized audio).
    pub max_value: Real,
}

impl Default for WaveformEncoderParams {
    fn default() -> Self {
        Self {
            window_size: 64,
            bits_per_sample: 16,
            active_bits_per_sample: 4,
            min_value: -1.0,
            max_value: 1.0,
        }
    }
}

/// Encodes raw waveform samples into SDR representations.
///
/// Each sample in the window gets a dedicated region in the output.
/// The amplitude determines the position of active bits within that region.
///
/// # Example
///
/// ```rust
/// use mokosh::encoders::{WaveformEncoder, WaveformEncoderParams, Encoder};
///
/// let encoder = WaveformEncoder::new(WaveformEncoderParams {
///     window_size: 8,
///     bits_per_sample: 16,
///     active_bits_per_sample: 4,
///     ..Default::default()
/// }).unwrap();
///
/// // Encode a simple sine-like pattern
/// let samples = vec![0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5];
/// let sdr = encoder.encode_to_sdr(samples).unwrap();
///
/// assert_eq!(sdr.get_sum(), 32); // 8 samples * 4 bits each
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct WaveformEncoder {
    window_size: usize,
    bits_per_sample: UInt,
    active_bits_per_sample: UInt,
    min_value: Real,
    max_value: Real,
    value_range: Real,
    size: UInt,
    dimensions: Vec<UInt>,
}

impl WaveformEncoder {
    /// Creates a new Waveform Encoder.
    pub fn new(params: WaveformEncoderParams) -> Result<Self> {
        if params.window_size == 0 {
            return Err(MokoshError::InvalidParameter {
                name: "window_size",
                message: "Must be > 0".to_string(),
            });
        }

        if params.active_bits_per_sample > params.bits_per_sample {
            return Err(MokoshError::InvalidParameter {
                name: "active_bits_per_sample",
                message: "Cannot exceed bits_per_sample".to_string(),
            });
        }

        if params.max_value <= params.min_value {
            return Err(MokoshError::InvalidParameter {
                name: "max_value",
                message: "Must be greater than min_value".to_string(),
            });
        }

        let size = params.window_size as UInt * params.bits_per_sample;

        Ok(Self {
            window_size: params.window_size,
            bits_per_sample: params.bits_per_sample,
            active_bits_per_sample: params.active_bits_per_sample,
            min_value: params.min_value,
            max_value: params.max_value,
            value_range: params.max_value - params.min_value,
            size,
            dimensions: vec![size],
        })
    }

    /// Returns the window size.
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Normalizes a sample value to [0, 1] range.
    fn normalize(&self, value: Real) -> Real {
        let clamped = value.clamp(self.min_value, self.max_value);
        (clamped - self.min_value) / self.value_range
    }
}

impl Encoder<Vec<Real>> for WaveformEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, samples: Vec<Real>, output: &mut Sdr) -> Result<()> {
        if samples.len() != self.window_size {
            return Err(MokoshError::InvalidParameter {
                name: "samples",
                message: format!(
                    "Expected {} samples, got {}",
                    self.window_size,
                    samples.len()
                ),
            });
        }

        if output.dimensions() != self.dimensions.as_slice() {
            return Err(MokoshError::DimensionMismatch {
                expected: self.dimensions.clone(),
                actual: output.dimensions().to_vec(),
            });
        }

        let mut sparse = Vec::with_capacity(self.window_size * self.active_bits_per_sample as usize);

        for (sample_idx, &value) in samples.iter().enumerate() {
            let normalized = self.normalize(value);
            let sample_offset = sample_idx as UInt * self.bits_per_sample;

            // Calculate starting position within this sample's region
            let num_positions = self.bits_per_sample - self.active_bits_per_sample + 1;
            let start_pos = (normalized * (num_positions - 1) as Real).round() as UInt;

            for i in 0..self.active_bits_per_sample {
                sparse.push(sample_offset + start_pos + i);
            }
        }

        sparse.sort_unstable();
        output.set_sparse_unchecked(sparse);

        Ok(())
    }
}

impl Encoder<&[Real]> for WaveformEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, samples: &[Real], output: &mut Sdr) -> Result<()> {
        self.encode(samples.to_vec(), output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_encoder() {
        let encoder = WaveformEncoder::new(WaveformEncoderParams {
            window_size: 32,
            bits_per_sample: 20,
            active_bits_per_sample: 5,
            ..Default::default()
        })
        .unwrap();

        assert_eq!(encoder.window_size(), 32);
        assert_eq!(Encoder::<Vec<Real>>::size(&encoder), 640);
    }

    #[test]
    fn test_encode_waveform() {
        let encoder = WaveformEncoder::new(WaveformEncoderParams {
            window_size: 4,
            bits_per_sample: 16,
            active_bits_per_sample: 4,
            ..Default::default()
        })
        .unwrap();

        let samples = vec![0.0, 0.5, -0.5, 1.0];
        let sdr = encoder.encode_to_sdr(samples).unwrap();

        assert_eq!(sdr.get_sum(), 16); // 4 samples * 4 bits
    }

    #[test]
    fn test_amplitude_position() {
        let encoder = WaveformEncoder::new(WaveformEncoderParams {
            window_size: 1,
            bits_per_sample: 20,
            active_bits_per_sample: 4,
            min_value: -1.0,
            max_value: 1.0,
        })
        .unwrap();

        let sdr_min = encoder.encode_to_sdr(vec![-1.0]).unwrap();
        let sdr_zero = encoder.encode_to_sdr(vec![0.0]).unwrap();
        let sdr_max = encoder.encode_to_sdr(vec![1.0]).unwrap();

        let min_start = *sdr_min.get_sparse().first().unwrap();
        let zero_start = *sdr_zero.get_sparse().first().unwrap();
        let max_start = *sdr_max.get_sparse().first().unwrap();

        assert!(min_start < zero_start);
        assert!(zero_start < max_start);
    }

    #[test]
    fn test_similar_waveforms_overlap() {
        let encoder = WaveformEncoder::new(WaveformEncoderParams {
            window_size: 4,
            bits_per_sample: 20,
            active_bits_per_sample: 5,
            ..Default::default()
        })
        .unwrap();

        let wave1 = vec![0.0, 0.5, 1.0, 0.5];
        let wave2 = vec![0.0, 0.5, 0.95, 0.5]; // Slightly different
        let wave3 = vec![-1.0, -0.5, 0.0, 0.5]; // Very different

        let sdr1 = encoder.encode_to_sdr(wave1).unwrap();
        let sdr2 = encoder.encode_to_sdr(wave2).unwrap();
        let sdr3 = encoder.encode_to_sdr(wave3).unwrap();

        let similar_overlap = sdr1.get_overlap(&sdr2);
        let different_overlap = sdr1.get_overlap(&sdr3);

        assert!(similar_overlap > different_overlap);
    }

    #[test]
    fn test_wrong_window_size() {
        let encoder = WaveformEncoder::new(WaveformEncoderParams {
            window_size: 8,
            ..Default::default()
        })
        .unwrap();

        let result = encoder.encode_to_sdr(vec![0.0, 0.0, 0.0]); // Wrong size
        assert!(result.is_err());
    }
}
