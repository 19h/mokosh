//! Spectrogram Encoder implementation.
//!
//! Encodes frequency bin data from FFT/mel spectrograms into SDRs,
//! preserving frequency band relationships.

use crate::encoders::Encoder;
use crate::error::{MokoshError, Result};
use crate::types::{Real, Sdr, UInt};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Parameters for creating a Spectrogram Encoder.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SpectrogramEncoderParams {
    /// Number of frequency bins in input.
    pub num_bins: usize,

    /// Bits allocated per frequency bin.
    pub bits_per_bin: UInt,

    /// Number of active bits per bin when that bin is active.
    pub active_bits_per_bin: UInt,

    /// Minimum magnitude threshold (below this, bin is silent).
    pub min_magnitude: Real,

    /// Maximum expected magnitude for normalization.
    pub max_magnitude: Real,

    /// Whether to use log scale for magnitudes.
    pub log_scale: bool,
}

impl Default for SpectrogramEncoderParams {
    fn default() -> Self {
        Self {
            num_bins: 64,
            bits_per_bin: 20,
            active_bits_per_bin: 5,
            min_magnitude: 0.001,
            max_magnitude: 1.0,
            log_scale: true,
        }
    }
}

/// Encodes spectrogram frequency bins into SDR representations.
///
/// Each frequency bin gets a dedicated region in the output SDR.
/// The magnitude of each bin determines which bits within that region are active.
///
/// # Example
///
/// ```rust
/// use mokosh::encoders::{SpectrogramEncoder, SpectrogramEncoderParams, Encoder};
///
/// let encoder = SpectrogramEncoder::new(SpectrogramEncoderParams {
///     num_bins: 8,
///     bits_per_bin: 20,
///     active_bits_per_bin: 5,
///     ..Default::default()
/// }).unwrap();
///
/// // Encode a simple spectrum (8 frequency bins)
/// let spectrum = vec![0.1, 0.5, 0.9, 0.3, 0.0, 0.0, 0.2, 0.1];
/// let sdr = encoder.encode_to_sdr(spectrum).unwrap();
///
/// // Active bins contribute active_bits_per_bin each
/// assert!(sdr.get_sum() > 0);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SpectrogramEncoder {
    num_bins: usize,
    bits_per_bin: UInt,
    active_bits_per_bin: UInt,
    min_magnitude: Real,
    max_magnitude: Real,
    log_scale: bool,
    size: UInt,
    dimensions: Vec<UInt>,
}

impl SpectrogramEncoder {
    /// Creates a new Spectrogram Encoder.
    pub fn new(params: SpectrogramEncoderParams) -> Result<Self> {
        if params.num_bins == 0 {
            return Err(MokoshError::InvalidParameter {
                name: "num_bins",
                message: "Must be > 0".to_string(),
            });
        }

        if params.active_bits_per_bin > params.bits_per_bin {
            return Err(MokoshError::InvalidParameter {
                name: "active_bits_per_bin",
                message: "Cannot exceed bits_per_bin".to_string(),
            });
        }

        if params.max_magnitude <= params.min_magnitude {
            return Err(MokoshError::InvalidParameter {
                name: "max_magnitude",
                message: "Must be greater than min_magnitude".to_string(),
            });
        }

        let size = params.num_bins as UInt * params.bits_per_bin;

        Ok(Self {
            num_bins: params.num_bins,
            bits_per_bin: params.bits_per_bin,
            active_bits_per_bin: params.active_bits_per_bin,
            min_magnitude: params.min_magnitude,
            max_magnitude: params.max_magnitude,
            log_scale: params.log_scale,
            size,
            dimensions: vec![size],
        })
    }

    /// Returns the number of frequency bins.
    pub fn num_bins(&self) -> usize {
        self.num_bins
    }

    /// Normalizes a magnitude value to [0, 1] range.
    fn normalize_magnitude(&self, mag: Real) -> Real {
        if mag < self.min_magnitude {
            return 0.0;
        }

        let clamped = mag.clamp(self.min_magnitude, self.max_magnitude);

        if self.log_scale {
            let log_min = self.min_magnitude.ln();
            let log_max = self.max_magnitude.ln();
            let log_val = clamped.ln();
            (log_val - log_min) / (log_max - log_min)
        } else {
            (clamped - self.min_magnitude) / (self.max_magnitude - self.min_magnitude)
        }
    }
}

impl Encoder<Vec<Real>> for SpectrogramEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, spectrum: Vec<Real>, output: &mut Sdr) -> Result<()> {
        if spectrum.len() != self.num_bins {
            return Err(MokoshError::InvalidParameter {
                name: "spectrum",
                message: format!(
                    "Expected {} bins, got {}",
                    self.num_bins,
                    spectrum.len()
                ),
            });
        }

        if output.dimensions() != self.dimensions.as_slice() {
            return Err(MokoshError::DimensionMismatch {
                expected: self.dimensions.clone(),
                actual: output.dimensions().to_vec(),
            });
        }

        let mut sparse = Vec::new();

        for (bin_idx, &magnitude) in spectrum.iter().enumerate() {
            let normalized = self.normalize_magnitude(magnitude);

            if normalized > 0.0 {
                // Calculate which bits to activate within this bin's region
                let bin_offset = bin_idx as UInt * self.bits_per_bin;
                let num_positions = self.bits_per_bin - self.active_bits_per_bin + 1;
                let start_pos = (normalized * (num_positions - 1) as Real).round() as UInt;

                for i in 0..self.active_bits_per_bin {
                    sparse.push(bin_offset + start_pos + i);
                }
            }
        }

        sparse.sort_unstable();
        output.set_sparse_unchecked(sparse);

        Ok(())
    }
}

impl Encoder<&[Real]> for SpectrogramEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, spectrum: &[Real], output: &mut Sdr) -> Result<()> {
        self.encode(spectrum.to_vec(), output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_encoder() {
        let encoder = SpectrogramEncoder::new(SpectrogramEncoderParams {
            num_bins: 32,
            bits_per_bin: 16,
            active_bits_per_bin: 4,
            ..Default::default()
        })
        .unwrap();

        assert_eq!(encoder.num_bins(), 32);
        assert_eq!(Encoder::<Vec<Real>>::size(&encoder), 512);
    }

    #[test]
    fn test_encode_spectrum() {
        let encoder = SpectrogramEncoder::new(SpectrogramEncoderParams {
            num_bins: 4,
            bits_per_bin: 20,
            active_bits_per_bin: 5,
            min_magnitude: 0.01,
            max_magnitude: 1.0,
            log_scale: false,
        })
        .unwrap();

        let spectrum = vec![0.5, 0.0, 1.0, 0.25];
        let sdr = encoder.encode_to_sdr(spectrum).unwrap();

        // 3 active bins (0.0 is below threshold), 5 bits each = 15
        assert_eq!(sdr.get_sum(), 15);
    }

    #[test]
    fn test_silent_spectrum() {
        let encoder = SpectrogramEncoder::new(SpectrogramEncoderParams {
            num_bins: 4,
            bits_per_bin: 10,
            active_bits_per_bin: 3,
            min_magnitude: 0.1,
            ..Default::default()
        })
        .unwrap();

        // All below threshold
        let spectrum = vec![0.0, 0.05, 0.0, 0.0];
        let sdr = encoder.encode_to_sdr(spectrum).unwrap();

        assert_eq!(sdr.get_sum(), 0);
    }

    #[test]
    fn test_magnitude_position() {
        let encoder = SpectrogramEncoder::new(SpectrogramEncoderParams {
            num_bins: 1,
            bits_per_bin: 20,
            active_bits_per_bin: 5,
            min_magnitude: 0.0,
            max_magnitude: 1.0,
            log_scale: false,
        })
        .unwrap();

        let sdr_low = encoder.encode_to_sdr(vec![0.1]).unwrap();
        let sdr_high = encoder.encode_to_sdr(vec![0.9]).unwrap();

        // Low magnitude should have lower bit indices
        let low_min = *sdr_low.get_sparse().first().unwrap();
        let high_min = *sdr_high.get_sparse().first().unwrap();

        assert!(low_min < high_min);
    }

    #[test]
    fn test_wrong_bin_count() {
        let encoder = SpectrogramEncoder::new(SpectrogramEncoderParams {
            num_bins: 8,
            ..Default::default()
        })
        .unwrap();

        let result = encoder.encode_to_sdr(vec![0.5, 0.5]); // Wrong size
        assert!(result.is_err());
    }

    #[test]
    fn test_log_scale() {
        let encoder = SpectrogramEncoder::new(SpectrogramEncoderParams {
            num_bins: 1,
            bits_per_bin: 100,
            active_bits_per_bin: 10,
            min_magnitude: 0.001,
            max_magnitude: 1.0,
            log_scale: true,
        })
        .unwrap();

        // With log scale, 0.01 should be roughly 1/3 of the way (not 1/100)
        let sdr = encoder.encode_to_sdr(vec![0.01]).unwrap();
        let first_bit = *sdr.get_sparse().first().unwrap();

        // Should be somewhere in the lower-middle range, not at 0
        assert!(first_bit > 10);
        assert!(first_bit < 60);
    }
}
