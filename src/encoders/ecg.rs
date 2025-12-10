//! ECG Encoder implementation.
//!
//! Encodes ECG waveform morphology features into SDRs.

use crate::encoders::Encoder;
use crate::error::{MokoshError, Result};
use crate::types::{Real, Sdr, UInt};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Parameters for creating an ECG Encoder.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EcgEncoderParams {
    /// Number of samples in the input waveform window.
    pub window_size: usize,

    /// Bits per sample for amplitude encoding.
    pub bits_per_sample: UInt,

    /// Active bits per sample.
    pub active_per_sample: UInt,

    /// Additional bits for waveform statistics.
    pub stats_bits: UInt,

    /// Active bits for statistics.
    pub stats_active: UInt,

    /// Expected amplitude range (in mV typically).
    pub amplitude_range: (Real, Real),
}

impl Default for EcgEncoderParams {
    fn default() -> Self {
        Self {
            window_size: 50, // ~200ms at 250Hz
            bits_per_sample: 10,
            active_per_sample: 3,
            stats_bits: 60,
            stats_active: 15,
            amplitude_range: (-2.0, 2.0), // mV
        }
    }
}

/// ECG waveform statistics.
#[derive(Debug, Clone, Copy)]
pub struct EcgStats {
    /// Maximum amplitude in window.
    pub max_amplitude: Real,
    /// Minimum amplitude in window.
    pub min_amplitude: Real,
    /// Mean amplitude.
    pub mean: Real,
    /// Standard deviation.
    pub std_dev: Real,
    /// Number of zero crossings.
    pub zero_crossings: usize,
}

impl EcgStats {
    /// Computes statistics from waveform samples.
    pub fn from_samples(samples: &[Real]) -> Self {
        if samples.is_empty() {
            return Self {
                max_amplitude: 0.0,
                min_amplitude: 0.0,
                mean: 0.0,
                std_dev: 0.0,
                zero_crossings: 0,
            };
        }

        let n = samples.len() as Real;
        let mean = samples.iter().sum::<Real>() / n;

        let max_amplitude = samples.iter().cloned().fold(Real::MIN, Real::max);
        let min_amplitude = samples.iter().cloned().fold(Real::MAX, Real::min);

        let variance = samples.iter().map(|&x| (x - mean).powi(2)).sum::<Real>() / n;
        let std_dev = variance.sqrt();

        let mut zero_crossings = 0;
        for window in samples.windows(2) {
            if (window[0] >= 0.0) != (window[1] >= 0.0) {
                zero_crossings += 1;
            }
        }

        Self {
            max_amplitude,
            min_amplitude,
            mean,
            std_dev,
            zero_crossings,
        }
    }
}

/// Encodes ECG waveforms into SDR representations.
///
/// Captures both the raw waveform morphology and derived statistics
/// relevant for ECG analysis.
///
/// # Example
///
/// ```rust
/// use mokosh::encoders::{EcgEncoder, EcgEncoderParams, Encoder};
///
/// let encoder = EcgEncoder::new(EcgEncoderParams {
///     window_size: 10,
///     bits_per_sample: 10,
///     active_per_sample: 3,
///     stats_bits: 40,
///     stats_active: 10,
///     ..Default::default()
/// }).unwrap();
///
/// // Simulate a simple ECG-like waveform
/// let samples: Vec<f32> = (0..10)
///     .map(|i| (i as f32 * 0.5).sin())
///     .collect();
///
/// let sdr = encoder.encode_to_sdr(samples).unwrap();
/// assert!(sdr.get_sum() > 0);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EcgEncoder {
    window_size: usize,
    bits_per_sample: UInt,
    active_per_sample: UInt,
    stats_bits: UInt,
    stats_active: UInt,
    amplitude_range: (Real, Real),
    waveform_size: UInt,
    size: UInt,
    dimensions: Vec<UInt>,
}

impl EcgEncoder {
    /// Creates a new ECG Encoder.
    pub fn new(params: EcgEncoderParams) -> Result<Self> {
        if params.window_size == 0 {
            return Err(MokoshError::InvalidParameter {
                name: "window_size",
                message: "Must be > 0".to_string(),
            });
        }

        if params.active_per_sample > params.bits_per_sample {
            return Err(MokoshError::InvalidParameter {
                name: "active_per_sample",
                message: "Cannot exceed bits_per_sample".to_string(),
            });
        }

        if params.stats_active > params.stats_bits {
            return Err(MokoshError::InvalidParameter {
                name: "stats_active",
                message: "Cannot exceed stats_bits".to_string(),
            });
        }

        let waveform_size = params.window_size as UInt * params.bits_per_sample;
        let size = waveform_size + params.stats_bits;

        Ok(Self {
            window_size: params.window_size,
            bits_per_sample: params.bits_per_sample,
            active_per_sample: params.active_per_sample,
            stats_bits: params.stats_bits,
            stats_active: params.stats_active,
            amplitude_range: params.amplitude_range,
            waveform_size,
            size,
            dimensions: vec![size],
        })
    }

    /// Returns the window size.
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Normalizes an amplitude value.
    fn normalize_amplitude(&self, value: Real) -> Real {
        let (min, max) = self.amplitude_range;
        ((value - min) / (max - min)).clamp(0.0, 1.0)
    }
}

impl Encoder<Vec<Real>> for EcgEncoder {
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

        let mut sparse = Vec::new();

        // Encode waveform samples
        for (idx, &sample) in samples.iter().enumerate() {
            let offset = idx as UInt * self.bits_per_sample;
            let normalized = self.normalize_amplitude(sample);
            let positions = self.bits_per_sample - self.active_per_sample + 1;
            let start = (normalized * (positions - 1) as Real).round() as UInt;

            for i in 0..self.active_per_sample {
                sparse.push(offset + start + i);
            }
        }

        // Encode statistics
        let stats = EcgStats::from_samples(&samples);
        let stats_offset = self.waveform_size;

        // We'll encode 5 statistics, each getting stats_bits/5 bits
        let bits_per_stat = self.stats_bits / 5;
        let active_per_stat = self.stats_active / 5;

        // Max amplitude
        let norm_max = self.normalize_amplitude(stats.max_amplitude);
        let positions = bits_per_stat - active_per_stat + 1;
        let start = (norm_max * (positions - 1) as Real).round() as UInt;
        for i in 0..active_per_stat {
            sparse.push(stats_offset + start + i);
        }

        // Min amplitude
        let norm_min = self.normalize_amplitude(stats.min_amplitude);
        let start = (norm_min * (positions - 1) as Real).round() as UInt;
        for i in 0..active_per_stat {
            sparse.push(stats_offset + bits_per_stat + start + i);
        }

        // Mean
        let norm_mean = self.normalize_amplitude(stats.mean);
        let start = (norm_mean * (positions - 1) as Real).round() as UInt;
        for i in 0..active_per_stat {
            sparse.push(stats_offset + 2 * bits_per_stat + start + i);
        }

        // Std dev (normalized to amplitude range)
        let range = self.amplitude_range.1 - self.amplitude_range.0;
        let norm_std = (stats.std_dev / range).clamp(0.0, 1.0);
        let start = (norm_std * (positions - 1) as Real).round() as UInt;
        for i in 0..active_per_stat {
            sparse.push(stats_offset + 3 * bits_per_stat + start + i);
        }

        // Zero crossings (normalized to window size)
        let norm_zc = (stats.zero_crossings as Real / self.window_size as Real).clamp(0.0, 1.0);
        let start = (norm_zc * (positions - 1) as Real).round() as UInt;
        for i in 0..active_per_stat {
            sparse.push(stats_offset + 4 * bits_per_stat + start + i);
        }

        sparse.sort_unstable();
        output.set_sparse_unchecked(sparse);

        Ok(())
    }
}

impl Encoder<&[Real]> for EcgEncoder {
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
        let encoder = EcgEncoder::new(EcgEncoderParams {
            window_size: 50,
            bits_per_sample: 10,
            active_per_sample: 3,
            stats_bits: 50,
            stats_active: 10,
            ..Default::default()
        })
        .unwrap();

        assert_eq!(encoder.window_size(), 50);
        // 50 * 10 + 50 = 550
        assert_eq!(Encoder::<Vec<Real>>::size(&encoder), 550);
    }

    #[test]
    fn test_ecg_stats() {
        let samples = vec![-1.0, 0.5, 1.0, 0.0, -0.5];
        let stats = EcgStats::from_samples(&samples);

        assert!((stats.max_amplitude - 1.0).abs() < 0.01);
        assert!((stats.min_amplitude - (-1.0)).abs() < 0.01);
        assert!((stats.mean - 0.0).abs() < 0.01);
        assert!(stats.std_dev > 0.0);
        assert!(stats.zero_crossings >= 2);
    }

    #[test]
    fn test_encode_ecg() {
        let encoder = EcgEncoder::new(EcgEncoderParams {
            window_size: 10,
            bits_per_sample: 10,
            active_per_sample: 3,
            stats_bits: 50,
            stats_active: 10,
            ..Default::default()
        })
        .unwrap();

        let samples: Vec<Real> = (0..10).map(|i| (i as Real * 0.3).sin()).collect();
        let sdr = encoder.encode_to_sdr(samples).unwrap();

        // 10 samples * 3 bits + 10 stats bits = 40
        assert_eq!(sdr.get_sum(), 40);
    }

    #[test]
    fn test_similar_waveforms_overlap() {
        let encoder = EcgEncoder::new(EcgEncoderParams {
            window_size: 20,
            bits_per_sample: 10,
            active_per_sample: 3,
            stats_bits: 40,
            stats_active: 8,
            ..Default::default()
        })
        .unwrap();

        // Two similar waveforms
        let wave1: Vec<Real> = (0..20).map(|i| (i as Real * 0.2).sin()).collect();
        let wave2: Vec<Real> = (0..20).map(|i| (i as Real * 0.2).sin() + 0.05).collect();

        // Different waveform
        let wave3: Vec<Real> = (0..20).map(|i| (i as Real * 0.5).cos()).collect();

        let sdr1 = encoder.encode_to_sdr(wave1).unwrap();
        let sdr2 = encoder.encode_to_sdr(wave2).unwrap();
        let sdr3 = encoder.encode_to_sdr(wave3).unwrap();

        let similar_overlap = sdr1.get_overlap(&sdr2);
        let different_overlap = sdr1.get_overlap(&sdr3);

        assert!(similar_overlap > different_overlap);
    }

    #[test]
    fn test_wrong_window_size() {
        let encoder = EcgEncoder::new(EcgEncoderParams {
            window_size: 10,
            ..Default::default()
        })
        .unwrap();

        let result = encoder.encode_to_sdr(vec![0.0, 0.0, 0.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_deterministic() {
        let encoder = EcgEncoder::new(EcgEncoderParams::default()).unwrap();

        let samples: Vec<Real> = (0..50).map(|i| (i as Real * 0.1).sin()).collect();

        let sdr1 = encoder.encode_to_sdr(samples.clone()).unwrap();
        let sdr2 = encoder.encode_to_sdr(samples).unwrap();

        assert_eq!(sdr1.get_sparse(), sdr2.get_sparse());
    }
}
