//! Distribution Encoder implementation.
//!
//! Encodes probability distributions (not just point values) into SDRs.

use crate::encoders::Encoder;
use crate::error::{MokoshError, Result};
use crate::types::{Real, Sdr, UInt};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Parameters for creating a Distribution Encoder.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DistributionEncoderParams {
    /// Number of bins for discretizing the distribution.
    pub num_bins: usize,

    /// Bits per bin.
    pub bits_per_bin: UInt,

    /// Minimum probability threshold for a bin to be active.
    pub min_probability: Real,

    /// Value range (min, max).
    pub value_range: (Real, Real),
}

impl Default for DistributionEncoderParams {
    fn default() -> Self {
        Self {
            num_bins: 50,
            bits_per_bin: 10,
            min_probability: 0.01, // 1%
            value_range: (0.0, 100.0),
        }
    }
}

/// A discrete probability distribution.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Distribution {
    /// Probabilities for each bin (should sum to ~1.0).
    pub probabilities: Vec<Real>,
}

impl Distribution {
    /// Creates a distribution from raw probabilities.
    pub fn new(probabilities: Vec<Real>) -> Self {
        Self { probabilities }
    }

    /// Creates a uniform distribution.
    pub fn uniform(num_bins: usize) -> Self {
        let p = 1.0 / num_bins as Real;
        Self {
            probabilities: vec![p; num_bins],
        }
    }

    /// Creates a Gaussian distribution.
    pub fn gaussian(num_bins: usize, mean_bin: Real, std_bins: Real) -> Self {
        let mut probs = Vec::with_capacity(num_bins);
        let mut total = 0.0;

        for i in 0..num_bins {
            let x = i as Real;
            let z = (x - mean_bin) / std_bins;
            let p = (-0.5 * z * z).exp();
            probs.push(p);
            total += p;
        }

        // Normalize
        for p in probs.iter_mut() {
            *p /= total;
        }

        Self { probabilities: probs }
    }

    /// Creates a distribution from a point estimate with uncertainty.
    pub fn from_point_with_uncertainty(
        num_bins: usize,
        value: Real,
        value_range: (Real, Real),
        uncertainty: Real, // as fraction of range
    ) -> Self {
        let range = value_range.1 - value_range.0;
        let normalized = (value - value_range.0) / range;
        let mean_bin = normalized * num_bins as Real;
        let std_bins = uncertainty * num_bins as Real;

        Self::gaussian(num_bins, mean_bin, std_bins.max(0.5))
    }

    /// Returns the entropy of the distribution.
    pub fn entropy(&self) -> Real {
        let mut h = 0.0;
        for &p in &self.probabilities {
            if p > 0.0 {
                h -= p * p.ln();
            }
        }
        h
    }

    /// Returns the bin with maximum probability.
    pub fn mode(&self) -> usize {
        self.probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Returns the expected value (mean bin index).
    pub fn mean(&self) -> Real {
        self.probabilities
            .iter()
            .enumerate()
            .map(|(i, &p)| i as Real * p)
            .sum()
    }
}

/// Encodes probability distributions into SDR representations.
///
/// Unlike point encoders, this captures the full shape of uncertainty.
/// More certain (peaked) distributions have sparser encodings.
///
/// # Example
///
/// ```rust
/// use mokosh::encoders::{DistributionEncoder, DistributionEncoderParams, Distribution, Encoder};
///
/// let encoder = DistributionEncoder::new(DistributionEncoderParams {
///     num_bins: 20,
///     bits_per_bin: 5,
///     min_probability: 0.05,
///     value_range: (0.0, 100.0),
/// }).unwrap();
///
/// // A peaked distribution (high certainty)
/// let certain = Distribution::gaussian(20, 10.0, 1.0);
///
/// // A broad distribution (low certainty)
/// let uncertain = Distribution::gaussian(20, 10.0, 5.0);
///
/// let sdr_certain = encoder.encode_to_sdr(certain).unwrap();
/// let sdr_uncertain = encoder.encode_to_sdr(uncertain).unwrap();
///
/// // Certain distribution has fewer active bits
/// assert!(sdr_certain.get_sum() < sdr_uncertain.get_sum());
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DistributionEncoder {
    num_bins: usize,
    bits_per_bin: UInt,
    min_probability: Real,
    value_range: (Real, Real),
    size: UInt,
    dimensions: Vec<UInt>,
}

impl DistributionEncoder {
    /// Creates a new Distribution Encoder.
    pub fn new(params: DistributionEncoderParams) -> Result<Self> {
        if params.num_bins == 0 {
            return Err(MokoshError::InvalidParameter {
                name: "num_bins",
                message: "Must be > 0".to_string(),
            });
        }

        if params.bits_per_bin == 0 {
            return Err(MokoshError::InvalidParameter {
                name: "bits_per_bin",
                message: "Must be > 0".to_string(),
            });
        }

        if params.min_probability < 0.0 || params.min_probability > 1.0 {
            return Err(MokoshError::InvalidParameter {
                name: "min_probability",
                message: "Must be in [0, 1]".to_string(),
            });
        }

        let size = params.num_bins as UInt * params.bits_per_bin;

        Ok(Self {
            num_bins: params.num_bins,
            bits_per_bin: params.bits_per_bin,
            min_probability: params.min_probability,
            value_range: params.value_range,
            size,
            dimensions: vec![size],
        })
    }

    /// Returns the number of bins.
    pub fn num_bins(&self) -> usize {
        self.num_bins
    }

    /// Returns the value range.
    pub fn value_range(&self) -> (Real, Real) {
        self.value_range
    }
}

impl Encoder<Distribution> for DistributionEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, dist: Distribution, output: &mut Sdr) -> Result<()> {
        if dist.probabilities.len() != self.num_bins {
            return Err(MokoshError::InvalidParameter {
                name: "distribution",
                message: format!(
                    "Expected {} bins, got {}",
                    self.num_bins,
                    dist.probabilities.len()
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

        // Find max probability for scaling
        let max_prob = dist
            .probabilities
            .iter()
            .cloned()
            .fold(0.0, Real::max);

        if max_prob == 0.0 {
            // Empty distribution
            output.set_sparse_unchecked(sparse);
            return Ok(());
        }

        for (bin, &prob) in dist.probabilities.iter().enumerate() {
            if prob >= self.min_probability {
                let bin_offset = bin as UInt * self.bits_per_bin;

                // Number of active bits proportional to probability
                let normalized = prob / max_prob;
                let active = (normalized * self.bits_per_bin as Real).round() as UInt;
                let active = active.max(1).min(self.bits_per_bin);

                // Center the active bits in the bin
                let start = (self.bits_per_bin - active) / 2;

                for i in 0..active {
                    sparse.push(bin_offset + start + i);
                }
            }
        }

        sparse.sort_unstable();
        output.set_sparse_unchecked(sparse);

        Ok(())
    }
}

impl Encoder<Vec<Real>> for DistributionEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, probabilities: Vec<Real>, output: &mut Sdr) -> Result<()> {
        self.encode(Distribution::new(probabilities), output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_encoder() {
        let encoder = DistributionEncoder::new(DistributionEncoderParams {
            num_bins: 20,
            bits_per_bin: 10,
            ..Default::default()
        })
        .unwrap();

        assert_eq!(encoder.num_bins(), 20);
        assert_eq!(Encoder::<Distribution>::size(&encoder), 200);
    }

    #[test]
    fn test_uniform_distribution() {
        let dist = Distribution::uniform(10);
        assert_eq!(dist.probabilities.len(), 10);

        let sum: Real = dist.probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_gaussian_distribution() {
        let dist = Distribution::gaussian(20, 10.0, 2.0);

        // Should be centered around bin 10
        assert_eq!(dist.mode(), 10);

        let sum: Real = dist.probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_entropy() {
        // Uniform has high entropy
        let uniform = Distribution::uniform(10);

        // Peaked has low entropy
        let mut peaked = vec![0.0; 10];
        peaked[5] = 1.0;
        let peaked = Distribution::new(peaked);

        assert!(uniform.entropy() > peaked.entropy());
    }

    #[test]
    fn test_encode_distribution() {
        let encoder = DistributionEncoder::new(DistributionEncoderParams {
            num_bins: 10,
            bits_per_bin: 5,
            min_probability: 0.05,
            ..Default::default()
        })
        .unwrap();

        let dist = Distribution::gaussian(10, 5.0, 1.5);
        let sdr = encoder.encode_to_sdr(dist).unwrap();

        assert!(sdr.get_sum() > 0);
    }

    #[test]
    fn test_certain_vs_uncertain() {
        let encoder = DistributionEncoder::new(DistributionEncoderParams {
            num_bins: 20,
            bits_per_bin: 5,
            min_probability: 0.02,
            ..Default::default()
        })
        .unwrap();

        // Peaked (certain)
        let certain = Distribution::gaussian(20, 10.0, 1.0);
        let sdr_certain = encoder.encode_to_sdr(certain).unwrap();

        // Broad (uncertain)
        let uncertain = Distribution::gaussian(20, 10.0, 5.0);
        let sdr_uncertain = encoder.encode_to_sdr(uncertain).unwrap();

        // Certain should have fewer active bits (fewer bins above threshold)
        assert!(sdr_certain.get_sum() < sdr_uncertain.get_sum());
    }

    #[test]
    fn test_similar_distributions_overlap() {
        let encoder = DistributionEncoder::new(DistributionEncoderParams {
            num_bins: 20,
            bits_per_bin: 5,
            min_probability: 0.02,
            ..Default::default()
        })
        .unwrap();

        let dist1 = Distribution::gaussian(20, 10.0, 2.0);
        let dist2 = Distribution::gaussian(20, 11.0, 2.0); // Slightly shifted
        let dist3 = Distribution::gaussian(20, 18.0, 2.0); // Very different

        let sdr1 = encoder.encode_to_sdr(dist1).unwrap();
        let sdr2 = encoder.encode_to_sdr(dist2).unwrap();
        let sdr3 = encoder.encode_to_sdr(dist3).unwrap();

        let near_overlap = sdr1.get_overlap(&sdr2);
        let far_overlap = sdr1.get_overlap(&sdr3);

        assert!(near_overlap > far_overlap);
    }

    #[test]
    fn test_encode_vec() {
        let encoder = DistributionEncoder::new(DistributionEncoderParams {
            num_bins: 5,
            bits_per_bin: 4,
            min_probability: 0.1,
            ..Default::default()
        })
        .unwrap();

        let probs = vec![0.1, 0.2, 0.4, 0.2, 0.1];
        let sdr = encoder.encode_to_sdr(probs).unwrap();

        assert!(sdr.get_sum() > 0);
    }

    #[test]
    fn test_wrong_bin_count() {
        let encoder = DistributionEncoder::new(DistributionEncoderParams {
            num_bins: 10,
            ..Default::default()
        })
        .unwrap();

        let result = encoder.encode_to_sdr(Distribution::uniform(5));
        assert!(result.is_err());
    }

    #[test]
    fn test_deterministic() {
        let encoder = DistributionEncoder::new(DistributionEncoderParams::default()).unwrap();

        let dist = Distribution::gaussian(50, 25.0, 5.0);

        let sdr1 = encoder.encode_to_sdr(dist.clone()).unwrap();
        let sdr2 = encoder.encode_to_sdr(dist).unwrap();

        assert_eq!(sdr1.get_sparse(), sdr2.get_sparse());
    }
}
