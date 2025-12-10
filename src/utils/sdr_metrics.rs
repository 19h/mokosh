//! SDR metrics and analysis utilities.
//!
//! This module provides tools for analyzing SDR properties like sparsity,
//! overlap, and activation frequency.

use crate::types::{Real, Sdr, UInt};

/// Metrics for analyzing SDR properties over time.
#[derive(Debug, Clone)]
pub struct SdrMetrics {
    /// Number of samples seen.
    num_samples: usize,

    /// Running sum of sparsity values.
    sparsity_sum: f64,

    /// Running sum of squared sparsity values (for variance).
    sparsity_sq_sum: f64,

    /// Min and max sparsity seen.
    sparsity_min: f64,
    sparsity_max: f64,

    /// Activation counts per bit.
    activation_counts: Vec<usize>,

    /// Expected dimensions for SDRs.
    dimensions: Vec<UInt>,

    /// Size of the SDR.
    size: usize,

    /// Previous SDR for overlap computation.
    previous: Option<Sdr>,

    /// Running overlap statistics.
    overlap_sum: usize,
    overlap_min: usize,
    overlap_max: usize,
}

impl SdrMetrics {
    /// Creates a new metrics tracker for SDRs of the given dimensions.
    #[must_use]
    pub fn new(dimensions: &[UInt]) -> Self {
        let size: usize = dimensions.iter().map(|&d| d as usize).product();

        Self {
            num_samples: 0,
            sparsity_sum: 0.0,
            sparsity_sq_sum: 0.0,
            sparsity_min: f64::MAX,
            sparsity_max: f64::MIN,
            activation_counts: vec![0; size],
            dimensions: dimensions.to_vec(),
            size,
            previous: None,
            overlap_sum: 0,
            overlap_min: usize::MAX,
            overlap_max: 0,
        }
    }

    /// Adds an SDR observation.
    ///
    /// # Panics
    ///
    /// Panics if the SDR dimensions don't match.
    pub fn add_sample(&mut self, sdr: &Sdr) {
        assert_eq!(
            sdr.dimensions(),
            self.dimensions.as_slice(),
            "SDR dimensions mismatch"
        );

        self.num_samples += 1;

        // Update sparsity statistics
        let sparsity = sdr.get_sparsity() as f64;
        self.sparsity_sum += sparsity;
        self.sparsity_sq_sum += sparsity * sparsity;
        self.sparsity_min = self.sparsity_min.min(sparsity);
        self.sparsity_max = self.sparsity_max.max(sparsity);

        // Update activation counts
        for &idx in &sdr.get_sparse() {
            self.activation_counts[idx as usize] += 1;
        }

        // Update overlap statistics
        if let Some(ref prev) = self.previous {
            let overlap = sdr.get_overlap(prev);
            self.overlap_sum += overlap;
            self.overlap_min = self.overlap_min.min(overlap);
            self.overlap_max = self.overlap_max.max(overlap);
        }

        self.previous = Some(sdr.clone());
    }

    /// Returns the number of samples observed.
    #[must_use]
    pub fn num_samples(&self) -> usize {
        self.num_samples
    }

    /// Returns the mean sparsity.
    #[must_use]
    pub fn mean_sparsity(&self) -> Real {
        if self.num_samples == 0 {
            return 0.0;
        }
        (self.sparsity_sum / self.num_samples as f64) as Real
    }

    /// Returns the standard deviation of sparsity.
    #[must_use]
    pub fn std_sparsity(&self) -> Real {
        if self.num_samples < 2 {
            return 0.0;
        }
        let mean = self.sparsity_sum / self.num_samples as f64;
        let variance = (self.sparsity_sq_sum / self.num_samples as f64) - (mean * mean);
        variance.max(0.0).sqrt() as Real
    }

    /// Returns the minimum sparsity observed.
    #[must_use]
    pub fn min_sparsity(&self) -> Real {
        if self.num_samples == 0 {
            return 0.0;
        }
        self.sparsity_min as Real
    }

    /// Returns the maximum sparsity observed.
    #[must_use]
    pub fn max_sparsity(&self) -> Real {
        if self.num_samples == 0 {
            return 0.0;
        }
        self.sparsity_max as Real
    }

    /// Returns the mean overlap between consecutive SDRs.
    #[must_use]
    pub fn mean_overlap(&self) -> Real {
        if self.num_samples < 2 {
            return 0.0;
        }
        self.overlap_sum as Real / (self.num_samples - 1) as Real
    }

    /// Returns the minimum overlap observed.
    #[must_use]
    pub fn min_overlap(&self) -> usize {
        if self.num_samples < 2 {
            return 0;
        }
        self.overlap_min
    }

    /// Returns the maximum overlap observed.
    #[must_use]
    pub fn max_overlap(&self) -> usize {
        if self.num_samples < 2 {
            return 0;
        }
        self.overlap_max
    }

    /// Returns the activation frequency for each bit.
    #[must_use]
    pub fn activation_frequencies(&self) -> Vec<Real> {
        if self.num_samples == 0 {
            return vec![0.0; self.size];
        }
        self.activation_counts
            .iter()
            .map(|&count| count as Real / self.num_samples as Real)
            .collect()
    }

    /// Returns the entropy of the activation distribution.
    ///
    /// Higher entropy indicates more uniform activation across all bits.
    #[must_use]
    pub fn activation_entropy(&self) -> Real {
        if self.num_samples == 0 {
            return 0.0;
        }

        let total: usize = self.activation_counts.iter().sum();
        if total == 0 {
            return 0.0;
        }

        let mut entropy = 0.0;
        for &count in &self.activation_counts {
            if count > 0 {
                let p = count as f64 / total as f64;
                entropy -= p * p.log2();
            }
        }

        entropy as Real
    }

    /// Returns the number of bits that have never been active.
    #[must_use]
    pub fn num_dead_bits(&self) -> usize {
        self.activation_counts.iter().filter(|&&c| c == 0).count()
    }

    /// Returns the fraction of bits that have never been active.
    #[must_use]
    pub fn dead_bit_ratio(&self) -> Real {
        self.num_dead_bits() as Real / self.size as Real
    }

    /// Resets all statistics.
    pub fn reset(&mut self) {
        self.num_samples = 0;
        self.sparsity_sum = 0.0;
        self.sparsity_sq_sum = 0.0;
        self.sparsity_min = f64::MAX;
        self.sparsity_max = f64::MIN;
        self.activation_counts.fill(0);
        self.previous = None;
        self.overlap_sum = 0;
        self.overlap_min = usize::MAX;
        self.overlap_max = 0;
    }
}

impl std::fmt::Display for SdrMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "SDR Metrics ({} samples):", self.num_samples)?;
        writeln!(
            f,
            "  Sparsity: mean={:.4}, std={:.4}, min={:.4}, max={:.4}",
            self.mean_sparsity(),
            self.std_sparsity(),
            self.min_sparsity(),
            self.max_sparsity()
        )?;
        writeln!(
            f,
            "  Overlap: mean={:.2}, min={}, max={}",
            self.mean_overlap(),
            self.min_overlap(),
            self.max_overlap()
        )?;
        writeln!(
            f,
            "  Activation: entropy={:.4}, dead_bits={} ({:.2}%)",
            self.activation_entropy(),
            self.num_dead_bits(),
            self.dead_bit_ratio() * 100.0
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_metrics() {
        let mut metrics = SdrMetrics::new(&[100]);

        let mut sdr = Sdr::new(&[100]);
        sdr.set_sparse(&[0, 1, 2, 3, 4]).unwrap();
        metrics.add_sample(&sdr);

        assert_eq!(metrics.num_samples(), 1);
        assert!((metrics.mean_sparsity() - 0.05).abs() < 0.001);
    }

    #[test]
    fn test_overlap_metrics() {
        let mut metrics = SdrMetrics::new(&[100]);

        let mut sdr1 = Sdr::new(&[100]);
        let mut sdr2 = Sdr::new(&[100]);

        sdr1.set_sparse(&[0, 1, 2, 3, 4]).unwrap();
        sdr2.set_sparse(&[2, 3, 4, 5, 6]).unwrap();

        metrics.add_sample(&sdr1);
        metrics.add_sample(&sdr2);

        assert_eq!(metrics.num_samples(), 2);
        assert_eq!(metrics.mean_overlap(), 3.0);
    }

    #[test]
    fn test_activation_frequency() {
        let mut metrics = SdrMetrics::new(&[10]);

        let mut sdr = Sdr::new(&[10]);
        sdr.set_sparse(&[0, 1]).unwrap();
        metrics.add_sample(&sdr);

        sdr.set_sparse(&[0, 2]).unwrap();
        metrics.add_sample(&sdr);

        let freqs = metrics.activation_frequencies();
        assert!((freqs[0] - 1.0).abs() < 0.001); // Bit 0 active both times
        assert!((freqs[1] - 0.5).abs() < 0.001); // Bit 1 active once
        assert!((freqs[2] - 0.5).abs() < 0.001); // Bit 2 active once
    }

    #[test]
    fn test_dead_bits() {
        let mut metrics = SdrMetrics::new(&[10]);

        let mut sdr = Sdr::new(&[10]);
        sdr.set_sparse(&[0, 1, 2]).unwrap();
        metrics.add_sample(&sdr);

        assert_eq!(metrics.num_dead_bits(), 7);
        assert!((metrics.dead_bit_ratio() - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_reset() {
        let mut metrics = SdrMetrics::new(&[10]);

        let mut sdr = Sdr::new(&[10]);
        sdr.set_sparse(&[0, 1, 2]).unwrap();
        metrics.add_sample(&sdr);

        metrics.reset();
        assert_eq!(metrics.num_samples(), 0);
        assert_eq!(metrics.mean_sparsity(), 0.0);
    }
}
