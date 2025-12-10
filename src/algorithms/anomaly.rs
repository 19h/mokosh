//! Anomaly detection algorithms.
//!
//! This module provides tools for detecting anomalies in data streams
//! using HTM predictions.

use crate::types::{Real, Sdr, UInt};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Computes raw anomaly scores.
///
/// The anomaly score is the fraction of active bits that were not predicted.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Anomaly {
    /// Mode for computing anomaly.
    mode: AnomalyComputeMode,
}

/// Mode for anomaly computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum AnomalyComputeMode {
    /// Pure anomaly score.
    #[default]
    Pure,
    /// Use predicted active cells.
    Active,
}

impl Anomaly {
    /// Creates a new Anomaly detector.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an Anomaly detector with the specified mode.
    pub fn with_mode(mode: AnomalyComputeMode) -> Self {
        Self { mode }
    }

    /// Computes the anomaly score.
    ///
    /// # Arguments
    ///
    /// * `active` - The active bits (e.g., active columns)
    /// * `predicted` - The predicted bits
    ///
    /// # Returns
    ///
    /// Anomaly score between 0.0 (fully predicted) and 1.0 (fully anomalous).
    pub fn compute(&self, active: &Sdr, predicted: &Sdr) -> Real {
        let active_sparse = active.get_sparse();

        if active_sparse.is_empty() {
            return 0.0;
        }

        let predicted_set: std::collections::HashSet<_> =
            predicted.get_sparse().into_iter().collect();

        let num_predicted_active = active_sparse
            .iter()
            .filter(|&a| predicted_set.contains(a))
            .count();

        1.0 - (num_predicted_active as Real / active_sparse.len() as Real)
    }

    /// Computes anomaly given active cells and predictive cells.
    ///
    /// Converts cell predictions to column predictions and computes
    /// the fraction of unpredicted active columns.
    pub fn compute_from_cells(
        &self,
        active_columns: &Sdr,
        predictive_cells: &Sdr,
        cells_per_column: UInt,
    ) -> Real {
        let active_cols = active_columns.get_sparse();

        if active_cols.is_empty() {
            return 0.0;
        }

        // Convert predictive cells to predicted columns
        let predicted_columns: std::collections::HashSet<u32> = predictive_cells
            .get_sparse()
            .into_iter()
            .map(|cell| cell / cells_per_column)
            .collect();

        let num_predicted = active_cols
            .iter()
            .filter(|&col| predicted_columns.contains(col))
            .count();

        1.0 - (num_predicted as Real / active_cols.len() as Real)
    }
}

/// Computes anomaly likelihood based on historical anomaly scores.
///
/// The anomaly likelihood is a measure of how unusual the current
/// anomaly score is compared to recent history.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AnomalyLikelihood {
    /// Window size for computing statistics.
    learning_period: usize,

    /// Estimation samples.
    estimation_samples: usize,

    /// Reestimation period.
    reestimation_period: usize,

    /// Historical anomaly scores.
    historical_scores: Vec<Real>,

    /// Running sum for mean calculation.
    running_sum: f64,

    /// Running sum of squares for variance.
    running_sum_sq: f64,

    /// Current mean of anomaly scores.
    mean: f64,

    /// Current variance of anomaly scores.
    variance: f64,

    /// Iteration counter.
    iteration: usize,

    /// Last reestimation iteration.
    last_reestimation: usize,
}

impl AnomalyLikelihood {
    /// Creates a new AnomalyLikelihood calculator.
    ///
    /// # Arguments
    ///
    /// * `learning_period` - Number of samples to collect before computing statistics
    /// * `estimation_samples` - Number of samples to use for estimation
    /// * `reestimation_period` - How often to reestimate parameters
    pub fn new(learning_period: usize, estimation_samples: usize, reestimation_period: usize) -> Self {
        Self {
            learning_period,
            estimation_samples,
            reestimation_period,
            historical_scores: Vec::with_capacity(estimation_samples),
            running_sum: 0.0,
            running_sum_sq: 0.0,
            mean: 0.5,
            variance: 0.0001,
            iteration: 0,
            last_reestimation: 0,
        }
    }

    /// Creates an AnomalyLikelihood with default parameters.
    pub fn with_defaults() -> Self {
        Self::new(288, 100, 100)
    }

    /// Updates the likelihood model and returns the anomaly likelihood.
    ///
    /// # Arguments
    ///
    /// * `anomaly_score` - The raw anomaly score (0.0 to 1.0)
    ///
    /// # Returns
    ///
    /// Anomaly likelihood between 0.0 and 1.0.
    pub fn anomaly_probability(&mut self, anomaly_score: Real) -> Real {
        self.iteration += 1;

        // Add to historical scores
        if self.historical_scores.len() >= self.estimation_samples {
            let removed = self.historical_scores.remove(0);
            self.running_sum -= removed as f64;
            self.running_sum_sq -= (removed as f64) * (removed as f64);
        }

        self.historical_scores.push(anomaly_score);
        self.running_sum += anomaly_score as f64;
        self.running_sum_sq += (anomaly_score as f64) * (anomaly_score as f64);

        // Reestimate parameters periodically
        if self.iteration >= self.learning_period
            && self.iteration - self.last_reestimation >= self.reestimation_period
        {
            self.estimate_parameters();
            self.last_reestimation = self.iteration;
        }

        // During learning period, return 0.5 (neutral)
        if self.iteration < self.learning_period {
            return 0.5;
        }

        // Compute likelihood using Gaussian tail probability
        self.compute_log_likelihood(anomaly_score as f64)
    }

    /// Estimates mean and variance from historical data.
    fn estimate_parameters(&mut self) {
        let n = self.historical_scores.len() as f64;
        if n < 2.0 {
            return;
        }

        self.mean = self.running_sum / n;
        self.variance = (self.running_sum_sq / n) - (self.mean * self.mean);
        self.variance = self.variance.max(0.0001); // Minimum variance
    }

    /// Computes the log-likelihood based anomaly probability.
    fn compute_log_likelihood(&self, value: f64) -> Real {
        // Gaussian tail probability
        let z = (value - self.mean) / self.variance.sqrt();

        // Use complementary error function approximation for tail probability
        let prob = 0.5 * erfc(z / std::f64::consts::SQRT_2);

        // Convert to anomaly likelihood (high probability = high likelihood of anomaly)
        (1.0 - prob) as Real
    }
}

impl Default for AnomalyLikelihood {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// Complementary error function approximation.
fn erfc(x: f64) -> f64 {
    // Approximation from Abramowitz and Stegun
    let t = 1.0 / (1.0 + 0.5 * x.abs());

    let tau = t
        * (-x * x - 1.26551223
            + t * (1.00002368
                + t * (0.37409196
                    + t * (0.09678418
                        + t * (-0.18628806
                            + t * (0.27886807
                                + t * (-1.13520398
                                    + t * (1.48851587
                                        + t * (-0.82215223 + t * 0.17087277)))))))))
        .exp();

    if x >= 0.0 {
        tau
    } else {
        2.0 - tau
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anomaly_basic() {
        let anomaly = Anomaly::new();

        // Fully predicted - no anomaly
        let mut active = Sdr::new(&[10]);
        let mut predicted = Sdr::new(&[10]);

        active.set_sparse(&[1, 2, 3]).unwrap();
        predicted.set_sparse(&[1, 2, 3]).unwrap();

        let score = anomaly.compute(&active, &predicted);
        assert!((score - 0.0).abs() < 0.01);

        // Fully anomalous - nothing predicted
        predicted.set_sparse(&[4, 5, 6]).unwrap();
        let score = anomaly.compute(&active, &predicted);
        assert!((score - 1.0).abs() < 0.01);

        // Half predicted
        predicted.set_sparse(&[1, 4, 5]).unwrap(); // 1 out of 3 predicted
        let score = anomaly.compute(&active, &predicted);
        assert!((score - 0.67).abs() < 0.1);
    }

    #[test]
    fn test_anomaly_empty() {
        let anomaly = Anomaly::new();

        let active = Sdr::new(&[10]);
        let predicted = Sdr::new(&[10]);

        let score = anomaly.compute(&active, &predicted);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_anomaly_likelihood() {
        let mut likelihood = AnomalyLikelihood::new(10, 50, 10);

        // Feed normal anomaly scores during learning
        for _ in 0..20 {
            let prob = likelihood.anomaly_probability(0.1);
            // During learning period, should return 0.5
            if likelihood.iteration < 10 {
                assert!((prob - 0.5).abs() < 0.01);
            }
        }

        // After learning, low anomaly should have low likelihood
        let prob_low = likelihood.anomaly_probability(0.1);

        // High anomaly should have higher likelihood
        let prob_high = likelihood.anomaly_probability(0.9);

        assert!(prob_high > prob_low);
    }

    #[test]
    fn test_anomaly_from_cells() {
        let anomaly = Anomaly::new();

        // 10 columns, 4 cells per column = 40 cells
        let mut active_columns = Sdr::new(&[10]);
        let mut predictive_cells = Sdr::new(&[40]);

        active_columns.set_sparse(&[0, 1, 2]).unwrap();

        // Cells 0-3 are column 0, cells 4-7 are column 1, cells 8-11 are column 2
        // Predict cells in columns 0 and 1 only
        predictive_cells.set_sparse(&[0, 4]).unwrap();

        let score = anomaly.compute_from_cells(&active_columns, &predictive_cells, 4);

        // 2 out of 3 columns predicted
        assert!((score - 0.33).abs() < 0.1);
    }
}
