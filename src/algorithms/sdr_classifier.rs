//! SDR Classifier implementation.
//!
//! The SDR Classifier maps SDR patterns to output classes or values.
//! It can be used for both classification and prediction tasks.

use crate::types::{Real, Sdr, UInt};
use std::collections::{HashMap, VecDeque};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Parameters for creating an SDR Classifier.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SdrClassifierParams {
    /// Learning rate (alpha).
    pub alpha: Real,

    /// Number of prediction steps.
    /// E.g., `vec![1]` for 1-step prediction, `vec![1, 5]` for 1 and 5-step.
    pub steps: Vec<UInt>,
}

impl Default for SdrClassifierParams {
    fn default() -> Self {
        Self {
            alpha: 0.001,
            steps: vec![1],
        }
    }
}

/// A classifier that maps SDR patterns to output classes.
///
/// The SDR Classifier learns associations between SDR input patterns
/// and output categories. It uses online learning and can predict
/// at multiple future steps.
///
/// # Example
///
/// ```rust
/// use mokosh::algorithms::{SdrClassifier, SdrClassifierParams};
/// use mokosh::types::Sdr;
///
/// let mut classifier = SdrClassifier::new(SdrClassifierParams {
///     alpha: 0.001,
///     steps: vec![1],
/// });
///
/// let mut pattern = Sdr::new(&[100]);
/// pattern.set_sparse(&[1, 5, 10, 20]).unwrap();
///
/// // Learn pattern associated with bucket 3
/// classifier.learn(&pattern, 3);
///
/// // Infer bucket from pattern
/// let predictions = classifier.infer(&pattern);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct SdrClassifier {
    /// Learning rate.
    alpha: Real,

    /// Prediction steps.
    steps: Vec<UInt>,

    /// Weight matrix: step -> (input_bit -> (bucket -> weight)).
    weights: HashMap<UInt, HashMap<u32, HashMap<u32, Real>>>,

    /// History of active patterns for multi-step prediction.
    pattern_history: VecDeque<(u64, Vec<u32>)>,

    /// Category counts can differ until each prediction horizon is trained.
    num_buckets: HashMap<UInt, usize>,
    input_size: Option<usize>,

    /// Maximum bucket index seen.
    max_bucket: u32,

    /// Iteration count.
    iteration: usize,
}

impl SdrClassifier {
    /// Validates persisted horizons, history and sparse weight indexes.
    pub fn validate(&self) -> crate::Result<()> {
        let invalid = || crate::MokoshError::InvalidSdrData("Invalid classifier state".into());
        if !self.alpha.is_finite()
            || self.alpha <= 0.0
            || self.steps.is_empty()
            || self.steps.windows(2).any(|w| w[0] >= w[1])
            || self.input_size == Some(0)
        {
            return Err(invalid());
        }
        let mut previous = None;
        for (record, pattern) in &self.pattern_history {
            if previous.is_some_and(|last| last >= *record)
                || pattern.windows(2).any(|w| w[0] >= w[1])
                || pattern
                    .iter()
                    .any(|&i| self.input_size.is_none_or(|n| i as usize >= n))
            {
                return Err(invalid());
            }
            previous = Some(*record);
        }
        for (&step, rows) in &self.weights {
            if self.steps.binary_search(&step).is_err() {
                return Err(invalid());
            }
            let count = self.num_buckets.get(&step).copied().unwrap_or(0);
            for (&bit, weights) in rows {
                if self.input_size.is_none_or(|n| bit as usize >= n)
                    || weights
                        .iter()
                        .any(|(&category, &value)| category as usize >= count || !value.is_finite())
                {
                    return Err(invalid());
                }
            }
        }
        if self.num_buckets.iter().any(|(&step, &count)| {
            self.steps.binary_search(&step).is_err() || count > self.max_bucket as usize + 1
        }) {
            return Err(invalid());
        }
        Ok(())
    }

    /// Creates a new SDR Classifier.
    pub fn new(params: SdrClassifierParams) -> Self {
        assert!(
            params.alpha.is_finite() && params.alpha > 0.0,
            "Learning rate must be finite and positive"
        );
        assert!(
            !params.steps.is_empty(),
            "At least one prediction step is required"
        );
        let mut params = params;
        params.steps.sort_unstable();
        params.steps.dedup();
        let mut weights = HashMap::new();
        for &step in &params.steps {
            weights.insert(step, HashMap::new());
        }

        Self {
            alpha: params.alpha,
            steps: params.steps,
            weights,
            pattern_history: VecDeque::new(),
            num_buckets: HashMap::new(),
            input_size: None,
            max_bucket: 0,
            iteration: 0,
        }
    }

    /// Learns a pattern associated with a bucket index.
    ///
    /// # Arguments
    ///
    /// * `pattern` - The SDR pattern
    /// * `bucket_idx` - The target bucket index
    pub fn learn(&mut self, pattern: &Sdr, bucket_idx: u32) {
        let record = self.pattern_history.back().map_or(0, |(n, _)| {
            n.checked_add(1).expect("Record number overflow")
        });
        self.learn_at(record, pattern, &[bucket_idx]);
    }

    /// Learns one or more equally weighted categories at a monotonic record number.
    /// Gaps represent missing observations, not adjacent time steps. Repeated
    /// record numbers refine the same input record, as in htm.core Predictor.
    pub fn learn_at(&mut self, record: u64, pattern: &Sdr, buckets: &[u32]) {
        assert!(
            pattern.size() > 0 && !buckets.is_empty(),
            "Training needs an input and categories"
        );
        if let Some(size) = self.input_size {
            assert_eq!(pattern.size(), size, "Classifier input size changed");
        }
        self.input_size = Some(pattern.size());
        if let Some((last, _)) = self.pattern_history.back() {
            assert!(record >= *last, "Record numbers must be monotonic");
        }
        let mut buckets = buckets.to_vec();
        buckets.sort_unstable();
        buckets.dedup();
        let max_bucket = *buckets.last().unwrap();
        assert!(max_bucket < u32::MAX, "Category count must fit u32");
        self.max_bucket = self.max_bucket.max(max_bucket);
        self.iteration += 1;
        if self.pattern_history.back().map(|(n, _)| *n) != Some(record) {
            self.pattern_history
                .push_back((record, pattern.get_sparse()));
        }
        let max_step = u64::from(*self.steps.last().unwrap());
        while self
            .pattern_history
            .front()
            .is_some_and(|(n, _)| record - *n > max_step)
        {
            self.pattern_history.pop_front();
        }
        let history: Vec<_> = self.pattern_history.iter().cloned().collect();
        for (past_record, past_pattern) in history {
            let delta = record - past_record;
            if delta <= u64::from(u32::MAX) && self.steps.binary_search(&(delta as UInt)).is_ok() {
                let step = delta as UInt;
                let count = self.num_buckets.entry(step).or_default();
                *count = (*count).max(max_bucket as usize + 1);
                let predictions = self.compute_predictions(&past_pattern, step);
                self.update_weights(&past_pattern, &buckets, &predictions, step);
            }
        }
    }

    /// Infers bucket probabilities from a pattern.
    ///
    /// # Arguments
    ///
    /// * `pattern` - The SDR pattern
    ///
    /// # Returns
    ///
    /// A map from prediction step to bucket probability distributions.
    pub fn infer(&self, pattern: &Sdr) -> HashMap<UInt, Vec<Real>> {
        assert!(pattern.size() > 0, "Classifier input must be nonempty");
        if let Some(size) = self.input_size {
            assert_eq!(pattern.size(), size, "Classifier input size changed");
        }
        let sparse = pattern.get_sparse();
        let mut results = HashMap::new();

        for &step in &self.steps {
            let predictions = self.compute_predictions(&sparse, step);
            results.insert(step, predictions);
        }

        results
    }

    /// Computes raw predictions for a pattern.
    fn compute_predictions(&self, sparse: &[u32], step: UInt) -> Vec<Real> {
        let num_buckets = self.num_buckets.get(&step).copied().unwrap_or(0);
        let mut predictions = vec![0.0; num_buckets];

        if let Some(step_weights) = self.weights.get(&step) {
            for &input_bit in sparse {
                if let Some(bit_weights) = step_weights.get(&input_bit) {
                    for (&bucket, &weight) in bit_weights {
                        if (bucket as usize) < num_buckets {
                            predictions[bucket as usize] += weight;
                        }
                    }
                }
            }
        }

        // Apply softmax normalization
        self.softmax(&mut predictions);
        predictions
    }

    /// Updates weights based on prediction error.
    fn update_weights(
        &mut self,
        sparse: &[u32],
        actual_buckets: &[u32],
        predictions: &[Real],
        step: UInt,
    ) {
        let step_weights = self.weights.entry(step).or_default();

        for &input_bit in sparse {
            let bit_weights = step_weights.entry(input_bit).or_default();

            for bucket in 0..predictions.len() as u32 {
                let target = if actual_buckets.binary_search(&bucket).is_ok() {
                    1.0 / actual_buckets.len() as Real
                } else {
                    0.0
                };
                let prediction = predictions.get(bucket as usize).copied().unwrap_or(0.0);
                let error = target - prediction;

                let weight = bit_weights.entry(bucket).or_insert(0.0);
                *weight += self.alpha * error;
            }
        }
    }

    /// Applies softmax normalization to convert scores to probabilities.
    fn softmax(&self, scores: &mut [Real]) {
        if scores.is_empty() {
            return;
        }

        // Find max for numerical stability
        let max_score = scores.iter().copied().fold(Real::NEG_INFINITY, Real::max);

        // Compute exp and sum
        let mut sum = 0.0;
        for score in scores.iter_mut() {
            *score = (*score - max_score).exp();
            sum += *score;
        }

        // Normalize
        if sum > 0.0 {
            for score in scores.iter_mut() {
                *score /= sum;
            }
        }
    }

    /// Returns the most likely bucket for a pattern.
    pub fn infer_single(&self, pattern: &Sdr, step: UInt) -> Option<u32> {
        let predictions = self.infer(pattern);
        let step_predictions = predictions.get(&step)?;

        step_predictions
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(&a.0)))
            .map(|(idx, _)| idx as u32)
    }

    /// Returns the learning rate.
    pub fn alpha(&self) -> Real {
        self.alpha
    }

    /// Returns the prediction steps.
    pub fn steps(&self) -> &[UInt] {
        &self.steps
    }

    /// Returns the maximum bucket index.
    pub fn max_bucket(&self) -> u32 {
        self.max_bucket
    }

    /// Returns the iteration count.
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Resets the classifier state (but keeps learned weights).
    pub fn reset(&mut self) {
        self.pattern_history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_classifier() {
        let classifier = SdrClassifier::new(SdrClassifierParams::default());
        assert!((classifier.alpha() - 0.001).abs() < 0.0001);
        assert_eq!(classifier.steps(), &[1]);
    }

    #[test]
    fn test_learn_and_infer() {
        let mut classifier = SdrClassifier::new(SdrClassifierParams {
            alpha: 0.1,
            steps: vec![0],
        });

        // Create distinct patterns for different buckets
        let mut pattern_a = Sdr::new(&[100]);
        let mut pattern_b = Sdr::new(&[100]);

        pattern_a.set_sparse(&[0, 1, 2, 3, 4]).unwrap();
        pattern_b.set_sparse(&[50, 51, 52, 53, 54]).unwrap();

        // Learn associations
        for _ in 0..100 {
            classifier.learn(&pattern_a, 0);
            classifier.learn(&pattern_b, 1);
        }

        // Infer
        let pred_a = classifier.infer_single(&pattern_a, 0);
        let pred_b = classifier.infer_single(&pattern_b, 0);

        assert_eq!(pred_a, Some(0));
        assert_eq!(pred_b, Some(1));
    }

    #[test]
    fn test_multi_step() {
        let mut classifier = SdrClassifier::new(SdrClassifierParams {
            alpha: 0.1,
            steps: vec![1, 2],
        });

        let mut pattern = Sdr::new(&[50]);
        pattern.set_sparse(&[1, 2, 3]).unwrap();

        classifier.learn(&pattern, 0);
        classifier.learn(&pattern, 0);
        classifier.learn(&pattern, 1);

        let predictions = classifier.infer(&pattern);
        assert!(predictions.contains_key(&1));
        assert!(predictions.contains_key(&2));
    }

    #[test]
    fn test_probability_distribution() {
        let mut classifier = SdrClassifier::new(SdrClassifierParams {
            alpha: 0.1,
            steps: vec![0],
        });

        let mut pattern = Sdr::new(&[50]);
        pattern.set_sparse(&[1, 2, 3]).unwrap();

        // Learn pattern for bucket 0
        for _ in 0..50 {
            classifier.learn(&pattern, 0);
        }

        let predictions = classifier.infer(&pattern);
        let probs = predictions.get(&0).unwrap();

        // Should sum to 1.0
        let sum: Real = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);

        // Bucket 0 should have highest probability
        let max_idx = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(&a.0)))
            .map(|(idx, _)| idx);
        assert_eq!(max_idx, Some(0));
    }
}

#[cfg(feature = "serde")]
mod checked_serde {
    use super::*;
    #[derive(Deserialize)]
    #[serde(remote = "SdrClassifier")]
    struct State {
        /// Learning rate.
        alpha: Real,

        /// Prediction steps.
        steps: Vec<UInt>,

        /// Weight matrix: step -> (input_bit -> (bucket -> weight)).
        weights: HashMap<UInt, HashMap<u32, HashMap<u32, Real>>>,

        /// History of active patterns for multi-step prediction.
        pattern_history: VecDeque<(u64, Vec<u32>)>,

        /// Category counts can differ until each prediction horizon is trained.
        num_buckets: HashMap<UInt, usize>,
        input_size: Option<usize>,

        /// Maximum bucket index seen.
        max_bucket: u32,

        /// Iteration count.
        iteration: usize,
    }
    impl<'de> Deserialize<'de> for SdrClassifier {
        fn deserialize<D: serde::Deserializer<'de>>(
            deserializer: D,
        ) -> std::result::Result<Self, D::Error> {
            let value = State::deserialize(deserializer)?;
            value.validate().map_err(serde::de::Error::custom)?;
            Ok(value)
        }
    }
}
