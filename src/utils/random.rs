//! Random number generator compatible with htm.core's Random class.
//!
//! This module provides a deterministic pseudo-random number generator that
//! produces the same sequences as the C++ implementation when given the same seed.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

/// A pseudo-random number generator compatible with htm.core.
///
/// Uses ChaCha20 for high-quality randomness with deterministic behavior
/// when seeded. The implementation aims to be reproducible across platforms.
///
/// # Example
///
/// ```rust
/// use mokosh::utils::Random;
///
/// let mut rng = Random::new(42);
///
/// // Generate random integers
/// let n = rng.get_uint32();
///
/// // Generate random floats
/// let f = rng.get_real64();
///
/// // Sample from a range
/// let idx = rng.get_uint32_range(0, 100);
///
/// // Shuffle a vector
/// let mut items: Vec<u32> = (0..10).collect();
/// rng.shuffle(&mut items);
/// ```
pub struct Random {
    rng: ChaCha20Rng,
    seed: u64,
    /// Number of random values generated (for state reconstruction).
    steps: u64,
}

// Custom serialization for Random - we serialize seed and steps,
// then reconstruct the RNG state on deserialization.
#[cfg(feature = "serde")]
mod serde_impl {
    use super::*;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    #[derive(Serialize, Deserialize)]
    struct RandomState {
        seed: u64,
        steps: u64,
    }

    impl Serialize for Random {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            let state = RandomState {
                seed: self.seed,
                steps: self.steps,
            };
            state.serialize(serializer)
        }
    }

    impl<'de> Deserialize<'de> for Random {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            let state = RandomState::deserialize(deserializer)?;
            let mut rng = ChaCha20Rng::seed_from_u64(state.seed);
            // Advance the RNG to match the serialized state
            for _ in 0..state.steps {
                let _: u64 = rng.gen();
            }
            Ok(Random {
                rng,
                seed: state.seed,
                steps: state.steps,
            })
        }
    }
}

impl Random {
    /// Creates a new random number generator with the given seed.
    ///
    /// If seed is 0, uses a default seed (0 is a valid seed that produces
    /// deterministic output).
    #[must_use]
    pub fn new(seed: i64) -> Self {
        let actual_seed = if seed < 0 {
            // Negative seed means use system randomness
            rand::thread_rng().gen()
        } else {
            seed as u64
        };

        Self {
            rng: ChaCha20Rng::seed_from_u64(actual_seed),
            seed: actual_seed,
            steps: 0,
        }
    }

    /// Creates a random number generator with a random seed.
    #[must_use]
    pub fn with_random_seed() -> Self {
        Self::new(-1)
    }

    /// Returns the seed used for this generator.
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Returns the number of random values generated.
    #[must_use]
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Generates a random u32.
    pub fn get_uint32(&mut self) -> u32 {
        self.steps += 1;
        self.rng.gen()
    }

    /// Generates a random u64.
    pub fn get_uint64(&mut self) -> u64 {
        self.steps += 1;
        self.rng.gen()
    }

    /// Generates a random u32 in the range [min, max).
    pub fn get_uint32_range(&mut self, min: u32, max: u32) -> u32 {
        self.steps += 1;
        if min >= max {
            return min;
        }
        self.rng.gen_range(min..max)
    }

    /// Generates a random u64 in the range [min, max).
    pub fn get_uint64_range(&mut self, min: u64, max: u64) -> u64 {
        self.steps += 1;
        if min >= max {
            return min;
        }
        self.rng.gen_range(min..max)
    }

    /// Generates a random usize in the range [0, n).
    pub fn get_usize(&mut self, n: usize) -> usize {
        self.steps += 1;
        if n == 0 {
            return 0;
        }
        self.rng.gen_range(0..n)
    }

    /// Generates a random f32 in [0, 1).
    pub fn get_real32(&mut self) -> f32 {
        self.steps += 1;
        self.rng.gen()
    }

    /// Generates a random f64 in [0, 1).
    pub fn get_real64(&mut self) -> f64 {
        self.steps += 1;
        self.rng.gen()
    }

    /// Generates a random f64 in the range [min, max).
    pub fn get_real64_range(&mut self, min: f64, max: f64) -> f64 {
        if min >= max {
            return min;
        }
        min + (max - min) * self.get_real64()
    }

    /// Generates a random boolean with 50% probability.
    pub fn get_bool(&mut self) -> bool {
        self.steps += 1;
        self.rng.gen()
    }

    /// Generates a random boolean with the given probability of being true.
    pub fn get_bool_with_prob(&mut self, probability: f64) -> bool {
        self.get_real64() < probability
    }

    /// Shuffles a slice in place using Fisher-Yates algorithm.
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        let n = slice.len();
        if n <= 1 {
            return;
        }

        for i in (1..n).rev() {
            let j = self.get_usize(i + 1);
            slice.swap(i, j);
        }
    }

    /// Samples `k` unique items from a collection without replacement.
    ///
    /// Returns a vector of `k` randomly selected items from the input.
    /// If `k >= items.len()`, returns a shuffled copy of all items.
    pub fn sample<T: Clone>(&mut self, mut items: Vec<T>, k: usize) -> Vec<T> {
        let n = items.len();
        if k >= n {
            self.shuffle(&mut items);
            return items;
        }

        // Use partial Fisher-Yates for efficiency when k << n
        for i in 0..k {
            let j = self.get_usize(n - i) + i;
            items.swap(i, j);
        }

        items.truncate(k);
        items
    }

    /// Samples `k` indices from `0..n` without replacement.
    pub fn sample_indices(&mut self, n: usize, k: usize) -> Vec<usize> {
        if k >= n {
            let mut indices: Vec<usize> = (0..n).collect();
            self.shuffle(&mut indices);
            return indices;
        }

        // For small k relative to n, use selection sampling
        if k < n / 3 {
            let mut selected = std::collections::HashSet::with_capacity(k);
            let mut result = Vec::with_capacity(k);

            while result.len() < k {
                let idx = self.get_usize(n);
                if selected.insert(idx) {
                    result.push(idx);
                }
            }

            return result;
        }

        // For larger k, use partial Fisher-Yates
        let mut indices: Vec<usize> = (0..n).collect();
        for i in 0..k {
            let j = self.get_usize(n - i) + i;
            indices.swap(i, j);
        }
        indices.truncate(k);
        indices
    }

    /// Returns a normally distributed random number using Box-Muller transform.
    pub fn normal(&mut self, mean: f64, std_dev: f64) -> f64 {
        // Box-Muller transform
        let u1 = self.get_real64();
        let u2 = self.get_real64();

        let mag = std_dev * (-2.0 * u1.ln()).sqrt();
        let z0 = mag * (2.0 * std::f64::consts::PI * u2).cos();

        mean + z0
    }

    /// Returns an exponentially distributed random number.
    pub fn exponential(&mut self, lambda: f64) -> f64 {
        -self.get_real64().ln() / lambda
    }
}

impl Default for Random {
    fn default() -> Self {
        Self::new(0)
    }
}

impl Clone for Random {
    fn clone(&self) -> Self {
        Self {
            rng: self.rng.clone(),
            seed: self.seed,
            steps: self.steps,
        }
    }
}

impl std::fmt::Debug for Random {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Random")
            .field("seed", &self.seed)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic() {
        let mut rng1 = Random::new(42);
        let mut rng2 = Random::new(42);

        for _ in 0..100 {
            assert_eq!(rng1.get_uint32(), rng2.get_uint32());
        }
    }

    #[test]
    fn test_different_seeds() {
        let mut rng1 = Random::new(42);
        let mut rng2 = Random::new(43);

        let mut same = true;
        for _ in 0..100 {
            if rng1.get_uint32() != rng2.get_uint32() {
                same = false;
                break;
            }
        }
        assert!(!same);
    }

    #[test]
    fn test_range() {
        let mut rng = Random::new(42);

        for _ in 0..1000 {
            let v = rng.get_uint32_range(10, 20);
            assert!(v >= 10 && v < 20);
        }
    }

    #[test]
    fn test_real_range() {
        let mut rng = Random::new(42);

        for _ in 0..1000 {
            let v = rng.get_real64();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn test_shuffle() {
        let mut rng = Random::new(42);
        let original: Vec<u32> = (0..100).collect();
        let mut shuffled = original.clone();
        rng.shuffle(&mut shuffled);

        // Should be different from original
        assert_ne!(original, shuffled);

        // Should contain same elements
        let mut sorted = shuffled.clone();
        sorted.sort();
        assert_eq!(original, sorted);
    }

    #[test]
    fn test_sample() {
        let mut rng = Random::new(42);
        let items: Vec<u32> = (0..100).collect();
        let sampled = rng.sample(items.clone(), 10);

        assert_eq!(sampled.len(), 10);

        // All sampled items should be unique
        let mut unique = sampled.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), 10);

        // All items should be from original set
        for item in &sampled {
            assert!(items.contains(item));
        }
    }

    #[test]
    fn test_sample_indices() {
        let mut rng = Random::new(42);
        let indices = rng.sample_indices(100, 10);

        assert_eq!(indices.len(), 10);

        // All should be in range
        for &idx in &indices {
            assert!(idx < 100);
        }

        // All should be unique
        let mut unique = indices.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), 10);
    }

    #[test]
    fn test_normal_distribution() {
        let mut rng = Random::new(42);
        let mut sum = 0.0;
        let n = 10000;

        for _ in 0..n {
            sum += rng.normal(0.0, 1.0);
        }

        let mean = sum / n as f64;
        // Mean should be close to 0 with high probability
        assert!(mean.abs() < 0.1);
    }

    #[test]
    fn test_bool_with_prob() {
        let mut rng = Random::new(42);
        let mut count = 0;
        let n = 10000;

        for _ in 0..n {
            if rng.get_bool_with_prob(0.3) {
                count += 1;
            }
        }

        let ratio = count as f64 / n as f64;
        // Should be close to 0.3
        assert!((ratio - 0.3).abs() < 0.05);
    }

    #[test]
    fn test_empty_operations() {
        let mut rng = Random::new(42);

        // Empty shuffle should not panic
        let mut empty: Vec<u32> = Vec::new();
        rng.shuffle(&mut empty);

        // Sample more than available
        let items: Vec<u32> = (0..5).collect();
        let sampled = rng.sample(items.clone(), 10);
        assert_eq!(sampled.len(), 5);

        // Sample indices more than available
        let indices = rng.sample_indices(5, 10);
        assert_eq!(indices.len(), 5);
    }
}
