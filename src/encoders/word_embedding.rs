//! Word Embedding Encoder implementation.
//!
//! Converts dense word embedding vectors (word2vec, GloVe, etc.) into SDRs.

use crate::encoders::Encoder;
use crate::error::{MokoshError, Result};
use crate::types::{Real, Sdr, UInt};
use std::collections::HashSet;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Parameters for creating a Word Embedding Encoder.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct WordEmbeddingEncoderParams {
    /// Dimension of input embedding vectors.
    pub embedding_dim: usize,

    /// Total number of bits in output SDR.
    pub size: UInt,

    /// Number of active bits in output SDR.
    pub active_bits: UInt,

    /// Number of random hyperplanes for LSH.
    /// More hyperplanes = more precision but less overlap for similar embeddings.
    pub num_hyperplanes: usize,
}

impl Default for WordEmbeddingEncoderParams {
    fn default() -> Self {
        Self {
            embedding_dim: 300, // Common for word2vec/GloVe
            size: 2048,
            active_bits: 41,
            num_hyperplanes: 128,
        }
    }
}

/// Encodes dense word embeddings into SDR representations.
///
/// Uses locality-sensitive hashing (LSH) with random hyperplanes
/// to convert continuous vectors into sparse binary representations
/// while preserving cosine similarity.
///
/// # Example
///
/// ```rust
/// use mokosh::encoders::{WordEmbeddingEncoder, WordEmbeddingEncoderParams, Encoder};
///
/// let encoder = WordEmbeddingEncoder::new(WordEmbeddingEncoderParams {
///     embedding_dim: 4,
///     size: 100,
///     active_bits: 10,
///     num_hyperplanes: 32,
/// }).unwrap();
///
/// // Two similar embeddings
/// let embed1 = vec![0.5, 0.3, 0.1, 0.8];
/// let embed2 = vec![0.6, 0.35, 0.15, 0.75];
///
/// // A different embedding
/// let embed3 = vec![-0.5, -0.3, 0.9, -0.1];
///
/// let sdr1 = encoder.encode_to_sdr(embed1).unwrap();
/// let sdr2 = encoder.encode_to_sdr(embed2).unwrap();
/// let sdr3 = encoder.encode_to_sdr(embed3).unwrap();
///
/// // Similar embeddings should have more overlap
/// assert!(sdr1.get_overlap(&sdr2) > sdr1.get_overlap(&sdr3));
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct WordEmbeddingEncoder {
    embedding_dim: usize,
    size: UInt,
    active_bits: UInt,
    num_hyperplanes: usize,
    /// Random hyperplanes for LSH (flattened: num_hyperplanes x embedding_dim).
    hyperplanes: Vec<Real>,
    dimensions: Vec<UInt>,
}

impl WordEmbeddingEncoder {
    /// Creates a new Word Embedding Encoder.
    pub fn new(params: WordEmbeddingEncoderParams) -> Result<Self> {
        Self::with_seed(params, 42)
    }

    /// Creates a new Word Embedding Encoder with a specific seed.
    pub fn with_seed(params: WordEmbeddingEncoderParams, seed: u64) -> Result<Self> {
        if params.embedding_dim == 0 {
            return Err(MokoshError::InvalidParameter {
                name: "embedding_dim",
                message: "Must be > 0".to_string(),
            });
        }

        if params.active_bits > params.size {
            return Err(MokoshError::InvalidParameter {
                name: "active_bits",
                message: "Cannot exceed size".to_string(),
            });
        }

        if params.num_hyperplanes == 0 {
            return Err(MokoshError::InvalidParameter {
                name: "num_hyperplanes",
                message: "Must be > 0".to_string(),
            });
        }

        // Generate random hyperplanes using a simple LCG for reproducibility
        let mut hyperplanes =
            Vec::with_capacity(params.num_hyperplanes * params.embedding_dim);

        let mut state = seed;
        for _ in 0..(params.num_hyperplanes * params.embedding_dim) {
            // Simple LCG: state = (a * state + c) mod m
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            // Convert to [-1, 1] range
            let value = ((state >> 33) as Real / (u32::MAX as Real / 2.0)) - 1.0;
            hyperplanes.push(value);
        }

        Ok(Self {
            embedding_dim: params.embedding_dim,
            size: params.size,
            active_bits: params.active_bits,
            num_hyperplanes: params.num_hyperplanes,
            hyperplanes,
            dimensions: vec![params.size],
        })
    }

    /// Returns the embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Computes the LSH hash for an embedding.
    fn compute_lsh_hash(&self, embedding: &[Real]) -> u128 {
        let mut hash: u128 = 0;

        for hp_idx in 0..self.num_hyperplanes.min(128) {
            let hp_start = hp_idx * self.embedding_dim;
            let hyperplane = &self.hyperplanes[hp_start..hp_start + self.embedding_dim];

            // Compute dot product
            let dot: Real = embedding
                .iter()
                .zip(hyperplane.iter())
                .map(|(&e, &h)| e * h)
                .sum();

            if dot >= 0.0 {
                hash |= 1u128 << hp_idx;
            }
        }

        hash
    }
}

impl Encoder<Vec<Real>> for WordEmbeddingEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, embedding: Vec<Real>, output: &mut Sdr) -> Result<()> {
        if embedding.len() != self.embedding_dim {
            return Err(MokoshError::InvalidParameter {
                name: "embedding",
                message: format!(
                    "Expected {} dimensions, got {}",
                    self.embedding_dim,
                    embedding.len()
                ),
            });
        }

        if output.dimensions() != self.dimensions.as_slice() {
            return Err(MokoshError::DimensionMismatch {
                expected: self.dimensions.clone(),
                actual: output.dimensions().to_vec(),
            });
        }

        let lsh_hash = self.compute_lsh_hash(&embedding);

        // Use the LSH hash to deterministically select active bits
        let mut active_bits = HashSet::new();
        let mut state = lsh_hash as u64;

        while active_bits.len() < self.active_bits as usize {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let bit = (state % self.size as u64) as UInt;
            active_bits.insert(bit);
        }

        let mut sparse: Vec<UInt> = active_bits.into_iter().collect();
        sparse.sort_unstable();
        output.set_sparse_unchecked(sparse);

        Ok(())
    }
}

impl Encoder<&[Real]> for WordEmbeddingEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, embedding: &[Real], output: &mut Sdr) -> Result<()> {
        self.encode(embedding.to_vec(), output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_encoder() {
        let encoder = WordEmbeddingEncoder::new(WordEmbeddingEncoderParams {
            embedding_dim: 100,
            size: 500,
            active_bits: 25,
            num_hyperplanes: 64,
        })
        .unwrap();

        assert_eq!(encoder.embedding_dim(), 100);
        assert_eq!(Encoder::<Vec<Real>>::size(&encoder), 500);
    }

    #[test]
    fn test_encode_embedding() {
        let encoder = WordEmbeddingEncoder::new(WordEmbeddingEncoderParams {
            embedding_dim: 10,
            size: 200,
            active_bits: 20,
            num_hyperplanes: 32,
        })
        .unwrap();

        let embedding = vec![0.1, 0.2, 0.3, 0.4, 0.5, -0.1, -0.2, -0.3, -0.4, -0.5];
        let sdr = encoder.encode_to_sdr(embedding).unwrap();

        assert_eq!(sdr.get_sum(), 20);
    }

    #[test]
    fn test_similar_embeddings_overlap() {
        let encoder = WordEmbeddingEncoder::new(WordEmbeddingEncoderParams {
            embedding_dim: 8,
            size: 500,
            active_bits: 25,
            num_hyperplanes: 64,
        })
        .unwrap();

        // Very similar embeddings
        let embed1 = vec![0.5, 0.3, 0.1, 0.8, 0.2, 0.4, 0.6, 0.1];
        let embed2 = vec![0.5, 0.3, 0.1, 0.8, 0.2, 0.4, 0.6, 0.1]; // Identical

        // Different embedding
        let embed3 = vec![-0.5, -0.3, -0.1, -0.8, -0.2, -0.4, -0.6, -0.1];

        let sdr1 = encoder.encode_to_sdr(embed1).unwrap();
        let sdr2 = encoder.encode_to_sdr(embed2).unwrap();
        let sdr3 = encoder.encode_to_sdr(embed3).unwrap();

        // Identical should have full overlap
        assert_eq!(sdr1.get_overlap(&sdr2), 25);

        // Opposite should have less overlap (or different bits entirely)
        let diff_overlap = sdr1.get_overlap(&sdr3);
        assert!(diff_overlap < 25);
    }

    #[test]
    fn test_deterministic() {
        let encoder = WordEmbeddingEncoder::new(WordEmbeddingEncoderParams {
            embedding_dim: 5,
            size: 100,
            active_bits: 10,
            num_hyperplanes: 16,
        })
        .unwrap();

        let embedding = vec![0.1, 0.2, 0.3, 0.4, 0.5];

        let sdr1 = encoder.encode_to_sdr(embedding.clone()).unwrap();
        let sdr2 = encoder.encode_to_sdr(embedding).unwrap();

        assert_eq!(sdr1.get_sparse(), sdr2.get_sparse());
    }

    #[test]
    fn test_wrong_dimension() {
        let encoder = WordEmbeddingEncoder::new(WordEmbeddingEncoderParams {
            embedding_dim: 10,
            ..Default::default()
        })
        .unwrap();

        let result = encoder.encode_to_sdr(vec![0.1, 0.2, 0.3]); // Wrong size
        assert!(result.is_err());
    }

    #[test]
    fn test_with_seed() {
        let encoder1 = WordEmbeddingEncoder::with_seed(
            WordEmbeddingEncoderParams {
                embedding_dim: 5,
                size: 100,
                active_bits: 10,
                num_hyperplanes: 16,
            },
            123,
        )
        .unwrap();

        let encoder2 = WordEmbeddingEncoder::with_seed(
            WordEmbeddingEncoderParams {
                embedding_dim: 5,
                size: 100,
                active_bits: 10,
                num_hyperplanes: 16,
            },
            123,
        )
        .unwrap();

        let embedding = vec![0.1, 0.2, 0.3, 0.4, 0.5];

        let sdr1 = encoder1.encode_to_sdr(embedding.clone()).unwrap();
        let sdr2 = encoder2.encode_to_sdr(embedding).unwrap();

        assert_eq!(sdr1.get_sparse(), sdr2.get_sparse());
    }
}
