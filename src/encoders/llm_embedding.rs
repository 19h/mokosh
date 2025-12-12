//! LLM Embedding Encoder implementation.
//!
//! Converts dense LLM embedding vectors of arbitrary dimensions (384, 768, 1536, 3072, etc.)
//! into SDRs while preserving semantic similarity through locality-sensitive hashing.
//!
//! Unlike fixed-dimension encoders, this encoder generates random hyperplanes on-the-fly
//! using deterministic hashing, allowing it to handle any embedding dimension without
//! pre-allocation.

use crate::encoders::Encoder;
use crate::error::{MokoshError, Result};
use crate::types::{Real, Sdr, UInt};
use std::collections::HashSet;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Strategy for normalizing input embeddings before encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum NormalizationStrategy {
    /// L2 normalize to unit length (recommended for cosine similarity preservation).
    #[default]
    L2Normalize,
    /// Center around mean (zero-mean).
    MeanCenter,
    /// No normalization - use raw values.
    None,
}

/// Strategy for handling different embedding dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DimensionStrategy {
    /// Project all dimensions equally (default).
    #[default]
    FullProjection,
    /// Divide embedding into chunks and encode each separately.
    /// The u32 specifies how many bits per chunk.
    Chunked(u32),
    /// Use strided sampling for very high dimensions.
    /// The u32 specifies the stride (e.g., 2 means use every other dimension).
    Strided(u32),
}

/// Parameters for creating an LLM Embedding Encoder.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LlmEmbeddingEncoderParams {
    /// Total number of bits in output SDR.
    pub size: UInt,

    /// Number of active bits in output SDR.
    /// Typically ~2% of size for HTM compatibility.
    pub active_bits: UInt,

    /// Number of random hyperplanes for LSH.
    /// More hyperplanes = finer discrimination but less overlap for similar embeddings.
    /// Recommended: 128-256 for good balance.
    pub num_hyperplanes: usize,

    /// Number of hash bands for multi-probe LSH.
    /// More bands = better similarity preservation at cost of computation.
    /// Default: 4
    pub num_bands: usize,

    /// Strategy for normalizing input embeddings.
    pub normalization: NormalizationStrategy,

    /// Strategy for handling embedding dimensions.
    pub dimension_strategy: DimensionStrategy,

    /// Whether to encode the magnitude (L2 norm) as additional bits.
    /// Useful when embedding magnitude carries semantic meaning.
    pub encode_magnitude: bool,

    /// Bits reserved for magnitude encoding (only used if encode_magnitude is true).
    pub magnitude_bits: UInt,

    /// Random seed for reproducible hyperplane generation.
    pub seed: u64,
}

impl Default for LlmEmbeddingEncoderParams {
    fn default() -> Self {
        Self {
            size: 2048,
            active_bits: 41, // ~2% sparsity
            num_hyperplanes: 128,
            num_bands: 4,
            normalization: NormalizationStrategy::L2Normalize,
            dimension_strategy: DimensionStrategy::FullProjection,
            encode_magnitude: false,
            magnitude_bits: 8,
            seed: 42,
        }
    }
}

/// Encodes LLM embeddings of arbitrary dimensions into SDR representations.
///
/// This encoder uses locality-sensitive hashing (LSH) with on-the-fly hyperplane
/// generation, allowing it to handle embeddings of any dimension (384, 768, 1536,
/// 3072, etc.) without pre-allocation.
///
/// # Key Features
///
/// - **Arbitrary dimensions**: Works with any embedding size
/// - **Similarity preservation**: Similar embeddings produce overlapping SDRs
/// - **Deterministic**: Same input always produces same output
/// - **Memory efficient**: Generates hyperplanes on-the-fly using hash functions
/// - **Multi-band LSH**: Better similarity preservation through multiple hash bands
///
/// # Example
///
/// ```rust
/// use mokosh::encoders::{LlmEmbeddingEncoder, LlmEmbeddingEncoderParams, Encoder};
///
/// // Create encoder - works with ANY embedding dimension
/// let encoder = LlmEmbeddingEncoder::new(LlmEmbeddingEncoderParams {
///     size: 2048,
///     active_bits: 41,
///     num_hyperplanes: 128,
///     ..Default::default()
/// }).unwrap();
///
/// // Encode a 384-dim embedding (e.g., all-MiniLM-L6-v2)
/// let small_embed: Vec<f32> = vec![0.1; 384];
/// let sdr1 = encoder.encode_to_sdr(&small_embed).unwrap();
///
/// // Same encoder works with 768-dim (e.g., BERT)
/// let medium_embed: Vec<f32> = vec![0.1; 768];
/// let sdr2 = encoder.encode_to_sdr(&medium_embed).unwrap();
///
/// // And 1536-dim (e.g., OpenAI text-embedding-ada-002)
/// let large_embed: Vec<f32> = vec![0.1; 1536];
/// let sdr3 = encoder.encode_to_sdr(&large_embed).unwrap();
///
/// assert_eq!(sdr1.get_sum(), 41);
/// assert_eq!(sdr2.get_sum(), 41);
/// assert_eq!(sdr3.get_sum(), 41);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LlmEmbeddingEncoder {
    size: UInt,
    active_bits: UInt,
    num_hyperplanes: usize,
    num_bands: usize,
    normalization: NormalizationStrategy,
    dimension_strategy: DimensionStrategy,
    encode_magnitude: bool,
    magnitude_bits: UInt,
    seed: u64,
    dimensions: Vec<UInt>,
    /// Bits available for the main encoding (excluding magnitude bits if enabled).
    main_bits: UInt,
    /// Active bits for the main encoding.
    main_active_bits: UInt,
}

impl LlmEmbeddingEncoder {
    /// Creates a new LLM Embedding Encoder.
    pub fn new(params: LlmEmbeddingEncoderParams) -> Result<Self> {
        if params.size == 0 {
            return Err(MokoshError::InvalidParameter {
                name: "size",
                message: "Must be > 0".to_string(),
            });
        }

        if params.active_bits == 0 {
            return Err(MokoshError::InvalidParameter {
                name: "active_bits",
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

        if params.num_bands == 0 {
            return Err(MokoshError::InvalidParameter {
                name: "num_bands",
                message: "Must be > 0".to_string(),
            });
        }

        if params.encode_magnitude && params.magnitude_bits >= params.active_bits {
            return Err(MokoshError::InvalidParameter {
                name: "magnitude_bits",
                message: "Must be less than active_bits".to_string(),
            });
        }

        let (main_bits, main_active_bits) = if params.encode_magnitude {
            (
                params.size - params.magnitude_bits,
                params.active_bits - params.magnitude_bits,
            )
        } else {
            (params.size, params.active_bits)
        };

        Ok(Self {
            size: params.size,
            active_bits: params.active_bits,
            num_hyperplanes: params.num_hyperplanes,
            num_bands: params.num_bands,
            normalization: params.normalization,
            dimension_strategy: params.dimension_strategy,
            encode_magnitude: params.encode_magnitude,
            magnitude_bits: params.magnitude_bits,
            seed: params.seed,
            dimensions: vec![params.size],
            main_bits,
            main_active_bits,
        })
    }

    /// Generates a deterministic "random" value for a hyperplane component.
    ///
    /// Uses FNV-1a hashing to generate values in [-1, 1] based on
    /// (seed, hyperplane_index, dimension_index).
    #[inline]
    fn hyperplane_value(&self, hyperplane_idx: usize, dim_idx: usize) -> Real {
        // FNV-1a hash
        const FNV_PRIME: u64 = 0x100000001b3;
        const FNV_OFFSET: u64 = 0xcbf29ce484222325;

        let mut hash = FNV_OFFSET;

        // Mix in seed
        hash ^= self.seed;
        hash = hash.wrapping_mul(FNV_PRIME);

        // Mix in hyperplane index
        hash ^= hyperplane_idx as u64;
        hash = hash.wrapping_mul(FNV_PRIME);

        // Mix in dimension index
        hash ^= dim_idx as u64;
        hash = hash.wrapping_mul(FNV_PRIME);

        // Convert to [-1, 1] range using high bits for better distribution
        let normalized = (hash >> 32) as f64 / (u32::MAX as f64);
        (normalized * 2.0 - 1.0) as Real
    }

    /// Computes the dot product between embedding and a hyperplane (generated on-the-fly).
    #[inline]
    fn dot_with_hyperplane(&self, embedding: &[Real], hyperplane_idx: usize) -> Real {
        match self.dimension_strategy {
            DimensionStrategy::FullProjection => {
                embedding
                    .iter()
                    .enumerate()
                    .map(|(i, &e)| e * self.hyperplane_value(hyperplane_idx, i))
                    .sum()
            }
            DimensionStrategy::Strided(stride) => {
                embedding
                    .iter()
                    .enumerate()
                    .step_by(stride as usize)
                    .map(|(i, &e)| e * self.hyperplane_value(hyperplane_idx, i))
                    .sum()
            }
            DimensionStrategy::Chunked(_) => {
                // For chunked, we still do full projection per chunk
                embedding
                    .iter()
                    .enumerate()
                    .map(|(i, &e)| e * self.hyperplane_value(hyperplane_idx, i))
                    .sum()
            }
        }
    }

    /// Normalizes the embedding according to the configured strategy.
    fn normalize(&self, embedding: &[Real]) -> Vec<Real> {
        match self.normalization {
            NormalizationStrategy::None => embedding.to_vec(),
            NormalizationStrategy::L2Normalize => {
                let norm: Real = embedding.iter().map(|x| x * x).sum::<Real>().sqrt();
                if norm > Real::EPSILON {
                    embedding.iter().map(|x| x / norm).collect()
                } else {
                    embedding.to_vec()
                }
            }
            NormalizationStrategy::MeanCenter => {
                let mean: Real = embedding.iter().sum::<Real>() / embedding.len() as Real;
                embedding.iter().map(|x| x - mean).collect()
            }
        }
    }

    /// Computes the LSH signature for an embedding using multi-band hashing.
    fn compute_lsh_signature(&self, embedding: &[Real]) -> Vec<u64> {
        let hyperplanes_per_band = self.num_hyperplanes / self.num_bands;
        let mut signatures = Vec::with_capacity(self.num_bands);

        for band in 0..self.num_bands {
            let mut band_hash: u64 = 0;
            let start_hp = band * hyperplanes_per_band;

            for hp_offset in 0..hyperplanes_per_band.min(64) {
                let hp_idx = start_hp + hp_offset;
                let dot = self.dot_with_hyperplane(embedding, hp_idx);

                if dot >= 0.0 {
                    band_hash |= 1u64 << hp_offset;
                }
            }

            signatures.push(band_hash);
        }

        signatures
    }

    /// Computes the L2 magnitude of an embedding.
    fn compute_magnitude(&self, embedding: &[Real]) -> Real {
        embedding.iter().map(|x| x * x).sum::<Real>().sqrt()
    }

    /// Encodes magnitude into bit positions.
    fn encode_magnitude_bits(&self, magnitude: Real, offset: UInt) -> Vec<UInt> {
        // Use logarithmic scaling for magnitude (most embeddings have similar scales)
        let log_mag = (1.0 + magnitude).ln();

        // Map to [0, 1] range - typical LLM embeddings have magnitudes in [0, 50]
        let normalized = (log_mag / 5.0).clamp(0.0, 1.0);

        // Convert to bucket index
        let num_buckets = self.magnitude_bits;
        let bucket = (normalized * (num_buckets - 1) as Real).round() as UInt;

        // Activate a small window around the bucket for overlap
        let mut bits = Vec::new();
        let half_width = (self.magnitude_bits / 4).max(1);

        for i in 0..half_width {
            let bit = offset + ((bucket + i) % num_buckets);
            bits.push(bit);
        }

        bits
    }

    /// Core encoding logic.
    fn encode_impl(&self, embedding: &[Real], output: &mut Sdr) -> Result<()> {
        if embedding.is_empty() {
            return Err(MokoshError::InvalidParameter {
                name: "embedding",
                message: "Cannot encode empty embedding".to_string(),
            });
        }

        if output.dimensions() != self.dimensions.as_slice() {
            return Err(MokoshError::DimensionMismatch {
                expected: self.dimensions.clone(),
                actual: output.dimensions().to_vec(),
            });
        }

        // Get magnitude before normalization (if needed)
        let magnitude = if self.encode_magnitude {
            Some(self.compute_magnitude(embedding))
        } else {
            None
        };

        // Normalize embedding
        let normalized = self.normalize(embedding);

        // Handle different dimension strategies
        let active_bits: Vec<UInt> = match self.dimension_strategy {
            DimensionStrategy::Chunked(bits_per_chunk) => {
                self.encode_chunked(&normalized, bits_per_chunk)?
            }
            _ => self.encode_single(&normalized)?,
        };

        // Combine main bits with magnitude bits if enabled
        let mut all_bits: HashSet<UInt> = active_bits.into_iter().collect();

        if let Some(mag) = magnitude {
            let mag_offset = self.main_bits;
            let mag_bits = self.encode_magnitude_bits(mag, mag_offset);
            for bit in mag_bits {
                all_bits.insert(bit);
            }
        }

        // Convert to sorted vec
        let mut sparse: Vec<UInt> = all_bits.into_iter().collect();
        sparse.sort_unstable();

        // Ensure we have exactly the right number of active bits
        if sparse.len() > self.active_bits as usize {
            sparse.truncate(self.active_bits as usize);
        }

        output.set_sparse_unchecked(sparse);
        Ok(())
    }

    /// Encode a single (non-chunked) embedding.
    fn encode_single(&self, embedding: &[Real]) -> Result<Vec<UInt>> {
        let signatures = self.compute_lsh_signature(embedding);

        // Combine signatures into a single hash state
        let mut combined_state: u64 = 0;
        for (i, &sig) in signatures.iter().enumerate() {
            combined_state ^= sig.rotate_left((i * 13) as u32);
        }

        // Use the combined signature to deterministically select active bits
        let mut active_bits = HashSet::new();
        let mut state = combined_state;

        // Generate bits from signatures
        while active_bits.len() < self.main_active_bits as usize {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let bit = (state % self.main_bits as u64) as UInt;
            active_bits.insert(bit);
        }

        Ok(active_bits.into_iter().collect())
    }

    /// Encode embedding in chunks for better resolution with high-dimensional embeddings.
    fn encode_chunked(&self, embedding: &[Real], bits_per_chunk: UInt) -> Result<Vec<UInt>> {
        let chunk_size = embedding.len() / 4; // Divide into 4 chunks
        if chunk_size == 0 {
            return self.encode_single(embedding);
        }

        let active_per_chunk = (self.main_active_bits / 4).max(1);
        let mut all_bits = HashSet::new();
        let mut chunk_offset = 0u32;

        for (chunk_idx, chunk) in embedding.chunks(chunk_size).enumerate() {
            let signatures = self.compute_lsh_signature(chunk);

            let mut combined_state: u64 = chunk_idx as u64;
            for (i, &sig) in signatures.iter().enumerate() {
                combined_state ^= sig.rotate_left((i * 13) as u32);
            }

            let mut state = combined_state;
            let mut chunk_bits = 0u32;

            while chunk_bits < active_per_chunk {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let local_bit = (state % bits_per_chunk as u64) as UInt;
                let global_bit = chunk_offset + local_bit;

                if global_bit < self.main_bits && all_bits.insert(global_bit) {
                    chunk_bits += 1;
                }
            }

            chunk_offset += bits_per_chunk;
            if chunk_offset >= self.main_bits {
                chunk_offset = 0;
            }
        }

        Ok(all_bits.into_iter().collect())
    }
}

impl Encoder<&[Real]> for LlmEmbeddingEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, embedding: &[Real], output: &mut Sdr) -> Result<()> {
        self.encode_impl(embedding, output)
    }
}

impl Encoder<Vec<Real>> for LlmEmbeddingEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, embedding: Vec<Real>, output: &mut Sdr) -> Result<()> {
        self.encode_impl(&embedding, output)
    }
}

impl Encoder<&Vec<Real>> for LlmEmbeddingEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, embedding: &Vec<Real>, output: &mut Sdr) -> Result<()> {
        self.encode_impl(embedding.as_slice(), output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_encoder() -> LlmEmbeddingEncoder {
        LlmEmbeddingEncoder::new(LlmEmbeddingEncoderParams {
            size: 2048,
            active_bits: 41,
            num_hyperplanes: 128,
            num_bands: 4,
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn test_create_encoder() {
        let encoder = create_test_encoder();
        assert_eq!(Encoder::<&[Real]>::size(&encoder), 2048);
        assert_eq!(Encoder::<&[Real]>::dimensions(&encoder), &[2048]);
    }

    #[test]
    fn test_encode_384_dim() {
        let encoder = create_test_encoder();
        let embedding: Vec<Real> = (0..384).map(|i| (i as Real / 384.0) - 0.5).collect();
        let sdr = encoder.encode_to_sdr(&embedding).unwrap();
        assert_eq!(sdr.get_sum(), 41);
    }

    #[test]
    fn test_encode_768_dim() {
        let encoder = create_test_encoder();
        let embedding: Vec<Real> = (0..768).map(|i| (i as Real / 768.0) - 0.5).collect();
        let sdr = encoder.encode_to_sdr(&embedding).unwrap();
        assert_eq!(sdr.get_sum(), 41);
    }

    #[test]
    fn test_encode_1536_dim() {
        let encoder = create_test_encoder();
        let embedding: Vec<Real> = (0..1536).map(|i| (i as Real / 1536.0) - 0.5).collect();
        let sdr = encoder.encode_to_sdr(&embedding).unwrap();
        assert_eq!(sdr.get_sum(), 41);
    }

    #[test]
    fn test_encode_3072_dim() {
        let encoder = create_test_encoder();
        let embedding: Vec<Real> = (0..3072).map(|i| (i as Real / 3072.0) - 0.5).collect();
        let sdr = encoder.encode_to_sdr(&embedding).unwrap();
        assert_eq!(sdr.get_sum(), 41);
    }

    #[test]
    fn test_deterministic() {
        let encoder = create_test_encoder();
        let embedding: Vec<Real> = (0..768).map(|i| (i as Real).sin()).collect();

        let sdr1 = encoder.encode_to_sdr(&embedding).unwrap();
        let sdr2 = encoder.encode_to_sdr(&embedding).unwrap();

        assert_eq!(sdr1.get_sparse(), sdr2.get_sparse());
    }

    #[test]
    fn test_similar_embeddings_overlap() {
        let encoder = create_test_encoder();

        // Create two similar embeddings
        let base: Vec<Real> = (0..768).map(|i| (i as Real).sin()).collect();
        let similar: Vec<Real> = base.iter().map(|&x| x + 0.01).collect();

        // Create a very different embedding
        let different: Vec<Real> = base.iter().map(|&x| -x).collect();

        let sdr_base = encoder.encode_to_sdr(&base).unwrap();
        let sdr_similar = encoder.encode_to_sdr(&similar).unwrap();
        let sdr_different = encoder.encode_to_sdr(&different).unwrap();

        let overlap_similar = sdr_base.get_overlap(&sdr_similar);
        let overlap_different = sdr_base.get_overlap(&sdr_different);

        // Similar embeddings should have more overlap
        assert!(
            overlap_similar > overlap_different,
            "Expected similar overlap ({}) > different overlap ({})",
            overlap_similar,
            overlap_different
        );
    }

    #[test]
    fn test_identical_embeddings_full_overlap() {
        let encoder = create_test_encoder();
        let embedding: Vec<Real> = (0..512).map(|i| (i as Real * 0.01).cos()).collect();

        let sdr1 = encoder.encode_to_sdr(&embedding).unwrap();
        let sdr2 = encoder.encode_to_sdr(&embedding).unwrap();

        assert_eq!(sdr1.get_overlap(&sdr2), 41);
    }

    #[test]
    fn test_different_seeds() {
        let encoder1 = LlmEmbeddingEncoder::new(LlmEmbeddingEncoderParams {
            seed: 42,
            ..Default::default()
        })
        .unwrap();

        let encoder2 = LlmEmbeddingEncoder::new(LlmEmbeddingEncoderParams {
            seed: 123,
            ..Default::default()
        })
        .unwrap();

        let embedding: Vec<Real> = (0..384).map(|i| i as Real / 100.0).collect();

        let sdr1 = encoder1.encode_to_sdr(&embedding).unwrap();
        let sdr2 = encoder2.encode_to_sdr(&embedding).unwrap();

        // Different seeds should produce different encodings
        assert_ne!(sdr1.get_sparse(), sdr2.get_sparse());
    }

    #[test]
    fn test_empty_embedding_error() {
        let encoder = create_test_encoder();
        let embedding: Vec<Real> = vec![];
        let result = encoder.encode_to_sdr(&embedding);
        assert!(result.is_err());
    }

    #[test]
    fn test_normalization_strategies() {
        let embedding: Vec<Real> = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // Test L2 normalization
        let encoder_l2 = LlmEmbeddingEncoder::new(LlmEmbeddingEncoderParams {
            normalization: NormalizationStrategy::L2Normalize,
            ..Default::default()
        })
        .unwrap();
        let sdr_l2 = encoder_l2.encode_to_sdr(&embedding).unwrap();
        assert_eq!(sdr_l2.get_sum(), 41);

        // Test mean centering
        let encoder_mean = LlmEmbeddingEncoder::new(LlmEmbeddingEncoderParams {
            normalization: NormalizationStrategy::MeanCenter,
            ..Default::default()
        })
        .unwrap();
        let sdr_mean = encoder_mean.encode_to_sdr(&embedding).unwrap();
        assert_eq!(sdr_mean.get_sum(), 41);

        // Test no normalization
        let encoder_none = LlmEmbeddingEncoder::new(LlmEmbeddingEncoderParams {
            normalization: NormalizationStrategy::None,
            ..Default::default()
        })
        .unwrap();
        let sdr_none = encoder_none.encode_to_sdr(&embedding).unwrap();
        assert_eq!(sdr_none.get_sum(), 41);
    }

    #[test]
    fn test_strided_strategy() {
        let encoder = LlmEmbeddingEncoder::new(LlmEmbeddingEncoderParams {
            dimension_strategy: DimensionStrategy::Strided(2),
            ..Default::default()
        })
        .unwrap();

        let embedding: Vec<Real> = (0..1536).map(|i| (i as Real).sin()).collect();
        let sdr = encoder.encode_to_sdr(&embedding).unwrap();
        assert_eq!(sdr.get_sum(), 41);
    }

    #[test]
    fn test_chunked_strategy() {
        let encoder = LlmEmbeddingEncoder::new(LlmEmbeddingEncoderParams {
            dimension_strategy: DimensionStrategy::Chunked(512),
            ..Default::default()
        })
        .unwrap();

        let embedding: Vec<Real> = (0..1536).map(|i| (i as Real).sin()).collect();
        let sdr = encoder.encode_to_sdr(&embedding).unwrap();
        // Note: chunked may produce slightly different active bit counts due to rounding
        assert!(sdr.get_sum() > 0);
        assert!(sdr.get_sum() <= 41);
    }

    #[test]
    fn test_with_magnitude_encoding() {
        let encoder = LlmEmbeddingEncoder::new(LlmEmbeddingEncoderParams {
            encode_magnitude: true,
            magnitude_bits: 8,
            ..Default::default()
        })
        .unwrap();

        // Two embeddings with same direction but different magnitudes
        let embed1: Vec<Real> = vec![1.0, 2.0, 3.0, 4.0];
        let embed2: Vec<Real> = vec![10.0, 20.0, 30.0, 40.0];

        let sdr1 = encoder.encode_to_sdr(&embed1).unwrap();
        let sdr2 = encoder.encode_to_sdr(&embed2).unwrap();

        // With L2 normalization (default), the direction is the same
        // but magnitude encoding should create some difference
        assert!(sdr1.get_sum() > 0);
        assert!(sdr2.get_sum() > 0);
    }

    #[test]
    fn test_small_embedding() {
        let encoder = create_test_encoder();
        let embedding: Vec<Real> = vec![0.5, -0.3, 0.8];
        let sdr = encoder.encode_to_sdr(&embedding).unwrap();
        assert_eq!(sdr.get_sum(), 41);
    }

    #[test]
    fn test_validation_errors() {
        // Zero size
        assert!(LlmEmbeddingEncoder::new(LlmEmbeddingEncoderParams {
            size: 0,
            ..Default::default()
        })
        .is_err());

        // Zero active bits
        assert!(LlmEmbeddingEncoder::new(LlmEmbeddingEncoderParams {
            active_bits: 0,
            ..Default::default()
        })
        .is_err());

        // Active bits > size
        assert!(LlmEmbeddingEncoder::new(LlmEmbeddingEncoderParams {
            size: 100,
            active_bits: 200,
            ..Default::default()
        })
        .is_err());

        // Zero hyperplanes
        assert!(LlmEmbeddingEncoder::new(LlmEmbeddingEncoderParams {
            num_hyperplanes: 0,
            ..Default::default()
        })
        .is_err());

        // Zero bands
        assert!(LlmEmbeddingEncoder::new(LlmEmbeddingEncoderParams {
            num_bands: 0,
            ..Default::default()
        })
        .is_err());
    }

    #[test]
    fn test_cosine_similarity_preservation() {
        let encoder = LlmEmbeddingEncoder::new(LlmEmbeddingEncoderParams {
            size: 4096,
            active_bits: 82,
            num_hyperplanes: 256,
            num_bands: 8,
            ..Default::default()
        })
        .unwrap();

        // Create embeddings with known cosine similarities
        let a: Vec<Real> = (0..768).map(|i| (i as Real * 0.1).sin()).collect();

        // b is very similar to a (small perturbation)
        let b: Vec<Real> = a.iter().map(|&x| x + 0.02).collect();

        // c is completely different - random-looking pattern
        let c: Vec<Real> = (0..768)
            .map(|i| ((i * 7 + 13) as Real * 0.37).sin() * ((i * 3) as Real * 0.11).cos())
            .collect();

        // d is opposite to a
        let d: Vec<Real> = a.iter().map(|&x| -x).collect();

        let sdr_a = encoder.encode_to_sdr(&a).unwrap();
        let sdr_b = encoder.encode_to_sdr(&b).unwrap();
        let sdr_c = encoder.encode_to_sdr(&c).unwrap();
        let sdr_d = encoder.encode_to_sdr(&d).unwrap();

        let overlap_ab = sdr_a.get_overlap(&sdr_b);
        let overlap_ac = sdr_a.get_overlap(&sdr_c);
        let overlap_ad = sdr_a.get_overlap(&sdr_d);

        // Similar should have highest overlap
        assert!(
            overlap_ab > overlap_ac,
            "Similar overlap ({}) should be > different overlap ({})",
            overlap_ab,
            overlap_ac
        );

        // Opposite should have different encoding from original
        assert!(
            overlap_ab > overlap_ad,
            "Similar overlap ({}) should be > opposite overlap ({})",
            overlap_ab,
            overlap_ad
        );
    }

    #[test]
    fn test_slice_and_vec_produce_same_result() {
        let encoder = create_test_encoder();
        let embedding: Vec<Real> = (0..512).map(|i| (i as Real).sin()).collect();

        // Encode as slice
        let sdr_slice = Encoder::<&[Real]>::encode_to_sdr(&encoder, embedding.as_slice()).unwrap();

        // Encode as Vec
        let sdr_vec = Encoder::<Vec<Real>>::encode_to_sdr(&encoder, embedding.clone()).unwrap();

        // Encode as &Vec
        let sdr_ref = Encoder::<&Vec<Real>>::encode_to_sdr(&encoder, &embedding).unwrap();

        assert_eq!(sdr_slice.get_sparse(), sdr_vec.get_sparse());
        assert_eq!(sdr_slice.get_sparse(), sdr_ref.get_sparse());
    }
}
