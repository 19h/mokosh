//! Graph Node Encoder implementation.
//!
//! Encodes graph node embeddings into SDRs, preserving graph structure.

use crate::encoders::Encoder;
use crate::error::{MokoshError, Result};
use crate::types::{Real, Sdr, UInt};
use std::collections::HashSet;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Parameters for creating a Graph Node Encoder.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GraphNodeEncoderParams {
    /// Dimension of node embeddings.
    pub embedding_dim: usize,

    /// Total number of bits in output SDR.
    pub size: UInt,

    /// Number of active bits in output SDR.
    pub active_bits: UInt,

    /// Number of random hyperplanes for LSH.
    pub num_hyperplanes: usize,
}

impl Default for GraphNodeEncoderParams {
    fn default() -> Self {
        Self {
            embedding_dim: 64,
            size: 2048,
            active_bits: 41,
            num_hyperplanes: 128,
        }
    }
}

/// A graph node with its embedding and optional structural features.
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Node embedding vector.
    pub embedding: Vec<Real>,
    /// Node degree (optional, for structural encoding).
    pub degree: Option<usize>,
    /// Clustering coefficient (optional).
    pub clustering: Option<Real>,
}

impl GraphNode {
    /// Creates a new graph node with just an embedding.
    pub fn new(embedding: Vec<Real>) -> Self {
        Self {
            embedding,
            degree: None,
            clustering: None,
        }
    }

    /// Creates a graph node with structural features.
    pub fn with_structure(
        embedding: Vec<Real>,
        degree: usize,
        clustering: Real,
    ) -> Self {
        Self {
            embedding,
            degree: Some(degree),
            clustering: Some(clustering),
        }
    }
}

/// Encodes graph node embeddings into SDR representations.
///
/// Uses locality-sensitive hashing to convert node embeddings into
/// sparse representations while preserving similarity relationships.
///
/// # Example
///
/// ```rust
/// use mokosh::encoders::{GraphNodeEncoder, GraphNodeEncoderParams, GraphNode, Encoder};
///
/// let encoder = GraphNodeEncoder::new(GraphNodeEncoderParams {
///     embedding_dim: 4,
///     size: 256,
///     active_bits: 20,
///     num_hyperplanes: 32,
/// }).unwrap();
///
/// let node1 = GraphNode::new(vec![0.5, 0.3, 0.1, 0.8]);
/// let node2 = GraphNode::new(vec![0.5, 0.3, 0.1, 0.7]); // Similar
/// let node3 = GraphNode::new(vec![-0.5, -0.3, 0.9, -0.1]); // Different
///
/// let sdr1 = encoder.encode_to_sdr(node1).unwrap();
/// let sdr2 = encoder.encode_to_sdr(node2).unwrap();
/// let sdr3 = encoder.encode_to_sdr(node3).unwrap();
///
/// // Similar nodes should have more overlap
/// assert!(sdr1.get_overlap(&sdr2) > sdr1.get_overlap(&sdr3));
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GraphNodeEncoder {
    embedding_dim: usize,
    size: UInt,
    active_bits: UInt,
    num_hyperplanes: usize,
    /// Random hyperplanes for LSH.
    hyperplanes: Vec<Real>,
    dimensions: Vec<UInt>,
}

impl GraphNodeEncoder {
    /// Creates a new Graph Node Encoder.
    pub fn new(params: GraphNodeEncoderParams) -> Result<Self> {
        Self::with_seed(params, 42)
    }

    /// Creates a new Graph Node Encoder with a specific seed.
    pub fn with_seed(params: GraphNodeEncoderParams, seed: u64) -> Result<Self> {
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

        // Generate random hyperplanes
        let mut hyperplanes =
            Vec::with_capacity(params.num_hyperplanes * params.embedding_dim);

        let mut state = seed;
        for _ in 0..(params.num_hyperplanes * params.embedding_dim) {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
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

    /// Computes LSH hash for embedding.
    fn compute_lsh_hash(&self, embedding: &[Real]) -> u128 {
        let mut hash: u128 = 0;

        for hp_idx in 0..self.num_hyperplanes.min(128) {
            let hp_start = hp_idx * self.embedding_dim;
            let hyperplane = &self.hyperplanes[hp_start..hp_start + self.embedding_dim];

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

impl Encoder<GraphNode> for GraphNodeEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, node: GraphNode, output: &mut Sdr) -> Result<()> {
        if node.embedding.len() != self.embedding_dim {
            return Err(MokoshError::InvalidParameter {
                name: "embedding",
                message: format!(
                    "Expected {} dimensions, got {}",
                    self.embedding_dim,
                    node.embedding.len()
                ),
            });
        }

        if output.dimensions() != self.dimensions.as_slice() {
            return Err(MokoshError::DimensionMismatch {
                expected: self.dimensions.clone(),
                actual: output.dimensions().to_vec(),
            });
        }

        let lsh_hash = self.compute_lsh_hash(&node.embedding);

        // Mix in structural features if present
        let mut state = lsh_hash as u64;
        if let Some(degree) = node.degree {
            state = state.wrapping_add((degree as u64).wrapping_mul(0x517cc1b727220a95));
        }
        if let Some(clustering) = node.clustering {
            state = state.wrapping_add(((clustering * 1000.0) as u64).wrapping_mul(0x2545f4914f6cdd1d));
        }

        // Generate active bits
        let mut active_bits = HashSet::new();

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

impl Encoder<Vec<Real>> for GraphNodeEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, embedding: Vec<Real>, output: &mut Sdr) -> Result<()> {
        self.encode(GraphNode::new(embedding), output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_encoder() {
        let encoder = GraphNodeEncoder::new(GraphNodeEncoderParams {
            embedding_dim: 32,
            size: 512,
            active_bits: 25,
            num_hyperplanes: 64,
        })
        .unwrap();

        assert_eq!(encoder.embedding_dim(), 32);
        assert_eq!(Encoder::<GraphNode>::size(&encoder), 512);
    }

    #[test]
    fn test_encode_node() {
        let encoder = GraphNodeEncoder::new(GraphNodeEncoderParams {
            embedding_dim: 8,
            size: 256,
            active_bits: 20,
            num_hyperplanes: 32,
        })
        .unwrap();

        let node = GraphNode::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
        let sdr = encoder.encode_to_sdr(node).unwrap();

        assert_eq!(sdr.get_sum(), 20);
    }

    #[test]
    fn test_similar_nodes_overlap() {
        let encoder = GraphNodeEncoder::new(GraphNodeEncoderParams {
            embedding_dim: 8,
            size: 512,
            active_bits: 30,
            num_hyperplanes: 64,
        })
        .unwrap();

        let node1 = GraphNode::new(vec![0.5, 0.3, 0.1, 0.8, 0.2, 0.4, 0.6, 0.1]);
        let node2 = GraphNode::new(vec![0.5, 0.3, 0.1, 0.8, 0.2, 0.4, 0.6, 0.1]); // Identical
        let node3 = GraphNode::new(vec![-0.5, -0.3, -0.1, -0.8, -0.2, -0.4, -0.6, -0.1]); // Opposite

        let sdr1 = encoder.encode_to_sdr(node1).unwrap();
        let sdr2 = encoder.encode_to_sdr(node2).unwrap();
        let sdr3 = encoder.encode_to_sdr(node3).unwrap();

        // Identical should have full overlap
        assert_eq!(sdr1.get_overlap(&sdr2), 30);

        // Opposite should have less overlap
        assert!(sdr1.get_overlap(&sdr3) < 30);
    }

    #[test]
    fn test_with_structure() {
        let encoder = GraphNodeEncoder::new(GraphNodeEncoderParams {
            embedding_dim: 4,
            size: 256,
            active_bits: 20,
            num_hyperplanes: 32,
        })
        .unwrap();

        let node = GraphNode::with_structure(
            vec![0.1, 0.2, 0.3, 0.4],
            10,  // degree
            0.5, // clustering coefficient
        );

        let sdr = encoder.encode_to_sdr(node).unwrap();
        assert_eq!(sdr.get_sum(), 20);
    }

    #[test]
    fn test_encode_vec_directly() {
        let encoder = GraphNodeEncoder::new(GraphNodeEncoderParams {
            embedding_dim: 4,
            size: 128,
            active_bits: 15,
            num_hyperplanes: 16,
        })
        .unwrap();

        let sdr = encoder.encode_to_sdr(vec![0.1, 0.2, 0.3, 0.4]).unwrap();
        assert_eq!(sdr.get_sum(), 15);
    }

    #[test]
    fn test_wrong_dimension() {
        let encoder = GraphNodeEncoder::new(GraphNodeEncoderParams {
            embedding_dim: 8,
            ..Default::default()
        })
        .unwrap();

        let result = encoder.encode_to_sdr(GraphNode::new(vec![0.1, 0.2, 0.3]));
        assert!(result.is_err());
    }

    #[test]
    fn test_deterministic() {
        let encoder = GraphNodeEncoder::new(GraphNodeEncoderParams::default()).unwrap();

        let embedding: Vec<Real> = (0..64).map(|i| i as Real / 64.0).collect();
        let node1 = GraphNode::new(embedding.clone());
        let node2 = GraphNode::new(embedding);

        let sdr1 = encoder.encode_to_sdr(node1).unwrap();
        let sdr2 = encoder.encode_to_sdr(node2).unwrap();

        assert_eq!(sdr1.get_sparse(), sdr2.get_sparse());
    }
}
