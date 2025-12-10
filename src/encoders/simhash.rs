//! SimHash Document Encoder implementation.
//!
//! The SimHashDocumentEncoder encodes documents and text into SDRs where similar
//! documents will have similar representations (small Hamming distance, high overlap).
//!
//! This is bitwise similarity, not semantic similarity - encodings for "apple" and
//! "computer" will have no special relation.

use crate::encoders::Encoder;
use crate::error::{MokoshError, Result};
use crate::types::{Real, Sdr, UInt};
use std::collections::{HashMap, HashSet};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Parameters for creating a SimHash Document Encoder.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SimHashDocumentEncoderParams {
    /// Total number of bits in the encoded output SDR.
    pub size: UInt,

    /// Number of true bits in the encoded output SDR.
    /// Mutually exclusive with `sparsity`.
    pub active_bits: UInt,

    /// Fraction of bits in the output which will be active.
    /// Alternative to `active_bits`. Mutually exclusive with `active_bits`.
    pub sparsity: Real,

    /// Whether capitalized letters should differ from lowercase.
    /// - If true: "DOGS" and "dogs" will have completely different encodings.
    /// - If false (default): "DOGS" and "dogs" will share the same encoding.
    pub case_sensitivity: bool,

    /// If `vocabulary` is set, should we encode tokens not in that vocabulary?
    /// - If true: Unrecognized tokens will be added with weight=1.
    /// - If false (default): Unrecognized tokens will be discarded.
    pub encode_orphans: bool,

    /// List of tokens to discard when encoding.
    pub excludes: Vec<String>,

    /// Maximum number of times a token can be repeated in a document.
    /// Setting to 1 acts as token de-duplication.
    /// 0 means no limit.
    pub frequency_ceiling: UInt,

    /// Minimum number of times a token must occur before being encoded.
    /// 0 means no minimum.
    pub frequency_floor: UInt,

    /// Enable token-level similarity in addition to document similarity.
    /// - If true: Similar tokens ("cat", "cats") will have similar influence.
    /// - If false (default): Similar tokens will have unique, unrelated influence.
    pub token_similarity: bool,

    /// Map of possible document tokens with weights.
    /// Example: `{"what": 3, "is": 1, "up": 2}`.
    pub vocabulary: HashMap<String, UInt>,
}

impl Default for SimHashDocumentEncoderParams {
    fn default() -> Self {
        Self {
            size: 400,
            active_bits: 21,
            sparsity: 0.0,
            case_sensitivity: false,
            encode_orphans: false,
            excludes: Vec::new(),
            frequency_ceiling: 0,
            frequency_floor: 0,
            token_similarity: false,
            vocabulary: HashMap::new(),
        }
    }
}

/// SimHash Document Encoder.
///
/// Encodes documents and text into SDRs where similar documents will have
/// similar representations. Uses the SimHash algorithm with SHA3/SHAKE256
/// hashing.
///
/// # Example
///
/// ```rust
/// use mokosh::encoders::{SimHashDocumentEncoder, SimHashDocumentEncoderParams, Encoder};
///
/// let encoder = SimHashDocumentEncoder::new(SimHashDocumentEncoderParams {
///     size: 400,
///     active_bits: 21,
///     ..Default::default()
/// }).unwrap();
///
/// let tokens = vec!["hello".to_string(), "world".to_string()];
/// let sdr = encoder.encode_to_sdr(tokens).unwrap();
/// assert_eq!(sdr.get_sum(), 21);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SimHashDocumentEncoder {
    size: UInt,
    active_bits: UInt,
    case_sensitivity: bool,
    encode_orphans: bool,
    excludes: HashSet<String>,
    frequency_ceiling: UInt,
    frequency_floor: UInt,
    token_similarity: bool,
    vocabulary: HashMap<String, UInt>,

    /// Output dimensions.
    dimensions: Vec<UInt>,
}

impl SimHashDocumentEncoder {
    /// Creates a new SimHash Document Encoder.
    pub fn new(params: SimHashDocumentEncoderParams) -> Result<Self> {
        // Validate size
        if params.size == 0 {
            return Err(MokoshError::InvalidParameter {
                name: "size",
                message: "Size must be > 0".to_string(),
            });
        }

        // Validate active_bits/sparsity (mutually exclusive)
        let num_active_args = (params.active_bits > 0) as u8 + (params.sparsity > 0.0) as u8;
        if num_active_args == 0 {
            return Err(MokoshError::InvalidParameter {
                name: "active_bits/sparsity",
                message: "Need one of: 'active_bits' or 'sparsity'".to_string(),
            });
        }
        if num_active_args > 1 {
            return Err(MokoshError::InvalidParameter {
                name: "active_bits/sparsity",
                message: "Specify only one of: 'active_bits' or 'sparsity'".to_string(),
            });
        }

        // Validate sparsity range
        if params.sparsity > 0.0 && !(0.0..=1.0).contains(&params.sparsity) {
            return Err(MokoshError::InvalidParameter {
                name: "sparsity",
                message: "Sparsity must be between 0.0 and 1.0".to_string(),
            });
        }

        // Validate frequency parameters
        if params.frequency_ceiling > 0
            && params.frequency_floor > 0
            && params.frequency_ceiling <= params.frequency_floor
        {
            return Err(MokoshError::InvalidParameter {
                name: "frequency_ceiling",
                message: "frequency_ceiling must be greater than frequency_floor".to_string(),
            });
        }

        // Validate encode_orphans requires vocabulary
        if params.encode_orphans && params.vocabulary.is_empty() {
            return Err(MokoshError::InvalidParameter {
                name: "encode_orphans",
                message: "encode_orphans requires vocabulary to be set".to_string(),
            });
        }

        // Calculate active_bits from sparsity if needed
        let active_bits = if params.sparsity > 0.0 {
            (params.size as Real * params.sparsity).round() as UInt
        } else {
            params.active_bits
        };

        if active_bits == 0 || active_bits >= params.size {
            return Err(MokoshError::InvalidParameter {
                name: "active_bits",
                message: "active_bits must be > 0 and < size".to_string(),
            });
        }

        // Process excludes and vocabulary for case insensitivity
        let excludes: HashSet<String> = if params.case_sensitivity {
            params.excludes.into_iter().collect()
        } else {
            params.excludes.into_iter().map(|s| s.to_lowercase()).collect()
        };

        let vocabulary: HashMap<String, UInt> = if params.case_sensitivity {
            params.vocabulary
        } else {
            params
                .vocabulary
                .into_iter()
                .map(|(k, v)| (k.to_lowercase(), v))
                .collect()
        };

        Ok(Self {
            size: params.size,
            active_bits,
            case_sensitivity: params.case_sensitivity,
            encode_orphans: params.encode_orphans,
            excludes,
            frequency_ceiling: params.frequency_ceiling,
            frequency_floor: params.frequency_floor,
            token_similarity: params.token_similarity,
            vocabulary,
            dimensions: vec![params.size],
        })
    }

    /// Returns the size.
    pub fn size_param(&self) -> UInt {
        self.size
    }

    /// Returns the number of active bits.
    pub fn active_bits(&self) -> UInt {
        self.active_bits
    }

    /// Hash a string token into a bit vector using a simple hash function.
    ///
    /// We use a combination of multiple hash rounds to get enough bits.
    fn hash_token(&self, token: &str) -> Vec<i32> {
        let mut bits = vec![0i32; self.size as usize];

        // Use multiple hashing rounds to fill the bit vector
        // This is a simplified version - the C++ uses SHA3/SHAKE256
        let bytes = token.as_bytes();

        // Use bit spreading from the hash to get better distribution
        let mut bit_idx = 0;
        let mut round = 0u32;

        while bit_idx < self.size as usize {
            // Get a 64-bit hash for this round
            let h = self.fnv1a_hash(bytes, round);

            // Extract 64 bits from this hash
            for shift in 0..64 {
                if bit_idx >= self.size as usize {
                    break;
                }
                bits[bit_idx] = if ((h >> shift) & 1) == 1 { 1 } else { 0 };
                bit_idx += 1;
            }

            round += 1;
        }

        bits
    }

    /// FNV-1a hash function with seed.
    #[inline]
    fn fnv1a_hash(&self, data: &[u8], seed: u32) -> u64 {
        const FNV_PRIME: u64 = 0x00000100000001B3;
        const FNV_OFFSET: u64 = 0xcbf29ce484222325;

        let mut hash = FNV_OFFSET;

        // Mix in the seed
        hash ^= seed as u64;
        hash = hash.wrapping_mul(FNV_PRIME);

        for &byte in data {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }

        hash
    }

    /// Convert hash bits to weighted adders (0 -> -weight, 1 -> +weight).
    fn bits_to_weighted_adder(&self, bits: &mut [i32], weight: UInt) {
        let w = weight as i32;
        for bit in bits.iter_mut() {
            *bit = if *bit == 0 { -w } else { w };
        }
    }

    /// Perform SimHash on the accumulated adders.
    fn simhash_adders(&self, adders: &[Vec<i32>]) -> Vec<u32> {
        if adders.is_empty() {
            return Vec::new();
        }

        // Sum all adder columns
        let mut sums = vec![0i64; self.size as usize];
        for adder in adders {
            for (i, &val) in adder.iter().enumerate() {
                sums[i] += val as i64;
            }
        }

        // Find the top N (active_bits) highest sums
        let mut indices: Vec<(usize, i64)> = sums.iter().cloned().enumerate().collect();
        indices.sort_by(|a, b| b.1.cmp(&a.1)); // Sort descending by sum value

        // Create sparse representation
        let mut sparse: Vec<u32> = indices
            .iter()
            .take(self.active_bits as usize)
            .map(|(idx, _)| *idx as u32)
            .collect();

        sparse.sort_unstable();
        sparse
    }

    /// Encode a list of tokens.
    pub fn encode_tokens(&self, tokens: &[String], output: &mut Sdr) -> Result<()> {
        if output.dimensions() != self.dimensions.as_slice() {
            return Err(MokoshError::DimensionMismatch {
                expected: self.dimensions.clone(),
                actual: output.dimensions().to_vec(),
            });
        }

        if tokens.is_empty() {
            output.set_sparse(&[])?;
            return Ok(());
        }

        let mut adders: Vec<Vec<i32>> = Vec::new();
        let mut token_histogram: HashMap<String, UInt> = HashMap::new();

        for token in tokens {
            let token = if self.case_sensitivity {
                token.clone()
            } else {
                token.to_lowercase()
            };

            // Check excludes
            if self.excludes.contains(&token) {
                continue;
            }

            // Determine token weight
            let token_weight = if !self.vocabulary.is_empty() {
                if let Some(&weight) = self.vocabulary.get(&token) {
                    weight
                } else if self.encode_orphans {
                    1 // Default weight for orphans
                } else {
                    continue; // Discard non-vocab token
                }
            } else {
                1 // Default weight when no vocabulary
            };

            // Update token histogram
            let count = token_histogram.entry(token.clone()).or_insert(0);
            *count += 1;

            // Check frequency floor
            if self.frequency_floor > 0 && *count <= self.frequency_floor {
                continue;
            }

            // Check frequency ceiling
            if self.frequency_ceiling > 0 && *count > self.frequency_ceiling {
                continue;
            }

            // Token similarity: hash individual characters
            if self.token_similarity {
                let mut char_histogram: HashMap<char, UInt> = HashMap::new();

                for c in token.chars() {
                    let char_str = c.to_string();
                    let char_weight = self.vocabulary.get(&char_str).copied().unwrap_or(token_weight);

                    // Check character frequency ceiling
                    let char_count = char_histogram.entry(c).or_insert(0);
                    *char_count += 1;

                    if self.frequency_ceiling > 0 && *char_count > self.frequency_ceiling {
                        continue;
                    }

                    // Hash the character
                    let mut hash_bits = self.hash_token(&char_str);
                    self.bits_to_weighted_adder(&mut hash_bits, char_weight);
                    adders.push(hash_bits);
                }
            }

            // Hash the whole token
            let adjusted_weight = if self.token_similarity {
                (token_weight as f64 * 1.5) as UInt
            } else {
                token_weight
            };

            let mut hash_bits = self.hash_token(&token);
            self.bits_to_weighted_adder(&mut hash_bits, adjusted_weight);
            adders.push(hash_bits);
        }

        // Perform SimHash
        let sparse = self.simhash_adders(&adders);

        if sparse.is_empty() {
            output.set_sparse(&[])?;
        } else {
            output.set_sparse_unchecked(sparse);
        }

        Ok(())
    }

    /// Encode a string by splitting on whitespace.
    pub fn encode_string(&self, input: &str, output: &mut Sdr) -> Result<()> {
        let tokens: Vec<String> = input.split_whitespace().map(|s| s.to_string()).collect();
        self.encode_tokens(&tokens, output)
    }
}

impl Encoder<Vec<String>> for SimHashDocumentEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, value: Vec<String>, output: &mut Sdr) -> Result<()> {
        self.encode_tokens(&value, output)
    }
}

/// Convenience implementation for string slices.
impl Encoder<&str> for SimHashDocumentEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, value: &str, output: &mut Sdr) -> Result<()> {
        self.encode_string(value, output)
    }
}

/// Convenience implementation for String.
impl Encoder<String> for SimHashDocumentEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, value: String, output: &mut Sdr) -> Result<()> {
        self.encode_string(&value, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_encoder() {
        let encoder = SimHashDocumentEncoder::new(SimHashDocumentEncoderParams {
            size: 400,
            active_bits: 21,
            ..Default::default()
        })
        .unwrap();

        assert_eq!(encoder.size_param(), 400);
        assert_eq!(encoder.active_bits(), 21);
    }

    #[test]
    fn test_encode_tokens() {
        let encoder = SimHashDocumentEncoder::new(SimHashDocumentEncoderParams {
            size: 400,
            active_bits: 21,
            ..Default::default()
        })
        .unwrap();

        let tokens = vec!["hello".to_string(), "world".to_string()];
        let sdr = encoder.encode_to_sdr(tokens).unwrap();
        assert_eq!(sdr.get_sum(), 21);
    }

    #[test]
    fn test_encode_string() {
        let encoder = SimHashDocumentEncoder::new(SimHashDocumentEncoderParams {
            size: 400,
            active_bits: 21,
            ..Default::default()
        })
        .unwrap();

        let mut output = Sdr::new(&[400]);
        encoder.encode_string("hello world", &mut output).unwrap();
        assert_eq!(output.get_sum(), 21);
    }

    #[test]
    fn test_empty_input() {
        let encoder = SimHashDocumentEncoder::new(SimHashDocumentEncoderParams {
            size: 400,
            active_bits: 21,
            ..Default::default()
        })
        .unwrap();

        let tokens: Vec<String> = vec![];
        let sdr = encoder.encode_to_sdr(tokens).unwrap();
        assert_eq!(sdr.get_sum(), 0);
    }

    #[test]
    fn test_similar_documents() {
        let encoder = SimHashDocumentEncoder::new(SimHashDocumentEncoderParams {
            size: 400,
            active_bits: 21,
            ..Default::default()
        })
        .unwrap();

        let doc1 = vec!["the".to_string(), "quick".to_string(), "brown".to_string(), "fox".to_string()];
        let doc2 = vec!["the".to_string(), "quick".to_string(), "brown".to_string(), "dog".to_string()];
        let doc3 = vec!["completely".to_string(), "different".to_string(), "document".to_string()];

        let sdr1 = encoder.encode_to_sdr(doc1).unwrap();
        let sdr2 = encoder.encode_to_sdr(doc2).unwrap();
        let sdr3 = encoder.encode_to_sdr(doc3).unwrap();

        // Similar documents should have higher overlap
        let overlap_similar = sdr1.get_overlap(&sdr2);
        let overlap_different = sdr1.get_overlap(&sdr3);

        assert!(overlap_similar > overlap_different);
    }

    #[test]
    fn test_case_insensitivity() {
        let encoder = SimHashDocumentEncoder::new(SimHashDocumentEncoderParams {
            size: 400,
            active_bits: 21,
            case_sensitivity: false,
            ..Default::default()
        })
        .unwrap();

        let doc1 = vec!["HELLO".to_string(), "WORLD".to_string()];
        let doc2 = vec!["hello".to_string(), "world".to_string()];

        let sdr1 = encoder.encode_to_sdr(doc1).unwrap();
        let sdr2 = encoder.encode_to_sdr(doc2).unwrap();

        // Case-insensitive encoding should be identical
        assert_eq!(sdr1.get_sparse(), sdr2.get_sparse());
    }

    #[test]
    fn test_case_sensitivity() {
        let encoder = SimHashDocumentEncoder::new(SimHashDocumentEncoderParams {
            size: 400,
            active_bits: 21,
            case_sensitivity: true,
            ..Default::default()
        })
        .unwrap();

        let doc1 = vec!["HELLO".to_string(), "WORLD".to_string()];
        let doc2 = vec!["hello".to_string(), "world".to_string()];

        let sdr1 = encoder.encode_to_sdr(doc1).unwrap();
        let sdr2 = encoder.encode_to_sdr(doc2).unwrap();

        // Case-sensitive encoding should be different
        assert_ne!(sdr1.get_sparse(), sdr2.get_sparse());
    }

    #[test]
    fn test_excludes() {
        let encoder = SimHashDocumentEncoder::new(SimHashDocumentEncoderParams {
            size: 400,
            active_bits: 21,
            excludes: vec!["the".to_string(), "a".to_string()],
            ..Default::default()
        })
        .unwrap();

        let doc1 = vec!["the".to_string(), "quick".to_string(), "fox".to_string()];
        let doc2 = vec!["quick".to_string(), "fox".to_string()];

        let sdr1 = encoder.encode_to_sdr(doc1).unwrap();
        let sdr2 = encoder.encode_to_sdr(doc2).unwrap();

        // "the" is excluded, so encodings should be the same
        assert_eq!(sdr1.get_sparse(), sdr2.get_sparse());
    }

    #[test]
    fn test_vocabulary_weighting() {
        let mut vocab = HashMap::new();
        vocab.insert("important".to_string(), 5);
        vocab.insert("less".to_string(), 1);

        let encoder = SimHashDocumentEncoder::new(SimHashDocumentEncoderParams {
            size: 400,
            active_bits: 21,
            vocabulary: vocab,
            encode_orphans: true,
            ..Default::default()
        })
        .unwrap();

        let tokens = vec!["important".to_string(), "less".to_string()];
        let sdr = encoder.encode_to_sdr(tokens).unwrap();
        assert_eq!(sdr.get_sum(), 21);
    }

    #[test]
    fn test_frequency_ceiling() {
        let encoder = SimHashDocumentEncoder::new(SimHashDocumentEncoderParams {
            size: 400,
            active_bits: 21,
            frequency_ceiling: 1, // De-duplicate
            ..Default::default()
        })
        .unwrap();

        let doc1 = vec!["hello".to_string(), "hello".to_string(), "hello".to_string()];
        let doc2 = vec!["hello".to_string()];

        let sdr1 = encoder.encode_to_sdr(doc1).unwrap();
        let sdr2 = encoder.encode_to_sdr(doc2).unwrap();

        // With ceiling=1, repeated tokens are deduplicated
        assert_eq!(sdr1.get_sparse(), sdr2.get_sparse());
    }

    #[test]
    fn test_sparsity_param() {
        let encoder = SimHashDocumentEncoder::new(SimHashDocumentEncoderParams {
            size: 400,
            active_bits: 0,
            sparsity: 0.05, // 5% of 400 = 20 bits
            ..Default::default()
        })
        .unwrap();

        assert_eq!(encoder.active_bits(), 20);

        let tokens = vec!["hello".to_string()];
        let sdr = encoder.encode_to_sdr(tokens).unwrap();
        assert_eq!(sdr.get_sum(), 20);
    }

    #[test]
    fn test_token_similarity() {
        let encoder_without = SimHashDocumentEncoder::new(SimHashDocumentEncoderParams {
            size: 400,
            active_bits: 21,
            token_similarity: false,
            ..Default::default()
        })
        .unwrap();

        let encoder_with = SimHashDocumentEncoder::new(SimHashDocumentEncoderParams {
            size: 400,
            active_bits: 21,
            token_similarity: true,
            ..Default::default()
        })
        .unwrap();

        let tokens = vec!["cat".to_string()];
        let similar_tokens = vec!["cats".to_string()];

        // Without token similarity
        let sdr1 = encoder_without.encode_to_sdr(tokens.clone()).unwrap();
        let sdr2 = encoder_without.encode_to_sdr(similar_tokens.clone()).unwrap();
        let overlap_without = sdr1.get_overlap(&sdr2);

        // With token similarity
        let sdr3 = encoder_with.encode_to_sdr(tokens).unwrap();
        let sdr4 = encoder_with.encode_to_sdr(similar_tokens).unwrap();
        let overlap_with = sdr3.get_overlap(&sdr4);

        // Token similarity should increase overlap for similar tokens
        assert!(overlap_with > overlap_without);
    }

    #[test]
    fn test_invalid_params_no_size() {
        let result = SimHashDocumentEncoder::new(SimHashDocumentEncoderParams {
            size: 0,
            ..Default::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_params_both_active() {
        let result = SimHashDocumentEncoder::new(SimHashDocumentEncoderParams {
            size: 400,
            active_bits: 21,
            sparsity: 0.05,
            ..Default::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_deterministic() {
        let encoder = SimHashDocumentEncoder::new(SimHashDocumentEncoderParams {
            size: 400,
            active_bits: 21,
            ..Default::default()
        })
        .unwrap();

        let tokens = vec!["hello".to_string(), "world".to_string()];

        let sdr1 = encoder.encode_to_sdr(tokens.clone()).unwrap();
        let sdr2 = encoder.encode_to_sdr(tokens).unwrap();

        // Same input should produce same output
        assert_eq!(sdr1.get_sparse(), sdr2.get_sparse());
    }
}
