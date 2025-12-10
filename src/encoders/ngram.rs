//! N-Gram Encoder implementation.
//!
//! Encodes character or word n-grams into SDRs with overlapping
//! representations for similar sequences.

use crate::encoders::Encoder;
use crate::error::{MokoshError, Result};
use crate::types::{Sdr, UInt};
use std::collections::HashSet;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Parameters for creating an N-Gram Encoder.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NGramEncoderParams {
    /// Size of n-grams to extract.
    pub n: usize,

    /// Total number of bits in output SDR.
    pub size: UInt,

    /// Number of active bits per n-gram.
    pub bits_per_ngram: UInt,

    /// Maximum number of n-grams to encode (limits sparsity).
    pub max_ngrams: Option<usize>,

    /// Whether to use character-level (true) or word-level (false) n-grams.
    pub character_level: bool,
}

impl Default for NGramEncoderParams {
    fn default() -> Self {
        Self {
            n: 3,
            size: 2048,
            bits_per_ngram: 3,
            max_ngrams: Some(100),
            character_level: true,
        }
    }
}

/// Encodes text n-grams into SDR representations.
///
/// Extracts overlapping n-grams from input text and hashes each
/// to a set of bits, creating a representation that captures
/// local sequence structure.
///
/// # Example
///
/// ```rust
/// use mokosh::encoders::{NGramEncoder, NGramEncoderParams, Encoder};
///
/// let encoder = NGramEncoder::new(NGramEncoderParams {
///     n: 3,
///     size: 500,
///     bits_per_ngram: 5,
///     character_level: true,
///     ..Default::default()
/// }).unwrap();
///
/// let sdr1 = encoder.encode_to_sdr("hello").unwrap();
/// let sdr2 = encoder.encode_to_sdr("hella").unwrap();  // Similar
/// let sdr3 = encoder.encode_to_sdr("world").unwrap();  // Different
///
/// // Similar strings share n-grams
/// assert!(sdr1.get_overlap(&sdr2) > sdr1.get_overlap(&sdr3));
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NGramEncoder {
    n: usize,
    size: UInt,
    bits_per_ngram: UInt,
    max_ngrams: Option<usize>,
    character_level: bool,
    dimensions: Vec<UInt>,
}

impl NGramEncoder {
    /// Creates a new N-Gram Encoder.
    pub fn new(params: NGramEncoderParams) -> Result<Self> {
        if params.n == 0 {
            return Err(MokoshError::InvalidParameter {
                name: "n",
                message: "Must be > 0".to_string(),
            });
        }

        if params.bits_per_ngram == 0 {
            return Err(MokoshError::InvalidParameter {
                name: "bits_per_ngram",
                message: "Must be > 0".to_string(),
            });
        }

        if params.bits_per_ngram > params.size {
            return Err(MokoshError::InvalidParameter {
                name: "bits_per_ngram",
                message: "Cannot exceed size".to_string(),
            });
        }

        Ok(Self {
            n: params.n,
            size: params.size,
            bits_per_ngram: params.bits_per_ngram,
            max_ngrams: params.max_ngrams,
            character_level: params.character_level,
            dimensions: vec![params.size],
        })
    }

    /// Returns the n-gram size.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Hash function for n-grams.
    fn hash_ngram(&self, ngram: &str) -> Vec<UInt> {
        let mut result = Vec::with_capacity(self.bits_per_ngram as usize);
        let mut hash: u64 = 0xcbf29ce484222325; // FNV-1a offset
        let prime: u64 = 0x100000001b3;

        for byte in ngram.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(prime);
        }

        // Generate multiple bits from the hash
        for i in 0..self.bits_per_ngram {
            hash = hash.wrapping_mul(prime).wrapping_add(i as u64);
            let bit = (hash % self.size as u64) as UInt;
            result.push(bit);
        }

        result
    }

    /// Extracts character-level n-grams.
    fn extract_char_ngrams(&self, text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        if chars.len() < self.n {
            return vec![];
        }

        let mut ngrams = Vec::new();
        for window in chars.windows(self.n) {
            ngrams.push(window.iter().collect());
        }

        if let Some(max) = self.max_ngrams {
            ngrams.truncate(max);
        }

        ngrams
    }

    /// Extracts word-level n-grams.
    fn extract_word_ngrams(&self, text: &str) -> Vec<String> {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.len() < self.n {
            return vec![];
        }

        let mut ngrams = Vec::new();
        for window in words.windows(self.n) {
            ngrams.push(window.join(" "));
        }

        if let Some(max) = self.max_ngrams {
            ngrams.truncate(max);
        }

        ngrams
    }
}

impl Encoder<&str> for NGramEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, text: &str, output: &mut Sdr) -> Result<()> {
        if output.dimensions() != self.dimensions.as_slice() {
            return Err(MokoshError::DimensionMismatch {
                expected: self.dimensions.clone(),
                actual: output.dimensions().to_vec(),
            });
        }

        let ngrams = if self.character_level {
            self.extract_char_ngrams(text)
        } else {
            self.extract_word_ngrams(text)
        };

        let mut active_bits = HashSet::new();

        for ngram in ngrams {
            for bit in self.hash_ngram(&ngram) {
                active_bits.insert(bit);
            }
        }

        let mut sparse: Vec<UInt> = active_bits.into_iter().collect();
        sparse.sort_unstable();
        output.set_sparse_unchecked(sparse);

        Ok(())
    }
}

impl Encoder<String> for NGramEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, text: String, output: &mut Sdr) -> Result<()> {
        self.encode(text.as_str(), output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_encoder() {
        let encoder = NGramEncoder::new(NGramEncoderParams {
            n: 3,
            size: 1000,
            bits_per_ngram: 5,
            ..Default::default()
        })
        .unwrap();

        assert_eq!(encoder.n(), 3);
        assert_eq!(Encoder::<&str>::size(&encoder), 1000);
    }

    #[test]
    fn test_encode_text() {
        let encoder = NGramEncoder::new(NGramEncoderParams {
            n: 3,
            size: 500,
            bits_per_ngram: 3,
            character_level: true,
            ..Default::default()
        })
        .unwrap();

        let sdr = encoder.encode_to_sdr("hello").unwrap();

        // "hello" has 3 trigrams: "hel", "ell", "llo"
        // Each contributes bits_per_ngram bits (with possible overlap)
        assert!(sdr.get_sum() > 0);
        assert!(sdr.get_sum() <= 9); // At most 3 * 3 = 9 if no collisions
    }

    #[test]
    fn test_similar_text_overlap() {
        let encoder = NGramEncoder::new(NGramEncoderParams {
            n: 3,
            size: 1000,
            bits_per_ngram: 5,
            character_level: true,
            ..Default::default()
        })
        .unwrap();

        let sdr1 = encoder.encode_to_sdr("hello world").unwrap();
        let sdr2 = encoder.encode_to_sdr("hello there").unwrap();
        let sdr3 = encoder.encode_to_sdr("goodbye moon").unwrap();

        // Similar prefixes share n-grams
        let similar_overlap = sdr1.get_overlap(&sdr2);
        let different_overlap = sdr1.get_overlap(&sdr3);

        assert!(similar_overlap > different_overlap);
    }

    #[test]
    fn test_word_ngrams() {
        let encoder = NGramEncoder::new(NGramEncoderParams {
            n: 2,
            size: 500,
            bits_per_ngram: 4,
            character_level: false,
            ..Default::default()
        })
        .unwrap();

        let sdr1 = encoder.encode_to_sdr("the quick brown fox").unwrap();
        let sdr2 = encoder.encode_to_sdr("the quick red dog").unwrap();

        // Share "the quick" bigram
        assert!(sdr1.get_overlap(&sdr2) > 0);
    }

    #[test]
    fn test_short_text() {
        let encoder = NGramEncoder::new(NGramEncoderParams {
            n: 3,
            size: 500,
            bits_per_ngram: 5,
            character_level: true,
            ..Default::default()
        })
        .unwrap();

        // Text shorter than n should produce empty SDR
        let sdr = encoder.encode_to_sdr("ab").unwrap();
        assert_eq!(sdr.get_sum(), 0);
    }

    #[test]
    fn test_deterministic() {
        let encoder = NGramEncoder::new(NGramEncoderParams::default()).unwrap();

        let sdr1 = encoder.encode_to_sdr("test string").unwrap();
        let sdr2 = encoder.encode_to_sdr("test string").unwrap();

        assert_eq!(sdr1.get_sparse(), sdr2.get_sparse());
    }

    #[test]
    fn test_max_ngrams() {
        let encoder = NGramEncoder::new(NGramEncoderParams {
            n: 2,
            size: 1000,
            bits_per_ngram: 5,
            max_ngrams: Some(3),
            character_level: true,
        })
        .unwrap();

        // "abcdefgh" has 7 bigrams, but we limit to 3
        let sdr = encoder.encode_to_sdr("abcdefgh").unwrap();

        // At most 3 * 5 = 15 bits (likely fewer due to collisions)
        assert!(sdr.get_sum() <= 15);
    }

    #[test]
    fn test_encode_string() {
        let encoder = NGramEncoder::new(NGramEncoderParams::default()).unwrap();

        let text = String::from("hello");
        let sdr = encoder.encode_to_sdr(text).unwrap();
        assert!(sdr.get_sum() > 0);
    }
}
