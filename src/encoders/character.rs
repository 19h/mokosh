//! Character Encoder implementation.
//!
//! Encodes individual characters into SDR representations for
//! character-level sequence modeling.

use crate::encoders::Encoder;
use crate::error::{MokoshError, Result};
use crate::types::{Sdr, UInt};
use std::collections::HashMap;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Parameters for creating a Character Encoder.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CharacterEncoderParams {
    /// Character set to encode. If empty, uses ASCII printable characters.
    pub charset: Option<Vec<char>>,

    /// Number of active bits per character.
    pub active_bits: UInt,

    /// Whether to include an "unknown" character for out-of-vocabulary chars.
    pub include_unknown: bool,

    /// Whether similar characters should have overlapping representations.
    /// If true, adjacent characters in the charset share some bits.
    pub semantic_similarity: bool,

    /// Number of bits of overlap between adjacent characters (if semantic_similarity is true).
    pub overlap_bits: UInt,
}

impl Default for CharacterEncoderParams {
    fn default() -> Self {
        Self {
            charset: None,
            active_bits: 21,
            include_unknown: true,
            semantic_similarity: false,
            overlap_bits: 5,
        }
    }
}

/// Encodes individual characters into SDR representations.
///
/// Supports custom character sets and optional semantic similarity
/// where similar characters share some active bits.
///
/// # Example
///
/// ```rust
/// use mokosh::encoders::{CharacterEncoder, CharacterEncoderParams, Encoder};
///
/// let encoder = CharacterEncoder::new(CharacterEncoderParams {
///     active_bits: 10,
///     ..Default::default()
/// }).unwrap();
///
/// let sdr_a = encoder.encode_to_sdr('a').unwrap();
/// let sdr_b = encoder.encode_to_sdr('b').unwrap();
///
/// assert_eq!(sdr_a.get_sum(), 10);
/// assert_eq!(sdr_b.get_sum(), 10);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CharacterEncoder {
    /// Mapping from character to index.
    char_to_idx: HashMap<char, usize>,

    /// List of characters in order.
    charset: Vec<char>,

    /// Number of active bits.
    active_bits: UInt,

    /// Whether unknown characters are supported.
    include_unknown: bool,

    /// Whether to use semantic similarity.
    semantic_similarity: bool,

    /// Overlap bits for semantic similarity.
    overlap_bits: UInt,

    /// Size of the output SDR.
    size: UInt,

    /// Output dimensions.
    dimensions: Vec<UInt>,
}

impl CharacterEncoder {
    /// Creates a new Character Encoder.
    pub fn new(params: CharacterEncoderParams) -> Result<Self> {
        if params.active_bits == 0 {
            return Err(MokoshError::InvalidParameter {
                name: "active_bits",
                message: "Must be > 0".to_string(),
            });
        }

        if params.semantic_similarity && params.overlap_bits >= params.active_bits {
            return Err(MokoshError::InvalidParameter {
                name: "overlap_bits",
                message: "Must be less than active_bits for semantic similarity".to_string(),
            });
        }

        let charset: Vec<char> = params.charset.unwrap_or_else(|| {
            // ASCII printable characters (32-126) plus common control characters
            (32u8..=126u8).map(|c| c as char).collect()
        });

        if charset.is_empty() {
            return Err(MokoshError::InvalidParameter {
                name: "charset",
                message: "Must have at least one character".to_string(),
            });
        }

        let mut char_to_idx = HashMap::new();
        for (idx, &ch) in charset.iter().enumerate() {
            char_to_idx.insert(ch, idx);
        }

        let num_chars = charset.len() + if params.include_unknown { 1 } else { 0 };

        // Calculate size based on encoding method
        let size = if params.semantic_similarity {
            // With overlap, we need: first_char_bits + (num_chars - 1) * unique_bits_per_char
            let unique_bits = params.active_bits - params.overlap_bits;
            params.active_bits + (num_chars as UInt - 1) * unique_bits
        } else {
            // Non-overlapping: each character gets its own region
            num_chars as UInt * params.active_bits
        };

        Ok(Self {
            char_to_idx,
            charset,
            active_bits: params.active_bits,
            include_unknown: params.include_unknown,
            semantic_similarity: params.semantic_similarity,
            overlap_bits: params.overlap_bits,
            size,
            dimensions: vec![size],
        })
    }

    /// Returns the character set.
    pub fn charset(&self) -> &[char] {
        &self.charset
    }

    /// Returns the number of characters (including unknown if enabled).
    pub fn num_characters(&self) -> usize {
        self.charset.len() + if self.include_unknown { 1 } else { 0 }
    }

    /// Returns the index for a character.
    pub fn char_index(&self, ch: char) -> Option<usize> {
        self.char_to_idx.get(&ch).copied()
    }
}

impl Encoder<char> for CharacterEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, ch: char, output: &mut Sdr) -> Result<()> {
        if output.dimensions() != self.dimensions.as_slice() {
            return Err(MokoshError::DimensionMismatch {
                expected: self.dimensions.clone(),
                actual: output.dimensions().to_vec(),
            });
        }

        let idx = match self.char_to_idx.get(&ch) {
            Some(&i) => i,
            None => {
                if self.include_unknown {
                    self.charset.len() // Unknown character index
                } else {
                    return Err(MokoshError::InvalidParameter {
                        name: "character",
                        message: format!("Unknown character: {:?}", ch),
                    });
                }
            }
        };

        let sparse: Vec<UInt> = if self.semantic_similarity {
            let unique_bits = self.active_bits - self.overlap_bits;
            let start = idx as UInt * unique_bits;
            (start..start + self.active_bits).collect()
        } else {
            let start = idx as UInt * self.active_bits;
            (start..start + self.active_bits).collect()
        };

        output.set_sparse_unchecked(sparse);

        Ok(())
    }
}

impl Encoder<u8> for CharacterEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, byte: u8, output: &mut Sdr) -> Result<()> {
        self.encode(byte as char, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_encoder() {
        let encoder = CharacterEncoder::new(CharacterEncoderParams {
            active_bits: 15,
            ..Default::default()
        })
        .unwrap();

        // ASCII printable (95 chars) + unknown
        assert_eq!(encoder.num_characters(), 96);
    }

    #[test]
    fn test_custom_charset() {
        let encoder = CharacterEncoder::new(CharacterEncoderParams {
            charset: Some(vec!['a', 'b', 'c', 'd', 'e']),
            active_bits: 10,
            include_unknown: false,
            ..Default::default()
        })
        .unwrap();

        assert_eq!(encoder.num_characters(), 5);
        assert_eq!(Encoder::<char>::size(&encoder), 50);
    }

    #[test]
    fn test_encode_character() {
        let encoder = CharacterEncoder::new(CharacterEncoderParams {
            charset: Some(vec!['a', 'b', 'c']),
            active_bits: 10,
            include_unknown: false,
            semantic_similarity: false,
            ..Default::default()
        })
        .unwrap();

        let sdr_a = encoder.encode_to_sdr('a').unwrap();
        let sdr_b = encoder.encode_to_sdr('b').unwrap();
        let sdr_c = encoder.encode_to_sdr('c').unwrap();

        assert_eq!(sdr_a.get_sum(), 10);
        assert_eq!(sdr_b.get_sum(), 10);
        assert_eq!(sdr_c.get_sum(), 10);

        // Non-overlapping
        assert_eq!(sdr_a.get_overlap(&sdr_b), 0);
        assert_eq!(sdr_b.get_overlap(&sdr_c), 0);
    }

    #[test]
    fn test_semantic_similarity() {
        let encoder = CharacterEncoder::new(CharacterEncoderParams {
            charset: Some(vec!['a', 'b', 'c', 'd', 'e']),
            active_bits: 10,
            include_unknown: false,
            semantic_similarity: true,
            overlap_bits: 4,
        })
        .unwrap();

        let sdr_a = encoder.encode_to_sdr('a').unwrap();
        let sdr_b = encoder.encode_to_sdr('b').unwrap();
        let sdr_c = encoder.encode_to_sdr('c').unwrap();

        assert_eq!(sdr_a.get_sum(), 10);

        // Adjacent characters should overlap by overlap_bits
        assert_eq!(sdr_a.get_overlap(&sdr_b), 4);
        assert_eq!(sdr_b.get_overlap(&sdr_c), 4);

        // Non-adjacent have less overlap
        let a_c_overlap = sdr_a.get_overlap(&sdr_c);
        assert!(a_c_overlap < 4);
    }

    #[test]
    fn test_unknown_character() {
        let encoder = CharacterEncoder::new(CharacterEncoderParams {
            charset: Some(vec!['a', 'b', 'c']),
            active_bits: 10,
            include_unknown: true,
            ..Default::default()
        })
        .unwrap();

        // Unknown character should work
        let sdr = encoder.encode_to_sdr('z').unwrap();
        assert_eq!(sdr.get_sum(), 10);
    }

    #[test]
    fn test_unknown_rejected() {
        let encoder = CharacterEncoder::new(CharacterEncoderParams {
            charset: Some(vec!['a', 'b', 'c']),
            active_bits: 10,
            include_unknown: false,
            ..Default::default()
        })
        .unwrap();

        let result = encoder.encode_to_sdr('z');
        assert!(result.is_err());
    }

    #[test]
    fn test_encode_byte() {
        let encoder = CharacterEncoder::new(CharacterEncoderParams::default()).unwrap();

        let sdr = encoder.encode_to_sdr(65u8).unwrap(); // 'A'
        assert_eq!(sdr.get_sum(), 21);
    }

    #[test]
    fn test_deterministic() {
        let encoder = CharacterEncoder::new(CharacterEncoderParams::default()).unwrap();

        let sdr1 = encoder.encode_to_sdr('X').unwrap();
        let sdr2 = encoder.encode_to_sdr('X').unwrap();

        assert_eq!(sdr1.get_sparse(), sdr2.get_sparse());
    }
}
