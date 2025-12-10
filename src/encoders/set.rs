//! Set Encoder implementation.
//!
//! Encodes variable-size sets of items into fixed-size SDRs.

use crate::encoders::Encoder;
use crate::error::{MokoshError, Result};
use crate::types::{Sdr, UInt};
use std::collections::HashSet;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Parameters for creating a Set Encoder.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SetEncoderParams {
    /// Total number of bits in output SDR.
    pub size: UInt,

    /// Number of bits per set element.
    pub bits_per_element: UInt,

    /// Maximum number of elements to encode.
    /// If the set is larger, only this many are encoded.
    pub max_elements: usize,
}

impl Default for SetEncoderParams {
    fn default() -> Self {
        Self {
            size: 2048,
            bits_per_element: 5,
            max_elements: 50,
        }
    }
}

/// Encodes sets of items into SDR representations.
///
/// Each element in the set contributes a fixed number of bits
/// to the output, determined by hashing. Similar sets with
/// overlapping elements will have overlapping SDRs.
///
/// # Example
///
/// ```rust
/// use mokosh::encoders::{SetEncoder, SetEncoderParams, Encoder};
///
/// let encoder = SetEncoder::new(SetEncoderParams {
///     size: 500,
///     bits_per_element: 10,
///     max_elements: 20,
/// }).unwrap();
///
/// let set1 = vec!["apple", "banana", "orange"];
/// let set2 = vec!["apple", "banana", "grape"];
/// let set3 = vec!["car", "truck", "bus"];
///
/// let sdr1 = encoder.encode_to_sdr(set1).unwrap();
/// let sdr2 = encoder.encode_to_sdr(set2).unwrap();
/// let sdr3 = encoder.encode_to_sdr(set3).unwrap();
///
/// // Sets with shared elements have more overlap
/// assert!(sdr1.get_overlap(&sdr2) > sdr1.get_overlap(&sdr3));
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SetEncoder {
    size: UInt,
    bits_per_element: UInt,
    max_elements: usize,
    dimensions: Vec<UInt>,
}

impl SetEncoder {
    /// Creates a new Set Encoder.
    pub fn new(params: SetEncoderParams) -> Result<Self> {
        if params.bits_per_element == 0 {
            return Err(MokoshError::InvalidParameter {
                name: "bits_per_element",
                message: "Must be > 0".to_string(),
            });
        }

        if params.bits_per_element > params.size {
            return Err(MokoshError::InvalidParameter {
                name: "bits_per_element",
                message: "Cannot exceed size".to_string(),
            });
        }

        if params.max_elements == 0 {
            return Err(MokoshError::InvalidParameter {
                name: "max_elements",
                message: "Must be > 0".to_string(),
            });
        }

        Ok(Self {
            size: params.size,
            bits_per_element: params.bits_per_element,
            max_elements: params.max_elements,
            dimensions: vec![params.size],
        })
    }

    /// Returns the maximum number of elements.
    pub fn max_elements(&self) -> usize {
        self.max_elements
    }

    /// Hash function for set elements.
    fn hash_element(item: &str) -> u64 {
        let mut hash: u64 = 0xcbf29ce484222325;
        let prime: u64 = 0x100000001b3;

        for byte in item.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(prime);
        }

        hash
    }

    /// Gets the bits for a single element.
    fn get_element_bits(&self, item: &str) -> Vec<UInt> {
        let hash = Self::hash_element(item);
        let mut bits = Vec::with_capacity(self.bits_per_element as usize);
        let mut state = hash;

        while bits.len() < self.bits_per_element as usize {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let bit = (state % self.size as u64) as UInt;
            if !bits.contains(&bit) {
                bits.push(bit);
            }
        }

        bits
    }
}

impl Encoder<Vec<&str>> for SetEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, items: Vec<&str>, output: &mut Sdr) -> Result<()> {
        if output.dimensions() != self.dimensions.as_slice() {
            return Err(MokoshError::DimensionMismatch {
                expected: self.dimensions.clone(),
                actual: output.dimensions().to_vec(),
            });
        }

        let mut active_bits = HashSet::new();

        for item in items.iter().take(self.max_elements) {
            for bit in self.get_element_bits(item) {
                active_bits.insert(bit);
            }
        }

        let mut sparse: Vec<UInt> = active_bits.into_iter().collect();
        sparse.sort_unstable();
        output.set_sparse_unchecked(sparse);

        Ok(())
    }
}

impl Encoder<Vec<String>> for SetEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, items: Vec<String>, output: &mut Sdr) -> Result<()> {
        let refs: Vec<&str> = items.iter().map(|s| s.as_str()).collect();
        self.encode(refs, output)
    }
}

impl Encoder<HashSet<String>> for SetEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, items: HashSet<String>, output: &mut Sdr) -> Result<()> {
        let refs: Vec<&str> = items.iter().map(|s| s.as_str()).collect();
        self.encode(refs, output)
    }
}

impl Encoder<&[&str]> for SetEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, items: &[&str], output: &mut Sdr) -> Result<()> {
        self.encode(items.to_vec(), output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_encoder() {
        let encoder = SetEncoder::new(SetEncoderParams {
            size: 500,
            bits_per_element: 10,
            max_elements: 20,
        })
        .unwrap();

        assert_eq!(encoder.max_elements(), 20);
        assert_eq!(Encoder::<Vec<&str>>::size(&encoder), 500);
    }

    #[test]
    fn test_encode_set() {
        let encoder = SetEncoder::new(SetEncoderParams {
            size: 200,
            bits_per_element: 5,
            max_elements: 10,
        })
        .unwrap();

        let set = vec!["apple", "banana", "orange"];
        let sdr = encoder.encode_to_sdr(set).unwrap();

        // 3 elements * 5 bits = 15 (may be less due to collisions)
        assert!(sdr.get_sum() >= 10);
        assert!(sdr.get_sum() <= 15);
    }

    #[test]
    fn test_overlapping_sets() {
        let encoder = SetEncoder::new(SetEncoderParams {
            size: 500,
            bits_per_element: 10,
            max_elements: 20,
        })
        .unwrap();

        let set1 = vec!["a", "b", "c", "d"];
        let set2 = vec!["a", "b", "e", "f"]; // Shares a, b
        let set3 = vec!["x", "y", "z", "w"]; // No overlap

        let sdr1 = encoder.encode_to_sdr(set1).unwrap();
        let sdr2 = encoder.encode_to_sdr(set2).unwrap();
        let sdr3 = encoder.encode_to_sdr(set3).unwrap();

        let overlap_12 = sdr1.get_overlap(&sdr2);
        let overlap_13 = sdr1.get_overlap(&sdr3);

        // Should share bits from common elements
        assert!(overlap_12 > overlap_13);
        assert!(overlap_12 >= 15); // At least ~2 elements worth
    }

    #[test]
    fn test_empty_set() {
        let encoder = SetEncoder::new(SetEncoderParams::default()).unwrap();

        let sdr = encoder.encode_to_sdr(Vec::<&str>::new()).unwrap();
        assert_eq!(sdr.get_sum(), 0);
    }

    #[test]
    fn test_max_elements() {
        let encoder = SetEncoder::new(SetEncoderParams {
            size: 500,
            bits_per_element: 10,
            max_elements: 3,
        })
        .unwrap();

        // Large set, but only 3 should be encoded
        let set: Vec<&str> = (0..100).map(|i| if i == 0 { "a" } else { "b" }).collect();
        let sdr = encoder.encode_to_sdr(set).unwrap();

        // At most 3 * 10 = 30 bits
        assert!(sdr.get_sum() <= 30);
    }

    #[test]
    fn test_order_invariance() {
        let encoder = SetEncoder::new(SetEncoderParams::default()).unwrap();

        let set1 = vec!["a", "b", "c"];
        let set2 = vec!["c", "a", "b"]; // Same elements, different order

        let sdr1 = encoder.encode_to_sdr(set1).unwrap();
        let sdr2 = encoder.encode_to_sdr(set2).unwrap();

        // Same elements should produce same bits
        assert_eq!(sdr1.get_sparse(), sdr2.get_sparse());
    }

    #[test]
    fn test_encode_hashset() {
        let encoder = SetEncoder::new(SetEncoderParams::default()).unwrap();

        let mut set = HashSet::new();
        set.insert("apple".to_string());
        set.insert("banana".to_string());

        let sdr = encoder.encode_to_sdr(set).unwrap();
        assert!(sdr.get_sum() > 0);
    }

    #[test]
    fn test_encode_slice() {
        let encoder = SetEncoder::new(SetEncoderParams::default()).unwrap();

        let items: &[&str] = &["one", "two", "three"];
        let sdr = encoder.encode_to_sdr(items).unwrap();
        assert!(sdr.get_sum() > 0);
    }

    #[test]
    fn test_deterministic() {
        let encoder = SetEncoder::new(SetEncoderParams::default()).unwrap();

        let set = vec!["x", "y", "z"];

        let sdr1 = encoder.encode_to_sdr(set.clone()).unwrap();
        let sdr2 = encoder.encode_to_sdr(set).unwrap();

        assert_eq!(sdr1.get_sparse(), sdr2.get_sparse());
    }
}
