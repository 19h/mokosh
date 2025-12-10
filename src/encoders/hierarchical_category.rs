//! Hierarchical Category Encoder implementation.
//!
//! Encodes hierarchical/taxonomic categories where items at the same
//! level share representations with their parent categories.

use crate::encoders::Encoder;
use crate::error::{MokoshError, Result};
use crate::types::{Sdr, UInt};
use std::collections::HashMap;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Parameters for creating a Hierarchical Category Encoder.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HierarchicalCategoryEncoderParams {
    /// Separator used in category paths (e.g., "/" for "animal/mammal/dog").
    pub separator: String,

    /// Bits allocated per hierarchy level.
    pub bits_per_level: UInt,

    /// Active bits per level.
    pub active_per_level: UInt,

    /// Maximum depth of hierarchy.
    pub max_depth: usize,
}

impl Default for HierarchicalCategoryEncoderParams {
    fn default() -> Self {
        Self {
            separator: "/".to_string(),
            bits_per_level: 50,
            active_per_level: 10,
            max_depth: 5,
        }
    }
}

/// Encodes hierarchical categories into SDR representations.
///
/// Categories are specified as paths (e.g., "animal/mammal/dog") and
/// items sharing ancestors have overlapping representations.
///
/// # Example
///
/// ```rust
/// use mokosh::encoders::{HierarchicalCategoryEncoder, HierarchicalCategoryEncoderParams, Encoder};
///
/// let encoder = HierarchicalCategoryEncoder::new(HierarchicalCategoryEncoderParams {
///     separator: "/".to_string(),
///     bits_per_level: 30,
///     active_per_level: 6,
///     max_depth: 4,
/// }).unwrap();
///
/// let dog = encoder.encode_to_sdr("animal/mammal/dog").unwrap();
/// let cat = encoder.encode_to_sdr("animal/mammal/cat").unwrap();
/// let snake = encoder.encode_to_sdr("animal/reptile/snake").unwrap();
///
/// // Dog and cat share "animal/mammal" ancestry
/// assert!(dog.get_overlap(&cat) > dog.get_overlap(&snake));
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HierarchicalCategoryEncoder {
    separator: String,
    bits_per_level: UInt,
    active_per_level: UInt,
    max_depth: usize,
    size: UInt,
    dimensions: Vec<UInt>,
    /// Cache of category hashes at each level.
    #[cfg_attr(feature = "serde", serde(skip))]
    cache: HashMap<String, Vec<UInt>>,
}

impl HierarchicalCategoryEncoder {
    /// Creates a new Hierarchical Category Encoder.
    pub fn new(params: HierarchicalCategoryEncoderParams) -> Result<Self> {
        if params.max_depth == 0 {
            return Err(MokoshError::InvalidParameter {
                name: "max_depth",
                message: "Must be > 0".to_string(),
            });
        }

        if params.active_per_level > params.bits_per_level {
            return Err(MokoshError::InvalidParameter {
                name: "active_per_level",
                message: "Cannot exceed bits_per_level".to_string(),
            });
        }

        let size = params.max_depth as UInt * params.bits_per_level;

        Ok(Self {
            separator: params.separator,
            bits_per_level: params.bits_per_level,
            active_per_level: params.active_per_level,
            max_depth: params.max_depth,
            size,
            dimensions: vec![size],
            cache: HashMap::new(),
        })
    }

    /// Returns the maximum depth.
    pub fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// Hash function for category names.
    fn hash_category(name: &str, level: usize) -> u64 {
        let mut hash: u64 = 0xcbf29ce484222325;
        let prime: u64 = 0x100000001b3;

        // Include level in hash to differentiate same names at different levels
        hash ^= level as u64;
        hash = hash.wrapping_mul(prime);

        for byte in name.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(prime);
        }

        hash
    }

    /// Generates bits for a category at a specific level.
    fn get_level_bits(&self, category: &str, level: usize) -> Vec<UInt> {
        let level_offset = level as UInt * self.bits_per_level;
        let hash = Self::hash_category(category, level);

        let mut bits = Vec::with_capacity(self.active_per_level as usize);
        let mut state = hash;

        while bits.len() < self.active_per_level as usize {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let bit = level_offset + (state % self.bits_per_level as u64) as UInt;
            if !bits.contains(&bit) {
                bits.push(bit);
            }
        }

        bits
    }
}

impl Encoder<&str> for HierarchicalCategoryEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, path: &str, output: &mut Sdr) -> Result<()> {
        if output.dimensions() != self.dimensions.as_slice() {
            return Err(MokoshError::DimensionMismatch {
                expected: self.dimensions.clone(),
                actual: output.dimensions().to_vec(),
            });
        }

        let parts: Vec<&str> = path.split(&self.separator).collect();

        if parts.is_empty() {
            return Err(MokoshError::InvalidParameter {
                name: "path",
                message: "Empty category path".to_string(),
            });
        }

        let mut sparse = Vec::new();

        // Build cumulative path and encode each level
        let mut cumulative = String::new();

        for (level, part) in parts.iter().enumerate() {
            if level >= self.max_depth {
                break;
            }

            if !cumulative.is_empty() {
                cumulative.push_str(&self.separator);
            }
            cumulative.push_str(part);

            // Get bits for this level
            let level_bits = self.get_level_bits(&cumulative, level);
            sparse.extend(level_bits);
        }

        sparse.sort_unstable();
        sparse.dedup();
        output.set_sparse_unchecked(sparse);

        Ok(())
    }
}

impl Encoder<String> for HierarchicalCategoryEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, path: String, output: &mut Sdr) -> Result<()> {
        self.encode(path.as_str(), output)
    }
}

impl Encoder<Vec<&str>> for HierarchicalCategoryEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, path_parts: Vec<&str>, output: &mut Sdr) -> Result<()> {
        let path = path_parts.join(&self.separator);
        self.encode(path.as_str(), output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_encoder() {
        let encoder = HierarchicalCategoryEncoder::new(HierarchicalCategoryEncoderParams {
            max_depth: 4,
            bits_per_level: 50,
            active_per_level: 10,
            ..Default::default()
        })
        .unwrap();

        assert_eq!(encoder.max_depth(), 4);
        assert_eq!(Encoder::<&str>::size(&encoder), 200);
    }

    #[test]
    fn test_encode_category() {
        let encoder = HierarchicalCategoryEncoder::new(HierarchicalCategoryEncoderParams {
            max_depth: 3,
            bits_per_level: 30,
            active_per_level: 6,
            ..Default::default()
        })
        .unwrap();

        let sdr = encoder.encode_to_sdr("animal/mammal/dog").unwrap();

        // 3 levels * 6 active bits = 18 (may be less due to dedup)
        assert!(sdr.get_sum() <= 18);
        assert!(sdr.get_sum() >= 15); // Allow some overlap
    }

    #[test]
    fn test_hierarchy_overlap() {
        let encoder = HierarchicalCategoryEncoder::new(HierarchicalCategoryEncoderParams {
            max_depth: 4,
            bits_per_level: 40,
            active_per_level: 8,
            ..Default::default()
        })
        .unwrap();

        let dog = encoder.encode_to_sdr("animal/mammal/canine/dog").unwrap();
        let wolf = encoder.encode_to_sdr("animal/mammal/canine/wolf").unwrap();
        let cat = encoder.encode_to_sdr("animal/mammal/feline/cat").unwrap();
        let snake = encoder.encode_to_sdr("animal/reptile/snake").unwrap();

        // Dog and wolf share 3 levels (animal/mammal/canine)
        let dog_wolf = dog.get_overlap(&wolf);

        // Dog and cat share 2 levels (animal/mammal)
        let dog_cat = dog.get_overlap(&cat);

        // Dog and snake share 1 level (animal)
        let dog_snake = dog.get_overlap(&snake);

        assert!(dog_wolf > dog_cat);
        assert!(dog_cat > dog_snake);
    }

    #[test]
    fn test_same_category() {
        let encoder = HierarchicalCategoryEncoder::new(HierarchicalCategoryEncoderParams::default()).unwrap();

        let sdr1 = encoder.encode_to_sdr("product/electronics/phone").unwrap();
        let sdr2 = encoder.encode_to_sdr("product/electronics/phone").unwrap();

        assert_eq!(sdr1.get_sparse(), sdr2.get_sparse());
    }

    #[test]
    fn test_custom_separator() {
        let encoder = HierarchicalCategoryEncoder::new(HierarchicalCategoryEncoderParams {
            separator: "::".to_string(),
            ..Default::default()
        })
        .unwrap();

        let sdr = encoder.encode_to_sdr("root::child::grandchild").unwrap();
        assert!(sdr.get_sum() > 0);
    }

    #[test]
    fn test_encode_parts() {
        let encoder = HierarchicalCategoryEncoder::new(HierarchicalCategoryEncoderParams::default()).unwrap();

        let sdr1 = encoder.encode_to_sdr("a/b/c").unwrap();
        let sdr2 = encoder.encode_to_sdr(vec!["a", "b", "c"]).unwrap();

        assert_eq!(sdr1.get_sparse(), sdr2.get_sparse());
    }

    #[test]
    fn test_truncate_deep_path() {
        let encoder = HierarchicalCategoryEncoder::new(HierarchicalCategoryEncoderParams {
            max_depth: 2,
            bits_per_level: 30,
            active_per_level: 6,
            ..Default::default()
        })
        .unwrap();

        let sdr = encoder.encode_to_sdr("a/b/c/d/e").unwrap();

        // Only 2 levels encoded = 12 bits max
        assert!(sdr.get_sum() <= 12);
    }

    #[test]
    fn test_single_level() {
        let encoder = HierarchicalCategoryEncoder::new(HierarchicalCategoryEncoderParams {
            bits_per_level: 40,
            active_per_level: 10,
            ..Default::default()
        })
        .unwrap();

        let sdr = encoder.encode_to_sdr("root").unwrap();
        assert_eq!(sdr.get_sum(), 10);
    }

    #[test]
    fn test_deterministic() {
        let encoder = HierarchicalCategoryEncoder::new(HierarchicalCategoryEncoderParams::default()).unwrap();

        let sdr1 = encoder.encode_to_sdr("test/path/here").unwrap();
        let sdr2 = encoder.encode_to_sdr("test/path/here").unwrap();

        assert_eq!(sdr1.get_sparse(), sdr2.get_sparse());
    }
}
