//! Category Encoder implementation.
//!
//! The Category Encoder encodes discrete categories (strings or integers)
//! into non-overlapping SDR representations.

use crate::encoders::Encoder;
use crate::error::{MokoshError, Result};
use crate::types::{Sdr, UInt};
use std::collections::HashMap;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Parameters for creating a Category Encoder.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CategoryEncoderParams {
    /// List of category names. The order determines the encoding.
    pub categories: Vec<String>,

    /// Total number of bits in the output SDR.
    /// If 0, computed automatically from categories and active_bits.
    pub size: UInt,

    /// Number of active bits for each category encoding.
    pub active_bits: UInt,
}

impl Default for CategoryEncoderParams {
    fn default() -> Self {
        Self {
            categories: Vec::new(),
            size: 0,
            active_bits: 21,
        }
    }
}

/// Encodes discrete categories into SDR representations.
///
/// Each category is assigned a unique, non-overlapping block of bits.
/// Unknown categories can optionally be encoded or rejected.
///
/// # Example
///
/// ```rust
/// use mokosh::encoders::{CategoryEncoder, CategoryEncoderParams, Encoder};
///
/// let encoder = CategoryEncoder::new(CategoryEncoderParams {
///     categories: vec!["red".to_string(), "green".to_string(), "blue".to_string()],
///     active_bits: 10,
///     ..Default::default()
/// }).unwrap();
///
/// let sdr_red = encoder.encode_to_sdr("red").unwrap();
/// let sdr_blue = encoder.encode_to_sdr("blue").unwrap();
///
/// // Different categories have no overlap
/// assert_eq!(sdr_red.get_overlap(&sdr_blue), 0);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CategoryEncoder {
    /// Mapping from category name to index.
    category_map: HashMap<String, usize>,

    /// List of categories in order.
    categories: Vec<String>,

    /// Total size of output.
    size: UInt,

    /// Number of active bits per category.
    active_bits: UInt,

    /// Output dimensions.
    dimensions: Vec<UInt>,
}

impl CategoryEncoder {
    /// Creates a new Category Encoder.
    pub fn new(params: CategoryEncoderParams) -> Result<Self> {
        if params.categories.is_empty() {
            return Err(MokoshError::InvalidParameter {
                name: "categories",
                message: "Must provide at least one category".to_string(),
            });
        }

        if params.active_bits == 0 {
            return Err(MokoshError::InvalidParameter {
                name: "active_bits",
                message: "Must be > 0".to_string(),
            });
        }

        // Calculate size if not provided
        let size = if params.size > 0 {
            params.size
        } else {
            (params.categories.len() as UInt) * params.active_bits
        };

        // Validate size is sufficient
        let min_size = (params.categories.len() as UInt) * params.active_bits;
        if size < min_size {
            return Err(MokoshError::InvalidParameter {
                name: "size",
                message: format!(
                    "Size {} is too small for {} categories with {} active bits (need at least {})",
                    size,
                    params.categories.len(),
                    params.active_bits,
                    min_size
                ),
            });
        }

        // Build category map
        let mut category_map = HashMap::new();
        for (idx, cat) in params.categories.iter().enumerate() {
            if category_map.contains_key(cat) {
                return Err(MokoshError::InvalidParameter {
                    name: "categories",
                    message: format!("Duplicate category: {}", cat),
                });
            }
            category_map.insert(cat.clone(), idx);
        }

        Ok(Self {
            category_map,
            categories: params.categories,
            size,
            active_bits: params.active_bits,
            dimensions: vec![size],
        })
    }

    /// Returns the number of categories.
    pub fn num_categories(&self) -> usize {
        self.categories.len()
    }

    /// Returns the list of categories.
    pub fn categories(&self) -> &[String] {
        &self.categories
    }

    /// Returns the index for a category, if it exists.
    pub fn category_index(&self, category: &str) -> Option<usize> {
        self.category_map.get(category).copied()
    }

    /// Encodes a category by its index.
    pub fn encode_index(&self, index: usize, output: &mut Sdr) -> Result<()> {
        if index >= self.categories.len() {
            return Err(MokoshError::IndexOutOfBounds {
                index,
                size: self.categories.len(),
            });
        }

        if output.dimensions() != self.dimensions.as_slice() {
            return Err(MokoshError::DimensionMismatch {
                expected: self.dimensions.clone(),
                actual: output.dimensions().to_vec(),
            });
        }

        // Calculate the starting bit for this category
        let bits_per_category = self.size / self.categories.len() as UInt;
        let start = (index as UInt) * bits_per_category;

        // Generate active bits
        let sparse: Vec<UInt> = (start..start + self.active_bits).collect();
        output.set_sparse_unchecked(sparse);

        Ok(())
    }
}

impl Encoder<&str> for CategoryEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, value: &str, output: &mut Sdr) -> Result<()> {
        let index = self.category_map.get(value).ok_or_else(|| {
            MokoshError::InvalidParameter {
                name: "category",
                message: format!("Unknown category: {}", value),
            }
        })?;

        self.encode_index(*index, output)
    }
}

impl Encoder<String> for CategoryEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, value: String, output: &mut Sdr) -> Result<()> {
        self.encode(value.as_str(), output)
    }
}

impl Encoder<usize> for CategoryEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, value: usize, output: &mut Sdr) -> Result<()> {
        self.encode_index(value, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_encoder() {
        let encoder = CategoryEncoder::new(CategoryEncoderParams {
            categories: vec!["a".to_string(), "b".to_string(), "c".to_string()],
            active_bits: 10,
            ..Default::default()
        })
        .unwrap();

        assert_eq!(encoder.num_categories(), 3);
        assert_eq!(Encoder::<&str>::size(&encoder), 30);
    }

    #[test]
    fn test_encode_categories() {
        let encoder = CategoryEncoder::new(CategoryEncoderParams {
            categories: vec!["red".to_string(), "green".to_string(), "blue".to_string()],
            active_bits: 10,
            ..Default::default()
        })
        .unwrap();

        let sdr_red = encoder.encode_to_sdr("red").unwrap();
        let sdr_green = encoder.encode_to_sdr("green").unwrap();
        let sdr_blue = encoder.encode_to_sdr("blue").unwrap();

        // Each should have correct number of active bits
        assert_eq!(sdr_red.get_sum(), 10);
        assert_eq!(sdr_green.get_sum(), 10);
        assert_eq!(sdr_blue.get_sum(), 10);

        // Categories should not overlap
        assert_eq!(sdr_red.get_overlap(&sdr_green), 0);
        assert_eq!(sdr_red.get_overlap(&sdr_blue), 0);
        assert_eq!(sdr_green.get_overlap(&sdr_blue), 0);
    }

    #[test]
    fn test_encode_by_index() {
        let encoder = CategoryEncoder::new(CategoryEncoderParams {
            categories: vec!["a".to_string(), "b".to_string()],
            active_bits: 5,
            ..Default::default()
        })
        .unwrap();

        let sdr_by_name = encoder.encode_to_sdr("a").unwrap();
        let sdr_by_index: Sdr = Encoder::<usize>::encode_to_sdr(&encoder, 0).unwrap();

        assert_eq!(sdr_by_name.get_sparse(), sdr_by_index.get_sparse());
    }

    #[test]
    fn test_unknown_category() {
        let encoder = CategoryEncoder::new(CategoryEncoderParams {
            categories: vec!["a".to_string(), "b".to_string()],
            active_bits: 5,
            ..Default::default()
        })
        .unwrap();

        let result = encoder.encode_to_sdr("unknown");
        assert!(result.is_err());
    }

    #[test]
    fn test_duplicate_categories() {
        let result = CategoryEncoder::new(CategoryEncoderParams {
            categories: vec!["a".to_string(), "b".to_string(), "a".to_string()],
            active_bits: 5,
            ..Default::default()
        });

        assert!(result.is_err());
    }

    #[test]
    fn test_deterministic() {
        let encoder = CategoryEncoder::new(CategoryEncoderParams {
            categories: vec!["x".to_string(), "y".to_string(), "z".to_string()],
            active_bits: 7,
            ..Default::default()
        })
        .unwrap();

        let sdr1 = encoder.encode_to_sdr("y").unwrap();
        let sdr2 = encoder.encode_to_sdr("y").unwrap();

        assert_eq!(sdr1.get_sparse(), sdr2.get_sparse());
    }
}
