//! Ordinal Encoder implementation.
//!
//! Encodes ordered categories where adjacent categories have overlapping
//! representations (e.g., low < medium < high).

use crate::encoders::Encoder;
use crate::error::{MokoshError, Result};
use crate::types::{Sdr, UInt};
use std::collections::HashMap;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Parameters for creating an Ordinal Encoder.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OrdinalEncoderParams {
    /// Ordered list of categories (first is lowest, last is highest).
    pub categories: Vec<String>,

    /// Total number of bits in output SDR.
    pub size: UInt,

    /// Number of active bits.
    pub active_bits: UInt,
}

impl Default for OrdinalEncoderParams {
    fn default() -> Self {
        Self {
            categories: vec![
                "very_low".to_string(),
                "low".to_string(),
                "medium".to_string(),
                "high".to_string(),
                "very_high".to_string(),
            ],
            size: 100,
            active_bits: 21,
        }
    }
}

/// Encodes ordinal (ordered) categories into SDR representations.
///
/// Unlike categorical encoding, ordinal encoding preserves order:
/// adjacent categories share more overlap than distant ones.
///
/// # Example
///
/// ```rust
/// use mokosh::encoders::{OrdinalEncoder, OrdinalEncoderParams, Encoder};
///
/// let encoder = OrdinalEncoder::new(OrdinalEncoderParams {
///     categories: vec![
///         "low".to_string(),
///         "medium".to_string(),
///         "high".to_string(),
///     ],
///     size: 60,
///     active_bits: 15,
/// }).unwrap();
///
/// let low = encoder.encode_to_sdr("low").unwrap();
/// let medium = encoder.encode_to_sdr("medium").unwrap();
/// let high = encoder.encode_to_sdr("high").unwrap();
///
/// // Adjacent categories have more overlap
/// assert!(low.get_overlap(&medium) > low.get_overlap(&high));
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OrdinalEncoder {
    /// Ordered list of categories.
    categories: Vec<String>,

    /// Mapping from category to index.
    category_map: HashMap<String, usize>,

    /// Total size.
    size: UInt,

    /// Number of active bits.
    active_bits: UInt,

    /// Output dimensions.
    dimensions: Vec<UInt>,
}

impl OrdinalEncoder {
    /// Creates a new Ordinal Encoder.
    pub fn new(params: OrdinalEncoderParams) -> Result<Self> {
        if params.categories.is_empty() {
            return Err(MokoshError::InvalidParameter {
                name: "categories",
                message: "Must have at least one category".to_string(),
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
            categories: params.categories,
            category_map,
            size: params.size,
            active_bits: params.active_bits,
            dimensions: vec![params.size],
        })
    }

    /// Returns the number of categories.
    pub fn num_categories(&self) -> usize {
        self.categories.len()
    }

    /// Returns the categories.
    pub fn categories(&self) -> &[String] {
        &self.categories
    }

    /// Returns the index for a category.
    pub fn category_index(&self, category: &str) -> Option<usize> {
        self.category_map.get(category).copied()
    }
}

impl Encoder<&str> for OrdinalEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, category: &str, output: &mut Sdr) -> Result<()> {
        let index = self.category_map.get(category).ok_or_else(|| {
            MokoshError::InvalidParameter {
                name: "category",
                message: format!("Unknown category: {}", category),
            }
        })?;

        if output.dimensions() != self.dimensions.as_slice() {
            return Err(MokoshError::DimensionMismatch {
                expected: self.dimensions.clone(),
                actual: output.dimensions().to_vec(),
            });
        }

        // Calculate position based on category index
        let num_positions = self.size - self.active_bits + 1;
        let position = if self.categories.len() == 1 {
            0
        } else {
            (*index as f32 / (self.categories.len() - 1) as f32 * (num_positions - 1) as f32)
                .round() as UInt
        };

        let sparse: Vec<UInt> = (position..position + self.active_bits).collect();
        output.set_sparse_unchecked(sparse);

        Ok(())
    }
}

impl Encoder<String> for OrdinalEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, category: String, output: &mut Sdr) -> Result<()> {
        self.encode(category.as_str(), output)
    }
}

impl Encoder<usize> for OrdinalEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, index: usize, output: &mut Sdr) -> Result<()> {
        if index >= self.categories.len() {
            return Err(MokoshError::IndexOutOfBounds {
                index,
                size: self.categories.len(),
            });
        }
        self.encode(self.categories[index].as_str(), output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_encoder() {
        let encoder = OrdinalEncoder::new(OrdinalEncoderParams {
            categories: vec!["a".to_string(), "b".to_string(), "c".to_string()],
            size: 60,
            active_bits: 15,
        })
        .unwrap();

        assert_eq!(encoder.num_categories(), 3);
        assert_eq!(Encoder::<&str>::size(&encoder), 60);
    }

    #[test]
    fn test_encode_category() {
        let encoder = OrdinalEncoder::new(OrdinalEncoderParams {
            categories: vec!["low".to_string(), "medium".to_string(), "high".to_string()],
            size: 50,
            active_bits: 10,
        })
        .unwrap();

        let sdr = encoder.encode_to_sdr("medium").unwrap();
        assert_eq!(sdr.get_sum(), 10);
    }

    #[test]
    fn test_adjacent_overlap() {
        let encoder = OrdinalEncoder::new(OrdinalEncoderParams {
            categories: vec![
                "very_low".to_string(),
                "low".to_string(),
                "medium".to_string(),
                "high".to_string(),
                "very_high".to_string(),
            ],
            size: 100,
            active_bits: 21,
        })
        .unwrap();

        let very_low = encoder.encode_to_sdr("very_low").unwrap();
        let low = encoder.encode_to_sdr("low").unwrap();
        let medium = encoder.encode_to_sdr("medium").unwrap();
        let high = encoder.encode_to_sdr("high").unwrap();
        let very_high = encoder.encode_to_sdr("very_high").unwrap();

        // Adjacent should have more overlap than distant
        let adjacent = very_low.get_overlap(&low);
        let distant = very_low.get_overlap(&very_high);

        assert!(adjacent > distant);

        // Middle should overlap with both sides
        let med_low = medium.get_overlap(&low);
        let med_high = medium.get_overlap(&high);
        assert!(med_low > 0);
        assert!(med_high > 0);
    }

    #[test]
    fn test_extremes() {
        let encoder = OrdinalEncoder::new(OrdinalEncoderParams {
            categories: vec!["min".to_string(), "mid".to_string(), "max".to_string()],
            size: 40,
            active_bits: 10,
        })
        .unwrap();

        let min_sdr = encoder.encode_to_sdr("min").unwrap();
        let max_sdr = encoder.encode_to_sdr("max").unwrap();

        // Min should start at 0
        assert!(min_sdr.get_sparse().contains(&0));

        // Max should end at size-1
        assert!(max_sdr.get_sparse().contains(&39));
    }

    #[test]
    fn test_encode_by_index() {
        let encoder = OrdinalEncoder::new(OrdinalEncoderParams {
            categories: vec!["a".to_string(), "b".to_string(), "c".to_string()],
            size: 30,
            active_bits: 8,
        })
        .unwrap();

        let sdr_by_name = encoder.encode_to_sdr("b").unwrap();
        let sdr_by_index: Sdr = Encoder::<usize>::encode_to_sdr(&encoder, 1).unwrap();

        assert_eq!(sdr_by_name.get_sparse(), sdr_by_index.get_sparse());
    }

    #[test]
    fn test_unknown_category() {
        let encoder = OrdinalEncoder::new(OrdinalEncoderParams {
            categories: vec!["a".to_string(), "b".to_string()],
            size: 20,
            active_bits: 5,
        })
        .unwrap();

        let result = encoder.encode_to_sdr("unknown");
        assert!(result.is_err());
    }

    #[test]
    fn test_duplicate_category() {
        let result = OrdinalEncoder::new(OrdinalEncoderParams {
            categories: vec!["a".to_string(), "b".to_string(), "a".to_string()],
            size: 30,
            active_bits: 8,
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_single_category() {
        let encoder = OrdinalEncoder::new(OrdinalEncoderParams {
            categories: vec!["only".to_string()],
            size: 20,
            active_bits: 5,
        })
        .unwrap();

        let sdr = encoder.encode_to_sdr("only").unwrap();
        assert_eq!(sdr.get_sum(), 5);
    }

    #[test]
    fn test_deterministic() {
        let encoder = OrdinalEncoder::new(OrdinalEncoderParams::default()).unwrap();

        let sdr1 = encoder.encode_to_sdr("medium").unwrap();
        let sdr2 = encoder.encode_to_sdr("medium").unwrap();

        assert_eq!(sdr1.get_sparse(), sdr2.get_sparse());
    }
}
