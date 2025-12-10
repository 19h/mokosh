//! Multi Encoder implementation.
//!
//! The Multi Encoder combines multiple encoders into a single composite encoder,
//! concatenating their outputs into one SDR.

use crate::encoders::Encoder;
use crate::error::{MokoshError, Result};
use crate::types::{Sdr, UInt};

/// A field in a multi-encoder with its name, offset, and size.
#[derive(Debug, Clone)]
pub struct EncoderField {
    /// Name of this field.
    pub name: String,
    /// Starting bit offset in the combined SDR.
    pub offset: UInt,
    /// Size of this encoder's output in bits.
    pub size: UInt,
}

/// Combines multiple encoders into a single composite encoder.
///
/// Each sub-encoder's output is placed at a specific offset in the combined SDR.
/// This allows encoding multiple features of an input into a single representation.
///
/// # Example
///
/// ```rust
/// use mokosh::encoders::{MultiEncoder, ScalarEncoder, ScalarEncoderParams, Encoder};
///
/// // Create individual encoders
/// let temp_encoder = ScalarEncoder::new(ScalarEncoderParams {
///     minimum: -10.0,
///     maximum: 40.0,
///     size: 50,
///     active_bits: 5,
///     ..Default::default()
/// }).unwrap();
///
/// let humidity_encoder = ScalarEncoder::new(ScalarEncoderParams {
///     minimum: 0.0,
///     maximum: 100.0,
///     size: 50,
///     active_bits: 5,
///     ..Default::default()
/// }).unwrap();
///
/// // Combine them
/// let multi = MultiEncoder::new()
///     .add_encoder("temperature", Box::new(temp_encoder))
///     .add_encoder("humidity", Box::new(humidity_encoder))
///     .build()
///     .unwrap();
///
/// assert_eq!(multi.size(), 100);
/// ```
pub struct MultiEncoder<T> {
    /// List of encoders with their field metadata.
    encoders: Vec<(EncoderField, Box<dyn Encoder<T> + Send + Sync>)>,
    /// Total size of the combined output.
    size: UInt,
    /// Output dimensions.
    dimensions: Vec<UInt>,
}

impl<T> std::fmt::Debug for MultiEncoder<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiEncoder")
            .field("size", &self.size)
            .field("num_encoders", &self.encoders.len())
            .finish()
    }
}

/// Builder for constructing a MultiEncoder.
#[derive(Default)]
pub struct MultiEncoderBuilder<T> {
    encoders: Vec<(String, Box<dyn Encoder<T> + Send + Sync>)>,
}

impl<T> std::fmt::Debug for MultiEncoderBuilder<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiEncoderBuilder")
            .field("num_encoders", &self.encoders.len())
            .finish()
    }
}

impl<T> MultiEncoderBuilder<T> {
    /// Creates a new MultiEncoder builder.
    pub fn new() -> Self {
        Self {
            encoders: Vec::new(),
        }
    }

    /// Adds an encoder with a given name.
    pub fn add_encoder(mut self, name: &str, encoder: Box<dyn Encoder<T> + Send + Sync>) -> Self {
        self.encoders.push((name.to_string(), encoder));
        self
    }

    /// Builds the MultiEncoder.
    pub fn build(self) -> Result<MultiEncoder<T>> {
        if self.encoders.is_empty() {
            return Err(MokoshError::InvalidParameter {
                name: "encoders",
                message: "Must provide at least one encoder".to_string(),
            });
        }

        let mut offset: UInt = 0;
        let mut encoders_with_fields = Vec::new();

        for (name, encoder) in self.encoders {
            let size = encoder.size() as UInt;
            let field = EncoderField {
                name,
                offset,
                size,
            };
            offset += size;
            encoders_with_fields.push((field, encoder));
        }

        let total_size = offset;

        Ok(MultiEncoder {
            encoders: encoders_with_fields,
            size: total_size,
            dimensions: vec![total_size],
        })
    }
}

impl<T> MultiEncoder<T> {
    /// Creates a new MultiEncoder builder.
    pub fn new() -> MultiEncoderBuilder<T> {
        MultiEncoderBuilder::new()
    }

    /// Returns the number of sub-encoders.
    pub fn num_encoders(&self) -> usize {
        self.encoders.len()
    }

    /// Returns information about the encoder fields.
    pub fn fields(&self) -> Vec<&EncoderField> {
        self.encoders.iter().map(|(f, _)| f).collect()
    }

    /// Returns the field information by name.
    pub fn get_field(&self, name: &str) -> Option<&EncoderField> {
        self.encoders
            .iter()
            .find(|(f, _)| f.name == name)
            .map(|(f, _)| f)
    }
}

impl<T: Clone> Encoder<T> for MultiEncoder<T> {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, value: T, output: &mut Sdr) -> Result<()> {
        if output.dimensions() != self.dimensions.as_slice() {
            return Err(MokoshError::DimensionMismatch {
                expected: self.dimensions.clone(),
                actual: output.dimensions().to_vec(),
            });
        }

        let mut all_sparse: Vec<UInt> = Vec::new();

        for (field, encoder) in &self.encoders {
            // Create temporary SDR for this encoder
            let temp_sdr = encoder.encode_to_sdr(value.clone())?;

            // Add the bits with offset
            for bit in temp_sdr.get_sparse() {
                all_sparse.push(field.offset + bit);
            }
        }

        all_sparse.sort_unstable();
        output.set_sparse_unchecked(all_sparse);

        Ok(())
    }
}

/// A multi-encoder that accepts a vector of values, one for each sub-encoder.
pub struct VecMultiEncoder<T> {
    /// List of encoders with their field metadata.
    encoders: Vec<(EncoderField, Box<dyn Encoder<T> + Send + Sync>)>,
    /// Total size of the combined output.
    size: UInt,
    /// Output dimensions.
    dimensions: Vec<UInt>,
}

impl<T> std::fmt::Debug for VecMultiEncoder<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VecMultiEncoder")
            .field("size", &self.size)
            .field("num_encoders", &self.encoders.len())
            .finish()
    }
}

/// Builder for constructing a VecMultiEncoder.
#[derive(Default)]
pub struct VecMultiEncoderBuilder<T> {
    encoders: Vec<(String, Box<dyn Encoder<T> + Send + Sync>)>,
}

impl<T> std::fmt::Debug for VecMultiEncoderBuilder<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VecMultiEncoderBuilder")
            .field("num_encoders", &self.encoders.len())
            .finish()
    }
}

impl<T> VecMultiEncoderBuilder<T> {
    /// Creates a new builder.
    pub fn new() -> Self {
        Self {
            encoders: Vec::new(),
        }
    }

    /// Adds an encoder with a given name.
    pub fn add_encoder(mut self, name: &str, encoder: Box<dyn Encoder<T> + Send + Sync>) -> Self {
        self.encoders.push((name.to_string(), encoder));
        self
    }

    /// Builds the VecMultiEncoder.
    pub fn build(self) -> Result<VecMultiEncoder<T>> {
        if self.encoders.is_empty() {
            return Err(MokoshError::InvalidParameter {
                name: "encoders",
                message: "Must provide at least one encoder".to_string(),
            });
        }

        let mut offset: UInt = 0;
        let mut encoders_with_fields = Vec::new();

        for (name, encoder) in self.encoders {
            let size = encoder.size() as UInt;
            let field = EncoderField {
                name,
                offset,
                size,
            };
            offset += size;
            encoders_with_fields.push((field, encoder));
        }

        let total_size = offset;

        Ok(VecMultiEncoder {
            encoders: encoders_with_fields,
            size: total_size,
            dimensions: vec![total_size],
        })
    }
}

impl<T> VecMultiEncoder<T> {
    /// Creates a new builder.
    pub fn new() -> VecMultiEncoderBuilder<T> {
        VecMultiEncoderBuilder::new()
    }

    /// Returns the number of sub-encoders.
    pub fn num_encoders(&self) -> usize {
        self.encoders.len()
    }

    /// Returns information about the encoder fields.
    pub fn fields(&self) -> Vec<&EncoderField> {
        self.encoders.iter().map(|(f, _)| f).collect()
    }
}

impl<T> Encoder<Vec<T>> for VecMultiEncoder<T> {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, values: Vec<T>, output: &mut Sdr) -> Result<()> {
        if values.len() != self.encoders.len() {
            return Err(MokoshError::InvalidParameter {
                name: "values",
                message: format!(
                    "Expected {} values, got {}",
                    self.encoders.len(),
                    values.len()
                ),
            });
        }

        if output.dimensions() != self.dimensions.as_slice() {
            return Err(MokoshError::DimensionMismatch {
                expected: self.dimensions.clone(),
                actual: output.dimensions().to_vec(),
            });
        }

        let mut all_sparse: Vec<UInt> = Vec::new();

        for ((field, encoder), value) in self.encoders.iter().zip(values.into_iter()) {
            let temp_sdr = encoder.encode_to_sdr(value)?;

            for bit in temp_sdr.get_sparse() {
                all_sparse.push(field.offset + bit);
            }
        }

        all_sparse.sort_unstable();
        output.set_sparse_unchecked(all_sparse);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoders::{ScalarEncoder, ScalarEncoderParams};

    #[test]
    fn test_vec_multi_encoder() {
        let enc1 = ScalarEncoder::new(ScalarEncoderParams {
            minimum: 0.0,
            maximum: 100.0,
            size: 50,
            active_bits: 5,
            ..Default::default()
        })
        .unwrap();

        let enc2 = ScalarEncoder::new(ScalarEncoderParams {
            minimum: 0.0,
            maximum: 100.0,
            size: 50,
            active_bits: 5,
            ..Default::default()
        })
        .unwrap();

        let multi = VecMultiEncoder::new()
            .add_encoder("a", Box::new(enc1))
            .add_encoder("b", Box::new(enc2))
            .build()
            .unwrap();

        assert_eq!(multi.size(), 100);
        assert_eq!(multi.num_encoders(), 2);

        let sdr = multi.encode_to_sdr(vec![25.0, 75.0]).unwrap();
        assert_eq!(sdr.get_sum(), 10); // 5 + 5 active bits
    }

    #[test]
    fn test_fields() {
        let enc1 = ScalarEncoder::new(ScalarEncoderParams {
            minimum: 0.0,
            maximum: 100.0,
            size: 30,
            active_bits: 5,
            ..Default::default()
        })
        .unwrap();

        let enc2 = ScalarEncoder::new(ScalarEncoderParams {
            minimum: 0.0,
            maximum: 100.0,
            size: 40,
            active_bits: 5,
            ..Default::default()
        })
        .unwrap();

        let multi = VecMultiEncoder::new()
            .add_encoder("first", Box::new(enc1))
            .add_encoder("second", Box::new(enc2))
            .build()
            .unwrap();

        let fields = multi.fields();
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].name, "first");
        assert_eq!(fields[0].offset, 0);
        assert_eq!(fields[0].size, 30);
        assert_eq!(fields[1].name, "second");
        assert_eq!(fields[1].offset, 30);
        assert_eq!(fields[1].size, 40);
    }

    #[test]
    fn test_wrong_value_count() {
        let enc = ScalarEncoder::new(ScalarEncoderParams {
            minimum: 0.0,
            maximum: 100.0,
            size: 50,
            active_bits: 5,
            ..Default::default()
        })
        .unwrap();

        let multi = VecMultiEncoder::new()
            .add_encoder("a", Box::new(enc))
            .build()
            .unwrap();

        let result = multi.encode_to_sdr(vec![1.0, 2.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_encoder() {
        let result: Result<VecMultiEncoder<f64>> = VecMultiEncoder::new().build();
        assert!(result.is_err());
    }
}
