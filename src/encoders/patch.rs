//! Patch Encoder implementation.
//!
//! Encodes small image patches into SDRs, similar to Vision Transformer
//! (ViT) patch embeddings but in sparse distributed form.

use crate::encoders::Encoder;
use crate::error::{MokoshError, Result};
use crate::types::{Real, Sdr, UInt};
use std::collections::HashSet;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Parameters for creating a Patch Encoder.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PatchEncoderParams {
    /// Width of the patch in pixels.
    pub patch_width: usize,

    /// Height of the patch in pixels.
    pub patch_height: usize,

    /// Number of color channels (1 for grayscale, 3 for RGB).
    pub channels: usize,

    /// Total number of bits in output SDR.
    pub size: UInt,

    /// Number of active bits in output SDR.
    pub active_bits: UInt,

    /// Number of random projections for encoding.
    pub num_projections: usize,
}

impl Default for PatchEncoderParams {
    fn default() -> Self {
        Self {
            patch_width: 8,
            patch_height: 8,
            channels: 1,
            size: 1024,
            active_bits: 41,
            num_projections: 64,
        }
    }
}

/// An image patch to encode.
#[derive(Debug, Clone)]
pub struct ImagePatch {
    /// Pixel values in row-major order, channel-last format.
    /// Values should be normalized to [0, 1].
    pub pixels: Vec<Real>,
    /// Width of the patch.
    pub width: usize,
    /// Height of the patch.
    pub height: usize,
    /// Number of channels.
    pub channels: usize,
}

impl ImagePatch {
    /// Creates a new grayscale patch from pixel values.
    pub fn grayscale(pixels: Vec<Real>, width: usize, height: usize) -> Result<Self> {
        if pixels.len() != width * height {
            return Err(MokoshError::InvalidParameter {
                name: "pixels",
                message: format!(
                    "Expected {} pixels, got {}",
                    width * height,
                    pixels.len()
                ),
            });
        }

        Ok(Self {
            pixels,
            width,
            height,
            channels: 1,
        })
    }

    /// Creates a new RGB patch from pixel values.
    pub fn rgb(pixels: Vec<Real>, width: usize, height: usize) -> Result<Self> {
        if pixels.len() != width * height * 3 {
            return Err(MokoshError::InvalidParameter {
                name: "pixels",
                message: format!(
                    "Expected {} pixels, got {}",
                    width * height * 3,
                    pixels.len()
                ),
            });
        }

        Ok(Self {
            pixels,
            width,
            height,
            channels: 3,
        })
    }
}

/// Encodes image patches into SDR representations.
///
/// Uses random projections to convert pixel values into a sparse
/// distributed representation that preserves visual similarity.
///
/// # Example
///
/// ```rust
/// use mokosh::encoders::{PatchEncoder, PatchEncoderParams, ImagePatch, Encoder};
///
/// let encoder = PatchEncoder::new(PatchEncoderParams {
///     patch_width: 4,
///     patch_height: 4,
///     channels: 1,
///     size: 256,
///     active_bits: 20,
///     num_projections: 32,
/// }).unwrap();
///
/// // Create a simple 4x4 grayscale patch
/// let pixels = vec![0.5; 16];
/// let patch = ImagePatch::grayscale(pixels, 4, 4).unwrap();
///
/// let sdr = encoder.encode_to_sdr(patch).unwrap();
/// assert_eq!(sdr.get_sum(), 20);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PatchEncoder {
    patch_width: usize,
    patch_height: usize,
    channels: usize,
    input_size: usize,
    size: UInt,
    active_bits: UInt,
    num_projections: usize,
    /// Random projection matrix (flattened: num_projections x input_size).
    projections: Vec<Real>,
    dimensions: Vec<UInt>,
}

impl PatchEncoder {
    /// Creates a new Patch Encoder.
    pub fn new(params: PatchEncoderParams) -> Result<Self> {
        Self::with_seed(params, 42)
    }

    /// Creates a new Patch Encoder with a specific seed.
    pub fn with_seed(params: PatchEncoderParams, seed: u64) -> Result<Self> {
        if params.patch_width == 0 || params.patch_height == 0 {
            return Err(MokoshError::InvalidParameter {
                name: "patch_width/patch_height",
                message: "Must be > 0".to_string(),
            });
        }

        if params.channels == 0 {
            return Err(MokoshError::InvalidParameter {
                name: "channels",
                message: "Must be > 0".to_string(),
            });
        }

        if params.active_bits > params.size {
            return Err(MokoshError::InvalidParameter {
                name: "active_bits",
                message: "Cannot exceed size".to_string(),
            });
        }

        let input_size = params.patch_width * params.patch_height * params.channels;

        // Generate random projections
        let mut projections = Vec::with_capacity(params.num_projections * input_size);
        let mut state = seed;

        for _ in 0..(params.num_projections * input_size) {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let value = ((state >> 33) as Real / (u32::MAX as Real / 2.0)) - 1.0;
            projections.push(value);
        }

        Ok(Self {
            patch_width: params.patch_width,
            patch_height: params.patch_height,
            channels: params.channels,
            input_size,
            size: params.size,
            active_bits: params.active_bits,
            num_projections: params.num_projections,
            projections,
            dimensions: vec![params.size],
        })
    }

    /// Returns the expected input size (width * height * channels).
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Computes projection hash for the patch.
    fn compute_projection_hash(&self, pixels: &[Real]) -> u128 {
        let mut hash: u128 = 0;

        for proj_idx in 0..self.num_projections.min(128) {
            let proj_start = proj_idx * self.input_size;
            let projection = &self.projections[proj_start..proj_start + self.input_size];

            let dot: Real = pixels
                .iter()
                .zip(projection.iter())
                .map(|(&p, &proj)| p * proj)
                .sum();

            if dot >= 0.0 {
                hash |= 1u128 << proj_idx;
            }
        }

        hash
    }
}

impl Encoder<ImagePatch> for PatchEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, patch: ImagePatch, output: &mut Sdr) -> Result<()> {
        if patch.width != self.patch_width || patch.height != self.patch_height {
            return Err(MokoshError::InvalidParameter {
                name: "patch",
                message: format!(
                    "Expected {}x{} patch, got {}x{}",
                    self.patch_width, self.patch_height, patch.width, patch.height
                ),
            });
        }

        if patch.channels != self.channels {
            return Err(MokoshError::InvalidParameter {
                name: "patch",
                message: format!(
                    "Expected {} channels, got {}",
                    self.channels, patch.channels
                ),
            });
        }

        if output.dimensions() != self.dimensions.as_slice() {
            return Err(MokoshError::DimensionMismatch {
                expected: self.dimensions.clone(),
                actual: output.dimensions().to_vec(),
            });
        }

        let hash = self.compute_projection_hash(&patch.pixels);

        // Use hash to select active bits
        let mut active_bits = HashSet::new();
        let mut state = hash as u64;

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

impl Encoder<Vec<Real>> for PatchEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, pixels: Vec<Real>, output: &mut Sdr) -> Result<()> {
        if pixels.len() != self.input_size {
            return Err(MokoshError::InvalidParameter {
                name: "pixels",
                message: format!(
                    "Expected {} pixels, got {}",
                    self.input_size,
                    pixels.len()
                ),
            });
        }

        let patch = ImagePatch {
            pixels,
            width: self.patch_width,
            height: self.patch_height,
            channels: self.channels,
        };

        self.encode(patch, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_encoder() {
        let encoder = PatchEncoder::new(PatchEncoderParams {
            patch_width: 8,
            patch_height: 8,
            channels: 3,
            size: 512,
            active_bits: 25,
            num_projections: 32,
        })
        .unwrap();

        assert_eq!(encoder.input_size(), 192); // 8 * 8 * 3
        assert_eq!(Encoder::<ImagePatch>::size(&encoder), 512);
    }

    #[test]
    fn test_encode_grayscale_patch() {
        let encoder = PatchEncoder::new(PatchEncoderParams {
            patch_width: 4,
            patch_height: 4,
            channels: 1,
            size: 256,
            active_bits: 20,
            num_projections: 32,
        })
        .unwrap();

        let pixels = vec![0.5; 16];
        let patch = ImagePatch::grayscale(pixels, 4, 4).unwrap();

        let sdr = encoder.encode_to_sdr(patch).unwrap();
        assert_eq!(sdr.get_sum(), 20);
    }

    #[test]
    fn test_similar_patches_overlap() {
        let encoder = PatchEncoder::new(PatchEncoderParams {
            patch_width: 4,
            patch_height: 4,
            channels: 1,
            size: 512,
            active_bits: 30,
            num_projections: 64,
        })
        .unwrap();

        // Identical patches (all white)
        let patch1 = ImagePatch::grayscale(vec![1.0; 16], 4, 4).unwrap();
        let patch2 = ImagePatch::grayscale(vec![1.0; 16], 4, 4).unwrap();

        // Very different patch (all black)
        let patch3 = ImagePatch::grayscale(vec![0.0; 16], 4, 4).unwrap();

        let sdr1 = encoder.encode_to_sdr(patch1).unwrap();
        let sdr2 = encoder.encode_to_sdr(patch2).unwrap();
        let sdr3 = encoder.encode_to_sdr(patch3).unwrap();

        // Identical patches should have full overlap
        assert_eq!(sdr1.get_overlap(&sdr2), 30);

        // Very different patch should have less overlap than identical
        let diff_overlap = sdr1.get_overlap(&sdr3);
        assert!(diff_overlap < 30, "Different patches should not have full overlap, got {}", diff_overlap);
    }

    #[test]
    fn test_encode_rgb_patch() {
        let encoder = PatchEncoder::new(PatchEncoderParams {
            patch_width: 2,
            patch_height: 2,
            channels: 3,
            size: 256,
            active_bits: 15,
            num_projections: 32,
        })
        .unwrap();

        let pixels = vec![0.5; 12]; // 2 * 2 * 3 = 12
        let patch = ImagePatch::rgb(pixels, 2, 2).unwrap();

        let sdr = encoder.encode_to_sdr(patch).unwrap();
        assert_eq!(sdr.get_sum(), 15);
    }

    #[test]
    fn test_encode_vec_directly() {
        let encoder = PatchEncoder::new(PatchEncoderParams {
            patch_width: 2,
            patch_height: 2,
            channels: 1,
            size: 128,
            active_bits: 10,
            num_projections: 16,
        })
        .unwrap();

        let pixels = vec![0.1, 0.2, 0.3, 0.4];
        let sdr = encoder.encode_to_sdr(pixels).unwrap();
        assert_eq!(sdr.get_sum(), 10);
    }

    #[test]
    fn test_wrong_patch_size() {
        let encoder = PatchEncoder::new(PatchEncoderParams {
            patch_width: 4,
            patch_height: 4,
            channels: 1,
            ..Default::default()
        })
        .unwrap();

        let patch = ImagePatch::grayscale(vec![0.5; 9], 3, 3).unwrap(); // Wrong size
        let result = encoder.encode_to_sdr(patch);
        assert!(result.is_err());
    }

    #[test]
    fn test_deterministic() {
        let encoder = PatchEncoder::new(PatchEncoderParams::default()).unwrap();

        let pixels = vec![0.3; 64];
        let patch1 = ImagePatch::grayscale(pixels.clone(), 8, 8).unwrap();
        let patch2 = ImagePatch::grayscale(pixels, 8, 8).unwrap();

        let sdr1 = encoder.encode_to_sdr(patch1).unwrap();
        let sdr2 = encoder.encode_to_sdr(patch2).unwrap();

        assert_eq!(sdr1.get_sparse(), sdr2.get_sparse());
    }
}
