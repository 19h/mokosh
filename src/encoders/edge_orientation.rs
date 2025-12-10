//! Edge Orientation Encoder implementation.
//!
//! Encodes oriented edge features (similar to Gabor filters) into SDRs.

use crate::encoders::Encoder;
use crate::error::{MokoshError, Result};
use crate::types::{Real, Sdr, UInt};
use std::f32::consts::PI;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Parameters for creating an Edge Orientation Encoder.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EdgeOrientationEncoderParams {
    /// Number of orientation bins (typically 8 or 16).
    pub num_orientations: usize,

    /// Bits per orientation.
    pub bits_per_orientation: UInt,

    /// Active bits per orientation.
    pub active_bits_per_orientation: UInt,

    /// Bits for encoding magnitude.
    pub magnitude_bits: UInt,

    /// Active bits for magnitude.
    pub magnitude_active: UInt,

    /// Minimum magnitude threshold.
    pub min_magnitude: Real,

    /// Maximum magnitude for normalization.
    pub max_magnitude: Real,
}

impl Default for EdgeOrientationEncoderParams {
    fn default() -> Self {
        Self {
            num_orientations: 8,
            bits_per_orientation: 20,
            active_bits_per_orientation: 5,
            magnitude_bits: 40,
            magnitude_active: 10,
            min_magnitude: 0.01,
            max_magnitude: 1.0,
        }
    }
}

/// An oriented edge feature.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OrientedEdge {
    /// Orientation angle in radians [0, PI).
    pub orientation: Real,
    /// Magnitude/strength of the edge.
    pub magnitude: Real,
}

impl OrientedEdge {
    /// Creates a new oriented edge.
    pub fn new(orientation: Real, magnitude: Real) -> Self {
        Self {
            orientation: orientation.rem_euclid(PI),
            magnitude: magnitude.max(0.0),
        }
    }

    /// Creates from gradient components (dx, dy).
    pub fn from_gradient(dx: Real, dy: Real) -> Self {
        let magnitude = (dx * dx + dy * dy).sqrt();
        let orientation = dy.atan2(dx).rem_euclid(PI);
        Self::new(orientation, magnitude)
    }
}

/// Encodes oriented edge features into SDR representations.
///
/// Combines orientation (periodic over [0, PI)) with magnitude,
/// similar to how HOG (Histogram of Oriented Gradients) features work.
///
/// # Example
///
/// ```rust
/// use mokosh::encoders::{EdgeOrientationEncoder, EdgeOrientationEncoderParams, OrientedEdge, Encoder};
///
/// let encoder = EdgeOrientationEncoder::new(EdgeOrientationEncoderParams {
///     num_orientations: 8,
///     ..Default::default()
/// }).unwrap();
///
/// let horizontal = OrientedEdge::new(0.0, 1.0);
/// let vertical = OrientedEdge::new(std::f32::consts::FRAC_PI_2, 1.0);
///
/// let sdr_h = encoder.encode_to_sdr(horizontal).unwrap();
/// let sdr_v = encoder.encode_to_sdr(vertical).unwrap();
///
/// // Different orientations have some overlap (magnitude component)
/// // but less than identical orientations
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EdgeOrientationEncoder {
    num_orientations: usize,
    bits_per_orientation: UInt,
    active_bits_per_orientation: UInt,
    magnitude_bits: UInt,
    magnitude_active: UInt,
    min_magnitude: Real,
    max_magnitude: Real,
    orientation_size: UInt,
    size: UInt,
    dimensions: Vec<UInt>,
}

impl EdgeOrientationEncoder {
    /// Creates a new Edge Orientation Encoder.
    pub fn new(params: EdgeOrientationEncoderParams) -> Result<Self> {
        if params.num_orientations == 0 {
            return Err(MokoshError::InvalidParameter {
                name: "num_orientations",
                message: "Must be > 0".to_string(),
            });
        }

        if params.active_bits_per_orientation > params.bits_per_orientation {
            return Err(MokoshError::InvalidParameter {
                name: "active_bits_per_orientation",
                message: "Cannot exceed bits_per_orientation".to_string(),
            });
        }

        if params.magnitude_active > params.magnitude_bits {
            return Err(MokoshError::InvalidParameter {
                name: "magnitude_active",
                message: "Cannot exceed magnitude_bits".to_string(),
            });
        }

        let orientation_size = params.num_orientations as UInt * params.bits_per_orientation;
        let size = orientation_size + params.magnitude_bits;

        Ok(Self {
            num_orientations: params.num_orientations,
            bits_per_orientation: params.bits_per_orientation,
            active_bits_per_orientation: params.active_bits_per_orientation,
            magnitude_bits: params.magnitude_bits,
            magnitude_active: params.magnitude_active,
            min_magnitude: params.min_magnitude,
            max_magnitude: params.max_magnitude,
            orientation_size,
            size,
            dimensions: vec![size],
        })
    }

    /// Returns the number of orientation bins.
    pub fn num_orientations(&self) -> usize {
        self.num_orientations
    }
}

impl Encoder<OrientedEdge> for EdgeOrientationEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, edge: OrientedEdge, output: &mut Sdr) -> Result<()> {
        if output.dimensions() != self.dimensions.as_slice() {
            return Err(MokoshError::DimensionMismatch {
                expected: self.dimensions.clone(),
                actual: output.dimensions().to_vec(),
            });
        }

        let mut sparse = Vec::new();

        // Only encode if magnitude is above threshold
        if edge.magnitude >= self.min_magnitude {
            // Encode orientation (periodic over [0, PI))
            let orientation_normalized = edge.orientation / PI;
            let bin = (orientation_normalized * self.num_orientations as Real) as usize
                % self.num_orientations;

            let bin_offset = bin as UInt * self.bits_per_orientation;
            let positions_in_bin = self.bits_per_orientation - self.active_bits_per_orientation + 1;

            // Position within bin based on fine orientation
            let bin_start = bin as Real / self.num_orientations as Real;
            let bin_end = (bin + 1) as Real / self.num_orientations as Real;
            let within_bin = (orientation_normalized - bin_start) / (bin_end - bin_start);
            let start_pos = (within_bin * (positions_in_bin - 1) as Real).round() as UInt;

            for i in 0..self.active_bits_per_orientation {
                sparse.push(bin_offset + start_pos + i);
            }

            // Encode magnitude
            let mag_normalized = ((edge.magnitude - self.min_magnitude)
                / (self.max_magnitude - self.min_magnitude))
                .clamp(0.0, 1.0);

            let mag_offset = self.orientation_size;
            let mag_positions = self.magnitude_bits - self.magnitude_active + 1;
            let mag_start = (mag_normalized * (mag_positions - 1) as Real).round() as UInt;

            for i in 0..self.magnitude_active {
                sparse.push(mag_offset + mag_start + i);
            }
        }

        sparse.sort_unstable();
        output.set_sparse_unchecked(sparse);

        Ok(())
    }
}

impl Encoder<(Real, Real)> for EdgeOrientationEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, gradient: (Real, Real), output: &mut Sdr) -> Result<()> {
        self.encode(OrientedEdge::from_gradient(gradient.0, gradient.1), output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_encoder() {
        let encoder = EdgeOrientationEncoder::new(EdgeOrientationEncoderParams {
            num_orientations: 8,
            bits_per_orientation: 20,
            active_bits_per_orientation: 5,
            magnitude_bits: 40,
            magnitude_active: 10,
            ..Default::default()
        })
        .unwrap();

        assert_eq!(encoder.num_orientations(), 8);
        assert_eq!(Encoder::<OrientedEdge>::size(&encoder), 200); // 8*20 + 40
    }

    #[test]
    fn test_encode_edge() {
        let encoder = EdgeOrientationEncoder::new(EdgeOrientationEncoderParams::default()).unwrap();

        let edge = OrientedEdge::new(PI / 4.0, 0.5);
        let sdr = encoder.encode_to_sdr(edge).unwrap();

        // Should have orientation bits + magnitude bits
        let expected = 5 + 10;
        assert_eq!(sdr.get_sum(), expected);
    }

    #[test]
    fn test_from_gradient() {
        let edge = OrientedEdge::from_gradient(1.0, 0.0);
        assert!((edge.orientation - 0.0).abs() < 0.01);
        assert!((edge.magnitude - 1.0).abs() < 0.01);

        let edge = OrientedEdge::from_gradient(0.0, 1.0);
        assert!((edge.orientation - PI / 2.0).abs() < 0.01);
        assert!((edge.magnitude - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_similar_orientations_overlap() {
        let encoder = EdgeOrientationEncoder::new(EdgeOrientationEncoderParams {
            num_orientations: 16,
            bits_per_orientation: 20,
            active_bits_per_orientation: 5,
            ..Default::default()
        })
        .unwrap();

        let edge1 = OrientedEdge::new(0.0, 1.0);
        let edge2 = OrientedEdge::new(PI / 32.0, 1.0); // Slightly different
        let edge3 = OrientedEdge::new(PI / 2.0, 1.0); // Very different

        let sdr1 = encoder.encode_to_sdr(edge1).unwrap();
        let sdr2 = encoder.encode_to_sdr(edge2).unwrap();
        let sdr3 = encoder.encode_to_sdr(edge3).unwrap();

        // Similar orientations should overlap more
        let similar_overlap = sdr1.get_overlap(&sdr2);
        let different_overlap = sdr1.get_overlap(&sdr3);

        assert!(similar_overlap >= different_overlap);
    }

    #[test]
    fn test_weak_edge() {
        let encoder = EdgeOrientationEncoder::new(EdgeOrientationEncoderParams {
            min_magnitude: 0.1,
            ..Default::default()
        })
        .unwrap();

        // Below threshold
        let weak_edge = OrientedEdge::new(0.0, 0.05);
        let sdr = encoder.encode_to_sdr(weak_edge).unwrap();

        assert_eq!(sdr.get_sum(), 0);
    }

    #[test]
    fn test_encode_gradient_tuple() {
        let encoder = EdgeOrientationEncoder::new(EdgeOrientationEncoderParams::default()).unwrap();

        let sdr = encoder.encode_to_sdr((1.0 as Real, 1.0 as Real)).unwrap();
        assert!(sdr.get_sum() > 0);
    }

    #[test]
    fn test_orientation_periodic() {
        let encoder = EdgeOrientationEncoder::new(EdgeOrientationEncoderParams::default()).unwrap();

        // Orientation 0 and PI should be the same (edges are undirected)
        let edge1 = OrientedEdge::new(0.0, 1.0);
        let edge2 = OrientedEdge::new(PI, 1.0);

        let sdr1 = encoder.encode_to_sdr(edge1).unwrap();
        let sdr2 = encoder.encode_to_sdr(edge2).unwrap();

        // Should have very high overlap (same orientation bin)
        assert!(sdr1.get_overlap(&sdr2) >= 10);
    }

    #[test]
    fn test_deterministic() {
        let encoder = EdgeOrientationEncoder::new(EdgeOrientationEncoderParams::default()).unwrap();

        let edge = OrientedEdge::new(PI / 3.0, 0.7);

        let sdr1 = encoder.encode_to_sdr(edge).unwrap();
        let sdr2 = encoder.encode_to_sdr(edge).unwrap();

        assert_eq!(sdr1.get_sparse(), sdr2.get_sparse());
    }
}
