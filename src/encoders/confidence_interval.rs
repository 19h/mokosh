//! Confidence Interval Encoder implementation.
//!
//! Encodes values with uncertainty ranges into SDRs.

use crate::encoders::Encoder;
use crate::error::{MokoshError, Result};
use crate::types::{Real, Sdr, UInt};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Parameters for creating a Confidence Interval Encoder.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ConfidenceIntervalEncoderParams {
    /// Minimum value of the range.
    pub min_value: Real,

    /// Maximum value of the range.
    pub max_value: Real,

    /// Bits for encoding the central estimate.
    pub center_bits: UInt,

    /// Active bits for center.
    pub center_active: UInt,

    /// Bits for encoding the width (uncertainty).
    pub width_bits: UInt,

    /// Active bits for width.
    pub width_active: UInt,

    /// Maximum expected width (as fraction of total range).
    pub max_width_fraction: Real,
}

impl Default for ConfidenceIntervalEncoderParams {
    fn default() -> Self {
        Self {
            min_value: 0.0,
            max_value: 100.0,
            center_bits: 100,
            center_active: 21,
            width_bits: 50,
            width_active: 10,
            max_width_fraction: 0.5, // Up to 50% of range
        }
    }
}

/// A value with a confidence interval.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ConfidenceInterval {
    /// Lower bound of the interval.
    pub lower: Real,
    /// Upper bound of the interval.
    pub upper: Real,
}

impl ConfidenceInterval {
    /// Creates a new confidence interval.
    pub fn new(lower: Real, upper: Real) -> Result<Self> {
        if lower > upper {
            return Err(MokoshError::InvalidParameter {
                name: "interval",
                message: "Lower bound must be <= upper bound".to_string(),
            });
        }
        Ok(Self { lower, upper })
    }

    /// Creates from a point estimate (zero uncertainty).
    pub fn point(value: Real) -> Self {
        Self {
            lower: value,
            upper: value,
        }
    }

    /// Creates from center and margin of error.
    pub fn from_center_margin(center: Real, margin: Real) -> Self {
        Self {
            lower: center - margin,
            upper: center + margin,
        }
    }

    /// Creates from center and relative uncertainty (percentage).
    pub fn from_center_percent(center: Real, percent: Real) -> Self {
        let margin = center.abs() * percent / 100.0;
        Self::from_center_margin(center, margin)
    }

    /// Returns the center of the interval.
    pub fn center(&self) -> Real {
        (self.lower + self.upper) / 2.0
    }

    /// Returns the width of the interval.
    pub fn width(&self) -> Real {
        self.upper - self.lower
    }

    /// Returns the margin of error (half the width).
    pub fn margin(&self) -> Real {
        self.width() / 2.0
    }

    /// Returns whether this interval contains a value.
    pub fn contains(&self, value: Real) -> bool {
        value >= self.lower && value <= self.upper
    }

    /// Returns the overlap with another interval.
    pub fn overlap_with(&self, other: &ConfidenceInterval) -> Option<ConfidenceInterval> {
        let lower = self.lower.max(other.lower);
        let upper = self.upper.min(other.upper);

        if lower <= upper {
            Some(ConfidenceInterval { lower, upper })
        } else {
            None
        }
    }
}

/// Encodes values with confidence intervals into SDR representations.
///
/// Captures both the central estimate and the uncertainty range.
/// Narrower intervals (more certain) produce different encodings
/// than wider intervals (less certain) for the same center.
///
/// # Example
///
/// ```rust
/// use mokosh::encoders::{ConfidenceIntervalEncoder, ConfidenceIntervalEncoderParams, ConfidenceInterval, Encoder};
///
/// let encoder = ConfidenceIntervalEncoder::new(ConfidenceIntervalEncoderParams {
///     min_value: 0.0,
///     max_value: 100.0,
///     ..Default::default()
/// }).unwrap();
///
/// // High certainty (narrow interval)
/// let certain = ConfidenceInterval::from_center_margin(50.0, 2.0);
///
/// // Low certainty (wide interval)
/// let uncertain = ConfidenceInterval::from_center_margin(50.0, 20.0);
///
/// let sdr_certain = encoder.encode_to_sdr(certain).unwrap();
/// let sdr_uncertain = encoder.encode_to_sdr(uncertain).unwrap();
///
/// // Same center, different width = different encodings
/// assert!(sdr_certain.get_overlap(&sdr_uncertain) < sdr_certain.get_sum());
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ConfidenceIntervalEncoder {
    min_value: Real,
    max_value: Real,
    value_range: Real,
    center_bits: UInt,
    center_active: UInt,
    width_bits: UInt,
    width_active: UInt,
    max_width_fraction: Real,
    size: UInt,
    dimensions: Vec<UInt>,
}

impl ConfidenceIntervalEncoder {
    /// Creates a new Confidence Interval Encoder.
    pub fn new(params: ConfidenceIntervalEncoderParams) -> Result<Self> {
        if params.max_value <= params.min_value {
            return Err(MokoshError::InvalidParameter {
                name: "max_value",
                message: "Must be greater than min_value".to_string(),
            });
        }

        if params.center_active > params.center_bits {
            return Err(MokoshError::InvalidParameter {
                name: "center_active",
                message: "Cannot exceed center_bits".to_string(),
            });
        }

        if params.width_active > params.width_bits {
            return Err(MokoshError::InvalidParameter {
                name: "width_active",
                message: "Cannot exceed width_bits".to_string(),
            });
        }

        let size = params.center_bits + params.width_bits;

        Ok(Self {
            min_value: params.min_value,
            max_value: params.max_value,
            value_range: params.max_value - params.min_value,
            center_bits: params.center_bits,
            center_active: params.center_active,
            width_bits: params.width_bits,
            width_active: params.width_active,
            max_width_fraction: params.max_width_fraction,
            size,
            dimensions: vec![size],
        })
    }

    /// Normalizes a value to [0, 1].
    fn normalize_value(&self, value: Real) -> Real {
        ((value - self.min_value) / self.value_range).clamp(0.0, 1.0)
    }

    /// Normalizes a width to [0, 1].
    fn normalize_width(&self, width: Real) -> Real {
        let max_width = self.value_range * self.max_width_fraction;
        (width / max_width).clamp(0.0, 1.0)
    }
}

impl Encoder<ConfidenceInterval> for ConfidenceIntervalEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, interval: ConfidenceInterval, output: &mut Sdr) -> Result<()> {
        if output.dimensions() != self.dimensions.as_slice() {
            return Err(MokoshError::DimensionMismatch {
                expected: self.dimensions.clone(),
                actual: output.dimensions().to_vec(),
            });
        }

        let mut sparse = Vec::new();

        // Encode center
        let center = interval.center();
        let normalized_center = self.normalize_value(center);
        let positions = self.center_bits - self.center_active + 1;
        let start = (normalized_center * (positions - 1) as Real).round() as UInt;

        for i in 0..self.center_active {
            sparse.push(start + i);
        }

        // Encode width
        let width = interval.width();
        let normalized_width = self.normalize_width(width);
        let width_offset = self.center_bits;
        let positions = self.width_bits - self.width_active + 1;
        let start = (normalized_width * (positions - 1) as Real).round() as UInt;

        for i in 0..self.width_active {
            sparse.push(width_offset + start + i);
        }

        sparse.sort_unstable();
        output.set_sparse_unchecked(sparse);

        Ok(())
    }
}

impl Encoder<(Real, Real)> for ConfidenceIntervalEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, bounds: (Real, Real), output: &mut Sdr) -> Result<()> {
        let interval = ConfidenceInterval::new(bounds.0, bounds.1)?;
        self.encode(interval, output)
    }
}

impl Encoder<Real> for ConfidenceIntervalEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, value: Real, output: &mut Sdr) -> Result<()> {
        self.encode(ConfidenceInterval::point(value), output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_encoder() {
        let encoder = ConfidenceIntervalEncoder::new(ConfidenceIntervalEncoderParams {
            min_value: 0.0,
            max_value: 100.0,
            center_bits: 80,
            center_active: 16,
            width_bits: 40,
            width_active: 8,
            ..Default::default()
        })
        .unwrap();

        assert_eq!(Encoder::<ConfidenceInterval>::size(&encoder), 120);
    }

    #[test]
    fn test_confidence_interval() {
        let ci = ConfidenceInterval::new(10.0, 20.0).unwrap();

        assert!((ci.center() - 15.0).abs() < 0.01);
        assert!((ci.width() - 10.0).abs() < 0.01);
        assert!((ci.margin() - 5.0).abs() < 0.01);
        assert!(ci.contains(15.0));
        assert!(!ci.contains(5.0));
    }

    #[test]
    fn test_from_center_margin() {
        let ci = ConfidenceInterval::from_center_margin(50.0, 10.0);

        assert!((ci.lower - 40.0).abs() < 0.01);
        assert!((ci.upper - 60.0).abs() < 0.01);
    }

    #[test]
    fn test_from_center_percent() {
        let ci = ConfidenceInterval::from_center_percent(100.0, 10.0);

        assert!((ci.lower - 90.0).abs() < 0.01);
        assert!((ci.upper - 110.0).abs() < 0.01);
    }

    #[test]
    fn test_overlap() {
        let ci1 = ConfidenceInterval::new(10.0, 30.0).unwrap();
        let ci2 = ConfidenceInterval::new(20.0, 40.0).unwrap();
        let ci3 = ConfidenceInterval::new(50.0, 60.0).unwrap();

        let overlap = ci1.overlap_with(&ci2).unwrap();
        assert!((overlap.lower - 20.0).abs() < 0.01);
        assert!((overlap.upper - 30.0).abs() < 0.01);

        assert!(ci1.overlap_with(&ci3).is_none());
    }

    #[test]
    fn test_encode_interval() {
        let encoder = ConfidenceIntervalEncoder::new(ConfidenceIntervalEncoderParams::default())
            .unwrap();

        let ci = ConfidenceInterval::from_center_margin(50.0, 5.0);
        let sdr = encoder.encode_to_sdr(ci).unwrap();

        // 21 + 10 = 31 active bits
        assert_eq!(sdr.get_sum(), 31);
    }

    #[test]
    fn test_different_widths() {
        let encoder = ConfidenceIntervalEncoder::new(ConfidenceIntervalEncoderParams::default())
            .unwrap();

        let narrow = ConfidenceInterval::from_center_margin(50.0, 2.0);
        let wide = ConfidenceInterval::from_center_margin(50.0, 20.0);

        let sdr_narrow = encoder.encode_to_sdr(narrow).unwrap();
        let sdr_wide = encoder.encode_to_sdr(wide).unwrap();

        // Same center, different width should give partial overlap
        let overlap = sdr_narrow.get_overlap(&sdr_wide);

        // Should share center bits but not width bits
        assert!(overlap > 15); // Share most of center
        assert!(overlap < 31); // Don't share all width
    }

    #[test]
    fn test_similar_intervals_overlap() {
        let encoder = ConfidenceIntervalEncoder::new(ConfidenceIntervalEncoderParams::default())
            .unwrap();

        let ci1 = ConfidenceInterval::from_center_margin(50.0, 5.0);
        let ci2 = ConfidenceInterval::from_center_margin(52.0, 5.0);
        let ci3 = ConfidenceInterval::from_center_margin(80.0, 5.0);

        let sdr1 = encoder.encode_to_sdr(ci1).unwrap();
        let sdr2 = encoder.encode_to_sdr(ci2).unwrap();
        let sdr3 = encoder.encode_to_sdr(ci3).unwrap();

        let near_overlap = sdr1.get_overlap(&sdr2);
        let far_overlap = sdr1.get_overlap(&sdr3);

        assert!(near_overlap > far_overlap);
    }

    #[test]
    fn test_encode_tuple() {
        let encoder = ConfidenceIntervalEncoder::new(ConfidenceIntervalEncoderParams::default())
            .unwrap();

        let sdr = encoder.encode_to_sdr((40.0 as Real, 60.0 as Real)).unwrap();
        assert_eq!(sdr.get_sum(), 31);
    }

    #[test]
    fn test_encode_point() {
        let encoder = ConfidenceIntervalEncoder::new(ConfidenceIntervalEncoderParams::default())
            .unwrap();

        let sdr = encoder.encode_to_sdr(50.0 as Real).unwrap();
        assert_eq!(sdr.get_sum(), 31);
    }

    #[test]
    fn test_deterministic() {
        let encoder = ConfidenceIntervalEncoder::new(ConfidenceIntervalEncoderParams::default())
            .unwrap();

        let ci = ConfidenceInterval::from_center_margin(75.0, 10.0);

        let sdr1 = encoder.encode_to_sdr(ci).unwrap();
        let sdr2 = encoder.encode_to_sdr(ci).unwrap();

        assert_eq!(sdr1.get_sparse(), sdr2.get_sparse());
    }
}
