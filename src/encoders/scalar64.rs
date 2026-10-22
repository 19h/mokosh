//! Double-precision scalar encoding matching the htm.core configuration model.
use super::Encoder;
use crate::{
    error::{MokoshError, Result},
    types::{Sdr, UInt},
};

/// Full double-precision scalar configuration.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ScalarEncoderParams64 {
    /// Inclusive minimum.
    pub minimum: f64,
    /// Inclusive maximum (periodic maximum wraps to the minimum).
    pub maximum: f64,
    /// Output size; exclusive with radius, resolution and category.
    pub size: UInt,
    /// Active bits; exclusive with sparsity.
    pub active_bits: UInt,
    /// Active fraction when size is specified.
    pub sparsity: f32,
    /// Input separation giving non-overlapping encodings.
    pub radius: f64,
    /// Input distance per bit.
    pub resolution: f64,
    /// Integer category encoding.
    pub category: bool,
    /// Circular domain.
    pub periodic: bool,
    /// Clip out-of-range input.
    pub clip_input: bool,
}

/// Scalar encoder with f64 bounds, resolution and input.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct ScalarEncoder64 {
    params: ScalarEncoderParams64,
    dimensions: Vec<UInt>,
}
impl ScalarEncoder64 {
    /// Resolves and validates configuration without narrowing to f32.
    pub fn new(mut p: ScalarEncoderParams64) -> Result<Self> {
        let invalid = |message: &str| MokoshError::InvalidParameter {
            name: "scalar64",
            message: message.into(),
        };
        if !p.minimum.is_finite()
            || !p.maximum.is_finite()
            || p.minimum >= p.maximum
            || !(p.maximum - p.minimum).is_finite()
        {
            return Err(invalid("Bounds must define a finite positive interval"));
        }
        if !p.radius.is_finite()
            || !p.resolution.is_finite()
            || p.radius < 0.0
            || p.resolution < 0.0
            || !p.sparsity.is_finite()
            || !(0.0..=1.0).contains(&p.sparsity)
        {
            return Err(invalid("Invalid resolution, radius or sparsity"));
        }
        if u8::from(p.active_bits > 0) + u8::from(p.sparsity > 0.0) != 1
            || u8::from(p.size > 0)
                + u8::from(p.radius > 0.0)
                + u8::from(p.resolution > 0.0)
                + u8::from(p.category)
                != 1
        {
            return Err(invalid("Specify one active-bit mode and one size mode"));
        }
        if p.periodic && p.clip_input
            || p.category
                && (p.periodic
                    || p.clip_input
                    || p.minimum < 0.0
                    || p.minimum.fract() != 0.0
                    || p.maximum.fract() != 0.0)
        {
            return Err(invalid(
                "Incompatible clipping, periodic or category configuration",
            ));
        }
        if p.sparsity > 0.0 {
            if p.size == 0 {
                return Err(invalid("Sparsity requires size"));
            }
            p.active_bits = (p.size as f64 * f64::from(p.sparsity)).round() as u32;
        }
        if p.active_bits == 0 {
            return Err(invalid("At least one active bit is required"));
        }
        if p.category {
            p.radius = 1.0;
        }
        let inclusive_max = if p.maximum == 0.0 {
            f64::from_bits(1)
        } else if p.maximum > 0.0 {
            f64::from_bits(p.maximum.to_bits() + 1)
        } else {
            f64::from_bits(p.maximum.to_bits() - 1)
        };
        let extent = if p.periodic {
            p.maximum - p.minimum
        } else {
            inclusive_max - p.minimum
        };
        if !extent.is_finite() {
            return Err(invalid("Inclusive interval overflow"));
        }
        if p.size > 0 {
            if p.size <= p.active_bits {
                return Err(invalid("Size must exceed active bits"));
            }
            p.resolution = extent
                / if p.periodic {
                    p.size
                } else {
                    p.size - p.active_bits
                } as f64;
        } else {
            if p.radius > 0.0 {
                p.resolution = p.radius / p.active_bits as f64;
            }
            if p.resolution <= 0.0 || !p.resolution.is_finite() {
                return Err(invalid("Resolution underflow"));
            }
            let bands = (extent / p.resolution).ceil();
            let size = bands
                + if p.periodic {
                    0.0
                } else {
                    f64::from(p.active_bits - 1)
                };
            if size > u32::MAX as f64 || size <= p.active_bits as f64 {
                return Err(invalid("Unrepresentable output size"));
            }
            p.size = size as u32;
        }
        p.radius = p.resolution * p.active_bits as f64;
        p.sparsity = p.active_bits as f32 / p.size as f32;
        if p.resolution <= 0.0 || !p.resolution.is_finite() || !p.radius.is_finite() {
            return Err(invalid("Derived resolution or radius overflow"));
        }
        Ok(Self {
            dimensions: vec![p.size],
            params: p,
        })
    }
    /// Validates resolved geometry and cached dimensions.
    pub fn validate(&self) -> Result<()> {
        let p = &self.params;
        if !p.minimum.is_finite()
            || !p.maximum.is_finite()
            || p.minimum >= p.maximum
            || p.active_bits == 0
            || p.size <= p.active_bits
            || self.dimensions != [p.size]
            || !p.resolution.is_finite()
            || p.resolution <= 0.0
            || !p.radius.is_finite()
            || p.radius != p.resolution * p.active_bits as f64
            || p.category
                && (p.periodic
                    || p.clip_input
                    || p.minimum < 0.0
                    || p.minimum.fract() != 0.0
                    || p.maximum.fract() != 0.0)
            || p.periodic && p.clip_input
        {
            return Err(MokoshError::InvalidSdrData(
                "Invalid scalar64 geometry".into(),
            ));
        }
        Ok(())
    }
    /// Resolved parameters, including computed size, radius and sparsity.
    pub fn parameters(&self) -> &ScalarEncoderParams64 {
        &self.params
    }
    /// Output size in bits.
    pub fn size(&self) -> usize {
        self.params.size as usize
    }
}
impl Encoder<f64> for ScalarEncoder64 {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }
    fn size(&self) -> usize {
        self.size()
    }
    fn encode(&self, mut value: f64, output: &mut Sdr) -> Result<()> {
        let p = &self.params;
        if output.dimensions() != self.dimensions {
            return Err(MokoshError::DimensionMismatch {
                expected: self.dimensions.clone(),
                actual: output.dimensions().to_vec(),
            });
        }
        if value.is_nan() {
            output.zero();
            return Ok(());
        }
        if p.clip_input {
            value = value.clamp(p.minimum, p.maximum);
        }
        if !value.is_finite()
            || value < p.minimum
            || value > p.maximum
            || (p.category && (value < 0.0 || value.fract() != 0.0))
        {
            return Err(MokoshError::InvalidParameter {
                name: "value",
                message: "Input is outside the scalar domain".into(),
            });
        }
        let start = ((value - p.minimum) / p.resolution).round() as u64;
        let start = if p.periodic {
            start
        } else {
            start.min(u64::from(p.size - p.active_bits))
        };
        let mut sparse: Vec<_> = (0..p.active_bits)
            .map(|i| ((start + u64::from(i)) % u64::from(p.size)) as UInt)
            .collect();
        sparse.sort_unstable();
        output.set_sparse_owned(sparse)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for ScalarEncoder64 {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        struct State {
            params: ScalarEncoderParams64,
            dimensions: Vec<UInt>,
        }
        let state = State::deserialize(d)?;
        // Persist resolved state through its size mode, preserving f64 resolution.
        let mut p = state.params.clone();
        p.radius = 0.0;
        p.resolution = 0.0;
        p.sparsity = 0.0;
        p.category = false;
        let mut value = Self::new(p).map_err(serde::de::Error::custom)?;
        if state.dimensions != value.dimensions
            || !state.params.resolution.is_finite()
            || state.params.resolution <= 0.0
        {
            return Err(serde::de::Error::custom("Invalid scalar64 state"));
        }
        value.params = state.params;
        value.validate().map_err(serde::de::Error::custom)?;
        Ok(value)
    }
}
