//! Color Encoder implementation.
//!
//! Encodes colors (RGB, HSV, etc.) into SDRs with perceptual similarity.

use crate::encoders::Encoder;
use crate::error::{MokoshError, Result};
use crate::types::{Real, Sdr, UInt};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Parameters for creating a Color Encoder.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ColorEncoderParams {
    /// Bits for encoding hue (periodic).
    pub hue_bits: UInt,

    /// Active bits for hue.
    pub hue_active: UInt,

    /// Bits for encoding saturation.
    pub saturation_bits: UInt,

    /// Active bits for saturation.
    pub saturation_active: UInt,

    /// Bits for encoding value/brightness.
    pub value_bits: UInt,

    /// Active bits for value.
    pub value_active: UInt,
}

impl Default for ColorEncoderParams {
    fn default() -> Self {
        Self {
            hue_bits: 120,       // 10 bits per 30 degrees of hue
            hue_active: 21,
            saturation_bits: 50,
            saturation_active: 10,
            value_bits: 50,
            value_active: 10,
        }
    }
}

/// An RGB color with values in [0, 1].
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RgbColor {
    /// Red component [0, 1].
    pub r: Real,
    /// Green component [0, 1].
    pub g: Real,
    /// Blue component [0, 1].
    pub b: Real,
}

impl RgbColor {
    /// Creates a new RGB color.
    pub fn new(r: Real, g: Real, b: Real) -> Self {
        Self {
            r: r.clamp(0.0, 1.0),
            g: g.clamp(0.0, 1.0),
            b: b.clamp(0.0, 1.0),
        }
    }

    /// Creates from 8-bit RGB values (0-255).
    pub fn from_rgb8(r: u8, g: u8, b: u8) -> Self {
        Self::new(r as Real / 255.0, g as Real / 255.0, b as Real / 255.0)
    }

    /// Converts to HSV.
    pub fn to_hsv(&self) -> HsvColor {
        let max = self.r.max(self.g).max(self.b);
        let min = self.r.min(self.g).min(self.b);
        let delta = max - min;

        let v = max;

        let s = if max == 0.0 { 0.0 } else { delta / max };

        let h = if delta == 0.0 {
            0.0
        } else if max == self.r {
            60.0 * (((self.g - self.b) / delta) % 6.0)
        } else if max == self.g {
            60.0 * (((self.b - self.r) / delta) + 2.0)
        } else {
            60.0 * (((self.r - self.g) / delta) + 4.0)
        };

        let h = if h < 0.0 { h + 360.0 } else { h };

        HsvColor { h, s, v }
    }
}

/// An HSV color.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HsvColor {
    /// Hue in degrees [0, 360).
    pub h: Real,
    /// Saturation [0, 1].
    pub s: Real,
    /// Value/brightness [0, 1].
    pub v: Real,
}

impl HsvColor {
    /// Creates a new HSV color.
    pub fn new(h: Real, s: Real, v: Real) -> Self {
        Self {
            h: h.rem_euclid(360.0),
            s: s.clamp(0.0, 1.0),
            v: v.clamp(0.0, 1.0),
        }
    }

    /// Converts to RGB.
    pub fn to_rgb(&self) -> RgbColor {
        let c = self.v * self.s;
        let x = c * (1.0 - ((self.h / 60.0) % 2.0 - 1.0).abs());
        let m = self.v - c;

        let (r1, g1, b1) = if self.h < 60.0 {
            (c, x, 0.0)
        } else if self.h < 120.0 {
            (x, c, 0.0)
        } else if self.h < 180.0 {
            (0.0, c, x)
        } else if self.h < 240.0 {
            (0.0, x, c)
        } else if self.h < 300.0 {
            (x, 0.0, c)
        } else {
            (c, 0.0, x)
        };

        RgbColor::new(r1 + m, g1 + m, b1 + m)
    }
}

/// Encodes colors into SDR representations.
///
/// Uses HSV color space internally for perceptually meaningful
/// encodings. Hue is encoded periodically, while saturation
/// and value are encoded linearly.
///
/// # Example
///
/// ```rust
/// use mokosh::encoders::{ColorEncoder, ColorEncoderParams, RgbColor, Encoder};
///
/// let encoder = ColorEncoder::new(ColorEncoderParams::default()).unwrap();
///
/// let red = RgbColor::new(1.0, 0.0, 0.0);
/// let orange = RgbColor::new(1.0, 0.5, 0.0);
/// let blue = RgbColor::new(0.0, 0.0, 1.0);
///
/// let sdr_red = encoder.encode_to_sdr(red).unwrap();
/// let sdr_orange = encoder.encode_to_sdr(orange).unwrap();
/// let sdr_blue = encoder.encode_to_sdr(blue).unwrap();
///
/// // Red and orange are more similar than red and blue
/// assert!(sdr_red.get_overlap(&sdr_orange) > sdr_red.get_overlap(&sdr_blue));
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ColorEncoder {
    hue_bits: UInt,
    hue_active: UInt,
    saturation_bits: UInt,
    saturation_active: UInt,
    value_bits: UInt,
    value_active: UInt,
    size: UInt,
    dimensions: Vec<UInt>,
}

impl ColorEncoder {
    /// Creates a new Color Encoder.
    pub fn new(params: ColorEncoderParams) -> Result<Self> {
        if params.hue_active > params.hue_bits {
            return Err(MokoshError::InvalidParameter {
                name: "hue_active",
                message: "Cannot exceed hue_bits".to_string(),
            });
        }

        if params.saturation_active > params.saturation_bits {
            return Err(MokoshError::InvalidParameter {
                name: "saturation_active",
                message: "Cannot exceed saturation_bits".to_string(),
            });
        }

        if params.value_active > params.value_bits {
            return Err(MokoshError::InvalidParameter {
                name: "value_active",
                message: "Cannot exceed value_bits".to_string(),
            });
        }

        let size = params.hue_bits + params.saturation_bits + params.value_bits;

        Ok(Self {
            hue_bits: params.hue_bits,
            hue_active: params.hue_active,
            saturation_bits: params.saturation_bits,
            saturation_active: params.saturation_active,
            value_bits: params.value_bits,
            value_active: params.value_active,
            size,
            dimensions: vec![size],
        })
    }

    /// Returns the size of the hue encoding.
    pub fn hue_size(&self) -> UInt {
        self.hue_bits
    }

    /// Returns the size of the saturation encoding.
    pub fn saturation_size(&self) -> UInt {
        self.saturation_bits
    }

    /// Returns the size of the value encoding.
    pub fn value_size(&self) -> UInt {
        self.value_bits
    }
}

impl Encoder<HsvColor> for ColorEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, color: HsvColor, output: &mut Sdr) -> Result<()> {
        if output.dimensions() != self.dimensions.as_slice() {
            return Err(MokoshError::DimensionMismatch {
                expected: self.dimensions.clone(),
                actual: output.dimensions().to_vec(),
            });
        }

        let mut sparse = Vec::new();

        // Encode hue (periodic)
        let hue_normalized = color.h / 360.0;
        let hue_positions = self.hue_bits - self.hue_active + 1;
        let hue_start = (hue_normalized * hue_positions as Real).round() as UInt % hue_positions;

        for i in 0..self.hue_active {
            let bit = (hue_start + i) % self.hue_bits;
            sparse.push(bit);
        }

        // Encode saturation (linear)
        let sat_offset = self.hue_bits;
        let sat_positions = self.saturation_bits - self.saturation_active + 1;
        let sat_start = (color.s * (sat_positions - 1) as Real).round() as UInt;

        for i in 0..self.saturation_active {
            sparse.push(sat_offset + sat_start + i);
        }

        // Encode value (linear)
        let val_offset = self.hue_bits + self.saturation_bits;
        let val_positions = self.value_bits - self.value_active + 1;
        let val_start = (color.v * (val_positions - 1) as Real).round() as UInt;

        for i in 0..self.value_active {
            sparse.push(val_offset + val_start + i);
        }

        sparse.sort_unstable();
        sparse.dedup();
        output.set_sparse_unchecked(sparse);

        Ok(())
    }
}

impl Encoder<RgbColor> for ColorEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, color: RgbColor, output: &mut Sdr) -> Result<()> {
        self.encode(color.to_hsv(), output)
    }
}

impl Encoder<(Real, Real, Real)> for ColorEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, rgb: (Real, Real, Real), output: &mut Sdr) -> Result<()> {
        self.encode(RgbColor::new(rgb.0, rgb.1, rgb.2), output)
    }
}

impl Encoder<(u8, u8, u8)> for ColorEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, rgb: (u8, u8, u8), output: &mut Sdr) -> Result<()> {
        self.encode(RgbColor::from_rgb8(rgb.0, rgb.1, rgb.2), output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_encoder() {
        let encoder = ColorEncoder::new(ColorEncoderParams::default()).unwrap();

        assert_eq!(encoder.hue_size(), 120);
        assert_eq!(encoder.saturation_size(), 50);
        assert_eq!(encoder.value_size(), 50);
        assert_eq!(Encoder::<HsvColor>::size(&encoder), 220);
    }

    #[test]
    fn test_rgb_to_hsv() {
        let red = RgbColor::new(1.0, 0.0, 0.0);
        let hsv = red.to_hsv();
        assert!((hsv.h - 0.0).abs() < 1.0);
        assert!((hsv.s - 1.0).abs() < 0.01);
        assert!((hsv.v - 1.0).abs() < 0.01);

        let green = RgbColor::new(0.0, 1.0, 0.0);
        let hsv = green.to_hsv();
        assert!((hsv.h - 120.0).abs() < 1.0);

        let blue = RgbColor::new(0.0, 0.0, 1.0);
        let hsv = blue.to_hsv();
        assert!((hsv.h - 240.0).abs() < 1.0);
    }

    #[test]
    fn test_hsv_to_rgb() {
        let red_hsv = HsvColor::new(0.0, 1.0, 1.0);
        let rgb = red_hsv.to_rgb();
        assert!((rgb.r - 1.0).abs() < 0.01);
        assert!((rgb.g - 0.0).abs() < 0.01);
        assert!((rgb.b - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_encode_color() {
        let encoder = ColorEncoder::new(ColorEncoderParams::default()).unwrap();

        let red = RgbColor::new(1.0, 0.0, 0.0);
        let sdr = encoder.encode_to_sdr(red).unwrap();

        // Should have hue_active + saturation_active + value_active bits
        let expected = 21 + 10 + 10;
        assert!(sdr.get_sum() >= expected - 2 && sdr.get_sum() <= expected);
    }

    #[test]
    fn test_similar_colors_overlap() {
        let encoder = ColorEncoder::new(ColorEncoderParams::default()).unwrap();

        let red = RgbColor::new(1.0, 0.0, 0.0);
        let orange = RgbColor::new(1.0, 0.5, 0.0);
        let blue = RgbColor::new(0.0, 0.0, 1.0);

        let sdr_red = encoder.encode_to_sdr(red).unwrap();
        let sdr_orange = encoder.encode_to_sdr(orange).unwrap();
        let sdr_blue = encoder.encode_to_sdr(blue).unwrap();

        // Red and orange (similar hue) should overlap more than red and blue
        let red_orange_overlap = sdr_red.get_overlap(&sdr_orange);
        let red_blue_overlap = sdr_red.get_overlap(&sdr_blue);

        assert!(red_orange_overlap > red_blue_overlap);
    }

    #[test]
    fn test_encode_tuple() {
        let encoder = ColorEncoder::new(ColorEncoderParams::default()).unwrap();

        let sdr = encoder.encode_to_sdr((1.0 as Real, 0.5 as Real, 0.0 as Real)).unwrap();
        assert!(sdr.get_sum() > 0);
    }

    #[test]
    fn test_encode_u8_tuple() {
        let encoder = ColorEncoder::new(ColorEncoderParams::default()).unwrap();

        let sdr = encoder.encode_to_sdr((255u8, 128u8, 0u8)).unwrap();
        assert!(sdr.get_sum() > 0);
    }

    #[test]
    fn test_hue_periodic() {
        let encoder = ColorEncoder::new(ColorEncoderParams::default()).unwrap();

        // Red at 0 degrees and red at 360 degrees should be the same
        let color_0 = HsvColor::new(0.0, 1.0, 1.0);
        let color_360 = HsvColor::new(360.0, 1.0, 1.0);

        let sdr_0 = encoder.encode_to_sdr(color_0).unwrap();
        let sdr_360 = encoder.encode_to_sdr(color_360).unwrap();

        assert_eq!(sdr_0.get_sparse(), sdr_360.get_sparse());
    }

    #[test]
    fn test_deterministic() {
        let encoder = ColorEncoder::new(ColorEncoderParams::default()).unwrap();

        let color = RgbColor::new(0.3, 0.6, 0.9);

        let sdr1 = encoder.encode_to_sdr(color).unwrap();
        let sdr2 = encoder.encode_to_sdr(color).unwrap();

        assert_eq!(sdr1.get_sparse(), sdr2.get_sparse());
    }
}
