//! Accelerometer Encoder implementation.
//!
//! Encodes 3-axis accelerometer data into SDRs with orientation awareness.

use crate::encoders::Encoder;
use crate::error::{MokoshError, Result};
use crate::types::{Real, Sdr, UInt};
use std::f32::consts::PI;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Parameters for creating an Accelerometer Encoder.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AccelerometerEncoderParams {
    /// Bits for encoding each axis magnitude.
    pub axis_bits: UInt,

    /// Active bits per axis.
    pub axis_active: UInt,

    /// Bits for encoding total magnitude.
    pub magnitude_bits: UInt,

    /// Active bits for magnitude.
    pub magnitude_active: UInt,

    /// Bits for encoding orientation (tilt angles).
    pub orientation_bits: UInt,

    /// Active bits for orientation.
    pub orientation_active: UInt,

    /// Maximum expected acceleration (in g).
    pub max_acceleration: Real,
}

impl Default for AccelerometerEncoderParams {
    fn default() -> Self {
        Self {
            axis_bits: 32,
            axis_active: 8,
            magnitude_bits: 32,
            magnitude_active: 8,
            orientation_bits: 64, // For 2 angles
            orientation_active: 16,
            max_acceleration: 4.0, // 4g typical for activity tracking
        }
    }
}

/// 3-axis accelerometer reading.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AccelerometerReading {
    /// X-axis acceleration (typically lateral).
    pub x: Real,
    /// Y-axis acceleration (typically forward/backward).
    pub y: Real,
    /// Z-axis acceleration (typically vertical).
    pub z: Real,
}

impl AccelerometerReading {
    /// Creates a new accelerometer reading.
    pub fn new(x: Real, y: Real, z: Real) -> Self {
        Self { x, y, z }
    }

    /// Returns the total magnitude of acceleration.
    pub fn magnitude(&self) -> Real {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Returns the pitch angle (rotation around X axis) in radians.
    pub fn pitch(&self) -> Real {
        let mag = self.magnitude();
        if mag == 0.0 {
            return 0.0;
        }
        (self.y / mag).asin()
    }

    /// Returns the roll angle (rotation around Y axis) in radians.
    pub fn roll(&self) -> Real {
        self.x.atan2(self.z)
    }

    /// Returns whether the device is approximately level.
    pub fn is_level(&self, tolerance: Real) -> bool {
        let pitch = self.pitch().abs();
        let roll = self.roll().abs();
        pitch < tolerance && roll < tolerance
    }

    /// Returns whether this represents a significant motion event.
    pub fn is_motion(&self, threshold: Real) -> bool {
        // Deviation from 1g (stationary)
        (self.magnitude() - 1.0).abs() > threshold
    }
}

/// Encodes accelerometer data into SDR representations.
///
/// Captures both raw axis values and derived orientation/motion features.
///
/// # Example
///
/// ```rust
/// use mokosh::encoders::{AccelerometerEncoder, AccelerometerEncoderParams, AccelerometerReading, Encoder};
///
/// let encoder = AccelerometerEncoder::new(AccelerometerEncoderParams::default()).unwrap();
///
/// // Stationary, upright device (gravity on Z)
/// let stationary = AccelerometerReading::new(0.0, 0.0, 1.0);
///
/// // Tilted device
/// let tilted = AccelerometerReading::new(0.5, 0.0, 0.866);
///
/// let sdr1 = encoder.encode_to_sdr(stationary).unwrap();
/// let sdr2 = encoder.encode_to_sdr(tilted).unwrap();
///
/// // Different orientations have less overlap
/// assert!(sdr1.get_overlap(&sdr2) < sdr1.get_sum());
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AccelerometerEncoder {
    axis_bits: UInt,
    axis_active: UInt,
    magnitude_bits: UInt,
    magnitude_active: UInt,
    orientation_bits: UInt,
    orientation_active: UInt,
    max_acceleration: Real,
    size: UInt,
    dimensions: Vec<UInt>,
}

impl AccelerometerEncoder {
    /// Creates a new Accelerometer Encoder.
    pub fn new(params: AccelerometerEncoderParams) -> Result<Self> {
        if params.axis_active > params.axis_bits {
            return Err(MokoshError::InvalidParameter {
                name: "axis_active",
                message: "Cannot exceed axis_bits".to_string(),
            });
        }

        if params.magnitude_active > params.magnitude_bits {
            return Err(MokoshError::InvalidParameter {
                name: "magnitude_active",
                message: "Cannot exceed magnitude_bits".to_string(),
            });
        }

        if params.orientation_active > params.orientation_bits {
            return Err(MokoshError::InvalidParameter {
                name: "orientation_active",
                message: "Cannot exceed orientation_bits".to_string(),
            });
        }

        // 3 axes + magnitude + orientation
        let size = 3 * params.axis_bits + params.magnitude_bits + params.orientation_bits;

        Ok(Self {
            axis_bits: params.axis_bits,
            axis_active: params.axis_active,
            magnitude_bits: params.magnitude_bits,
            magnitude_active: params.magnitude_active,
            orientation_bits: params.orientation_bits,
            orientation_active: params.orientation_active,
            max_acceleration: params.max_acceleration,
            size,
            dimensions: vec![size],
        })
    }

    /// Normalizes an acceleration value.
    fn normalize_accel(&self, value: Real) -> Real {
        let normalized = (value + self.max_acceleration) / (2.0 * self.max_acceleration);
        normalized.clamp(0.0, 1.0)
    }

    /// Normalizes a magnitude value.
    fn normalize_magnitude(&self, value: Real) -> Real {
        (value / self.max_acceleration).clamp(0.0, 1.0)
    }

    /// Normalizes an angle (radians) to [0, 1].
    fn normalize_angle(&self, angle: Real, range: Real) -> Real {
        ((angle + range) / (2.0 * range)).clamp(0.0, 1.0)
    }
}

impl Encoder<AccelerometerReading> for AccelerometerEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, reading: AccelerometerReading, output: &mut Sdr) -> Result<()> {
        if output.dimensions() != self.dimensions.as_slice() {
            return Err(MokoshError::DimensionMismatch {
                expected: self.dimensions.clone(),
                actual: output.dimensions().to_vec(),
            });
        }

        let mut sparse = Vec::new();
        let mut offset: UInt = 0;

        // Encode X axis
        let norm_x = self.normalize_accel(reading.x);
        let positions = self.axis_bits - self.axis_active + 1;
        let start = (norm_x * (positions - 1) as Real).round() as UInt;
        for i in 0..self.axis_active {
            sparse.push(offset + start + i);
        }
        offset += self.axis_bits;

        // Encode Y axis
        let norm_y = self.normalize_accel(reading.y);
        let start = (norm_y * (positions - 1) as Real).round() as UInt;
        for i in 0..self.axis_active {
            sparse.push(offset + start + i);
        }
        offset += self.axis_bits;

        // Encode Z axis
        let norm_z = self.normalize_accel(reading.z);
        let start = (norm_z * (positions - 1) as Real).round() as UInt;
        for i in 0..self.axis_active {
            sparse.push(offset + start + i);
        }
        offset += self.axis_bits;

        // Encode magnitude
        let norm_mag = self.normalize_magnitude(reading.magnitude());
        let positions = self.magnitude_bits - self.magnitude_active + 1;
        let start = (norm_mag * (positions - 1) as Real).round() as UInt;
        for i in 0..self.magnitude_active {
            sparse.push(offset + start + i);
        }
        offset += self.magnitude_bits;

        // Encode orientation (pitch and roll)
        let pitch = reading.pitch();
        let roll = reading.roll();

        let orientation_per_angle = self.orientation_bits / 2;
        let active_per_angle = self.orientation_active / 2;

        // Pitch (-PI/2 to PI/2)
        let norm_pitch = self.normalize_angle(pitch, PI / 2.0);
        let positions = orientation_per_angle - active_per_angle + 1;
        let start = (norm_pitch * (positions - 1) as Real).round() as UInt;
        for i in 0..active_per_angle {
            sparse.push(offset + start + i);
        }

        // Roll (-PI to PI)
        let norm_roll = self.normalize_angle(roll, PI);
        let start = (norm_roll * (positions - 1) as Real).round() as UInt;
        for i in 0..active_per_angle {
            sparse.push(offset + orientation_per_angle + start + i);
        }

        sparse.sort_unstable();
        output.set_sparse_unchecked(sparse);

        Ok(())
    }
}

impl Encoder<(Real, Real, Real)> for AccelerometerEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, xyz: (Real, Real, Real), output: &mut Sdr) -> Result<()> {
        self.encode(AccelerometerReading::new(xyz.0, xyz.1, xyz.2), output)
    }
}

impl Encoder<[Real; 3]> for AccelerometerEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, xyz: [Real; 3], output: &mut Sdr) -> Result<()> {
        self.encode(AccelerometerReading::new(xyz[0], xyz[1], xyz[2]), output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_encoder() {
        let encoder = AccelerometerEncoder::new(AccelerometerEncoderParams::default()).unwrap();

        // 3*32 + 32 + 64 = 192
        assert_eq!(Encoder::<AccelerometerReading>::size(&encoder), 192);
    }

    #[test]
    fn test_reading_magnitude() {
        let reading = AccelerometerReading::new(0.0, 0.0, 1.0);
        assert!((reading.magnitude() - 1.0).abs() < 0.01);

        let reading = AccelerometerReading::new(1.0, 0.0, 0.0);
        assert!((reading.magnitude() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_reading_angles() {
        // Upright
        let upright = AccelerometerReading::new(0.0, 0.0, 1.0);
        assert!((upright.pitch()).abs() < 0.01);
        assert!((upright.roll()).abs() < 0.01);

        // Tilted forward
        let forward = AccelerometerReading::new(0.0, 0.5, 0.866);
        assert!(forward.pitch() > 0.0);
    }

    #[test]
    fn test_is_level() {
        let level = AccelerometerReading::new(0.0, 0.0, 1.0);
        assert!(level.is_level(0.1));

        let tilted = AccelerometerReading::new(0.5, 0.5, 0.7);
        assert!(!tilted.is_level(0.1));
    }

    #[test]
    fn test_encode_reading() {
        let encoder = AccelerometerEncoder::new(AccelerometerEncoderParams::default()).unwrap();

        let reading = AccelerometerReading::new(0.0, 0.0, 1.0);
        let sdr = encoder.encode_to_sdr(reading).unwrap();

        // 3*8 + 8 + 16 = 48
        assert_eq!(sdr.get_sum(), 48);
    }

    #[test]
    fn test_similar_readings_overlap() {
        let encoder = AccelerometerEncoder::new(AccelerometerEncoderParams::default()).unwrap();

        let reading1 = AccelerometerReading::new(0.0, 0.0, 1.0);
        let reading2 = AccelerometerReading::new(0.0, 0.0, 1.05); // Similar
        let reading3 = AccelerometerReading::new(1.0, 0.0, 0.0); // Different orientation

        let sdr1 = encoder.encode_to_sdr(reading1).unwrap();
        let sdr2 = encoder.encode_to_sdr(reading2).unwrap();
        let sdr3 = encoder.encode_to_sdr(reading3).unwrap();

        let similar_overlap = sdr1.get_overlap(&sdr2);
        let different_overlap = sdr1.get_overlap(&sdr3);

        assert!(similar_overlap > different_overlap);
    }

    #[test]
    fn test_encode_tuple() {
        let encoder = AccelerometerEncoder::new(AccelerometerEncoderParams::default()).unwrap();

        let sdr = encoder.encode_to_sdr((0.1 as Real, 0.2 as Real, 0.9 as Real)).unwrap();
        assert_eq!(sdr.get_sum(), 48);
    }

    #[test]
    fn test_encode_array() {
        let encoder = AccelerometerEncoder::new(AccelerometerEncoderParams::default()).unwrap();

        let sdr = encoder.encode_to_sdr([0.1 as Real, 0.2, 0.9]).unwrap();
        assert_eq!(sdr.get_sum(), 48);
    }

    #[test]
    fn test_deterministic() {
        let encoder = AccelerometerEncoder::new(AccelerometerEncoderParams::default()).unwrap();

        let reading = AccelerometerReading::new(0.5, -0.3, 0.8);

        let sdr1 = encoder.encode_to_sdr(reading).unwrap();
        let sdr2 = encoder.encode_to_sdr(reading).unwrap();

        assert_eq!(sdr1.get_sparse(), sdr2.get_sparse());
    }
}
