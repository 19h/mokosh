//! Geospatial Coordinate Encoder implementation.
//!
//! Encodes GPS coordinates (latitude/longitude) into SDRs with
//! locality-sensitive properties.

use crate::encoders::Encoder;
use crate::error::{MokoshError, Result};
use crate::types::{Real, Sdr, UInt};
use std::collections::HashSet;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Earth's radius in meters.
const EARTH_RADIUS_M: Real = 6_371_000.0;

/// Parameters for creating a Geospatial Coordinate Encoder.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GeospatialEncoderParams {
    /// Total number of bits in the output SDR.
    pub size: UInt,

    /// Number of active bits.
    pub active_bits: UInt,

    /// Scale in meters - coordinates within this distance share bits.
    pub scale: Real,

    /// Number of scales to encode (for multi-resolution encoding).
    pub num_scales: usize,
}

impl Default for GeospatialEncoderParams {
    fn default() -> Self {
        Self {
            size: 1000,
            active_bits: 21,
            scale: 100.0, // 100 meters
            num_scales: 1,
        }
    }
}

/// A GPS coordinate.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GpsCoordinate {
    /// Latitude in degrees (-90 to 90).
    pub latitude: Real,
    /// Longitude in degrees (-180 to 180).
    pub longitude: Real,
}

impl GpsCoordinate {
    /// Creates a new GPS coordinate.
    pub fn new(latitude: Real, longitude: Real) -> Result<Self> {
        if !(-90.0..=90.0).contains(&latitude) {
            return Err(MokoshError::InvalidParameter {
                name: "latitude",
                message: "Must be between -90 and 90".to_string(),
            });
        }
        if !(-180.0..=180.0).contains(&longitude) {
            return Err(MokoshError::InvalidParameter {
                name: "longitude",
                message: "Must be between -180 and 180".to_string(),
            });
        }
        Ok(Self {
            latitude,
            longitude,
        })
    }

    /// Calculates the Haversine distance in meters between two GPS coordinates.
    pub fn distance_to(&self, other: &GpsCoordinate) -> Real {
        let lat1 = self.latitude.to_radians();
        let lat2 = other.latitude.to_radians();
        let dlat = (other.latitude - self.latitude).to_radians();
        let dlon = (other.longitude - self.longitude).to_radians();

        let a = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
        let c = 2.0 * a.sqrt().asin();

        EARTH_RADIUS_M * c
    }
}

/// Encodes GPS coordinates into SDR representations.
///
/// Uses a locality-sensitive hashing approach where nearby coordinates
/// produce similar encodings. Supports multi-resolution encoding.
///
/// # Example
///
/// ```rust
/// use mokosh::encoders::{GeospatialEncoder, GeospatialEncoderParams, GpsCoordinate, Encoder};
///
/// let encoder = GeospatialEncoder::new(GeospatialEncoderParams {
///     size: 1000,
///     active_bits: 21,
///     scale: 100.0, // 100 meters
///     ..Default::default()
/// }).unwrap();
///
/// // Encode two nearby locations
/// let coord1 = GpsCoordinate::new(37.7749, -122.4194).unwrap(); // SF
/// let coord2 = GpsCoordinate::new(37.7750, -122.4194).unwrap(); // ~11m away
///
/// let sdr1 = encoder.encode_to_sdr(coord1).unwrap();
/// let sdr2 = encoder.encode_to_sdr(coord2).unwrap();
///
/// // Nearby coordinates should have high overlap
/// assert!(sdr1.get_overlap(&sdr2) > 15);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GeospatialEncoder {
    /// Total size of output.
    size: UInt,

    /// Number of active bits.
    active_bits: UInt,

    /// Scale in meters.
    scale: Real,

    /// Number of scales.
    num_scales: usize,

    /// Output dimensions.
    dimensions: Vec<UInt>,
}

impl GeospatialEncoder {
    /// Creates a new Geospatial Encoder.
    pub fn new(params: GeospatialEncoderParams) -> Result<Self> {
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

        if params.scale <= 0.0 {
            return Err(MokoshError::InvalidParameter {
                name: "scale",
                message: "Must be > 0".to_string(),
            });
        }

        if params.num_scales == 0 {
            return Err(MokoshError::InvalidParameter {
                name: "num_scales",
                message: "Must be > 0".to_string(),
            });
        }

        Ok(Self {
            size: params.size,
            active_bits: params.active_bits,
            scale: params.scale,
            num_scales: params.num_scales,
            dimensions: vec![params.size],
        })
    }

    /// Returns the scale in meters.
    pub fn scale(&self) -> Real {
        self.scale
    }

    /// Hash function for geospatial encoding.
    fn hash_cell(lat_cell: i64, lon_cell: i64, scale_idx: usize, bit_idx: u32) -> u64 {
        let mut hash: u64 = 0xcbf29ce484222325; // FNV-1a offset
        let prime: u64 = 0x100000001b3;

        hash ^= lat_cell as u64;
        hash = hash.wrapping_mul(prime);
        hash ^= lon_cell as u64;
        hash = hash.wrapping_mul(prime);
        hash ^= scale_idx as u64;
        hash = hash.wrapping_mul(prime);
        hash ^= bit_idx as u64;
        hash = hash.wrapping_mul(prime);

        hash
    }

    /// Converts latitude to meters from equator.
    fn lat_to_meters(lat: Real) -> Real {
        lat.to_radians() * EARTH_RADIUS_M
    }

    /// Converts longitude to meters from prime meridian at given latitude.
    fn lon_to_meters(lon: Real, lat: Real) -> Real {
        lon.to_radians() * EARTH_RADIUS_M * lat.to_radians().cos()
    }
}

impl Encoder<GpsCoordinate> for GeospatialEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, coord: GpsCoordinate, output: &mut Sdr) -> Result<()> {
        if output.dimensions() != self.dimensions.as_slice() {
            return Err(MokoshError::DimensionMismatch {
                expected: self.dimensions.clone(),
                actual: output.dimensions().to_vec(),
            });
        }

        let mut active_bits = HashSet::new();
        let bits_per_scale = self.active_bits / self.num_scales as UInt;

        for scale_idx in 0..self.num_scales {
            let current_scale = self.scale * (1 << scale_idx) as Real;

            // Convert coordinates to cell indices at this scale
            let lat_m = Self::lat_to_meters(coord.latitude);
            let lon_m = Self::lon_to_meters(coord.longitude, coord.latitude);

            let lat_cell = (lat_m / current_scale).floor() as i64;
            let lon_cell = (lon_m / current_scale).floor() as i64;

            // Generate bits for this scale
            let mut bit_idx = 0u32;
            let target_bits = if scale_idx == self.num_scales - 1 {
                // Last scale gets any remainder
                self.active_bits - (bits_per_scale * (self.num_scales - 1) as UInt)
            } else {
                bits_per_scale
            };

            while active_bits.len() < (scale_idx + 1) * bits_per_scale as usize
                && bit_idx < target_bits * 100
            {
                let hash = Self::hash_cell(lat_cell, lon_cell, scale_idx, bit_idx);
                let bit = (hash % self.size as u64) as UInt;
                active_bits.insert(bit);
                bit_idx += 1;

                if active_bits.len()
                    >= scale_idx * bits_per_scale as usize + target_bits as usize
                {
                    break;
                }
            }
        }

        let mut sparse: Vec<UInt> = active_bits.into_iter().collect();
        sparse.sort_unstable();
        output.set_sparse_unchecked(sparse);

        Ok(())
    }
}

impl Encoder<(Real, Real)> for GeospatialEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, value: (Real, Real), output: &mut Sdr) -> Result<()> {
        let coord = GpsCoordinate::new(value.0, value.1)?;
        self.encode(coord, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_encoder() {
        let encoder = GeospatialEncoder::new(GeospatialEncoderParams {
            size: 500,
            active_bits: 15,
            scale: 50.0,
            num_scales: 2,
        })
        .unwrap();

        assert_eq!(Encoder::<GpsCoordinate>::size(&encoder), 500);
        assert_eq!(encoder.scale(), 50.0);
    }

    #[test]
    fn test_gps_coordinate() {
        let coord = GpsCoordinate::new(37.7749, -122.4194).unwrap();
        assert_eq!(coord.latitude, 37.7749);
        assert_eq!(coord.longitude, -122.4194);
    }

    #[test]
    fn test_invalid_coordinates() {
        assert!(GpsCoordinate::new(91.0, 0.0).is_err());
        assert!(GpsCoordinate::new(-91.0, 0.0).is_err());
        assert!(GpsCoordinate::new(0.0, 181.0).is_err());
        assert!(GpsCoordinate::new(0.0, -181.0).is_err());
    }

    #[test]
    fn test_distance() {
        let sf = GpsCoordinate::new(37.7749, -122.4194).unwrap();
        let la = GpsCoordinate::new(34.0522, -118.2437).unwrap();

        let distance = sf.distance_to(&la);

        // SF to LA is roughly 560 km
        assert!(distance > 550_000.0);
        assert!(distance < 580_000.0);
    }

    #[test]
    fn test_encode() {
        let encoder = GeospatialEncoder::new(GeospatialEncoderParams::default()).unwrap();

        let coord = GpsCoordinate::new(37.7749, -122.4194).unwrap();
        let sdr = encoder.encode_to_sdr(coord).unwrap();

        assert_eq!(sdr.get_sum(), 21);
    }

    #[test]
    fn test_nearby_overlap() {
        let encoder = GeospatialEncoder::new(GeospatialEncoderParams {
            size: 1000,
            active_bits: 21,
            scale: 100.0,
            num_scales: 1,
        })
        .unwrap();

        // Two points about 11 meters apart
        let coord1 = GpsCoordinate::new(37.7749, -122.4194).unwrap();
        let coord2 = GpsCoordinate::new(37.7750, -122.4194).unwrap();

        // Two points far apart
        let coord3 = GpsCoordinate::new(40.7128, -74.0060).unwrap(); // NYC

        let sdr1 = encoder.encode_to_sdr(coord1).unwrap();
        let sdr2 = encoder.encode_to_sdr(coord2).unwrap();
        let sdr3 = encoder.encode_to_sdr(coord3).unwrap();

        let near_overlap = sdr1.get_overlap(&sdr2);
        let far_overlap = sdr1.get_overlap(&sdr3);

        assert!(near_overlap > far_overlap);
    }

    #[test]
    fn test_tuple_encoding() {
        let encoder = GeospatialEncoder::new(GeospatialEncoderParams::default()).unwrap();

        let sdr = encoder.encode_to_sdr((37.7749, -122.4194)).unwrap();
        assert_eq!(sdr.get_sum(), 21);
    }

    #[test]
    fn test_deterministic() {
        let encoder = GeospatialEncoder::new(GeospatialEncoderParams::default()).unwrap();

        let coord = GpsCoordinate::new(51.5074, -0.1278).unwrap();

        let sdr1 = encoder.encode_to_sdr(coord).unwrap();
        let sdr2 = encoder.encode_to_sdr(coord).unwrap();

        assert_eq!(sdr1.get_sparse(), sdr2.get_sparse());
    }

    #[test]
    fn test_multi_scale() {
        let encoder = GeospatialEncoder::new(GeospatialEncoderParams {
            size: 1000,
            active_bits: 20,
            scale: 10.0,
            num_scales: 4,
        })
        .unwrap();

        let coord = GpsCoordinate::new(48.8566, 2.3522).unwrap();
        let sdr = encoder.encode_to_sdr(coord).unwrap();

        // Should have approximately 20 active bits
        let sum = sdr.get_sum();
        assert!(sum >= 15 && sum <= 25);
    }
}
