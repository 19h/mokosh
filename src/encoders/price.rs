//! Price Encoder implementation.
//!
//! Encodes financial prices with log-scale awareness and volatility support.

use crate::encoders::Encoder;
use crate::error::{MokoshError, Result};
use crate::types::{Real, Sdr, UInt};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Parameters for creating a Price Encoder.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PriceEncoderParams {
    /// Minimum expected price.
    pub min_price: Real,

    /// Maximum expected price.
    pub max_price: Real,

    /// Bits for encoding the price level.
    pub price_bits: UInt,

    /// Active bits for price.
    pub price_active: UInt,

    /// Bits for encoding price change (if using stateful mode).
    pub change_bits: UInt,

    /// Active bits for change.
    pub change_active: UInt,

    /// Maximum expected percent change for change encoding.
    pub max_percent_change: Real,

    /// Whether to use log scale for price encoding.
    pub log_scale: bool,
}

impl Default for PriceEncoderParams {
    fn default() -> Self {
        Self {
            min_price: 0.01,
            max_price: 10000.0,
            price_bits: 100,
            price_active: 21,
            change_bits: 50,
            change_active: 10,
            max_percent_change: 10.0, // 10%
            log_scale: true,
        }
    }
}

/// A price with optional change information.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PriceData {
    /// Current price.
    pub price: Real,
    /// Price change (optional, as percent or absolute).
    pub change: Option<Real>,
    /// Whether change is expressed as percent (true) or absolute (false).
    pub change_is_percent: bool,
}

impl PriceData {
    /// Creates price data without change.
    pub fn new(price: Real) -> Self {
        Self {
            price,
            change: None,
            change_is_percent: true,
        }
    }

    /// Creates price data with percent change.
    pub fn with_percent_change(price: Real, percent_change: Real) -> Self {
        Self {
            price,
            change: Some(percent_change),
            change_is_percent: true,
        }
    }

    /// Creates price data from current and previous price.
    pub fn from_prices(current: Real, previous: Real) -> Self {
        let percent_change = if previous != 0.0 {
            ((current - previous) / previous) * 100.0
        } else {
            0.0
        };
        Self::with_percent_change(current, percent_change)
    }
}

/// Encodes financial prices into SDR representations.
///
/// Uses logarithmic scaling for better resolution across orders of magnitude,
/// and optionally includes price change information.
///
/// # Example
///
/// ```rust
/// use mokosh::encoders::{PriceEncoder, PriceEncoderParams, PriceData, Encoder};
///
/// let encoder = PriceEncoder::new(PriceEncoderParams {
///     min_price: 1.0,
///     max_price: 1000.0,
///     log_scale: true,
///     ..Default::default()
/// }).unwrap();
///
/// let price_10 = PriceData::new(10.0);
/// let price_100 = PriceData::new(100.0);
///
/// let sdr_10 = encoder.encode_to_sdr(price_10).unwrap();
/// let sdr_100 = encoder.encode_to_sdr(price_100).unwrap();
///
/// // Both encode successfully
/// assert!(sdr_10.get_sum() > 0);
/// assert!(sdr_100.get_sum() > 0);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PriceEncoder {
    min_price: Real,
    max_price: Real,
    price_bits: UInt,
    price_active: UInt,
    change_bits: UInt,
    change_active: UInt,
    max_percent_change: Real,
    log_scale: bool,
    log_min: Real,
    log_max: Real,
    size: UInt,
    dimensions: Vec<UInt>,
}

impl PriceEncoder {
    /// Creates a new Price Encoder.
    pub fn new(params: PriceEncoderParams) -> Result<Self> {
        if params.min_price <= 0.0 {
            return Err(MokoshError::InvalidParameter {
                name: "min_price",
                message: "Must be > 0".to_string(),
            });
        }

        if params.max_price <= params.min_price {
            return Err(MokoshError::InvalidParameter {
                name: "max_price",
                message: "Must be greater than min_price".to_string(),
            });
        }

        if params.price_active > params.price_bits {
            return Err(MokoshError::InvalidParameter {
                name: "price_active",
                message: "Cannot exceed price_bits".to_string(),
            });
        }

        let size = params.price_bits + params.change_bits;
        let log_min = params.min_price.ln();
        let log_max = params.max_price.ln();

        Ok(Self {
            min_price: params.min_price,
            max_price: params.max_price,
            price_bits: params.price_bits,
            price_active: params.price_active,
            change_bits: params.change_bits,
            change_active: params.change_active,
            max_percent_change: params.max_percent_change,
            log_scale: params.log_scale,
            log_min,
            log_max,
            size,
            dimensions: vec![size],
        })
    }

    /// Normalizes a price to [0, 1].
    fn normalize_price(&self, price: Real) -> Real {
        let clamped = price.clamp(self.min_price, self.max_price);

        if self.log_scale {
            let log_val = clamped.ln();
            (log_val - self.log_min) / (self.log_max - self.log_min)
        } else {
            (clamped - self.min_price) / (self.max_price - self.min_price)
        }
    }

    /// Normalizes a percent change to [0, 1].
    fn normalize_change(&self, change: Real) -> Real {
        let clamped = change.clamp(-self.max_percent_change, self.max_percent_change);
        (clamped + self.max_percent_change) / (2.0 * self.max_percent_change)
    }
}

impl Encoder<PriceData> for PriceEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, data: PriceData, output: &mut Sdr) -> Result<()> {
        if output.dimensions() != self.dimensions.as_slice() {
            return Err(MokoshError::DimensionMismatch {
                expected: self.dimensions.clone(),
                actual: output.dimensions().to_vec(),
            });
        }

        let mut sparse = Vec::new();

        // Encode price
        let normalized_price = self.normalize_price(data.price);
        let positions = self.price_bits - self.price_active + 1;
        let start = (normalized_price * (positions - 1) as Real).round() as UInt;

        for i in 0..self.price_active {
            sparse.push(start + i);
        }

        // Encode change if present
        if let Some(change) = data.change {
            let change_offset = self.price_bits;
            let change_val = if data.change_is_percent {
                change
            } else {
                // Convert absolute to percent
                if data.price != 0.0 {
                    (change / data.price) * 100.0
                } else {
                    0.0
                }
            };

            let normalized_change = self.normalize_change(change_val);
            let positions = self.change_bits - self.change_active + 1;
            let start = (normalized_change * (positions - 1) as Real).round() as UInt;

            for i in 0..self.change_active {
                sparse.push(change_offset + start + i);
            }
        }

        sparse.sort_unstable();
        output.set_sparse_unchecked(sparse);

        Ok(())
    }
}

impl Encoder<Real> for PriceEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, price: Real, output: &mut Sdr) -> Result<()> {
        self.encode(PriceData::new(price), output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_encoder() {
        let encoder = PriceEncoder::new(PriceEncoderParams {
            min_price: 1.0,
            max_price: 1000.0,
            price_bits: 100,
            price_active: 20,
            change_bits: 50,
            change_active: 10,
            ..Default::default()
        })
        .unwrap();

        assert_eq!(Encoder::<PriceData>::size(&encoder), 150);
    }

    #[test]
    fn test_encode_price() {
        let encoder = PriceEncoder::new(PriceEncoderParams::default()).unwrap();

        let price = PriceData::new(100.0);
        let sdr = encoder.encode_to_sdr(price).unwrap();

        // Only price bits (no change)
        assert_eq!(sdr.get_sum(), 21);
    }

    #[test]
    fn test_encode_with_change() {
        let encoder = PriceEncoder::new(PriceEncoderParams::default()).unwrap();

        let price = PriceData::with_percent_change(100.0, 2.5);
        let sdr = encoder.encode_to_sdr(price).unwrap();

        // Price + change bits
        assert_eq!(sdr.get_sum(), 31); // 21 + 10
    }

    #[test]
    fn test_log_scale_spacing() {
        let encoder = PriceEncoder::new(PriceEncoderParams {
            min_price: 1.0,
            max_price: 10000.0,
            price_bits: 200,
            price_active: 20,
            change_bits: 0,
            change_active: 0,
            log_scale: true,
            ..Default::default()
        })
        .unwrap();

        // Equal ratios should have equal representation spacing
        let sdr_1 = encoder.encode_to_sdr(1.0 as Real).unwrap();
        let sdr_10 = encoder.encode_to_sdr(10.0 as Real).unwrap();
        let sdr_100 = encoder.encode_to_sdr(100.0 as Real).unwrap();
        let sdr_1000 = encoder.encode_to_sdr(1000.0 as Real).unwrap();

        let overlap_1_10 = sdr_1.get_overlap(&sdr_10);
        let overlap_10_100 = sdr_10.get_overlap(&sdr_100);
        let overlap_100_1000 = sdr_100.get_overlap(&sdr_1000);

        // Overlaps should be approximately equal (log spacing)
        assert!((overlap_1_10 as i32 - overlap_10_100 as i32).abs() <= 2);
        assert!((overlap_10_100 as i32 - overlap_100_1000 as i32).abs() <= 2);
    }

    #[test]
    fn test_from_prices() {
        let data = PriceData::from_prices(110.0, 100.0);

        assert!((data.price - 110.0).abs() < 0.01);
        assert!((data.change.unwrap() - 10.0).abs() < 0.01); // 10% increase
    }

    #[test]
    fn test_similar_prices_overlap() {
        let encoder = PriceEncoder::new(PriceEncoderParams::default()).unwrap();

        let p1 = PriceData::new(100.0);
        let p2 = PriceData::new(105.0);
        let p3 = PriceData::new(500.0);

        let sdr1 = encoder.encode_to_sdr(p1).unwrap();
        let sdr2 = encoder.encode_to_sdr(p2).unwrap();
        let sdr3 = encoder.encode_to_sdr(p3).unwrap();

        let near_overlap = sdr1.get_overlap(&sdr2);
        let far_overlap = sdr1.get_overlap(&sdr3);

        assert!(near_overlap > far_overlap);
    }

    #[test]
    fn test_encode_real() {
        let encoder = PriceEncoder::new(PriceEncoderParams::default()).unwrap();

        let sdr = encoder.encode_to_sdr(50.0 as Real).unwrap();
        assert_eq!(sdr.get_sum(), 21);
    }

    #[test]
    fn test_clipping() {
        let encoder = PriceEncoder::new(PriceEncoderParams {
            min_price: 10.0,
            max_price: 100.0,
            ..Default::default()
        })
        .unwrap();

        // Below min
        let sdr_low = encoder.encode_to_sdr(1.0 as Real).unwrap();
        let sdr_min = encoder.encode_to_sdr(10.0 as Real).unwrap();
        assert_eq!(sdr_low.get_sparse(), sdr_min.get_sparse());

        // Above max
        let sdr_high = encoder.encode_to_sdr(1000.0 as Real).unwrap();
        let sdr_max = encoder.encode_to_sdr(100.0 as Real).unwrap();
        assert_eq!(sdr_high.get_sparse(), sdr_max.get_sparse());
    }

    #[test]
    fn test_deterministic() {
        let encoder = PriceEncoder::new(PriceEncoderParams::default()).unwrap();

        let sdr1 = encoder.encode_to_sdr(PriceData::new(123.45)).unwrap();
        let sdr2 = encoder.encode_to_sdr(PriceData::new(123.45)).unwrap();

        assert_eq!(sdr1.get_sparse(), sdr2.get_sparse());
    }
}
