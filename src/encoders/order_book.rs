//! Order Book Encoder implementation.
//!
//! Encodes order book depth (bid/ask levels) into SDRs.

use crate::encoders::Encoder;
use crate::error::{MokoshError, Result};
use crate::types::{Real, Sdr, UInt};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Parameters for creating an Order Book Encoder.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OrderBookEncoderParams {
    /// Number of price levels to encode on each side.
    pub depth: usize,

    /// Bits per price level.
    pub bits_per_level: UInt,

    /// Active bits per level.
    pub active_per_level: UInt,

    /// Bits for encoding the spread.
    pub spread_bits: UInt,

    /// Active bits for spread.
    pub spread_active: UInt,

    /// Bits for encoding imbalance (bid vs ask volume).
    pub imbalance_bits: UInt,

    /// Active bits for imbalance.
    pub imbalance_active: UInt,

    /// Maximum expected volume for normalization.
    pub max_volume: Real,

    /// Maximum expected spread (as percentage of mid price).
    pub max_spread_pct: Real,
}

impl Default for OrderBookEncoderParams {
    fn default() -> Self {
        Self {
            depth: 5,
            bits_per_level: 20,
            active_per_level: 4,
            spread_bits: 30,
            spread_active: 6,
            imbalance_bits: 30,
            imbalance_active: 6,
            max_volume: 1_000_000.0,
            max_spread_pct: 5.0, // 5%
        }
    }
}

/// A price level in the order book.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PriceLevel {
    /// Price at this level.
    pub price: Real,
    /// Volume/quantity at this level.
    pub volume: Real,
}

impl PriceLevel {
    /// Creates a new price level.
    pub fn new(price: Real, volume: Real) -> Self {
        Self { price, volume }
    }
}

/// Order book snapshot.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OrderBook {
    /// Bid levels (highest first).
    pub bids: Vec<PriceLevel>,
    /// Ask levels (lowest first).
    pub asks: Vec<PriceLevel>,
}

impl OrderBook {
    /// Creates a new order book.
    pub fn new(bids: Vec<PriceLevel>, asks: Vec<PriceLevel>) -> Self {
        Self { bids, asks }
    }

    /// Returns the best bid price.
    pub fn best_bid(&self) -> Option<Real> {
        self.bids.first().map(|l| l.price)
    }

    /// Returns the best ask price.
    pub fn best_ask(&self) -> Option<Real> {
        self.asks.first().map(|l| l.price)
    }

    /// Returns the mid price.
    pub fn mid_price(&self) -> Option<Real> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid + ask) / 2.0),
            _ => None,
        }
    }

    /// Returns the spread.
    pub fn spread(&self) -> Option<Real> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Returns the spread as percentage of mid price.
    pub fn spread_pct(&self) -> Option<Real> {
        match (self.spread(), self.mid_price()) {
            (Some(spread), Some(mid)) if mid > 0.0 => Some((spread / mid) * 100.0),
            _ => None,
        }
    }

    /// Returns total bid volume.
    pub fn total_bid_volume(&self) -> Real {
        self.bids.iter().map(|l| l.volume).sum()
    }

    /// Returns total ask volume.
    pub fn total_ask_volume(&self) -> Real {
        self.asks.iter().map(|l| l.volume).sum()
    }

    /// Returns order imbalance (-1 to 1, positive = more bids).
    pub fn imbalance(&self) -> Real {
        let bid_vol = self.total_bid_volume();
        let ask_vol = self.total_ask_volume();
        let total = bid_vol + ask_vol;

        if total == 0.0 {
            0.0
        } else {
            (bid_vol - ask_vol) / total
        }
    }
}

/// Encodes order book snapshots into SDR representations.
///
/// Captures the depth structure, spread, and order imbalance.
///
/// # Example
///
/// ```rust
/// use mokosh::encoders::{OrderBookEncoder, OrderBookEncoderParams, OrderBook, PriceLevel, Encoder};
///
/// let encoder = OrderBookEncoder::new(OrderBookEncoderParams {
///     depth: 3,
///     ..Default::default()
/// }).unwrap();
///
/// let book = OrderBook::new(
///     vec![
///         PriceLevel::new(100.0, 500.0),
///         PriceLevel::new(99.5, 300.0),
///         PriceLevel::new(99.0, 200.0),
///     ],
///     vec![
///         PriceLevel::new(100.5, 400.0),
///         PriceLevel::new(101.0, 350.0),
///         PriceLevel::new(101.5, 250.0),
///     ],
/// );
///
/// let sdr = encoder.encode_to_sdr(book).unwrap();
/// assert!(sdr.get_sum() > 0);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OrderBookEncoder {
    depth: usize,
    bits_per_level: UInt,
    active_per_level: UInt,
    spread_bits: UInt,
    spread_active: UInt,
    imbalance_bits: UInt,
    imbalance_active: UInt,
    max_volume: Real,
    max_spread_pct: Real,
    levels_size: UInt,
    size: UInt,
    dimensions: Vec<UInt>,
}

impl OrderBookEncoder {
    /// Creates a new Order Book Encoder.
    pub fn new(params: OrderBookEncoderParams) -> Result<Self> {
        if params.depth == 0 {
            return Err(MokoshError::InvalidParameter {
                name: "depth",
                message: "Must be > 0".to_string(),
            });
        }

        if params.active_per_level > params.bits_per_level {
            return Err(MokoshError::InvalidParameter {
                name: "active_per_level",
                message: "Cannot exceed bits_per_level".to_string(),
            });
        }

        // bid levels + ask levels + spread + imbalance
        let levels_size = 2 * params.depth as UInt * params.bits_per_level;
        let size = levels_size + params.spread_bits + params.imbalance_bits;

        Ok(Self {
            depth: params.depth,
            bits_per_level: params.bits_per_level,
            active_per_level: params.active_per_level,
            spread_bits: params.spread_bits,
            spread_active: params.spread_active,
            imbalance_bits: params.imbalance_bits,
            imbalance_active: params.imbalance_active,
            max_volume: params.max_volume,
            max_spread_pct: params.max_spread_pct,
            levels_size,
            size,
            dimensions: vec![size],
        })
    }

    /// Returns the depth.
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Normalizes volume to [0, 1].
    fn normalize_volume(&self, volume: Real) -> Real {
        (volume / self.max_volume).clamp(0.0, 1.0)
    }
}

impl Encoder<OrderBook> for OrderBookEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, book: OrderBook, output: &mut Sdr) -> Result<()> {
        if output.dimensions() != self.dimensions.as_slice() {
            return Err(MokoshError::DimensionMismatch {
                expected: self.dimensions.clone(),
                actual: output.dimensions().to_vec(),
            });
        }

        let mut sparse = Vec::new();

        // Encode bid levels
        for (i, level) in book.bids.iter().take(self.depth).enumerate() {
            let offset = i as UInt * self.bits_per_level;
            let normalized = self.normalize_volume(level.volume);
            let positions = self.bits_per_level - self.active_per_level + 1;
            let start = (normalized * (positions - 1) as Real).round() as UInt;

            for j in 0..self.active_per_level {
                sparse.push(offset + start + j);
            }
        }

        // Encode ask levels
        let ask_offset = self.depth as UInt * self.bits_per_level;
        for (i, level) in book.asks.iter().take(self.depth).enumerate() {
            let offset = ask_offset + i as UInt * self.bits_per_level;
            let normalized = self.normalize_volume(level.volume);
            let positions = self.bits_per_level - self.active_per_level + 1;
            let start = (normalized * (positions - 1) as Real).round() as UInt;

            for j in 0..self.active_per_level {
                sparse.push(offset + start + j);
            }
        }

        // Encode spread
        if let Some(spread_pct) = book.spread_pct() {
            let spread_offset = self.levels_size;
            let normalized = (spread_pct / self.max_spread_pct).clamp(0.0, 1.0);
            let positions = self.spread_bits - self.spread_active + 1;
            let start = (normalized * (positions - 1) as Real).round() as UInt;

            for i in 0..self.spread_active {
                sparse.push(spread_offset + start + i);
            }
        }

        // Encode imbalance
        let imbalance = book.imbalance();
        let imbalance_offset = self.levels_size + self.spread_bits;
        let normalized = (imbalance + 1.0) / 2.0; // Map [-1, 1] to [0, 1]
        let positions = self.imbalance_bits - self.imbalance_active + 1;
        let start = (normalized * (positions - 1) as Real).round() as UInt;

        for i in 0..self.imbalance_active {
            sparse.push(imbalance_offset + start + i);
        }

        sparse.sort_unstable();
        output.set_sparse_unchecked(sparse);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_encoder() {
        let encoder = OrderBookEncoder::new(OrderBookEncoderParams {
            depth: 5,
            bits_per_level: 20,
            active_per_level: 4,
            spread_bits: 30,
            spread_active: 6,
            imbalance_bits: 30,
            imbalance_active: 6,
            ..Default::default()
        })
        .unwrap();

        assert_eq!(encoder.depth(), 5);
        // 2*5*20 + 30 + 30 = 260
        assert_eq!(Encoder::<OrderBook>::size(&encoder), 260);
    }

    #[test]
    fn test_order_book_metrics() {
        let book = OrderBook::new(
            vec![PriceLevel::new(100.0, 500.0), PriceLevel::new(99.0, 300.0)],
            vec![PriceLevel::new(101.0, 400.0), PriceLevel::new(102.0, 200.0)],
        );

        assert!((book.best_bid().unwrap() - 100.0).abs() < 0.01);
        assert!((book.best_ask().unwrap() - 101.0).abs() < 0.01);
        assert!((book.mid_price().unwrap() - 100.5).abs() < 0.01);
        assert!((book.spread().unwrap() - 1.0).abs() < 0.01);
        assert!((book.total_bid_volume() - 800.0).abs() < 0.01);
        assert!((book.total_ask_volume() - 600.0).abs() < 0.01);
    }

    #[test]
    fn test_imbalance() {
        // Equal volume
        let balanced = OrderBook::new(
            vec![PriceLevel::new(100.0, 500.0)],
            vec![PriceLevel::new(101.0, 500.0)],
        );
        assert!((balanced.imbalance()).abs() < 0.01);

        // More bids
        let bid_heavy = OrderBook::new(
            vec![PriceLevel::new(100.0, 900.0)],
            vec![PriceLevel::new(101.0, 100.0)],
        );
        assert!(bid_heavy.imbalance() > 0.7);

        // More asks
        let ask_heavy = OrderBook::new(
            vec![PriceLevel::new(100.0, 100.0)],
            vec![PriceLevel::new(101.0, 900.0)],
        );
        assert!(ask_heavy.imbalance() < -0.7);
    }

    #[test]
    fn test_encode_order_book() {
        let encoder = OrderBookEncoder::new(OrderBookEncoderParams {
            depth: 3,
            bits_per_level: 20,
            active_per_level: 4,
            spread_bits: 20,
            spread_active: 4,
            imbalance_bits: 20,
            imbalance_active: 4,
            ..Default::default()
        })
        .unwrap();

        let book = OrderBook::new(
            vec![
                PriceLevel::new(100.0, 500.0),
                PriceLevel::new(99.5, 300.0),
                PriceLevel::new(99.0, 200.0),
            ],
            vec![
                PriceLevel::new(100.5, 400.0),
                PriceLevel::new(101.0, 350.0),
                PriceLevel::new(101.5, 250.0),
            ],
        );

        let sdr = encoder.encode_to_sdr(book).unwrap();

        // 6 levels * 4 + spread + imbalance = 24 + 4 + 4 = 32
        assert_eq!(sdr.get_sum(), 32);
    }

    #[test]
    fn test_partial_depth() {
        let encoder = OrderBookEncoder::new(OrderBookEncoderParams {
            depth: 5,
            ..Default::default()
        })
        .unwrap();

        // Only 2 levels on each side (less than depth)
        let book = OrderBook::new(
            vec![PriceLevel::new(100.0, 500.0), PriceLevel::new(99.0, 300.0)],
            vec![
                PriceLevel::new(101.0, 400.0),
                PriceLevel::new(102.0, 200.0),
            ],
        );

        let sdr = encoder.encode_to_sdr(book).unwrap();
        assert!(sdr.get_sum() > 0);
    }

    #[test]
    fn test_similar_books_overlap() {
        let encoder = OrderBookEncoder::new(OrderBookEncoderParams::default()).unwrap();

        let book1 = OrderBook::new(
            vec![PriceLevel::new(100.0, 500.0)],
            vec![PriceLevel::new(100.5, 500.0)],
        );

        let book2 = OrderBook::new(
            vec![PriceLevel::new(100.0, 520.0)], // Similar volume
            vec![PriceLevel::new(100.5, 480.0)],
        );

        let book3 = OrderBook::new(
            vec![PriceLevel::new(100.0, 10000.0)], // Very different volume
            vec![PriceLevel::new(100.5, 10.0)],
        );

        let sdr1 = encoder.encode_to_sdr(book1).unwrap();
        let sdr2 = encoder.encode_to_sdr(book2).unwrap();
        let sdr3 = encoder.encode_to_sdr(book3).unwrap();

        let similar_overlap = sdr1.get_overlap(&sdr2);
        let different_overlap = sdr1.get_overlap(&sdr3);

        assert!(similar_overlap > different_overlap);
    }

    #[test]
    fn test_deterministic() {
        let encoder = OrderBookEncoder::new(OrderBookEncoderParams::default()).unwrap();

        let book = OrderBook::new(
            vec![PriceLevel::new(100.0, 500.0)],
            vec![PriceLevel::new(101.0, 400.0)],
        );

        let sdr1 = encoder.encode_to_sdr(book.clone()).unwrap();
        let sdr2 = encoder.encode_to_sdr(book).unwrap();

        assert_eq!(sdr1.get_sparse(), sdr2.get_sparse());
    }
}
