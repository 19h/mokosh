//! Currency Pair Encoder implementation.
//!
//! Encodes forex currency pairs with awareness of related currencies.

use crate::encoders::Encoder;
use crate::error::{MokoshError, Result};
use crate::types::{Real, Sdr, UInt};
use std::collections::HashSet;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Parameters for creating a Currency Pair Encoder.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CurrencyPairEncoderParams {
    /// Bits for encoding the base currency.
    pub base_bits: UInt,

    /// Active bits for base currency.
    pub base_active: UInt,

    /// Bits for encoding the quote currency.
    pub quote_bits: UInt,

    /// Active bits for quote currency.
    pub quote_active: UInt,

    /// Bits for encoding the exchange rate.
    pub rate_bits: UInt,

    /// Active bits for rate.
    pub rate_active: UInt,

    /// Expected rate range (min, max).
    pub rate_range: (Real, Real),

    /// Whether to use log scale for rates.
    pub log_scale: bool,
}

impl Default for CurrencyPairEncoderParams {
    fn default() -> Self {
        Self {
            base_bits: 64,
            base_active: 16,
            quote_bits: 64,
            quote_active: 16,
            rate_bits: 100,
            rate_active: 21,
            rate_range: (0.0001, 10000.0),
            log_scale: true,
        }
    }
}

/// A currency pair with exchange rate.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CurrencyPair {
    /// Base currency (e.g., "EUR" in EUR/USD).
    pub base: String,
    /// Quote currency (e.g., "USD" in EUR/USD).
    pub quote: String,
    /// Exchange rate (optional).
    pub rate: Option<Real>,
}

impl CurrencyPair {
    /// Creates a new currency pair.
    pub fn new(base: &str, quote: &str) -> Self {
        Self {
            base: base.to_uppercase(),
            quote: quote.to_uppercase(),
            rate: None,
        }
    }

    /// Creates a currency pair with exchange rate.
    pub fn with_rate(base: &str, quote: &str, rate: Real) -> Self {
        Self {
            base: base.to_uppercase(),
            quote: quote.to_uppercase(),
            rate: Some(rate),
        }
    }

    /// Parses a currency pair from string (e.g., "EUR/USD" or "EURUSD").
    pub fn parse(s: &str) -> Result<Self> {
        let s = s.trim().to_uppercase();

        if s.contains('/') {
            let parts: Vec<&str> = s.split('/').collect();
            if parts.len() != 2 {
                return Err(MokoshError::InvalidParameter {
                    name: "currency_pair",
                    message: "Expected format: BASE/QUOTE".to_string(),
                });
            }
            Ok(Self::new(parts[0], parts[1]))
        } else if s.len() == 6 {
            // EURUSD format
            Ok(Self::new(&s[0..3], &s[3..6]))
        } else {
            Err(MokoshError::InvalidParameter {
                name: "currency_pair",
                message: "Expected format: EUR/USD or EURUSD".to_string(),
            })
        }
    }

    /// Returns the inverse pair.
    pub fn inverse(&self) -> Self {
        Self {
            base: self.quote.clone(),
            quote: self.base.clone(),
            rate: self.rate.map(|r| 1.0 / r),
        }
    }

    /// Returns the pair symbol (e.g., "EUR/USD").
    pub fn symbol(&self) -> String {
        format!("{}/{}", self.base, self.quote)
    }
}

/// Encodes currency pairs into SDR representations.
///
/// Pairs sharing a currency have overlapping representations,
/// which helps capture relationships between related pairs.
///
/// # Example
///
/// ```rust
/// use mokosh::encoders::{CurrencyPairEncoder, CurrencyPairEncoderParams, CurrencyPair, Encoder};
///
/// let encoder = CurrencyPairEncoder::new(CurrencyPairEncoderParams::default()).unwrap();
///
/// let eur_usd = CurrencyPair::with_rate("EUR", "USD", 1.10);
/// let eur_gbp = CurrencyPair::with_rate("EUR", "GBP", 0.85);
/// let usd_jpy = CurrencyPair::with_rate("USD", "JPY", 150.0);
///
/// let sdr1 = encoder.encode_to_sdr(eur_usd).unwrap();
/// let sdr2 = encoder.encode_to_sdr(eur_gbp).unwrap();
/// let sdr3 = encoder.encode_to_sdr(usd_jpy).unwrap();
///
/// // EUR pairs share base currency
/// assert!(sdr1.get_overlap(&sdr2) > sdr1.get_overlap(&sdr3));
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CurrencyPairEncoder {
    base_bits: UInt,
    base_active: UInt,
    quote_bits: UInt,
    quote_active: UInt,
    rate_bits: UInt,
    rate_active: UInt,
    rate_range: (Real, Real),
    log_scale: bool,
    log_min: Real,
    log_max: Real,
    size: UInt,
    dimensions: Vec<UInt>,
}

impl CurrencyPairEncoder {
    /// Creates a new Currency Pair Encoder.
    pub fn new(params: CurrencyPairEncoderParams) -> Result<Self> {
        if params.base_active > params.base_bits {
            return Err(MokoshError::InvalidParameter {
                name: "base_active",
                message: "Cannot exceed base_bits".to_string(),
            });
        }

        if params.quote_active > params.quote_bits {
            return Err(MokoshError::InvalidParameter {
                name: "quote_active",
                message: "Cannot exceed quote_bits".to_string(),
            });
        }

        if params.rate_active > params.rate_bits {
            return Err(MokoshError::InvalidParameter {
                name: "rate_active",
                message: "Cannot exceed rate_bits".to_string(),
            });
        }

        let size = params.base_bits + params.quote_bits + params.rate_bits;

        Ok(Self {
            base_bits: params.base_bits,
            base_active: params.base_active,
            quote_bits: params.quote_bits,
            quote_active: params.quote_active,
            rate_bits: params.rate_bits,
            rate_active: params.rate_active,
            rate_range: params.rate_range,
            log_scale: params.log_scale,
            log_min: params.rate_range.0.ln(),
            log_max: params.rate_range.1.ln(),
            size,
            dimensions: vec![size],
        })
    }

    /// Hash function for currency codes.
    fn hash_currency(code: &str) -> u64 {
        let mut hash: u64 = 0xcbf29ce484222325;
        let prime: u64 = 0x100000001b3;

        for byte in code.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(prime);
        }

        hash
    }

    /// Gets bits for a currency code.
    fn get_currency_bits(&self, code: &str, offset: UInt, bits: UInt, active: UInt) -> Vec<UInt> {
        let hash = Self::hash_currency(code);
        let mut result = HashSet::new();
        let mut state = hash;

        // Use hash as seed with different scrambling to generate unique bit positions
        while result.len() < active as usize {
            // PCG-style mixing for better distribution
            state ^= state >> 33;
            state = state.wrapping_mul(0xff51afd7ed558ccd);
            state ^= state >> 33;
            state = state.wrapping_mul(0xc4ceb9fe1a85ec53);
            state ^= state >> 33;

            let bit = offset + (state % bits as u64) as UInt;
            result.insert(bit);
        }

        result.into_iter().collect()
    }

    /// Normalizes a rate to [0, 1].
    fn normalize_rate(&self, rate: Real) -> Real {
        let clamped = rate.clamp(self.rate_range.0, self.rate_range.1);

        if self.log_scale {
            let log_val = clamped.ln();
            (log_val - self.log_min) / (self.log_max - self.log_min)
        } else {
            (clamped - self.rate_range.0) / (self.rate_range.1 - self.rate_range.0)
        }
    }
}

impl Encoder<CurrencyPair> for CurrencyPairEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, pair: CurrencyPair, output: &mut Sdr) -> Result<()> {
        if output.dimensions() != self.dimensions.as_slice() {
            return Err(MokoshError::DimensionMismatch {
                expected: self.dimensions.clone(),
                actual: output.dimensions().to_vec(),
            });
        }

        let mut sparse = Vec::new();

        // Encode base currency
        let base_bits = self.get_currency_bits(&pair.base, 0, self.base_bits, self.base_active);
        sparse.extend(base_bits);

        // Encode quote currency
        let quote_offset = self.base_bits;
        let quote_bits =
            self.get_currency_bits(&pair.quote, quote_offset, self.quote_bits, self.quote_active);
        sparse.extend(quote_bits);

        // Encode rate if present
        if let Some(rate) = pair.rate {
            let rate_offset = self.base_bits + self.quote_bits;
            let normalized = self.normalize_rate(rate);
            let positions = self.rate_bits - self.rate_active + 1;
            let start = (normalized * (positions - 1) as Real).round() as UInt;

            for i in 0..self.rate_active {
                sparse.push(rate_offset + start + i);
            }
        }

        sparse.sort_unstable();
        sparse.dedup();
        output.set_sparse_unchecked(sparse);

        Ok(())
    }
}

impl Encoder<&str> for CurrencyPairEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, pair_str: &str, output: &mut Sdr) -> Result<()> {
        let pair = CurrencyPair::parse(pair_str)?;
        self.encode(pair, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_encoder() {
        let encoder = CurrencyPairEncoder::new(CurrencyPairEncoderParams::default()).unwrap();

        // 64 + 64 + 100 = 228
        assert_eq!(Encoder::<CurrencyPair>::size(&encoder), 228);
    }

    #[test]
    fn test_parse_pair() {
        let pair1 = CurrencyPair::parse("EUR/USD").unwrap();
        assert_eq!(pair1.base, "EUR");
        assert_eq!(pair1.quote, "USD");

        let pair2 = CurrencyPair::parse("GBPJPY").unwrap();
        assert_eq!(pair2.base, "GBP");
        assert_eq!(pair2.quote, "JPY");
    }

    #[test]
    fn test_inverse() {
        let pair = CurrencyPair::with_rate("EUR", "USD", 1.10);
        let inverse = pair.inverse();

        assert_eq!(inverse.base, "USD");
        assert_eq!(inverse.quote, "EUR");
        assert!((inverse.rate.unwrap() - 0.909).abs() < 0.01);
    }

    #[test]
    fn test_encode_pair() {
        let encoder = CurrencyPairEncoder::new(CurrencyPairEncoderParams::default()).unwrap();

        let pair = CurrencyPair::with_rate("EUR", "USD", 1.10);
        let sdr = encoder.encode_to_sdr(pair).unwrap();

        // 16 + 16 + 21 = 53
        assert_eq!(sdr.get_sum(), 53);
    }

    #[test]
    fn test_shared_currency_overlap() {
        let encoder = CurrencyPairEncoder::new(CurrencyPairEncoderParams::default()).unwrap();

        let eur_usd = CurrencyPair::new("EUR", "USD");
        let eur_gbp = CurrencyPair::new("EUR", "GBP");
        let usd_jpy = CurrencyPair::new("USD", "JPY");

        let sdr1 = encoder.encode_to_sdr(eur_usd).unwrap();
        let sdr2 = encoder.encode_to_sdr(eur_gbp).unwrap();
        let sdr3 = encoder.encode_to_sdr(usd_jpy).unwrap();

        // EUR pairs share base currency bits
        let eur_overlap = sdr1.get_overlap(&sdr2);
        // No shared currencies in same position
        let no_overlap = sdr2.get_overlap(&sdr3);

        assert!(eur_overlap > no_overlap);
        assert!(eur_overlap >= 16); // At least base currency bits
    }

    #[test]
    fn test_encode_from_string() {
        let encoder = CurrencyPairEncoder::new(CurrencyPairEncoderParams::default()).unwrap();

        let sdr = encoder.encode_to_sdr("GBP/USD").unwrap();
        assert!(sdr.get_sum() > 0);
    }

    #[test]
    fn test_without_rate() {
        let encoder = CurrencyPairEncoder::new(CurrencyPairEncoderParams::default()).unwrap();

        let pair = CurrencyPair::new("EUR", "USD");
        let sdr = encoder.encode_to_sdr(pair).unwrap();

        // Only currency bits, no rate
        assert_eq!(sdr.get_sum(), 32); // 16 + 16
    }

    #[test]
    fn test_deterministic() {
        let encoder = CurrencyPairEncoder::new(CurrencyPairEncoderParams::default()).unwrap();

        let pair = CurrencyPair::with_rate("EUR", "USD", 1.10);

        let sdr1 = encoder.encode_to_sdr(pair.clone()).unwrap();
        let sdr2 = encoder.encode_to_sdr(pair).unwrap();

        assert_eq!(sdr1.get_sparse(), sdr2.get_sparse());
    }
}
