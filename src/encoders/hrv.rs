//! Heart Rate Variability (HRV) Encoder implementation.
//!
//! Encodes HRV metrics (R-R intervals, time domain, frequency domain) into SDRs.

use crate::encoders::Encoder;
use crate::error::{MokoshError, Result};
use crate::types::{Real, Sdr, UInt};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Parameters for creating an HRV Encoder.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HrvEncoderParams {
    /// Bits for encoding mean RR interval.
    pub mean_rr_bits: UInt,
    /// Active bits for mean RR.
    pub mean_rr_active: UInt,

    /// Bits for encoding SDNN (standard deviation of NN intervals).
    pub sdnn_bits: UInt,
    /// Active bits for SDNN.
    pub sdnn_active: UInt,

    /// Bits for encoding RMSSD (root mean square of successive differences).
    pub rmssd_bits: UInt,
    /// Active bits for RMSSD.
    pub rmssd_active: UInt,

    /// Bits for encoding pNN50 (percentage of successive intervals differing by >50ms).
    pub pnn50_bits: UInt,
    /// Active bits for pNN50.
    pub pnn50_active: UInt,

    /// Expected range for mean RR (in ms).
    pub mean_rr_range: (Real, Real),
    /// Expected range for SDNN (in ms).
    pub sdnn_range: (Real, Real),
    /// Expected range for RMSSD (in ms).
    pub rmssd_range: (Real, Real),
}

impl Default for HrvEncoderParams {
    fn default() -> Self {
        Self {
            mean_rr_bits: 50,
            mean_rr_active: 10,
            sdnn_bits: 40,
            sdnn_active: 8,
            rmssd_bits: 40,
            rmssd_active: 8,
            pnn50_bits: 30,
            pnn50_active: 6,
            // Typical physiological ranges
            mean_rr_range: (400.0, 1500.0), // 40-150 BPM
            sdnn_range: (10.0, 200.0),
            rmssd_range: (10.0, 150.0),
        }
    }
}

/// HRV metrics computed from R-R intervals.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HrvMetrics {
    /// Mean RR interval in milliseconds.
    pub mean_rr: Real,
    /// Standard deviation of NN intervals (SDNN) in milliseconds.
    pub sdnn: Real,
    /// Root mean square of successive differences (RMSSD) in milliseconds.
    pub rmssd: Real,
    /// Percentage of successive intervals differing by >50ms (pNN50).
    pub pnn50: Real,
}

impl HrvMetrics {
    /// Creates HRV metrics from raw values.
    pub fn new(mean_rr: Real, sdnn: Real, rmssd: Real, pnn50: Real) -> Self {
        Self {
            mean_rr,
            sdnn,
            rmssd,
            pnn50: pnn50.clamp(0.0, 100.0),
        }
    }

    /// Computes HRV metrics from a sequence of R-R intervals (in milliseconds).
    pub fn from_rr_intervals(intervals: &[Real]) -> Result<Self> {
        if intervals.len() < 2 {
            return Err(MokoshError::InvalidParameter {
                name: "intervals",
                message: "Need at least 2 RR intervals".to_string(),
            });
        }

        let n = intervals.len() as Real;

        // Mean RR
        let mean_rr: Real = intervals.iter().sum::<Real>() / n;

        // SDNN
        let variance: Real = intervals
            .iter()
            .map(|&rr| (rr - mean_rr).powi(2))
            .sum::<Real>()
            / (n - 1.0);
        let sdnn = variance.sqrt();

        // RMSSD and pNN50
        let mut sum_sq_diff = 0.0;
        let mut count_nn50 = 0;

        for window in intervals.windows(2) {
            let diff = (window[1] - window[0]).abs();
            sum_sq_diff += diff * diff;
            if diff > 50.0 {
                count_nn50 += 1;
            }
        }

        let rmssd = (sum_sq_diff / (n - 1.0)).sqrt();
        let pnn50 = (count_nn50 as Real / (intervals.len() - 1) as Real) * 100.0;

        Ok(Self {
            mean_rr,
            sdnn,
            rmssd,
            pnn50,
        })
    }

    /// Returns the heart rate in BPM.
    pub fn heart_rate_bpm(&self) -> Real {
        60000.0 / self.mean_rr
    }
}

/// Encodes HRV metrics into SDR representations.
///
/// Captures time-domain HRV features that reflect autonomic nervous
/// system activity and cardiovascular health.
///
/// # Example
///
/// ```rust
/// use mokosh::encoders::{HrvEncoder, HrvEncoderParams, HrvMetrics, Encoder};
///
/// let encoder = HrvEncoder::new(HrvEncoderParams::default()).unwrap();
///
/// // Create HRV metrics (typical resting values)
/// let hrv = HrvMetrics::new(
///     850.0,  // mean RR (~70 BPM)
///     50.0,   // SDNN
///     35.0,   // RMSSD
///     15.0,   // pNN50
/// );
///
/// let sdr = encoder.encode_to_sdr(hrv).unwrap();
/// assert!(sdr.get_sum() > 0);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HrvEncoder {
    mean_rr_bits: UInt,
    mean_rr_active: UInt,
    sdnn_bits: UInt,
    sdnn_active: UInt,
    rmssd_bits: UInt,
    rmssd_active: UInt,
    pnn50_bits: UInt,
    pnn50_active: UInt,
    mean_rr_range: (Real, Real),
    sdnn_range: (Real, Real),
    rmssd_range: (Real, Real),
    size: UInt,
    dimensions: Vec<UInt>,
}

impl HrvEncoder {
    /// Creates a new HRV Encoder.
    pub fn new(params: HrvEncoderParams) -> Result<Self> {
        if params.mean_rr_active > params.mean_rr_bits {
            return Err(MokoshError::InvalidParameter {
                name: "mean_rr_active",
                message: "Cannot exceed mean_rr_bits".to_string(),
            });
        }

        let size = params.mean_rr_bits
            + params.sdnn_bits
            + params.rmssd_bits
            + params.pnn50_bits;

        Ok(Self {
            mean_rr_bits: params.mean_rr_bits,
            mean_rr_active: params.mean_rr_active,
            sdnn_bits: params.sdnn_bits,
            sdnn_active: params.sdnn_active,
            rmssd_bits: params.rmssd_bits,
            rmssd_active: params.rmssd_active,
            pnn50_bits: params.pnn50_bits,
            pnn50_active: params.pnn50_active,
            mean_rr_range: params.mean_rr_range,
            sdnn_range: params.sdnn_range,
            rmssd_range: params.rmssd_range,
            size,
            dimensions: vec![size],
        })
    }

    /// Normalizes a value to [0, 1] given a range.
    fn normalize(value: Real, min: Real, max: Real) -> Real {
        ((value - min) / (max - min)).clamp(0.0, 1.0)
    }

    /// Encodes a single metric.
    fn encode_metric(
        &self,
        value: Real,
        min: Real,
        max: Real,
        offset: UInt,
        bits: UInt,
        active: UInt,
        sparse: &mut Vec<UInt>,
    ) {
        let normalized = Self::normalize(value, min, max);
        let positions = bits - active + 1;
        let start = (normalized * (positions - 1) as Real).round() as UInt;

        for i in 0..active {
            sparse.push(offset + start + i);
        }
    }
}

impl Encoder<HrvMetrics> for HrvEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, hrv: HrvMetrics, output: &mut Sdr) -> Result<()> {
        if output.dimensions() != self.dimensions.as_slice() {
            return Err(MokoshError::DimensionMismatch {
                expected: self.dimensions.clone(),
                actual: output.dimensions().to_vec(),
            });
        }

        let mut sparse = Vec::new();
        let mut offset = 0;

        // Encode mean RR
        self.encode_metric(
            hrv.mean_rr,
            self.mean_rr_range.0,
            self.mean_rr_range.1,
            offset,
            self.mean_rr_bits,
            self.mean_rr_active,
            &mut sparse,
        );
        offset += self.mean_rr_bits;

        // Encode SDNN
        self.encode_metric(
            hrv.sdnn,
            self.sdnn_range.0,
            self.sdnn_range.1,
            offset,
            self.sdnn_bits,
            self.sdnn_active,
            &mut sparse,
        );
        offset += self.sdnn_bits;

        // Encode RMSSD
        self.encode_metric(
            hrv.rmssd,
            self.rmssd_range.0,
            self.rmssd_range.1,
            offset,
            self.rmssd_bits,
            self.rmssd_active,
            &mut sparse,
        );
        offset += self.rmssd_bits;

        // Encode pNN50 (0-100%)
        self.encode_metric(
            hrv.pnn50,
            0.0,
            100.0,
            offset,
            self.pnn50_bits,
            self.pnn50_active,
            &mut sparse,
        );

        sparse.sort_unstable();
        output.set_sparse_unchecked(sparse);

        Ok(())
    }
}

impl Encoder<Vec<Real>> for HrvEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, rr_intervals: Vec<Real>, output: &mut Sdr) -> Result<()> {
        let hrv = HrvMetrics::from_rr_intervals(&rr_intervals)?;
        self.encode(hrv, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_encoder() {
        let encoder = HrvEncoder::new(HrvEncoderParams::default()).unwrap();

        // 50 + 40 + 40 + 30 = 160
        assert_eq!(Encoder::<HrvMetrics>::size(&encoder), 160);
    }

    #[test]
    fn test_compute_hrv_metrics() {
        // Simulate RR intervals around 800ms with some variability
        let intervals = vec![780.0, 820.0, 810.0, 790.0, 830.0, 800.0, 815.0, 785.0];

        let hrv = HrvMetrics::from_rr_intervals(&intervals).unwrap();

        assert!((hrv.mean_rr - 803.75).abs() < 1.0);
        assert!(hrv.sdnn > 0.0);
        assert!(hrv.rmssd > 0.0);
        assert!(hrv.pnn50 >= 0.0 && hrv.pnn50 <= 100.0);
    }

    #[test]
    fn test_heart_rate() {
        let hrv = HrvMetrics::new(857.0, 50.0, 35.0, 15.0);
        let hr = hrv.heart_rate_bpm();

        // 60000 / 857 ≈ 70 BPM
        assert!((hr - 70.0).abs() < 1.0);
    }

    #[test]
    fn test_encode_hrv() {
        let encoder = HrvEncoder::new(HrvEncoderParams::default()).unwrap();

        let hrv = HrvMetrics::new(850.0, 50.0, 35.0, 15.0);
        let sdr = encoder.encode_to_sdr(hrv).unwrap();

        // 10 + 8 + 8 + 6 = 32 active bits
        assert_eq!(sdr.get_sum(), 32);
    }

    #[test]
    fn test_similar_hrv_overlap() {
        let encoder = HrvEncoder::new(HrvEncoderParams::default()).unwrap();

        let hrv1 = HrvMetrics::new(850.0, 50.0, 35.0, 15.0);
        let hrv2 = HrvMetrics::new(860.0, 52.0, 36.0, 16.0); // Similar
        let hrv3 = HrvMetrics::new(600.0, 20.0, 15.0, 5.0); // Very different

        let sdr1 = encoder.encode_to_sdr(hrv1).unwrap();
        let sdr2 = encoder.encode_to_sdr(hrv2).unwrap();
        let sdr3 = encoder.encode_to_sdr(hrv3).unwrap();

        let similar_overlap = sdr1.get_overlap(&sdr2);
        let different_overlap = sdr1.get_overlap(&sdr3);

        assert!(similar_overlap > different_overlap);
    }

    #[test]
    fn test_encode_from_intervals() {
        let encoder = HrvEncoder::new(HrvEncoderParams::default()).unwrap();

        let intervals = vec![800.0, 810.0, 790.0, 820.0, 805.0];
        let sdr = encoder.encode_to_sdr(intervals).unwrap();

        assert_eq!(sdr.get_sum(), 32);
    }

    #[test]
    fn test_insufficient_intervals() {
        let result = HrvMetrics::from_rr_intervals(&[800.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_deterministic() {
        let encoder = HrvEncoder::new(HrvEncoderParams::default()).unwrap();

        let hrv = HrvMetrics::new(900.0, 60.0, 40.0, 20.0);

        let sdr1 = encoder.encode_to_sdr(hrv).unwrap();
        let sdr2 = encoder.encode_to_sdr(hrv).unwrap();

        assert_eq!(sdr1.get_sparse(), sdr2.get_sparse());
    }
}
