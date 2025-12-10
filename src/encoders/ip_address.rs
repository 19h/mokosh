//! IP Address Encoder implementation.
//!
//! Encodes IPv4 and IPv6 addresses into SDRs with subnet hierarchy awareness.

use crate::encoders::Encoder;
use crate::error::{MokoshError, Result};
use crate::types::{Sdr, UInt};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Parameters for creating an IP Address Encoder.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct IpAddressEncoderParams {
    /// Bits per octet/segment for hierarchical encoding.
    pub bits_per_segment: UInt,

    /// Active bits per segment.
    pub active_per_segment: UInt,

    /// Whether to include IPv6 support (increases size).
    pub support_ipv6: bool,
}

impl Default for IpAddressEncoderParams {
    fn default() -> Self {
        Self {
            bits_per_segment: 32,
            active_per_segment: 8,
            support_ipv6: false,
        }
    }
}

/// Encodes IP addresses into SDR representations.
///
/// Uses hierarchical encoding where each octet (IPv4) or segment (IPv6)
/// gets its own region. This preserves subnet relationships - IPs in
/// the same /24 subnet share the first 3 octets' encoding.
///
/// # Example
///
/// ```rust
/// use mokosh::encoders::{IpAddressEncoder, IpAddressEncoderParams, Encoder};
/// use std::net::Ipv4Addr;
///
/// let encoder = IpAddressEncoder::new(IpAddressEncoderParams::default()).unwrap();
///
/// let ip1 = Ipv4Addr::new(192, 168, 1, 100);
/// let ip2 = Ipv4Addr::new(192, 168, 1, 101);  // Same /24
/// let ip3 = Ipv4Addr::new(10, 0, 0, 1);       // Different network
///
/// let sdr1 = encoder.encode_to_sdr(ip1).unwrap();
/// let sdr2 = encoder.encode_to_sdr(ip2).unwrap();
/// let sdr3 = encoder.encode_to_sdr(ip3).unwrap();
///
/// // Same subnet should have high overlap (3 shared octets)
/// assert!(sdr1.get_overlap(&sdr2) > sdr1.get_overlap(&sdr3));
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct IpAddressEncoder {
    bits_per_segment: UInt,
    active_per_segment: UInt,
    support_ipv6: bool,
    num_segments: usize,
    size: UInt,
    dimensions: Vec<UInt>,
}

impl IpAddressEncoder {
    /// Creates a new IP Address Encoder.
    pub fn new(params: IpAddressEncoderParams) -> Result<Self> {
        if params.active_per_segment > params.bits_per_segment {
            return Err(MokoshError::InvalidParameter {
                name: "active_per_segment",
                message: "Cannot exceed bits_per_segment".to_string(),
            });
        }

        let num_segments = if params.support_ipv6 { 8 } else { 4 };
        let size = num_segments as UInt * params.bits_per_segment;

        Ok(Self {
            bits_per_segment: params.bits_per_segment,
            active_per_segment: params.active_per_segment,
            support_ipv6: params.support_ipv6,
            num_segments,
            size,
            dimensions: vec![size],
        })
    }

    /// Encodes a single segment value (0-255 for IPv4, 0-65535 for IPv6).
    fn encode_segment(&self, value: u16, segment_idx: usize, sparse: &mut Vec<UInt>) {
        let segment_offset = segment_idx as UInt * self.bits_per_segment;
        let max_value = if self.support_ipv6 { 65535.0 } else { 255.0 };
        let normalized = value as f32 / max_value;

        let positions = self.bits_per_segment - self.active_per_segment + 1;
        let start = (normalized * (positions - 1) as f32).round() as UInt;

        for i in 0..self.active_per_segment {
            sparse.push(segment_offset + start + i);
        }
    }
}

impl Encoder<Ipv4Addr> for IpAddressEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, ip: Ipv4Addr, output: &mut Sdr) -> Result<()> {
        if output.dimensions() != self.dimensions.as_slice() {
            return Err(MokoshError::DimensionMismatch {
                expected: self.dimensions.clone(),
                actual: output.dimensions().to_vec(),
            });
        }

        let octets = ip.octets();
        let mut sparse = Vec::new();

        for (idx, &octet) in octets.iter().enumerate() {
            self.encode_segment(octet as u16, idx, &mut sparse);
        }

        // If IPv6 mode, pad remaining segments with zeros
        if self.support_ipv6 {
            for idx in 4..8 {
                self.encode_segment(0, idx, &mut sparse);
            }
        }

        sparse.sort_unstable();
        output.set_sparse_unchecked(sparse);

        Ok(())
    }
}

impl Encoder<Ipv6Addr> for IpAddressEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, ip: Ipv6Addr, output: &mut Sdr) -> Result<()> {
        if !self.support_ipv6 {
            return Err(MokoshError::InvalidParameter {
                name: "ip",
                message: "IPv6 not supported by this encoder".to_string(),
            });
        }

        if output.dimensions() != self.dimensions.as_slice() {
            return Err(MokoshError::DimensionMismatch {
                expected: self.dimensions.clone(),
                actual: output.dimensions().to_vec(),
            });
        }

        let segments = ip.segments();
        let mut sparse = Vec::new();

        for (idx, &segment) in segments.iter().enumerate() {
            self.encode_segment(segment, idx, &mut sparse);
        }

        sparse.sort_unstable();
        output.set_sparse_unchecked(sparse);

        Ok(())
    }
}

impl Encoder<IpAddr> for IpAddressEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, ip: IpAddr, output: &mut Sdr) -> Result<()> {
        match ip {
            IpAddr::V4(v4) => self.encode(v4, output),
            IpAddr::V6(v6) => self.encode(v6, output),
        }
    }
}

impl Encoder<&str> for IpAddressEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, ip_str: &str, output: &mut Sdr) -> Result<()> {
        let ip: IpAddr = ip_str.parse().map_err(|_| MokoshError::InvalidParameter {
            name: "ip",
            message: format!("Invalid IP address: {}", ip_str),
        })?;
        self.encode(ip, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_encoder() {
        let encoder = IpAddressEncoder::new(IpAddressEncoderParams {
            bits_per_segment: 32,
            active_per_segment: 8,
            support_ipv6: false,
        })
        .unwrap();

        assert_eq!(Encoder::<Ipv4Addr>::size(&encoder), 128); // 4 * 32
    }

    #[test]
    fn test_create_ipv6_encoder() {
        let encoder = IpAddressEncoder::new(IpAddressEncoderParams {
            bits_per_segment: 32,
            active_per_segment: 8,
            support_ipv6: true,
        })
        .unwrap();

        assert_eq!(Encoder::<Ipv6Addr>::size(&encoder), 256); // 8 * 32
    }

    #[test]
    fn test_encode_ipv4() {
        let encoder = IpAddressEncoder::new(IpAddressEncoderParams::default()).unwrap();

        let ip = Ipv4Addr::new(192, 168, 1, 100);
        let sdr = encoder.encode_to_sdr(ip).unwrap();

        // 4 octets * 8 active bits each = 32
        assert_eq!(sdr.get_sum(), 32);
    }

    #[test]
    fn test_same_subnet_overlap() {
        let encoder = IpAddressEncoder::new(IpAddressEncoderParams::default()).unwrap();

        let ip1 = Ipv4Addr::new(192, 168, 1, 100);
        let ip2 = Ipv4Addr::new(192, 168, 1, 101);
        let ip3 = Ipv4Addr::new(10, 0, 0, 1);

        let sdr1 = encoder.encode_to_sdr(ip1).unwrap();
        let sdr2 = encoder.encode_to_sdr(ip2).unwrap();
        let sdr3 = encoder.encode_to_sdr(ip3).unwrap();

        // Same /24 subnet - first 3 octets identical = 24 bits overlap
        let same_subnet_overlap = sdr1.get_overlap(&sdr2);
        let diff_subnet_overlap = sdr1.get_overlap(&sdr3);

        assert!(same_subnet_overlap > 20); // Should share ~24 bits
        assert!(same_subnet_overlap > diff_subnet_overlap);
    }

    #[test]
    fn test_encode_from_string() {
        let encoder = IpAddressEncoder::new(IpAddressEncoderParams::default()).unwrap();

        let sdr = encoder.encode_to_sdr("192.168.1.1").unwrap();
        assert_eq!(sdr.get_sum(), 32);
    }

    #[test]
    fn test_invalid_string() {
        let encoder = IpAddressEncoder::new(IpAddressEncoderParams::default()).unwrap();

        let result = encoder.encode_to_sdr("not-an-ip");
        assert!(result.is_err());
    }

    #[test]
    fn test_encode_ipv6() {
        let encoder = IpAddressEncoder::new(IpAddressEncoderParams {
            bits_per_segment: 32,
            active_per_segment: 8,
            support_ipv6: true,
        })
        .unwrap();

        let ip = Ipv6Addr::new(0x2001, 0x0db8, 0x85a3, 0, 0, 0x8a2e, 0x0370, 0x7334);
        let sdr = encoder.encode_to_sdr(ip).unwrap();

        // 8 segments * 8 active bits each = 64
        assert_eq!(sdr.get_sum(), 64);
    }

    #[test]
    fn test_ipv6_rejected_without_support() {
        let encoder = IpAddressEncoder::new(IpAddressEncoderParams {
            support_ipv6: false,
            ..Default::default()
        })
        .unwrap();

        let ip = Ipv6Addr::LOCALHOST;
        let result = encoder.encode_to_sdr(ip);
        assert!(result.is_err());
    }

    #[test]
    fn test_deterministic() {
        let encoder = IpAddressEncoder::new(IpAddressEncoderParams::default()).unwrap();

        let ip = Ipv4Addr::new(8, 8, 8, 8);

        let sdr1 = encoder.encode_to_sdr(ip).unwrap();
        let sdr2 = encoder.encode_to_sdr(ip).unwrap();

        assert_eq!(sdr1.get_sparse(), sdr2.get_sparse());
    }
}
