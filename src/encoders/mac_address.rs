//! MAC Address Encoder implementation.
//!
//! Encodes MAC addresses into SDRs with vendor prefix (OUI) awareness.

use crate::encoders::Encoder;
use crate::error::{MokoshError, Result};
use crate::types::{Sdr, UInt};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Parameters for creating a MAC Address Encoder.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MacAddressEncoderParams {
    /// Bits per octet.
    pub bits_per_octet: UInt,

    /// Active bits per octet.
    pub active_per_octet: UInt,

    /// Extra bits for the OUI (vendor prefix) - first 3 octets.
    /// This allows better differentiation of vendor prefixes.
    pub oui_extra_bits: UInt,

    /// Active bits for OUI encoding.
    pub oui_active: UInt,
}

impl Default for MacAddressEncoderParams {
    fn default() -> Self {
        Self {
            bits_per_octet: 32,
            active_per_octet: 8,
            oui_extra_bits: 64,
            oui_active: 16,
        }
    }
}

/// A MAC address (6 octets).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MacAddress {
    /// The 6 octets of the MAC address.
    pub octets: [u8; 6],
}

impl MacAddress {
    /// Creates a new MAC address from octets.
    pub fn new(octets: [u8; 6]) -> Self {
        Self { octets }
    }

    /// Parses a MAC address from string (supports : and - separators).
    pub fn parse(s: &str) -> Result<Self> {
        let s = s.trim();
        let parts: Vec<&str> = if s.contains(':') {
            s.split(':').collect()
        } else if s.contains('-') {
            s.split('-').collect()
        } else {
            return Err(MokoshError::InvalidParameter {
                name: "mac_address",
                message: "Invalid format, expected XX:XX:XX:XX:XX:XX".to_string(),
            });
        };

        if parts.len() != 6 {
            return Err(MokoshError::InvalidParameter {
                name: "mac_address",
                message: format!("Expected 6 octets, got {}", parts.len()),
            });
        }

        let mut octets = [0u8; 6];
        for (i, part) in parts.iter().enumerate() {
            octets[i] = u8::from_str_radix(part, 16).map_err(|_| {
                MokoshError::InvalidParameter {
                    name: "mac_address",
                    message: format!("Invalid hex value: {}", part),
                }
            })?;
        }

        Ok(Self { octets })
    }

    /// Returns the OUI (Organizationally Unique Identifier) - first 3 octets.
    pub fn oui(&self) -> [u8; 3] {
        [self.octets[0], self.octets[1], self.octets[2]]
    }

    /// Returns whether this is a locally administered address.
    pub fn is_local(&self) -> bool {
        (self.octets[0] & 0x02) != 0
    }

    /// Returns whether this is a multicast address.
    pub fn is_multicast(&self) -> bool {
        (self.octets[0] & 0x01) != 0
    }
}

impl std::fmt::Display for MacAddress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:02X}:{:02X}:{:02X}:{:02X}:{:02X}:{:02X}",
            self.octets[0],
            self.octets[1],
            self.octets[2],
            self.octets[3],
            self.octets[4],
            self.octets[5]
        )
    }
}

/// Encodes MAC addresses into SDR representations.
///
/// Uses hierarchical encoding with special treatment for the OUI
/// (vendor prefix). MACs with the same vendor share significant overlap.
///
/// # Example
///
/// ```rust
/// use mokosh::encoders::{MacAddressEncoder, MacAddressEncoderParams, MacAddress, Encoder};
///
/// let encoder = MacAddressEncoder::new(MacAddressEncoderParams::default()).unwrap();
///
/// let mac1 = MacAddress::parse("AA:BB:CC:11:22:33").unwrap();
/// let mac2 = MacAddress::parse("AA:BB:CC:44:55:66").unwrap();  // Same OUI
/// let mac3 = MacAddress::parse("11:22:33:44:55:66").unwrap();  // Different OUI
///
/// let sdr1 = encoder.encode_to_sdr(mac1).unwrap();
/// let sdr2 = encoder.encode_to_sdr(mac2).unwrap();
/// let sdr3 = encoder.encode_to_sdr(mac3).unwrap();
///
/// // Same vendor should have high overlap
/// assert!(sdr1.get_overlap(&sdr2) > sdr1.get_overlap(&sdr3));
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MacAddressEncoder {
    bits_per_octet: UInt,
    active_per_octet: UInt,
    oui_extra_bits: UInt,
    oui_active: UInt,
    size: UInt,
    dimensions: Vec<UInt>,
}

impl MacAddressEncoder {
    /// Creates a new MAC Address Encoder.
    pub fn new(params: MacAddressEncoderParams) -> Result<Self> {
        if params.active_per_octet > params.bits_per_octet {
            return Err(MokoshError::InvalidParameter {
                name: "active_per_octet",
                message: "Cannot exceed bits_per_octet".to_string(),
            });
        }

        if params.oui_active > params.oui_extra_bits {
            return Err(MokoshError::InvalidParameter {
                name: "oui_active",
                message: "Cannot exceed oui_extra_bits".to_string(),
            });
        }

        // Size = 6 octets + OUI extra bits
        let size = 6 * params.bits_per_octet + params.oui_extra_bits;

        Ok(Self {
            bits_per_octet: params.bits_per_octet,
            active_per_octet: params.active_per_octet,
            oui_extra_bits: params.oui_extra_bits,
            oui_active: params.oui_active,
            size,
            dimensions: vec![size],
        })
    }

    /// Hash function for OUI encoding.
    fn hash_oui(&self, oui: [u8; 3]) -> u64 {
        let mut hash: u64 = 0xcbf29ce484222325;
        let prime: u64 = 0x100000001b3;

        for byte in oui {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(prime);
        }

        hash
    }
}

impl Encoder<MacAddress> for MacAddressEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, mac: MacAddress, output: &mut Sdr) -> Result<()> {
        if output.dimensions() != self.dimensions.as_slice() {
            return Err(MokoshError::DimensionMismatch {
                expected: self.dimensions.clone(),
                actual: output.dimensions().to_vec(),
            });
        }

        let mut sparse = Vec::new();

        // Encode each octet
        for (idx, &octet) in mac.octets.iter().enumerate() {
            let offset = idx as UInt * self.bits_per_octet;
            let normalized = octet as f32 / 255.0;
            let positions = self.bits_per_octet - self.active_per_octet + 1;
            let start = (normalized * (positions - 1) as f32).round() as UInt;

            for i in 0..self.active_per_octet {
                sparse.push(offset + start + i);
            }
        }

        // Encode OUI hash
        let oui_offset = 6 * self.bits_per_octet;
        let oui_hash = self.hash_oui(mac.oui());

        let mut state = oui_hash;
        let mut oui_bits_added = 0;

        while oui_bits_added < self.oui_active as usize {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let bit = oui_offset + (state % self.oui_extra_bits as u64) as UInt;
            if !sparse.contains(&bit) {
                sparse.push(bit);
                oui_bits_added += 1;
            }
        }

        sparse.sort_unstable();
        output.set_sparse_unchecked(sparse);

        Ok(())
    }
}

impl Encoder<&str> for MacAddressEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, mac_str: &str, output: &mut Sdr) -> Result<()> {
        let mac = MacAddress::parse(mac_str)?;
        self.encode(mac, output)
    }
}

impl Encoder<[u8; 6]> for MacAddressEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, octets: [u8; 6], output: &mut Sdr) -> Result<()> {
        self.encode(MacAddress::new(octets), output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_encoder() {
        let encoder = MacAddressEncoder::new(MacAddressEncoderParams::default()).unwrap();

        // 6 * 32 + 64 = 256
        assert_eq!(Encoder::<MacAddress>::size(&encoder), 256);
    }

    #[test]
    fn test_parse_mac() {
        let mac1 = MacAddress::parse("AA:BB:CC:DD:EE:FF").unwrap();
        assert_eq!(mac1.octets, [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF]);

        let mac2 = MacAddress::parse("aa-bb-cc-dd-ee-ff").unwrap();
        assert_eq!(mac2.octets, [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF]);
    }

    #[test]
    fn test_mac_oui() {
        let mac = MacAddress::parse("AA:BB:CC:DD:EE:FF").unwrap();
        assert_eq!(mac.oui(), [0xAA, 0xBB, 0xCC]);
    }

    #[test]
    fn test_mac_flags() {
        // Locally administered
        let local = MacAddress::new([0x02, 0x00, 0x00, 0x00, 0x00, 0x00]);
        assert!(local.is_local());
        assert!(!local.is_multicast());

        // Multicast
        let multicast = MacAddress::new([0x01, 0x00, 0x00, 0x00, 0x00, 0x00]);
        assert!(!multicast.is_local());
        assert!(multicast.is_multicast());
    }

    #[test]
    fn test_encode_mac() {
        let encoder = MacAddressEncoder::new(MacAddressEncoderParams::default()).unwrap();

        let mac = MacAddress::parse("AA:BB:CC:DD:EE:FF").unwrap();
        let sdr = encoder.encode_to_sdr(mac).unwrap();

        // 6 * 8 + 16 = 64 active bits
        assert_eq!(sdr.get_sum(), 64);
    }

    #[test]
    fn test_same_oui_overlap() {
        let encoder = MacAddressEncoder::new(MacAddressEncoderParams::default()).unwrap();

        let mac1 = MacAddress::parse("AA:BB:CC:11:22:33").unwrap();
        let mac2 = MacAddress::parse("AA:BB:CC:44:55:66").unwrap();
        let mac3 = MacAddress::parse("11:22:33:44:55:66").unwrap();

        let sdr1 = encoder.encode_to_sdr(mac1).unwrap();
        let sdr2 = encoder.encode_to_sdr(mac2).unwrap();
        let sdr3 = encoder.encode_to_sdr(mac3).unwrap();

        // Same OUI = share OUI bits + first 3 octets
        let same_oui_overlap = sdr1.get_overlap(&sdr2);
        let diff_oui_overlap = sdr1.get_overlap(&sdr3);

        assert!(same_oui_overlap > diff_oui_overlap);
    }

    #[test]
    fn test_encode_from_string() {
        let encoder = MacAddressEncoder::new(MacAddressEncoderParams::default()).unwrap();

        let sdr = encoder.encode_to_sdr("DE:AD:BE:EF:CA:FE").unwrap();
        assert_eq!(sdr.get_sum(), 64);
    }

    #[test]
    fn test_encode_from_array() {
        let encoder = MacAddressEncoder::new(MacAddressEncoderParams::default()).unwrap();

        let sdr = encoder.encode_to_sdr([0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE]).unwrap();
        assert_eq!(sdr.get_sum(), 64);
    }

    #[test]
    fn test_invalid_mac() {
        let result = MacAddress::parse("not-a-mac");
        assert!(result.is_err());

        let result = MacAddress::parse("AA:BB:CC");
        assert!(result.is_err());
    }

    #[test]
    fn test_deterministic() {
        let encoder = MacAddressEncoder::new(MacAddressEncoderParams::default()).unwrap();

        let mac = MacAddress::parse("00:11:22:33:44:55").unwrap();

        let sdr1 = encoder.encode_to_sdr(mac).unwrap();
        let sdr2 = encoder.encode_to_sdr(mac).unwrap();

        assert_eq!(sdr1.get_sparse(), sdr2.get_sparse());
    }
}
