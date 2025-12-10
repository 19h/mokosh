//! Base encoder trait and utilities.

use crate::error::Result;
use crate::types::{Sdr, UInt};

/// Trait for all encoders.
///
/// Encoders convert input values into SDR representations.
pub trait Encoder<T> {
    /// Returns the dimensions of the output SDR.
    fn dimensions(&self) -> &[UInt];

    /// Returns the total size of the output SDR.
    fn size(&self) -> usize;

    /// Encodes a value into an SDR.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to encode
    /// * `output` - The SDR to write the encoding to
    fn encode(&self, value: T, output: &mut Sdr) -> Result<()>;

    /// Encodes a value and returns a new SDR.
    fn encode_to_sdr(&self, value: T) -> Result<Sdr> {
        let dims = self.dimensions().to_vec();
        let mut sdr = Sdr::new(&dims);
        self.encode(value, &mut sdr)?;
        Ok(sdr)
    }
}
