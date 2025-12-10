//! Pitch Encoder implementation.
//!
//! Encodes musical pitches/frequencies into SDRs with octave-aware
//! periodic representations.

use crate::encoders::Encoder;
use crate::error::{MokoshError, Result};
use crate::types::{Real, Sdr, UInt};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Standard A4 frequency in Hz.
const A4_FREQUENCY: Real = 440.0;

/// MIDI note number for A4.
const A4_MIDI: Real = 69.0;

/// Parameters for creating a Pitch Encoder.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PitchEncoderParams {
    /// Bits for encoding pitch class (0-11, the note within octave).
    pub pitch_class_bits: UInt,

    /// Active bits for pitch class encoding.
    pub pitch_class_active: UInt,

    /// Bits for encoding octave.
    pub octave_bits: UInt,

    /// Active bits for octave encoding.
    pub octave_active: UInt,

    /// Minimum octave to encode (e.g., 0 for C0 ~16Hz).
    pub min_octave: i32,

    /// Maximum octave to encode (e.g., 8 for C8 ~4186Hz).
    pub max_octave: i32,

    /// Resolution in cents (100 cents = 1 semitone).
    pub cents_resolution: Real,
}

impl Default for PitchEncoderParams {
    fn default() -> Self {
        Self {
            pitch_class_bits: 120,  // 10 bits per semitone
            pitch_class_active: 21,
            octave_bits: 72,        // 8 bits per octave (9 octaves)
            octave_active: 8,
            min_octave: 0,
            max_octave: 8,
            cents_resolution: 10.0, // 10 cent resolution
        }
    }
}

/// A musical pitch representation.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Pitch {
    /// Frequency in Hz.
    pub frequency: Real,
}

impl Pitch {
    /// Creates a pitch from frequency in Hz.
    pub fn from_frequency(frequency: Real) -> Result<Self> {
        if frequency <= 0.0 {
            return Err(MokoshError::InvalidParameter {
                name: "frequency",
                message: "Must be > 0".to_string(),
            });
        }
        Ok(Self { frequency })
    }

    /// Creates a pitch from MIDI note number.
    pub fn from_midi(midi_note: Real) -> Self {
        let frequency = A4_FREQUENCY * (2.0 as Real).powf((midi_note - A4_MIDI) / 12.0);
        Self { frequency }
    }

    /// Creates a pitch from note name (e.g., "A4", "C#5", "Bb3").
    pub fn from_note_name(name: &str) -> Result<Self> {
        let bytes = name.as_bytes();
        if bytes.is_empty() {
            return Err(MokoshError::InvalidParameter {
                name: "note_name",
                message: "Empty note name".to_string(),
            });
        }

        let note_letter = bytes[0].to_ascii_uppercase();
        let base_semitone = match note_letter {
            b'C' => 0,
            b'D' => 2,
            b'E' => 4,
            b'F' => 5,
            b'G' => 7,
            b'A' => 9,
            b'B' => 11,
            _ => {
                return Err(MokoshError::InvalidParameter {
                    name: "note_name",
                    message: format!("Invalid note letter: {}", note_letter as char),
                })
            }
        };

        let mut idx = 1;
        let mut semitone_offset = 0i32;

        // Check for accidentals
        if idx < bytes.len() {
            match bytes[idx] {
                b'#' => {
                    semitone_offset = 1;
                    idx += 1;
                }
                b'b' => {
                    semitone_offset = -1;
                    idx += 1;
                }
                _ => {}
            }
        }

        // Parse octave
        let octave_str = &name[idx..];
        let octave: i32 = octave_str.parse().map_err(|_| MokoshError::InvalidParameter {
            name: "note_name",
            message: format!("Invalid octave: {}", octave_str),
        })?;

        let midi_note = (octave + 1) * 12 + base_semitone + semitone_offset;
        Ok(Self::from_midi(midi_note as Real))
    }

    /// Returns the MIDI note number.
    pub fn to_midi(&self) -> Real {
        A4_MIDI + 12.0 * (self.frequency / A4_FREQUENCY).log2()
    }

    /// Returns the pitch class (0-11, where 0=C, 1=C#, etc.).
    pub fn pitch_class(&self) -> Real {
        let midi = self.to_midi();
        midi.rem_euclid(12.0)
    }

    /// Returns the octave number.
    pub fn octave(&self) -> i32 {
        let midi = self.to_midi();
        ((midi / 12.0).floor() as i32) - 1
    }
}

/// Encodes musical pitches into SDR representations.
///
/// Uses separate encodings for pitch class (periodic within octave)
/// and octave number, allowing the system to learn both absolute
/// pitch and relative pitch relationships.
///
/// # Example
///
/// ```rust
/// use mokosh::encoders::{PitchEncoder, PitchEncoderParams, Pitch, Encoder};
///
/// let encoder = PitchEncoder::new(PitchEncoderParams::default()).unwrap();
///
/// let a4 = Pitch::from_frequency(440.0).unwrap();
/// let a5 = Pitch::from_frequency(880.0).unwrap(); // One octave up
/// let e4 = Pitch::from_frequency(329.63).unwrap(); // Perfect fifth down
///
/// let sdr_a4 = encoder.encode_to_sdr(a4).unwrap();
/// let sdr_a5 = encoder.encode_to_sdr(a5).unwrap();
/// let sdr_e4 = encoder.encode_to_sdr(e4).unwrap();
///
/// // Same pitch class (A) should have more overlap than different pitch classes
/// let same_class_overlap = sdr_a4.get_overlap(&sdr_a5);
/// let diff_class_overlap = sdr_a4.get_overlap(&sdr_e4);
/// // Both have some overlap due to shared octave or pitch class
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PitchEncoder {
    pitch_class_bits: UInt,
    pitch_class_active: UInt,
    octave_bits: UInt,
    octave_active: UInt,
    min_octave: i32,
    max_octave: i32,
    num_octaves: UInt,
    size: UInt,
    dimensions: Vec<UInt>,
}

impl PitchEncoder {
    /// Creates a new Pitch Encoder.
    pub fn new(params: PitchEncoderParams) -> Result<Self> {
        if params.pitch_class_active > params.pitch_class_bits {
            return Err(MokoshError::InvalidParameter {
                name: "pitch_class_active",
                message: "Cannot exceed pitch_class_bits".to_string(),
            });
        }

        if params.octave_active > params.octave_bits {
            return Err(MokoshError::InvalidParameter {
                name: "octave_active",
                message: "Cannot exceed octave_bits".to_string(),
            });
        }

        if params.max_octave <= params.min_octave {
            return Err(MokoshError::InvalidParameter {
                name: "max_octave",
                message: "Must be greater than min_octave".to_string(),
            });
        }

        let num_octaves = (params.max_octave - params.min_octave + 1) as UInt;
        let size = params.pitch_class_bits + params.octave_bits;

        Ok(Self {
            pitch_class_bits: params.pitch_class_bits,
            pitch_class_active: params.pitch_class_active,
            octave_bits: params.octave_bits,
            octave_active: params.octave_active,
            min_octave: params.min_octave,
            max_octave: params.max_octave,
            num_octaves,
            size,
            dimensions: vec![size],
        })
    }

    /// Returns the size of the pitch class encoding.
    pub fn pitch_class_size(&self) -> UInt {
        self.pitch_class_bits
    }

    /// Returns the size of the octave encoding.
    pub fn octave_size(&self) -> UInt {
        self.octave_bits
    }
}

impl Encoder<Pitch> for PitchEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, pitch: Pitch, output: &mut Sdr) -> Result<()> {
        if output.dimensions() != self.dimensions.as_slice() {
            return Err(MokoshError::DimensionMismatch {
                expected: self.dimensions.clone(),
                actual: output.dimensions().to_vec(),
            });
        }

        let mut sparse = Vec::new();

        // Encode pitch class (periodic, wraps around)
        let pitch_class = pitch.pitch_class(); // 0.0 to 12.0
        let pc_normalized = pitch_class / 12.0;

        let pc_positions = self.pitch_class_bits - self.pitch_class_active + 1;
        let pc_start = (pc_normalized * pc_positions as Real).round() as UInt % pc_positions;

        for i in 0..self.pitch_class_active {
            let bit = (pc_start + i) % self.pitch_class_bits;
            sparse.push(bit);
        }

        // Encode octave (linear)
        let octave = pitch.octave().clamp(self.min_octave, self.max_octave);
        let octave_normalized =
            (octave - self.min_octave) as Real / (self.num_octaves - 1) as Real;

        let oct_positions = self.octave_bits - self.octave_active + 1;
        let oct_start = (octave_normalized * (oct_positions - 1) as Real).round() as UInt;
        let oct_offset = self.pitch_class_bits;

        for i in 0..self.octave_active {
            sparse.push(oct_offset + oct_start + i);
        }

        sparse.sort_unstable();
        sparse.dedup();
        output.set_sparse_unchecked(sparse);

        Ok(())
    }
}

impl Encoder<Real> for PitchEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, frequency: Real, output: &mut Sdr) -> Result<()> {
        let pitch = Pitch::from_frequency(frequency)?;
        self.encode(pitch, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_encoder() {
        let encoder = PitchEncoder::new(PitchEncoderParams::default()).unwrap();

        assert_eq!(encoder.pitch_class_size(), 120);
        assert_eq!(encoder.octave_size(), 72);
        assert_eq!(Encoder::<Pitch>::size(&encoder), 192);
    }

    #[test]
    fn test_pitch_from_frequency() {
        let a4 = Pitch::from_frequency(440.0).unwrap();
        assert!((a4.to_midi() - 69.0).abs() < 0.01);
        assert!((a4.pitch_class() - 9.0).abs() < 0.01); // A is pitch class 9
        assert_eq!(a4.octave(), 4);
    }

    #[test]
    fn test_pitch_from_midi() {
        let c4 = Pitch::from_midi(60.0);
        assert!((c4.frequency - 261.63).abs() < 1.0);
        assert!((c4.pitch_class()).abs() < 0.01); // C is pitch class 0
        assert_eq!(c4.octave(), 4);
    }

    #[test]
    fn test_pitch_from_note_name() {
        let a4 = Pitch::from_note_name("A4").unwrap();
        assert!((a4.frequency - 440.0).abs() < 1.0);

        let cs5 = Pitch::from_note_name("C#5").unwrap();
        assert!((cs5.to_midi() - 73.0).abs() < 0.01);

        let bb3 = Pitch::from_note_name("Bb3").unwrap();
        assert!((bb3.to_midi() - 58.0).abs() < 0.01);
    }

    #[test]
    fn test_encode_pitch() {
        let encoder = PitchEncoder::new(PitchEncoderParams::default()).unwrap();

        let a4 = Pitch::from_frequency(440.0).unwrap();
        let sdr = encoder.encode_to_sdr(a4).unwrap();

        // Should have pitch_class_active + octave_active bits
        let expected = 21 + 8;
        assert!(sdr.get_sum() >= expected - 2 && sdr.get_sum() <= expected);
    }

    #[test]
    fn test_octave_equivalence() {
        let encoder = PitchEncoder::new(PitchEncoderParams::default()).unwrap();

        let a4 = Pitch::from_frequency(440.0).unwrap();
        let a5 = Pitch::from_frequency(880.0).unwrap();
        let e4 = Pitch::from_frequency(329.63).unwrap();

        let sdr_a4 = encoder.encode_to_sdr(a4).unwrap();
        let sdr_a5 = encoder.encode_to_sdr(a5).unwrap();
        let sdr_e4 = encoder.encode_to_sdr(e4).unwrap();

        // A4 and A5 share pitch class, should have overlap in pitch class region
        let a4_a5_overlap = sdr_a4.get_overlap(&sdr_a5);

        // A4 and E4 share octave, should have overlap in octave region
        let a4_e4_overlap = sdr_a4.get_overlap(&sdr_e4);

        // Both should have some meaningful overlap
        assert!(a4_a5_overlap > 0);
        assert!(a4_e4_overlap > 0);
    }

    #[test]
    fn test_encode_frequency_directly() {
        let encoder = PitchEncoder::new(PitchEncoderParams::default()).unwrap();

        let sdr = encoder.encode_to_sdr(440.0 as Real).unwrap();
        assert!(sdr.get_sum() > 0);
    }

    #[test]
    fn test_invalid_frequency() {
        let result = Pitch::from_frequency(-100.0);
        assert!(result.is_err());

        let result = Pitch::from_frequency(0.0);
        assert!(result.is_err());
    }
}
