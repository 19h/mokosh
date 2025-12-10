//! Serialization support for mokosh types.
//!
//! This module provides serialization and deserialization capabilities for HTM
//! algorithms and data structures. When the `serde` feature is enabled, all
//! major types implement `Serialize` and `Deserialize`.
//!
//! # Supported Formats
//!
//! - **Binary** - Fast binary serialization using bincode (default)
//! - **JSON** - Human-readable JSON format using serde_json
//!
//! # Example
//!
//! ```rust,ignore
//! use mokosh::algorithms::SpatialPooler;
//! use mokosh::serialization::{Serializable, SerializableFormat};
//!
//! let sp = SpatialPooler::new(/* ... */);
//!
//! // Save to binary file
//! sp.save_to_file("model.bin", SerializableFormat::Binary)?;
//!
//! // Load from file
//! let sp2 = SpatialPooler::load_from_file("model.bin", SerializableFormat::Binary)?;
//!
//! // Serialize to bytes
//! let bytes = sp.to_bytes(SerializableFormat::Binary)?;
//!
//! // Serialize to JSON string
//! let json = sp.to_json()?;
//! ```

use crate::error::{MokoshError, Result};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

#[cfg(feature = "serde")]
use serde::{de::DeserializeOwned, Serialize};

/// Serialization format options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SerializableFormat {
    /// Fast binary serialization (default).
    /// Platform-specific, most efficient for storage and speed.
    Binary,

    /// Human-readable JSON format.
    /// Useful for debugging, configuration, and interoperability.
    Json,
}

impl Default for SerializableFormat {
    fn default() -> Self {
        SerializableFormat::Binary
    }
}

impl std::fmt::Display for SerializableFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SerializableFormat::Binary => write!(f, "BINARY"),
            SerializableFormat::Json => write!(f, "JSON"),
        }
    }
}

impl std::str::FromStr for SerializableFormat {
    type Err = MokoshError;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_uppercase().as_str() {
            "BINARY" | "BIN" => Ok(SerializableFormat::Binary),
            "JSON" => Ok(SerializableFormat::Json),
            _ => Err(MokoshError::InvalidParameter {
                name: "format",
                message: format!("Unknown format '{}'. Expected: BINARY, JSON", s),
            }),
        }
    }
}

/// Trait for types that can be serialized and deserialized.
///
/// This trait provides a unified interface for saving and loading HTM
/// components to/from various formats and destinations.
#[cfg(feature = "serde")]
pub trait Serializable: Serialize + DeserializeOwned + Sized {
    /// Serializes to a byte vector.
    fn to_bytes(&self, format: SerializableFormat) -> Result<Vec<u8>> {
        match format {
            SerializableFormat::Binary => {
                bincode::serialize(self).map_err(|e| MokoshError::SerializationError {
                    message: format!("Binary serialization failed: {}", e),
                })
            }
            SerializableFormat::Json => {
                serde_json::to_vec_pretty(self).map_err(|e| MokoshError::SerializationError {
                    message: format!("JSON serialization failed: {}", e),
                })
            }
        }
    }

    /// Deserializes from a byte slice.
    fn from_bytes(bytes: &[u8], format: SerializableFormat) -> Result<Self> {
        match format {
            SerializableFormat::Binary => {
                bincode::deserialize(bytes).map_err(|e| MokoshError::SerializationError {
                    message: format!("Binary deserialization failed: {}", e),
                })
            }
            SerializableFormat::Json => {
                serde_json::from_slice(bytes).map_err(|e| MokoshError::SerializationError {
                    message: format!("JSON deserialization failed: {}", e),
                })
            }
        }
    }

    /// Serializes to a JSON string.
    fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| MokoshError::SerializationError {
            message: format!("JSON serialization failed: {}", e),
        })
    }

    /// Deserializes from a JSON string.
    fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|e| MokoshError::SerializationError {
            message: format!("JSON deserialization failed: {}", e),
        })
    }

    /// Serializes to a writer.
    fn save<W: Write>(&self, writer: W, format: SerializableFormat) -> Result<()> {
        let mut writer = BufWriter::new(writer);
        match format {
            SerializableFormat::Binary => {
                bincode::serialize_into(&mut writer, self).map_err(|e| {
                    MokoshError::SerializationError {
                        message: format!("Binary serialization failed: {}", e),
                    }
                })
            }
            SerializableFormat::Json => {
                serde_json::to_writer_pretty(&mut writer, self).map_err(|e| {
                    MokoshError::SerializationError {
                        message: format!("JSON serialization failed: {}", e),
                    }
                })
            }
        }
    }

    /// Deserializes from a reader.
    fn load<R: Read>(reader: R, format: SerializableFormat) -> Result<Self> {
        let mut reader = BufReader::new(reader);
        match format {
            SerializableFormat::Binary => {
                bincode::deserialize_from(&mut reader).map_err(|e| {
                    MokoshError::SerializationError {
                        message: format!("Binary deserialization failed: {}", e),
                    }
                })
            }
            SerializableFormat::Json => {
                serde_json::from_reader(&mut reader).map_err(|e| {
                    MokoshError::SerializationError {
                        message: format!("JSON deserialization failed: {}", e),
                    }
                })
            }
        }
    }

    /// Saves to a file.
    fn save_to_file<P: AsRef<Path>>(&self, path: P, format: SerializableFormat) -> Result<()> {
        let file = File::create(path.as_ref()).map_err(|e| MokoshError::IoError {
            message: format!("Failed to create file: {}", e),
        })?;
        self.save(file, format)
    }

    /// Loads from a file.
    fn load_from_file<P: AsRef<Path>>(path: P, format: SerializableFormat) -> Result<Self> {
        let file = File::open(path.as_ref()).map_err(|e| MokoshError::IoError {
            message: format!("Failed to open file: {}", e),
        })?;
        Self::load(file, format)
    }

    /// Saves to a file, inferring format from the file extension.
    ///
    /// - `.json` -> JSON format
    /// - All other extensions -> Binary format
    fn save_to_file_auto<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let format = infer_format_from_path(path.as_ref());
        self.save_to_file(path, format)
    }

    /// Loads from a file, inferring format from the file extension.
    ///
    /// - `.json` -> JSON format
    /// - All other extensions -> Binary format
    fn load_from_file_auto<P: AsRef<Path>>(path: P) -> Result<Self> {
        let format = infer_format_from_path(path.as_ref());
        Self::load_from_file(path, format)
    }
}

/// Blanket implementation for all types that implement Serialize + DeserializeOwned.
#[cfg(feature = "serde")]
impl<T> Serializable for T where T: Serialize + DeserializeOwned + Sized {}

/// Infers serialization format from file extension.
fn infer_format_from_path(path: &Path) -> SerializableFormat {
    match path.extension().and_then(|e| e.to_str()) {
        Some("json") => SerializableFormat::Json,
        _ => SerializableFormat::Binary,
    }
}

#[cfg(all(test, feature = "serde"))]
mod tests {
    use super::*;
    use crate::types::Sdr;

    #[test]
    fn test_format_parsing() {
        assert_eq!(
            "BINARY".parse::<SerializableFormat>().unwrap(),
            SerializableFormat::Binary
        );
        assert_eq!(
            "json".parse::<SerializableFormat>().unwrap(),
            SerializableFormat::Json
        );
        assert_eq!(
            "JSON".parse::<SerializableFormat>().unwrap(),
            SerializableFormat::Json
        );
        assert!("unknown".parse::<SerializableFormat>().is_err());
    }

    #[test]
    fn test_format_display() {
        assert_eq!(SerializableFormat::Binary.to_string(), "BINARY");
        assert_eq!(SerializableFormat::Json.to_string(), "JSON");
    }

    #[test]
    fn test_sdr_binary_serialization() {
        let sdr = Sdr::new(&[100]);
        let sparse = vec![1, 5, 10, 50, 99];
        let mut sdr = sdr;
        sdr.set_sparse(&sparse).unwrap();

        // Serialize to bytes
        let bytes = sdr.to_bytes(SerializableFormat::Binary).unwrap();
        assert!(!bytes.is_empty());

        // Deserialize
        let restored: Sdr = Sdr::from_bytes(&bytes, SerializableFormat::Binary).unwrap();
        assert_eq!(restored.get_sparse(), sparse);
    }

    #[test]
    fn test_sdr_json_serialization() {
        let sdr = Sdr::new(&[100]);
        let sparse = vec![1, 5, 10, 50, 99];
        let mut sdr = sdr;
        sdr.set_sparse(&sparse).unwrap();

        // Serialize to JSON
        let json = sdr.to_json().unwrap();
        assert!(json.contains("dimensions"));
        assert!(json.contains("sparse"));

        // Deserialize
        let restored: Sdr = Sdr::from_json(&json).unwrap();
        assert_eq!(restored.get_sparse(), sparse);
    }

    #[test]
    fn test_infer_format() {
        assert_eq!(
            infer_format_from_path(Path::new("model.json")),
            SerializableFormat::Json
        );
        assert_eq!(
            infer_format_from_path(Path::new("model.bin")),
            SerializableFormat::Binary
        );
        assert_eq!(
            infer_format_from_path(Path::new("model")),
            SerializableFormat::Binary
        );
        assert_eq!(
            infer_format_from_path(Path::new("path/to/model.JSON")),
            SerializableFormat::Binary // case-sensitive
        );
    }
}
