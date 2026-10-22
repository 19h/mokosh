//! Encoders for converting data into SDR representations.
//!
//! Encoders transform various data types (scalars, dates, categories, text, coordinates)
//! into Sparse Distributed Representations suitable for HTM processing.
//!
//! # Available Encoders
//!
//! ## Scalar Encoders
//! - [`ScalarEncoder`]: Encodes numeric values as contiguous blocks of active bits
//! - [`RandomDistributedScalarEncoder`]: Encodes numeric values using hash-based random bit placement
//! - [`LogEncoder`]: Encodes positive values on a logarithmic scale
//! - [`DeltaEncoder`]: Encodes rate of change between consecutive values
//!
//! ## Categorical Encoders
//! - [`CategoryEncoder`]: Encodes discrete categories into non-overlapping patterns
//! - [`BooleanEncoder`]: Encodes true/false values
//! - [`HierarchicalCategoryEncoder`]: Encodes categories in a taxonomy with parent-child sharing
//! - [`OrdinalEncoder`]: Encodes ordered categories with adjacency overlap
//! - [`SetEncoder`]: Encodes variable-size sets of items
//!
//! ## Temporal Encoders
//! - [`DateEncoder`]: Encodes date/time values with multiple configurable attributes
//!
//! ## Spatial Encoders
//! - [`CoordinateEncoder`]: Encodes N-dimensional coordinates with locality sensitivity
//! - [`GeospatialEncoder`]: Encodes GPS coordinates (latitude/longitude)
//! - [`GridCellEncoder`]: Biologically-inspired 2D position encoder
//!
//! ## Text Encoders
//! - [`SimHashDocumentEncoder`]: Encodes documents/text with similarity preservation
//! - [`WordEmbeddingEncoder`]: Converts dense word vectors (word2vec/GloVe) to SDRs
//! - [`LlmEmbeddingEncoder`]: Converts LLM embeddings of arbitrary dimensions (384-3072+) to SDRs
//! - [`CharacterEncoder`]: Character-level encoding with optional semantic similarity
//! - [`NGramEncoder`]: Encodes character or word n-grams
//!
//! ## Audio Encoders
//! - [`SpectrogramEncoder`]: Encodes audio frequency spectra (FFT/mel)
//! - [`WaveformEncoder`]: Encodes raw audio samples
//! - [`PitchEncoder`]: Encodes musical pitch with octave and pitch class
//!
//! ## Vision Encoders
//! - [`PatchEncoder`]: Encodes image patches using random projections
//! - [`ColorEncoder`]: Encodes colors in HSV space
//! - [`EdgeOrientationEncoder`]: Encodes oriented edge features
//!
//! ## Network Encoders
//! - [`IpAddressEncoder`]: Encodes IPv4/IPv6 addresses with subnet awareness
//! - [`MacAddressEncoder`]: Encodes MAC addresses with OUI (vendor) awareness
//! - [`GraphNodeEncoder`]: Encodes graph node embeddings
//!
//! ## Biometric Encoders
//! - [`HrvEncoder`]: Encodes heart rate variability metrics
//! - [`EcgEncoder`]: Encodes ECG waveforms
//! - [`AccelerometerEncoder`]: Encodes 3-axis accelerometer data
//!
//! ## Financial Encoders
//! - [`PriceEncoder`]: Encodes financial prices with log-scale support
//! - [`CurrencyPairEncoder`]: Encodes forex currency pairs
//! - [`OrderBookEncoder`]: Encodes order book depth snapshots
//!
//! ## Probabilistic Encoders
//! - [`DistributionEncoder`]: Encodes probability distributions
//! - [`ConfidenceIntervalEncoder`]: Encodes values with uncertainty ranges
//!
//! ## Composite Encoders
//! - [`MultiEncoder`]: Combines multiple encoders for the same input type
//! - [`VecMultiEncoder`]: Combines multiple encoders for different input values
//! - [`PassThroughEncoder`]: Passes pre-encoded sparse indices through
//!
//! # Example
//!
//! ```rust
//! use mokosh::encoders::{ScalarEncoder, ScalarEncoderParams, Encoder};
//!
//! let encoder = ScalarEncoder::new(ScalarEncoderParams {
//!     minimum: 0.0,
//!     maximum: 100.0,
//!     size: 100,
//!     active_bits: 10,
//!     ..Default::default()
//! }).unwrap();
//!
//! let sdr = encoder.encode_to_sdr(50.0).unwrap();
//! assert_eq!(sdr.get_sum(), 10);
//! ```

mod base;
mod boolean;
mod category;
mod coordinate;
pub mod date;
mod delta;
mod geospatial;
mod grid;
mod log;
mod multi;
mod passthrough;
mod rdse;
mod scalar;
mod simhash;

// Audio encoders
mod pitch;
mod spectrogram;
mod waveform;

// Text/NLP encoders
mod character;
mod llm_embedding;
mod ngram;
mod word_embedding;

// Vision encoders
mod color;
mod edge_orientation;
mod patch;

// Network encoders
mod graph_node;
mod ip_address;
mod mac_address;

// Biometric encoders
mod accelerometer;
mod ecg;
mod hrv;

// Categorical encoders
mod hierarchical_category;
mod ordinal;
mod set;

// Financial encoders
mod currency_pair;
mod order_book;
mod price;

// Probabilistic encoders
mod confidence_interval;
mod distribution;

pub use base::Encoder;
pub use boolean::{BooleanEncoder, BooleanEncoderParams};
pub use category::{CategoryEncoder, CategoryEncoderParams};
pub use coordinate::{Coord2D, Coord3D, CoordinateEncoder, CoordinateEncoderParams};
pub use date::{DateEncoder, DateEncoderParams, DateTime, DayOfWeek, Holiday};
pub use delta::{DeltaEncoder, DeltaEncoderParams};
pub use geospatial::{GeospatialEncoder, GeospatialEncoderParams, GpsCoordinate};
pub use grid::{GridCellEncoder, GridCellEncoderParams};
pub use log::{LogEncoder, LogEncoderParams};
pub use multi::{
    EncoderField, MultiEncoder, MultiEncoderBuilder, VecMultiEncoder, VecMultiEncoderBuilder,
};
pub use passthrough::{PassThroughEncoder, PassThroughEncoderParams};
pub use rdse::{RandomDistributedScalarEncoder, Rdse, RdseParams};
pub use scalar::{ScalarEncoder, ScalarEncoderParams};
pub use simhash::{SimHashDocumentEncoder, SimHashDocumentEncoderParams};

// Audio encoders
pub use pitch::{Pitch, PitchEncoder, PitchEncoderParams};
pub use spectrogram::{SpectrogramEncoder, SpectrogramEncoderParams};
pub use waveform::{WaveformEncoder, WaveformEncoderParams};

// Text/NLP encoders
pub use character::{CharacterEncoder, CharacterEncoderParams};
pub use llm_embedding::{
    DimensionStrategy, LlmEmbeddingEncoder, LlmEmbeddingEncoderParams, NormalizationStrategy,
};
pub use ngram::{NGramEncoder, NGramEncoderParams};
pub use word_embedding::{WordEmbeddingEncoder, WordEmbeddingEncoderParams};

// Vision encoders
pub use color::{ColorEncoder, ColorEncoderParams, HsvColor, RgbColor};
pub use edge_orientation::{EdgeOrientationEncoder, EdgeOrientationEncoderParams, OrientedEdge};
pub use patch::{ImagePatch, PatchEncoder, PatchEncoderParams};

// Network encoders
pub use graph_node::{GraphNode, GraphNodeEncoder, GraphNodeEncoderParams};
pub use ip_address::{IpAddressEncoder, IpAddressEncoderParams};
pub use mac_address::{MacAddress, MacAddressEncoder, MacAddressEncoderParams};

// Biometric encoders
pub use accelerometer::{AccelerometerEncoder, AccelerometerEncoderParams, AccelerometerReading};
pub use ecg::{EcgEncoder, EcgEncoderParams, EcgStats};
pub use hrv::{HrvEncoder, HrvEncoderParams, HrvMetrics};

// Categorical encoders
pub use hierarchical_category::{HierarchicalCategoryEncoder, HierarchicalCategoryEncoderParams};
pub use ordinal::{OrdinalEncoder, OrdinalEncoderParams};
pub use set::{SetEncoder, SetEncoderParams};

// Financial encoders
pub use currency_pair::{CurrencyPair, CurrencyPairEncoder, CurrencyPairEncoderParams};
pub use order_book::{OrderBook, OrderBookEncoder, OrderBookEncoderParams, PriceLevel};
pub use price::{PriceData, PriceEncoder, PriceEncoderParams};

// Probabilistic encoders
pub use confidence_interval::{
    ConfidenceInterval, ConfidenceIntervalEncoder, ConfidenceIntervalEncoderParams,
};
pub use distribution::{Distribution, DistributionEncoder, DistributionEncoderParams};

mod scalar64;
pub use scalar64::{ScalarEncoder64, ScalarEncoderParams64};
