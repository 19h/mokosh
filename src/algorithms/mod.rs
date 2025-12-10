//! HTM algorithms implementation.
//!
//! This module contains the core algorithms that implement Hierarchical Temporal Memory:
//!
//! - **Connections**: The synaptic connectivity graph
//! - **Spatial Pooler**: Creates sparse representations from input patterns
//! - **Temporal Memory**: Learns temporal sequences
//! - **Anomaly Detection**: Identifies unusual patterns
//! - **SDR Classifier**: Maps SDR patterns to output classes

mod connections;
mod spatial_pooler;
mod temporal_memory;
mod anomaly;
mod sdr_classifier;

pub use connections::{Connections, ConnectionsParams, SegmentData, SynapseData};
pub use spatial_pooler::{SpatialPooler, SpatialPoolerParams};
pub use temporal_memory::{TemporalMemory, TemporalMemoryParams, AnomalyMode};
pub use anomaly::{Anomaly, AnomalyLikelihood};
pub use sdr_classifier::{SdrClassifier, SdrClassifierParams};
