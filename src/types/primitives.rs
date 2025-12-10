//! Primitive type definitions for HTM algorithms.
//!
//! This module provides type aliases that match the semantics of the original C++ implementation
//! while leveraging Rust's type system for safety.

/// 8-bit signed byte.
pub type Byte = i8;

/// 16-bit signed integer.
pub type Int16 = i16;

/// 16-bit unsigned integer.
pub type UInt16 = u16;

/// 32-bit signed integer.
pub type Int32 = i32;

/// 32-bit unsigned integer.
pub type UInt32 = u32;

/// 64-bit signed integer.
pub type Int64 = i64;

/// 64-bit unsigned integer.
pub type UInt64 = u64;

/// 32-bit floating point number.
pub type Real32 = f32;

/// 64-bit floating point number.
pub type Real64 = f64;

/// Default signed integer type.
/// Uses 32-bit by default, can be changed to 64-bit with feature flag.
#[cfg(not(feature = "big_integer"))]
pub type Int = Int32;

/// Default signed integer type (64-bit variant).
#[cfg(feature = "big_integer")]
pub type Int = Int64;

/// Default unsigned integer type.
/// Uses 32-bit by default, can be changed to 64-bit with feature flag.
#[cfg(not(feature = "big_integer"))]
pub type UInt = UInt32;

/// Default unsigned integer type (64-bit variant).
#[cfg(feature = "big_integer")]
pub type UInt = UInt64;

/// Default floating point type.
/// Uses 32-bit by default, can be changed to 64-bit with feature flag.
#[cfg(not(feature = "double_precision"))]
pub type Real = Real32;

/// Default floating point type (64-bit variant).
#[cfg(feature = "double_precision")]
pub type Real = Real64;

/// Index type for cells in the connections graph.
/// Must match `ElemSparse` for SDR compatibility.
pub type CellIdx = UInt32;

/// Index type for segments within a cell.
pub type SegmentIdx = UInt16;

/// Index type for synapses within a segment.
pub type SynapseIdx = UInt16;

/// Unique identifier for a segment in the connections flat list.
pub type Segment = UInt32;

/// Unique identifier for a synapse in the connections flat list.
pub type Synapse = UInt32;

/// Synapse permanence value (0.0 to 1.0).
pub type Permanence = Real32;

/// Minimum permanence value.
pub const MIN_PERMANENCE: Permanence = 0.0;

/// Maximum permanence value.
pub const MAX_PERMANENCE: Permanence = 1.0;

/// Epsilon for floating point comparisons.
pub const EPSILON: Permanence = 1e-6;

/// Element type for dense SDR representation.
pub type ElemDense = u8;

/// Element type for sparse SDR representation (indices).
pub type ElemSparse = UInt32;

/// Basic type enumeration for runtime type checking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum BasicType {
    /// 8-bit byte.
    Byte = 0,
    /// 16-bit signed integer.
    Int16 = 1,
    /// 16-bit unsigned integer.
    UInt16 = 2,
    /// 32-bit signed integer.
    Int32 = 3,
    /// 32-bit unsigned integer.
    UInt32 = 4,
    /// 64-bit signed integer.
    Int64 = 5,
    /// 64-bit unsigned integer.
    UInt64 = 6,
    /// 32-bit float.
    Real32 = 7,
    /// 64-bit float.
    Real64 = 8,
    /// Boolean.
    Bool = 9,
    /// SDR type.
    Sdr = 10,
    /// String type.
    Str = 11,
}

impl BasicType {
    /// Returns the size in bytes of this type.
    #[must_use]
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::Byte => 1,
            Self::Int16 | Self::UInt16 => 2,
            Self::Int32 | Self::UInt32 | Self::Real32 => 4,
            Self::Int64 | Self::UInt64 | Self::Real64 => 8,
            Self::Bool => 1,
            Self::Sdr | Self::Str => core::mem::size_of::<usize>(), // Pointer size
        }
    }

    /// Returns the name of this type.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Byte => "Byte",
            Self::Int16 => "Int16",
            Self::UInt16 => "UInt16",
            Self::Int32 => "Int32",
            Self::UInt32 => "UInt32",
            Self::Int64 => "Int64",
            Self::UInt64 => "UInt64",
            Self::Real32 => "Real32",
            Self::Real64 => "Real64",
            Self::Bool => "Bool",
            Self::Sdr => "SDR",
            Self::Str => "Str",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_sizes() {
        assert_eq!(core::mem::size_of::<CellIdx>(), 4);
        assert_eq!(core::mem::size_of::<SegmentIdx>(), 2);
        assert_eq!(core::mem::size_of::<SynapseIdx>(), 2);
        assert_eq!(core::mem::size_of::<Segment>(), 4);
        assert_eq!(core::mem::size_of::<Synapse>(), 4);
        assert_eq!(core::mem::size_of::<Permanence>(), 4);
    }

    #[test]
    fn test_permanence_bounds() {
        assert!(MIN_PERMANENCE < MAX_PERMANENCE);
        assert!(EPSILON > 0.0);
        assert!(EPSILON < 0.001);
    }

    #[test]
    fn test_basic_type_sizes() {
        assert_eq!(BasicType::Byte.size_bytes(), 1);
        assert_eq!(BasicType::Int32.size_bytes(), 4);
        assert_eq!(BasicType::Real64.size_bytes(), 8);
    }
}
