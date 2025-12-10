//! Grid Cell Encoder implementation.
//!
//! The Grid Cell Encoder is inspired by biological grid cells in the entorhinal cortex.
//! It uses multiple modules with different scales and orientations to encode 2D positions.

use crate::encoders::Encoder;
use crate::error::{MokoshError, Result};
use crate::types::{Real, Sdr, UInt};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use std::f32::consts::PI;

/// Parameters for creating a Grid Cell Encoder.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GridCellEncoderParams {
    /// Number of grid cell modules.
    pub num_modules: usize,

    /// Number of cells per module (should be a perfect square).
    pub cells_per_module: UInt,

    /// Base scale (smallest grid period in input units).
    pub base_scale: Real,

    /// Scale factor between successive modules.
    pub scale_ratio: Real,

    /// Orientation offset between modules (in radians).
    pub orientation_offset: Real,
}

impl Default for GridCellEncoderParams {
    fn default() -> Self {
        Self {
            num_modules: 4,
            cells_per_module: 16, // 4x4 grid
            base_scale: 1.0,
            scale_ratio: 1.5,
            orientation_offset: PI / 6.0, // 30 degrees
        }
    }
}

/// Biologically-inspired 2D position encoder using grid cells.
///
/// Each module represents a hexagonal grid at a different scale and orientation.
/// A position activates one cell in each module, creating a unique code.
///
/// # Example
///
/// ```rust
/// use mokosh::encoders::{GridCellEncoder, GridCellEncoderParams, Encoder};
///
/// let encoder = GridCellEncoder::new(GridCellEncoderParams {
///     num_modules: 4,
///     cells_per_module: 16,
///     base_scale: 1.0,
///     ..Default::default()
/// }).unwrap();
///
/// let sdr1 = encoder.encode_to_sdr((5.0, 3.0)).unwrap();
/// let sdr2 = encoder.encode_to_sdr((5.1, 3.0)).unwrap();
///
/// // Close positions have high overlap
/// assert!(sdr1.get_overlap(&sdr2) >= 2);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GridCellEncoder {
    /// Number of modules.
    num_modules: usize,

    /// Cells per module.
    cells_per_module: UInt,

    /// Grid dimension (sqrt of cells_per_module).
    grid_dim: UInt,

    /// Scale for each module.
    scales: Vec<Real>,

    /// Orientation for each module (in radians).
    orientations: Vec<Real>,

    /// Total size of output.
    size: UInt,

    /// Output dimensions.
    dimensions: Vec<UInt>,
}

impl GridCellEncoder {
    /// Creates a new Grid Cell Encoder.
    pub fn new(params: GridCellEncoderParams) -> Result<Self> {
        if params.num_modules == 0 {
            return Err(MokoshError::InvalidParameter {
                name: "num_modules",
                message: "Must be > 0".to_string(),
            });
        }

        if params.cells_per_module == 0 {
            return Err(MokoshError::InvalidParameter {
                name: "cells_per_module",
                message: "Must be > 0".to_string(),
            });
        }

        // Check if cells_per_module is a perfect square
        let grid_dim = (params.cells_per_module as f64).sqrt() as UInt;
        if grid_dim * grid_dim != params.cells_per_module {
            return Err(MokoshError::InvalidParameter {
                name: "cells_per_module",
                message: "Must be a perfect square".to_string(),
            });
        }

        if params.base_scale <= 0.0 {
            return Err(MokoshError::InvalidParameter {
                name: "base_scale",
                message: "Must be > 0".to_string(),
            });
        }

        if params.scale_ratio <= 1.0 {
            return Err(MokoshError::InvalidParameter {
                name: "scale_ratio",
                message: "Must be > 1".to_string(),
            });
        }

        // Compute scales and orientations for each module
        let mut scales = Vec::with_capacity(params.num_modules);
        let mut orientations = Vec::with_capacity(params.num_modules);

        for i in 0..params.num_modules {
            scales.push(params.base_scale * params.scale_ratio.powi(i as i32));
            orientations.push(params.orientation_offset * i as Real);
        }

        let size = params.num_modules as UInt * params.cells_per_module;

        Ok(Self {
            num_modules: params.num_modules,
            cells_per_module: params.cells_per_module,
            grid_dim,
            scales,
            orientations,
            size,
            dimensions: vec![size],
        })
    }

    /// Returns the number of modules.
    pub fn num_modules(&self) -> usize {
        self.num_modules
    }

    /// Returns the number of cells per module.
    pub fn cells_per_module(&self) -> UInt {
        self.cells_per_module
    }

    /// Returns the number of active bits (one per module).
    pub fn active_bits(&self) -> UInt {
        self.num_modules as UInt
    }

    /// Computes which cell is active in a module for a given position.
    fn get_active_cell(&self, x: Real, y: Real, module_idx: usize) -> UInt {
        let scale = self.scales[module_idx];
        let orientation = self.orientations[module_idx];

        // Rotate coordinates by module orientation
        let cos_o = orientation.cos();
        let sin_o = orientation.sin();
        let rx = x * cos_o - y * sin_o;
        let ry = x * sin_o + y * cos_o;

        // Scale coordinates and wrap to grid period
        let grid_x = (rx / scale).rem_euclid(self.grid_dim as Real);
        let grid_y = (ry / scale).rem_euclid(self.grid_dim as Real);

        // Convert to cell indices
        let cell_x = (grid_x.floor() as UInt).min(self.grid_dim - 1);
        let cell_y = (grid_y.floor() as UInt).min(self.grid_dim - 1);

        // Convert 2D cell position to 1D index
        cell_y * self.grid_dim + cell_x
    }
}

impl Encoder<(Real, Real)> for GridCellEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, value: (Real, Real), output: &mut Sdr) -> Result<()> {
        if output.dimensions() != self.dimensions.as_slice() {
            return Err(MokoshError::DimensionMismatch {
                expected: self.dimensions.clone(),
                actual: output.dimensions().to_vec(),
            });
        }

        let (x, y) = value;
        let mut sparse = Vec::with_capacity(self.num_modules);

        for module_idx in 0..self.num_modules {
            let cell_in_module = self.get_active_cell(x, y, module_idx);
            let global_bit = module_idx as UInt * self.cells_per_module + cell_in_module;
            sparse.push(global_bit);
        }

        sparse.sort_unstable();
        output.set_sparse_unchecked(sparse);

        Ok(())
    }
}

impl Encoder<[Real; 2]> for GridCellEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, value: [Real; 2], output: &mut Sdr) -> Result<()> {
        self.encode((value[0], value[1]), output)
    }
}

impl Encoder<Vec<Real>> for GridCellEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.size as usize
    }

    fn encode(&self, value: Vec<Real>, output: &mut Sdr) -> Result<()> {
        if value.len() != 2 {
            return Err(MokoshError::InvalidParameter {
                name: "value",
                message: "GridCellEncoder requires exactly 2 coordinates".to_string(),
            });
        }
        self.encode((value[0], value[1]), output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_encoder() {
        let encoder = GridCellEncoder::new(GridCellEncoderParams {
            num_modules: 5,
            cells_per_module: 25,
            base_scale: 2.0,
            scale_ratio: 1.5,
            orientation_offset: PI / 8.0,
        })
        .unwrap();

        assert_eq!(encoder.num_modules(), 5);
        assert_eq!(encoder.cells_per_module(), 25);
        assert_eq!(Encoder::<(Real, Real)>::size(&encoder), 125);
        assert_eq!(encoder.active_bits(), 5);
    }

    #[test]
    fn test_encode() {
        let encoder = GridCellEncoder::new(GridCellEncoderParams::default()).unwrap();

        let sdr = encoder.encode_to_sdr((5.0, 3.0)).unwrap();

        // Should have one active bit per module
        assert_eq!(sdr.get_sum(), 4);
    }

    #[test]
    fn test_same_position() {
        let encoder = GridCellEncoder::new(GridCellEncoderParams::default()).unwrap();

        let sdr1 = encoder.encode_to_sdr((10.0, 20.0)).unwrap();
        let sdr2 = encoder.encode_to_sdr((10.0, 20.0)).unwrap();

        assert_eq!(sdr1.get_sparse(), sdr2.get_sparse());
    }

    #[test]
    fn test_nearby_overlap() {
        let encoder = GridCellEncoder::new(GridCellEncoderParams {
            num_modules: 6,
            cells_per_module: 16,
            base_scale: 0.1,
            scale_ratio: 2.0,
            orientation_offset: PI / 12.0,
        })
        .unwrap();

        let sdr1 = encoder.encode_to_sdr((0.0, 0.0)).unwrap();
        let sdr2 = encoder.encode_to_sdr((0.05, 0.0)).unwrap(); // Small shift
        let sdr3 = encoder.encode_to_sdr((100.0, 100.0)).unwrap(); // Far away

        let near_overlap = sdr1.get_overlap(&sdr2);
        let far_overlap = sdr1.get_overlap(&sdr3);

        // Nearby positions should share cells in larger-scale modules
        assert!(near_overlap >= far_overlap);
    }

    #[test]
    fn test_array_encoding() {
        let encoder = GridCellEncoder::new(GridCellEncoderParams::default()).unwrap();

        let sdr = encoder.encode_to_sdr([1.0, 2.0]).unwrap();
        assert_eq!(sdr.get_sum(), 4);
    }

    #[test]
    fn test_vec_encoding() {
        let encoder = GridCellEncoder::new(GridCellEncoderParams::default()).unwrap();

        let sdr = encoder.encode_to_sdr(vec![1.0, 2.0]).unwrap();
        assert_eq!(sdr.get_sum(), 4);
    }

    #[test]
    fn test_wrong_vec_length() {
        let encoder = GridCellEncoder::new(GridCellEncoderParams::default()).unwrap();

        let result = encoder.encode_to_sdr(vec![1.0, 2.0, 3.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_non_square_cells_error() {
        let result = GridCellEncoder::new(GridCellEncoderParams {
            cells_per_module: 15, // Not a perfect square
            ..Default::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_params() {
        assert!(GridCellEncoder::new(GridCellEncoderParams {
            num_modules: 0,
            ..Default::default()
        })
        .is_err());

        assert!(GridCellEncoder::new(GridCellEncoderParams {
            cells_per_module: 0,
            ..Default::default()
        })
        .is_err());

        assert!(GridCellEncoder::new(GridCellEncoderParams {
            base_scale: 0.0,
            ..Default::default()
        })
        .is_err());

        assert!(GridCellEncoder::new(GridCellEncoderParams {
            scale_ratio: 1.0,
            ..Default::default()
        })
        .is_err());
    }

    #[test]
    fn test_negative_coordinates() {
        let encoder = GridCellEncoder::new(GridCellEncoderParams::default()).unwrap();

        let sdr = encoder.encode_to_sdr((-5.0, -3.0)).unwrap();
        assert_eq!(sdr.get_sum(), 4);
    }
}
