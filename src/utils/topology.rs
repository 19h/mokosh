//! Topology utilities for spatial computations in HTM.
//!
//! This module provides functions for computing neighborhoods and relationships
//! between cells/columns in multi-dimensional topological spaces.

use crate::types::UInt;
use std::collections::HashMap;

/// Specifies how boundaries are handled in topological computations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WrappingMode {
    /// No wrapping - boundaries are hard limits.
    #[default]
    NoWrap,
    /// Wrap around - space is toroidal.
    Wrap,
}

/// Utilities for computing topological relationships.
pub struct Topology;

impl Topology {
    /// Converts a flat index to multi-dimensional coordinates.
    ///
    /// # Arguments
    ///
    /// * `index` - The flat index
    /// * `dimensions` - The shape of the space
    ///
    /// # Example
    ///
    /// ```rust
    /// use mokosh::utils::Topology;
    ///
    /// let coords = Topology::index_to_coordinates(5, &[3, 3]);
    /// assert_eq!(coords, vec![1, 2]);
    /// ```
    #[must_use]
    pub fn index_to_coordinates(index: usize, dimensions: &[UInt]) -> Vec<UInt> {
        let mut coords = vec![0; dimensions.len()];
        let mut idx = index;

        for i in (0..dimensions.len()).rev() {
            let dim_size = dimensions[i] as usize;
            coords[i] = (idx % dim_size) as UInt;
            idx /= dim_size;
        }

        coords
    }

    /// Converts multi-dimensional coordinates to a flat index.
    ///
    /// # Arguments
    ///
    /// * `coordinates` - The coordinates in each dimension
    /// * `dimensions` - The shape of the space
    #[must_use]
    pub fn coordinates_to_index(coordinates: &[UInt], dimensions: &[UInt]) -> usize {
        let mut index = 0;
        let mut multiplier = 1;

        for i in (0..dimensions.len()).rev() {
            index += coordinates[i] as usize * multiplier;
            multiplier *= dimensions[i] as usize;
        }

        index
    }

    /// Computes the neighborhood of a cell/column within a given radius.
    ///
    /// Returns all indices within the hypercube of the given radius centered
    /// on the specified center point.
    ///
    /// # Arguments
    ///
    /// * `center_index` - The flat index of the center point
    /// * `dimensions` - The shape of the space
    /// * `radius` - The radius of the neighborhood
    /// * `wrap` - Whether to wrap around boundaries
    /// * `include_center` - Whether to include the center point in results
    #[must_use]
    pub fn neighborhood(
        center_index: usize,
        dimensions: &[UInt],
        radius: UInt,
        wrap: WrappingMode,
        include_center: bool,
    ) -> Vec<usize> {
        let center_coords = Self::index_to_coordinates(center_index, dimensions);
        let mut neighbors = Vec::new();

        // Generate all points in the hypercube
        Self::neighborhood_recursive(
            &center_coords,
            dimensions,
            radius as i32,
            wrap,
            0,
            &mut vec![0; dimensions.len()],
            &mut neighbors,
        );

        // Remove center if not requested
        if !include_center {
            neighbors.retain(|&idx| idx != center_index);
        }

        neighbors
    }

    fn neighborhood_recursive(
        center: &[UInt],
        dimensions: &[UInt],
        radius: i32,
        wrap: WrappingMode,
        dim: usize,
        current: &mut Vec<UInt>,
        result: &mut Vec<usize>,
    ) {
        if dim == dimensions.len() {
            let idx = Self::coordinates_to_index(current, dimensions);
            result.push(idx);
            return;
        }

        let center_coord = center[dim] as i32;
        let dim_size = dimensions[dim] as i32;

        for offset in -radius..=radius {
            let coord = center_coord + offset;

            let valid_coord = match wrap {
                WrappingMode::NoWrap => {
                    if coord < 0 || coord >= dim_size {
                        continue;
                    }
                    coord as UInt
                }
                WrappingMode::Wrap => {
                    ((coord % dim_size) + dim_size) as UInt % dimensions[dim]
                }
            };

            current[dim] = valid_coord;
            Self::neighborhood_recursive(center, dimensions, radius, wrap, dim + 1, current, result);
        }
    }

    /// Maps a column index to an input index based on topology.
    ///
    /// This distributes columns uniformly over the input space.
    ///
    /// # Arguments
    ///
    /// * `column_index` - The column index
    /// * `column_dimensions` - The shape of the column space
    /// * `input_dimensions` - The shape of the input space
    #[must_use]
    pub fn map_column_to_input(
        column_index: usize,
        column_dimensions: &[UInt],
        input_dimensions: &[UInt],
    ) -> usize {
        let column_coords = Self::index_to_coordinates(column_index, column_dimensions);

        let mut input_coords = Vec::with_capacity(input_dimensions.len());

        for dim in 0..input_dimensions.len() {
            let col_coord = if dim < column_coords.len() {
                column_coords[dim] as f64
            } else {
                0.0
            };

            let col_dim = if dim < column_dimensions.len() {
                column_dimensions[dim] as f64
            } else {
                1.0
            };

            let input_dim = input_dimensions[dim] as f64;

            // Map column coordinate to input coordinate proportionally
            let input_coord = ((col_coord + 0.5) * input_dim / col_dim) as UInt;
            input_coords.push(input_coord.min(input_dimensions[dim] - 1));
        }

        Self::coordinates_to_index(&input_coords, input_dimensions)
    }

    /// Computes the potential pool for a column.
    ///
    /// Returns the input indices that a column could potentially connect to.
    ///
    /// # Arguments
    ///
    /// * `column_index` - The column index
    /// * `column_dimensions` - The shape of the column space
    /// * `input_dimensions` - The shape of the input space
    /// * `potential_radius` - The radius around the mapped input
    /// * `wrap` - Whether to wrap around boundaries
    #[must_use]
    pub fn map_potential_pool(
        column_index: usize,
        column_dimensions: &[UInt],
        input_dimensions: &[UInt],
        potential_radius: UInt,
        wrap: WrappingMode,
    ) -> Vec<usize> {
        let center = Self::map_column_to_input(column_index, column_dimensions, input_dimensions);

        Self::neighborhood(
            center,
            input_dimensions,
            potential_radius,
            wrap,
            true,
        )
    }

    /// Computes the total number of elements in a dimensional space.
    #[must_use]
    pub fn num_elements(dimensions: &[UInt]) -> usize {
        dimensions.iter().map(|&d| d as usize).product()
    }

    /// Computes the average receptive field radius.
    ///
    /// Given the number of connected synapses per column, estimates the
    /// average radius of the receptive field.
    #[must_use]
    pub fn average_receptive_field_radius(
        num_connected: &[UInt],
        input_dimensions: &[UInt],
    ) -> f64 {
        if num_connected.is_empty() {
            return 0.0;
        }

        let num_dims = input_dimensions.len();
        let avg_connected: f64 = num_connected.iter().map(|&n| n as f64).sum::<f64>()
            / num_connected.len() as f64;

        // Estimate radius from average connections
        // For a hypercube: connections ≈ (2*r+1)^num_dims
        // So r ≈ (connections^(1/num_dims) - 1) / 2
        let approx_radius = (avg_connected.powf(1.0 / num_dims as f64) - 1.0) / 2.0;
        approx_radius.max(0.0)
    }
}

/// Represents a cached neighborhood map for efficient lookups.
#[derive(Debug, Clone)]
pub struct Neighborhood {
    /// Pre-computed neighbors for each cell.
    neighbors: HashMap<usize, Vec<usize>>,
}

impl Neighborhood {
    /// Creates a new neighborhood cache.
    pub fn new() -> Self {
        Self {
            neighbors: HashMap::new(),
        }
    }

    /// Creates and populates a neighborhood cache for all cells.
    #[must_use]
    pub fn compute_all(
        dimensions: &[UInt],
        radius: UInt,
        wrap: WrappingMode,
        skip_center: bool,
    ) -> Self {
        let num_cells = Topology::num_elements(dimensions);
        let mut neighbors = HashMap::with_capacity(num_cells);

        for i in 0..num_cells {
            let cell_neighbors = Topology::neighborhood(
                i,
                dimensions,
                radius,
                wrap,
                !skip_center,
            );
            neighbors.insert(i, cell_neighbors);
        }

        Self { neighbors }
    }

    /// Updates the neighborhood cache with new parameters.
    pub fn update_all(
        &mut self,
        dimensions: &[UInt],
        radius: UInt,
        wrap: WrappingMode,
        skip_center: bool,
    ) {
        let updated = Self::compute_all(dimensions, radius, wrap, skip_center);
        self.neighbors = updated.neighbors;
    }

    /// Gets the neighbors for a cell.
    pub fn get(&self, cell: usize) -> Option<&Vec<usize>> {
        self.neighbors.get(&cell)
    }

    /// Returns the number of cells with cached neighbors.
    pub fn len(&self) -> usize {
        self.neighbors.len()
    }

    /// Returns whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.neighbors.is_empty()
    }

    /// Clears the cache.
    pub fn clear(&mut self) {
        self.neighbors.clear();
    }
}

impl Default for Neighborhood {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_to_coordinates() {
        // 1D
        assert_eq!(Topology::index_to_coordinates(5, &[10]), vec![5]);

        // 2D
        assert_eq!(Topology::index_to_coordinates(0, &[3, 3]), vec![0, 0]);
        assert_eq!(Topology::index_to_coordinates(4, &[3, 3]), vec![1, 1]);
        assert_eq!(Topology::index_to_coordinates(8, &[3, 3]), vec![2, 2]);

        // 3D
        assert_eq!(Topology::index_to_coordinates(13, &[2, 3, 4]), vec![1, 0, 1]);
    }

    #[test]
    fn test_coordinates_to_index() {
        // 1D
        assert_eq!(Topology::coordinates_to_index(&[5], &[10]), 5);

        // 2D
        assert_eq!(Topology::coordinates_to_index(&[0, 0], &[3, 3]), 0);
        assert_eq!(Topology::coordinates_to_index(&[1, 1], &[3, 3]), 4);
        assert_eq!(Topology::coordinates_to_index(&[2, 2], &[3, 3]), 8);

        // Round trip
        for i in 0..60 {
            let coords = Topology::index_to_coordinates(i, &[3, 4, 5]);
            let back = Topology::coordinates_to_index(&coords, &[3, 4, 5]);
            assert_eq!(i, back);
        }
    }

    #[test]
    fn test_neighborhood_1d() {
        let neighbors = Topology::neighborhood(5, &[10], 2, WrappingMode::NoWrap, true);
        assert!(neighbors.contains(&3));
        assert!(neighbors.contains(&4));
        assert!(neighbors.contains(&5));
        assert!(neighbors.contains(&6));
        assert!(neighbors.contains(&7));
        assert_eq!(neighbors.len(), 5);
    }

    #[test]
    fn test_neighborhood_1d_boundary() {
        // At beginning
        let neighbors = Topology::neighborhood(0, &[10], 2, WrappingMode::NoWrap, true);
        assert!(neighbors.contains(&0));
        assert!(neighbors.contains(&1));
        assert!(neighbors.contains(&2));
        assert!(!neighbors.contains(&10)); // Out of bounds
        assert_eq!(neighbors.len(), 3);

        // With wrapping
        let wrapped = Topology::neighborhood(0, &[10], 2, WrappingMode::Wrap, true);
        assert!(wrapped.contains(&8));
        assert!(wrapped.contains(&9));
        assert!(wrapped.contains(&0));
        assert!(wrapped.contains(&1));
        assert!(wrapped.contains(&2));
        assert_eq!(wrapped.len(), 5);
    }

    #[test]
    fn test_neighborhood_2d() {
        let neighbors = Topology::neighborhood(4, &[3, 3], 1, WrappingMode::NoWrap, true);
        // Center is (1,1), neighbors should be all 8 surrounding plus center
        assert_eq!(neighbors.len(), 9);

        // Without center
        let no_center = Topology::neighborhood(4, &[3, 3], 1, WrappingMode::NoWrap, false);
        assert_eq!(no_center.len(), 8);
        assert!(!no_center.contains(&4));
    }

    #[test]
    fn test_map_column_to_input() {
        // Simple 1:1 mapping
        let idx = Topology::map_column_to_input(0, &[10], &[10]);
        assert_eq!(idx, 0);

        // Scaled mapping
        let idx = Topology::map_column_to_input(0, &[5], &[10]);
        assert_eq!(idx, 1); // (0.5) * 10 / 5 = 1

        let idx = Topology::map_column_to_input(4, &[5], &[10]);
        assert_eq!(idx, 9); // (4.5) * 10 / 5 = 9
    }

    #[test]
    fn test_potential_pool() {
        let pool = Topology::map_potential_pool(
            0,
            &[5],
            &[10],
            2,
            WrappingMode::NoWrap,
        );

        // Column 0 maps to input 1, with radius 2
        assert!(pool.len() >= 3);
        assert!(pool.contains(&0) || pool.contains(&1) || pool.contains(&2));
    }

    #[test]
    fn test_num_elements() {
        assert_eq!(Topology::num_elements(&[10]), 10);
        assert_eq!(Topology::num_elements(&[3, 4]), 12);
        assert_eq!(Topology::num_elements(&[2, 3, 4]), 24);
    }

    #[test]
    fn test_neighborhood_cache() {
        // skip_center=true excludes the cell itself from its neighborhood
        let cache = Neighborhood::compute_all(&[5, 5], 1, WrappingMode::NoWrap, true);

        assert_eq!(cache.len(), 25);

        // Center cell (index 12 = position 2,2) should have 8 neighbors
        let center_neighbors = cache.get(12).unwrap();
        assert_eq!(center_neighbors.len(), 8);

        // Corner cell (index 0 = position 0,0) should have 3 neighbors
        let corner_neighbors = cache.get(0).unwrap();
        assert_eq!(corner_neighbors.len(), 3);
    }
}
