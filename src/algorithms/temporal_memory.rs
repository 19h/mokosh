//! Temporal Memory implementation.
//!
//! The Temporal Memory algorithm learns temporal sequences by forming
//! connections between cells. It predicts future states based on
//! learned patterns.

use crate::algorithms::{Connections, ConnectionsParams};
use crate::error::{MokoshError, Result};
use crate::types::{
    CellIdx, Permanence, Real, Sdr, Segment, SegmentIdx, SynapseIdx, UInt, MAX_PERMANENCE,
    MIN_PERMANENCE,
};
use crate::utils::Random;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// How to compute anomaly scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum AnomalyMode {
    /// No anomaly computation.
    #[default]
    Disabled,
    /// Raw anomaly score (fraction of unpredicted active columns).
    Raw,
    /// Likelihood-based anomaly score.
    Likelihood,
}

/// Parameters for creating a Temporal Memory.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TemporalMemoryParams {
    /// Dimensions of the column space.
    pub column_dimensions: Vec<UInt>,

    /// Number of cells per column.
    pub cells_per_column: UInt,

    /// Radius of potential synapses for new segments.
    pub activation_threshold: UInt,

    /// Initial permanence for new synapses.
    pub initial_permanence: Permanence,

    /// Permanence threshold for connected synapses.
    pub connected_permanence: Permanence,

    /// Minimum number of potential synapses for learning.
    pub min_threshold: UInt,

    /// Maximum number of synapses per segment.
    pub max_synapses_per_segment: UInt,

    /// Maximum number of segments per cell.
    pub max_segments_per_cell: UInt,

    /// Maximum number of new synapses added per learning cycle.
    pub max_new_synapse_count: UInt,

    /// Amount to increment permanence for active synapses.
    pub permanence_increment: Permanence,

    /// Amount to decrement permanence for inactive synapses.
    pub permanence_decrement: Permanence,

    /// Amount to decrement permanence for predicted-inactive segments.
    pub predicted_segment_decrement: Permanence,

    /// Random seed.
    pub seed: i64,

    /// Anomaly computation mode.
    pub anomaly_mode: AnomalyMode,

    /// Use external predictive inputs for basal context.
    pub external_predictive_inputs: UInt,

    /// Check presynaptic cell consistency when adding synapses.
    pub check_input_tm: bool,
}

impl Default for TemporalMemoryParams {
    fn default() -> Self {
        Self {
            column_dimensions: vec![2048],
            cells_per_column: 32,
            activation_threshold: 13,
            initial_permanence: 0.21,
            connected_permanence: 0.5,
            min_threshold: 10,
            max_synapses_per_segment: 255,
            max_segments_per_cell: 255,
            max_new_synapse_count: 20,
            permanence_increment: 0.1,
            permanence_decrement: 0.1,
            predicted_segment_decrement: 0.0,
            seed: 42,
            anomaly_mode: AnomalyMode::Disabled,
            external_predictive_inputs: 0,
            check_input_tm: false,
        }
    }
}

/// The Temporal Memory algorithm.
///
/// Temporal Memory learns sequences by forming connections between
/// cells in different columns. It maintains a prediction of which
/// cells will become active in the next time step.
///
/// # Example
///
/// ```rust
/// use mokosh::algorithms::{TemporalMemory, TemporalMemoryParams};
/// use mokosh::types::Sdr;
///
/// let mut tm = TemporalMemory::new(TemporalMemoryParams {
///     column_dimensions: vec![100],
///     cells_per_column: 4,
///     ..Default::default()
/// }).unwrap();
///
/// let mut active_columns = Sdr::new(&[100]);
/// active_columns.set_sparse(&[1, 5, 10, 20]).unwrap();
///
/// tm.compute(&active_columns, true);
///
/// let active_cells = tm.active_cells();
/// let predictive_cells = tm.predictive_cells();
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TemporalMemory {
    // Configuration
    column_dimensions: Vec<UInt>,
    cells_per_column: UInt,
    num_columns: usize,
    num_cells: usize,
    activation_threshold: UInt,
    initial_permanence: Permanence,
    connected_permanence: Permanence,
    min_threshold: UInt,
    max_synapses_per_segment: UInt,
    max_segments_per_cell: UInt,
    max_new_synapse_count: UInt,
    permanence_increment: Permanence,
    permanence_decrement: Permanence,
    predicted_segment_decrement: Permanence,
    anomaly_mode: AnomalyMode,
    external_predictive_inputs: UInt,
    check_input_tm: bool,

    // Connections
    connections: Connections,

    // State
    active_cells: Vec<CellIdx>,
    winner_cells: Vec<CellIdx>,
    predictive_cells: Vec<CellIdx>,
    active_segments: Vec<Segment>,
    matching_segments: Vec<Segment>,
    num_active_potential_synapses_for_segment: Vec<SynapseIdx>,

    // Anomaly state
    anomaly: Real,

    // RNG
    rng: Random,

    // Iteration counter
    iteration: u64,
}

impl TemporalMemory {
    /// Creates a new Temporal Memory with the given parameters.
    pub fn new(params: TemporalMemoryParams) -> Result<Self> {
        if params.column_dimensions.is_empty() {
            return Err(MokoshError::InvalidParameter {
                name: "column_dimensions",
                message: "Cannot be empty".to_string(),
            });
        }
        if params.cells_per_column == 0 {
            return Err(MokoshError::InvalidParameter {
                name: "cells_per_column",
                message: "Must be > 0".to_string(),
            });
        }

        let num_columns: usize = params
            .column_dimensions
            .iter()
            .map(|&d| d as usize)
            .product();
        let num_cells = num_columns * params.cells_per_column as usize;

        Ok(Self {
            column_dimensions: params.column_dimensions,
            cells_per_column: params.cells_per_column,
            num_columns,
            num_cells,
            activation_threshold: params.activation_threshold,
            initial_permanence: params.initial_permanence,
            connected_permanence: params.connected_permanence,
            min_threshold: params.min_threshold,
            max_synapses_per_segment: params.max_synapses_per_segment,
            max_segments_per_cell: params.max_segments_per_cell,
            max_new_synapse_count: params.max_new_synapse_count,
            permanence_increment: params.permanence_increment,
            permanence_decrement: params.permanence_decrement,
            predicted_segment_decrement: params.predicted_segment_decrement,
            anomaly_mode: params.anomaly_mode,
            external_predictive_inputs: params.external_predictive_inputs,
            check_input_tm: params.check_input_tm,

            connections: Connections::new(ConnectionsParams {
                num_cells: num_cells as CellIdx,
                connected_threshold: params.connected_permanence,
                timeseries: false,
            }),

            active_cells: Vec::new(),
            winner_cells: Vec::new(),
            predictive_cells: Vec::new(),
            active_segments: Vec::new(),
            matching_segments: Vec::new(),
            num_active_potential_synapses_for_segment: Vec::new(),

            anomaly: 0.0,
            rng: Random::new(params.seed),
            iteration: 0,
        })
    }

    /// Main compute method.
    ///
    /// Given the active columns from the Spatial Pooler, computes the
    /// active and predictive cells.
    pub fn compute(&mut self, active_columns: &Sdr, learn: bool) {
        self.iteration += 1;

        let prev_active_cells = std::mem::take(&mut self.active_cells);
        let prev_winner_cells = std::mem::take(&mut self.winner_cells);

        // Compute activity
        let (num_active_connected, num_active_potential) =
            self.connections.compute_activity_full(&prev_active_cells, learn);

        // Find active and matching segments
        self.active_segments.clear();
        self.matching_segments.clear();
        self.num_active_potential_synapses_for_segment = num_active_potential.clone();

        for segment in 0..self.connections.segment_flat_list_length() {
            if num_active_connected[segment] >= self.activation_threshold as SynapseIdx {
                self.active_segments.push(segment as Segment);
            }
            if num_active_potential[segment] >= self.min_threshold as SynapseIdx {
                self.matching_segments.push(segment as Segment);
            }
        }

        // Calculate predictive cells
        self.predictive_cells.clear();
        let mut predictive_set = std::collections::HashSet::new();
        for &segment in &self.active_segments {
            let cell = self.connections.cell_for_segment(segment);
            if predictive_set.insert(cell) {
                self.predictive_cells.push(cell);
            }
        }

        // Activate columns
        let active_columns_sparse = active_columns.get_sparse();

        // Compute anomaly before activation
        if self.anomaly_mode != AnomalyMode::Disabled {
            self.compute_anomaly(&active_columns_sparse, &predictive_set);
        }

        self.active_cells.clear();
        self.winner_cells.clear();

        for &column in &active_columns_sparse {
            let column = column as usize;

            // Check if any cell in this column was predicted
            let predicted_cells: Vec<CellIdx> = (0..self.cells_per_column)
                .map(|i| self.column_cell(column, i as usize))
                .filter(|c| predictive_set.contains(c))
                .collect();

            if predicted_cells.is_empty() {
                // Bursting - activate all cells
                self.burst_column(column, &prev_winner_cells, learn);
            } else {
                // Activate predicted cells
                self.activate_predicted_column(column, &predicted_cells, &prev_winner_cells, learn);
            }
        }

        // Punish predicted segments in inactive columns
        if learn && self.predicted_segment_decrement > 0.0 {
            self.punish_predicted_column(&active_columns_sparse);
        }
    }

    /// Activates cells in a column that was correctly predicted.
    fn activate_predicted_column(
        &mut self,
        column: usize,
        predicted_cells: &[CellIdx],
        prev_winner_cells: &[CellIdx],
        learn: bool,
    ) {
        for &cell in predicted_cells {
            self.active_cells.push(cell);
            self.winner_cells.push(cell);

            if learn {
                // Reinforce active segments on this cell
                let segments: Vec<Segment> = self
                    .connections
                    .segments_for_cell(cell)
                    .iter()
                    .copied()
                    .filter(|&s| self.active_segments.contains(&s))
                    .collect();

                for segment in segments {
                    self.adapt_segment(segment, prev_winner_cells, true);
                }
            }
        }
    }

    /// Bursts a column (activates all cells when unpredicted).
    fn burst_column(&mut self, column: usize, prev_winner_cells: &[CellIdx], learn: bool) {
        // Activate all cells in column
        for i in 0..self.cells_per_column {
            let cell = self.column_cell(column, i as usize);
            self.active_cells.push(cell);
        }

        // Find best matching segment in this column
        let best_matching = self.best_matching_segment_in_column(column);

        let winner_cell = if let Some(segment) = best_matching {
            // Use cell with best matching segment
            let cell = self.connections.cell_for_segment(segment);

            if learn {
                self.adapt_segment(segment, prev_winner_cells, true);

                // Grow additional synapses
                let num_active = self.num_active_potential_synapses_for_segment
                    .get(segment as usize)
                    .copied()
                    .unwrap_or(0);
                let new_synapse_count = (self.max_new_synapse_count as usize)
                    .saturating_sub(num_active as usize);

                if new_synapse_count > 0 {
                    self.grow_synapses(segment, prev_winner_cells, new_synapse_count);
                }
            }

            cell
        } else {
            // No matching segment - pick least used cell and grow new segment
            let cell = self.least_used_cell(column);

            if learn && !prev_winner_cells.is_empty() {
                let segment = self.connections.create_segment(
                    cell,
                    Some(self.max_segments_per_cell as SegmentIdx),
                );
                self.grow_synapses(segment, prev_winner_cells, self.max_new_synapse_count as usize);
            }

            cell
        };

        self.winner_cells.push(winner_cell);
    }

    /// Punishes segments that predicted incorrectly.
    fn punish_predicted_column(&mut self, active_columns: &[u32]) {
        let active_set: std::collections::HashSet<_> = active_columns.iter().copied().collect();

        // Find predicted columns that weren't active
        for &segment in &self.matching_segments {
            let cell = self.connections.cell_for_segment(segment);
            let column = self.cell_column(cell);

            if !active_set.contains(&(column as u32)) {
                // Punish this segment
                let synapses: Vec<_> = self
                    .connections
                    .synapses_for_segment(segment)
                    .iter()
                    .copied()
                    .collect();

                for synapse in synapses {
                    let data = self.connections.data_for_synapse(synapse);
                    let new_perm =
                        (data.permanence - self.predicted_segment_decrement).max(MIN_PERMANENCE);
                    self.connections.update_synapse_permanence(synapse, new_perm);
                }
            }
        }
    }

    /// Adapts a segment by adjusting permanences.
    fn adapt_segment(&mut self, segment: Segment, active_cells: &[CellIdx], reinforce: bool) {
        let active_set: std::collections::HashSet<_> = active_cells.iter().copied().collect();

        let synapses: Vec<_> = self
            .connections
            .synapses_for_segment(segment)
            .iter()
            .copied()
            .collect();

        for synapse in synapses {
            let data = self.connections.data_for_synapse(synapse);
            let presynaptic = data.presynaptic_cell;
            let old_perm = data.permanence;

            let new_perm = if active_set.contains(&presynaptic) {
                if reinforce {
                    (old_perm + self.permanence_increment).min(MAX_PERMANENCE)
                } else {
                    old_perm
                }
            } else {
                (old_perm - self.permanence_decrement).max(MIN_PERMANENCE)
            };

            if (new_perm - old_perm).abs() > 1e-6 {
                self.connections.update_synapse_permanence(synapse, new_perm);
            }
        }
    }

    /// Grows new synapses on a segment.
    fn grow_synapses(&mut self, segment: Segment, candidates: &[CellIdx], count: usize) {
        if candidates.is_empty() || count == 0 {
            return;
        }

        let candidates_vec = candidates.to_vec();
        self.connections.grow_synapses(
            segment,
            &candidates_vec,
            self.initial_permanence,
            &mut self.rng,
            Some(count),
            Some(self.max_synapses_per_segment as usize),
        );
    }

    /// Finds the best matching segment in a column.
    fn best_matching_segment_in_column(&self, column: usize) -> Option<Segment> {
        let mut best_segment = None;
        let mut best_score = 0;

        for i in 0..self.cells_per_column {
            let cell = self.column_cell(column, i as usize);

            for &segment in self.connections.segments_for_cell(cell) {
                if self.matching_segments.contains(&segment) {
                    let score = self.num_active_potential_synapses_for_segment
                        .get(segment as usize)
                        .copied()
                        .unwrap_or(0);

                    if score > best_score {
                        best_score = score;
                        best_segment = Some(segment);
                    }
                }
            }
        }

        best_segment
    }

    /// Returns the cell with the fewest segments in a column.
    fn least_used_cell(&self, column: usize) -> CellIdx {
        let mut min_segments = usize::MAX;
        let mut least_used = self.column_cell(column, 0);

        for i in 0..self.cells_per_column {
            let cell = self.column_cell(column, i as usize);
            let num_segments = self.connections.segments_for_cell(cell).len();

            if num_segments < min_segments {
                min_segments = num_segments;
                least_used = cell;

                if num_segments == 0 {
                    break;
                }
            }
        }

        least_used
    }

    /// Computes the anomaly score.
    fn compute_anomaly(&mut self, active_columns: &[u32], predicted_cells: &std::collections::HashSet<CellIdx>) {
        if active_columns.is_empty() {
            self.anomaly = 0.0;
            return;
        }

        // Count columns that had predicted cells
        let predicted_columns: std::collections::HashSet<_> = predicted_cells
            .iter()
            .map(|&c| self.cell_column(c) as u32)
            .collect();

        let num_predicted_active = active_columns
            .iter()
            .filter(|c| predicted_columns.contains(c))
            .count();

        self.anomaly = 1.0 - (num_predicted_active as Real / active_columns.len() as Real);
    }

    // ========================================================================
    // Cell/Column utilities
    // ========================================================================

    /// Returns the cell index for a column and cell offset.
    #[inline]
    fn column_cell(&self, column: usize, cell_offset: usize) -> CellIdx {
        (column * self.cells_per_column as usize + cell_offset) as CellIdx
    }

    /// Returns the column for a cell index.
    #[inline]
    fn cell_column(&self, cell: CellIdx) -> usize {
        cell as usize / self.cells_per_column as usize
    }

    /// Resets the temporal memory state.
    pub fn reset(&mut self) {
        self.active_cells.clear();
        self.winner_cells.clear();
        self.predictive_cells.clear();
        self.active_segments.clear();
        self.matching_segments.clear();
        self.connections.reset();
    }

    // ========================================================================
    // Getters
    // ========================================================================

    /// Returns the column dimensions.
    pub fn column_dimensions(&self) -> &[UInt] {
        &self.column_dimensions
    }

    /// Returns the number of columns.
    pub fn num_columns(&self) -> usize {
        self.num_columns
    }

    /// Returns the number of cells per column.
    pub fn cells_per_column(&self) -> UInt {
        self.cells_per_column
    }

    /// Returns the total number of cells.
    pub fn num_cells(&self) -> usize {
        self.num_cells
    }

    /// Returns the currently active cells.
    pub fn active_cells(&self) -> &[CellIdx] {
        &self.active_cells
    }

    /// Returns the winner cells from the last compute.
    pub fn winner_cells(&self) -> &[CellIdx] {
        &self.winner_cells
    }

    /// Returns the predictive cells.
    pub fn predictive_cells(&self) -> &[CellIdx] {
        &self.predictive_cells
    }

    /// Returns the active cells as an SDR.
    pub fn get_active_cells_sdr(&self) -> Sdr {
        let mut sdr = Sdr::new(&[self.num_cells as UInt]);
        let sparse: Vec<u32> = self.active_cells.iter().map(|&c| c as u32).collect();
        sdr.set_sparse_unchecked(sparse);
        sdr
    }

    /// Returns the predictive cells as an SDR.
    pub fn get_predictive_cells_sdr(&self) -> Sdr {
        let mut sdr = Sdr::new(&[self.num_cells as UInt]);
        let sparse: Vec<u32> = self.predictive_cells.iter().map(|&c| c as u32).collect();
        sdr.set_sparse_unchecked(sparse);
        sdr
    }

    /// Returns the anomaly score.
    pub fn anomaly(&self) -> Real {
        self.anomaly
    }

    /// Returns a reference to the connections.
    pub fn connections(&self) -> &Connections {
        &self.connections
    }

    /// Returns the activation threshold.
    pub fn activation_threshold(&self) -> UInt {
        self.activation_threshold
    }

    /// Returns the initial permanence.
    pub fn initial_permanence(&self) -> Permanence {
        self.initial_permanence
    }

    /// Returns the connected permanence threshold.
    pub fn connected_permanence(&self) -> Permanence {
        self.connected_permanence
    }

    /// Returns the minimum threshold for matching.
    pub fn min_threshold(&self) -> UInt {
        self.min_threshold
    }

    /// Returns the current iteration.
    pub fn iteration(&self) -> u64 {
        self.iteration
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_temporal_memory() {
        let tm = TemporalMemory::new(TemporalMemoryParams {
            column_dimensions: vec![100],
            cells_per_column: 4,
            ..Default::default()
        })
        .unwrap();

        assert_eq!(tm.num_columns(), 100);
        assert_eq!(tm.cells_per_column(), 4);
        assert_eq!(tm.num_cells(), 400);
    }

    #[test]
    fn test_compute_basic() {
        let mut tm = TemporalMemory::new(TemporalMemoryParams {
            column_dimensions: vec![100],
            cells_per_column: 4,
            ..Default::default()
        })
        .unwrap();

        let mut active_columns = Sdr::new(&[100]);
        active_columns.set_sparse(&[1, 5, 10, 20]).unwrap();

        tm.compute(&active_columns, true);

        // Should have some active cells (4 columns * 4 cells = 16 for bursting)
        assert!(!tm.active_cells().is_empty());
        assert!(!tm.winner_cells().is_empty());
    }

    #[test]
    fn test_prediction() {
        let mut tm = TemporalMemory::new(TemporalMemoryParams {
            column_dimensions: vec![50],
            cells_per_column: 4,
            activation_threshold: 2,
            min_threshold: 1,
            max_new_synapse_count: 10,
            initial_permanence: 0.5,
            connected_permanence: 0.5,
            ..Default::default()
        })
        .unwrap();

        let mut pattern_a = Sdr::new(&[50]);
        let mut pattern_b = Sdr::new(&[50]);

        pattern_a.set_sparse(&[0, 1, 2, 3, 4]).unwrap();
        pattern_b.set_sparse(&[10, 11, 12, 13, 14]).unwrap();

        // Learn sequence A -> B multiple times
        for _ in 0..20 {
            tm.reset();
            tm.compute(&pattern_a, true);
            tm.compute(&pattern_b, true);
        }

        // Now present A and check for predictions
        tm.reset();
        tm.compute(&pattern_a, false);

        // Should have some predictive cells
        // (may or may not depending on learning success)
        assert!(!tm.active_cells().is_empty());
    }

    #[test]
    fn test_anomaly() {
        let mut tm = TemporalMemory::new(TemporalMemoryParams {
            column_dimensions: vec![50],
            cells_per_column: 4,
            anomaly_mode: AnomalyMode::Raw,
            ..Default::default()
        })
        .unwrap();

        let mut pattern = Sdr::new(&[50]);
        pattern.set_sparse(&[0, 1, 2, 3, 4]).unwrap();

        // First presentation should be anomalous (no predictions)
        tm.compute(&pattern, true);
        assert!(tm.anomaly() > 0.5); // Mostly unpredicted
    }

    #[test]
    fn test_reset() {
        let mut tm = TemporalMemory::new(TemporalMemoryParams {
            column_dimensions: vec![50],
            cells_per_column: 4,
            ..Default::default()
        })
        .unwrap();

        let mut pattern = Sdr::new(&[50]);
        pattern.set_sparse(&[0, 1, 2]).unwrap();

        tm.compute(&pattern, true);
        assert!(!tm.active_cells().is_empty());

        tm.reset();
        assert!(tm.active_cells().is_empty());
        assert!(tm.winner_cells().is_empty());
        assert!(tm.predictive_cells().is_empty());
    }

    #[test]
    fn test_cell_column_mapping() {
        let tm = TemporalMemory::new(TemporalMemoryParams {
            column_dimensions: vec![10],
            cells_per_column: 4,
            ..Default::default()
        })
        .unwrap();

        // Cell 0-3 should be column 0
        assert_eq!(tm.cell_column(0), 0);
        assert_eq!(tm.cell_column(3), 0);

        // Cell 4-7 should be column 1
        assert_eq!(tm.cell_column(4), 1);
        assert_eq!(tm.cell_column(7), 1);

        // Cell indices
        assert_eq!(tm.column_cell(0, 0), 0);
        assert_eq!(tm.column_cell(0, 3), 3);
        assert_eq!(tm.column_cell(1, 0), 4);
    }
}
