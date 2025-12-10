//! Connections - The synaptic connectivity graph for HTM.
//!
//! The Connections class is a data structure that represents the connections
//! of a collection of cells. It stores and provides access to the connectivity
//! between cells (segments and synapses) used by both Spatial Pooler and
//! Temporal Memory.

use crate::error::{MokoshError, Result};
use crate::types::{
    CellIdx, ElemSparse, Permanence, Real, Segment, SegmentIdx, Sdr, Synapse, SynapseIdx,
    MAX_PERMANENCE, MIN_PERMANENCE,
};
use crate::utils::Random;

use ahash::AHashMap;
use smallvec::SmallVec;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Data associated with a synapse.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SynapseData {
    /// The presynaptic cell this synapse connects to.
    pub presynaptic_cell: CellIdx,

    /// The permanence strength of this synapse.
    pub permanence: Permanence,

    /// The segment this synapse belongs to.
    pub segment: Segment,

    /// Index in the presynaptic map (for efficient removal).
    presynaptic_map_index: usize,
}

impl SynapseData {
    fn new(presynaptic_cell: CellIdx, permanence: Permanence, segment: Segment) -> Self {
        Self {
            presynaptic_cell,
            permanence,
            segment,
            presynaptic_map_index: 0,
        }
    }
}

/// Data associated with a segment.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SegmentData {
    /// The synapses on this segment.
    pub synapses: SmallVec<[Synapse; 32]>,

    /// The cell this segment belongs to.
    pub cell: CellIdx,

    /// Number of connected synapses (permanence >= threshold).
    pub num_connected: SynapseIdx,
}

impl SegmentData {
    fn new(cell: CellIdx) -> Self {
        Self {
            synapses: SmallVec::new(),
            cell,
            num_connected: 0,
        }
    }
}

/// Data associated with a cell.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CellData {
    /// The segments on this cell.
    pub segments: SmallVec<[Segment; 8]>,
}

/// Parameters for creating a Connections instance.
#[derive(Debug, Clone)]
pub struct ConnectionsParams {
    /// Number of cells in the connections graph.
    pub num_cells: CellIdx,

    /// Permanence threshold for a synapse to be considered connected.
    pub connected_threshold: Permanence,

    /// If true, don't apply the same learning update on consecutive cycles.
    /// Useful for time-series data with highly correlated inputs.
    pub timeseries: bool,
}

impl Default for ConnectionsParams {
    fn default() -> Self {
        Self {
            num_cells: 0,
            connected_threshold: 0.5,
            timeseries: false,
        }
    }
}

/// The Connections class manages the synaptic connections between cells.
///
/// This is the core data structure used by both Spatial Pooler and Temporal Memory
/// to store and manipulate the synaptic connectivity.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Connections {
    /// All cells in the connections graph.
    cells: Vec<CellData>,

    /// All segments (indexed by Segment).
    segments: Vec<SegmentData>,

    /// Destroyed segment indices (available for reuse).
    destroyed_segments: Vec<Segment>,

    /// All synapses (indexed by Synapse).
    synapses: Vec<SynapseData>,

    /// Destroyed synapse indices (available for reuse).
    destroyed_synapses: Vec<Synapse>,

    /// Permanence threshold for connected synapses.
    connected_threshold: Permanence,

    /// Iteration counter (incremented in compute_activity).
    iteration: u32,

    // Presynaptic lookup maps for efficient activity computation
    /// Maps presynaptic cell -> potential synapses (all synapses from that cell).
    potential_synapses_for_presynaptic_cell: AHashMap<CellIdx, Vec<Synapse>>,

    /// Maps presynaptic cell -> connected synapses.
    connected_synapses_for_presynaptic_cell: AHashMap<CellIdx, Vec<Synapse>>,

    /// Maps presynaptic cell -> potential segments.
    potential_segments_for_presynaptic_cell: AHashMap<CellIdx, Vec<Segment>>,

    /// Maps presynaptic cell -> connected segments.
    connected_segments_for_presynaptic_cell: AHashMap<CellIdx, Vec<Segment>>,

    /// Time-series mode for correlated data.
    timeseries: bool,

    /// Previous permanence updates (for time-series mode).
    previous_updates: Vec<Permanence>,

    /// Current permanence updates (for time-series mode).
    current_updates: Vec<Permanence>,

    /// Statistics: number of pruned synapses.
    pruned_synapses: usize,

    /// Statistics: number of pruned segments.
    pruned_segments: usize,
}

impl Connections {
    /// Creates a new Connections instance with the given parameters.
    pub fn new(params: ConnectionsParams) -> Self {
        let num_cells = params.num_cells as usize;

        Self {
            cells: vec![CellData::default(); num_cells],
            segments: Vec::new(),
            destroyed_segments: Vec::new(),
            synapses: Vec::new(),
            destroyed_synapses: Vec::new(),
            connected_threshold: params.connected_threshold,
            iteration: 0,
            potential_synapses_for_presynaptic_cell: AHashMap::new(),
            connected_synapses_for_presynaptic_cell: AHashMap::new(),
            potential_segments_for_presynaptic_cell: AHashMap::new(),
            connected_segments_for_presynaptic_cell: AHashMap::new(),
            timeseries: params.timeseries,
            previous_updates: Vec::new(),
            current_updates: Vec::new(),
            pruned_synapses: 0,
            pruned_segments: 0,
        }
    }

    /// Creates a Connections instance with default parameters.
    pub fn with_cells(num_cells: CellIdx) -> Self {
        Self::new(ConnectionsParams {
            num_cells,
            ..Default::default()
        })
    }

    /// Returns the number of cells.
    #[inline]
    pub fn num_cells(&self) -> usize {
        self.cells.len()
    }

    /// Returns the connected threshold.
    #[inline]
    pub fn connected_threshold(&self) -> Permanence {
        self.connected_threshold
    }

    /// Returns the current iteration count.
    #[inline]
    pub fn iteration(&self) -> u32 {
        self.iteration
    }

    /// Returns the number of segments (excluding destroyed).
    pub fn num_segments(&self) -> usize {
        self.segments.len() - self.destroyed_segments.len()
    }

    /// Returns the number of segments on a specific cell.
    pub fn num_segments_on_cell(&self, cell: CellIdx) -> usize {
        self.cells[cell as usize].segments.len()
    }

    /// Returns the number of synapses (excluding destroyed).
    pub fn num_synapses(&self) -> usize {
        self.synapses.len() - self.destroyed_synapses.len()
    }

    /// Returns the number of synapses on a specific segment.
    pub fn num_synapses_on_segment(&self, segment: Segment) -> usize {
        self.segments[segment as usize].synapses.len()
    }

    /// Returns the flat list length for segment indexing.
    pub fn segment_flat_list_length(&self) -> usize {
        self.segments.len()
    }

    // ========================================================================
    // Segment operations
    // ========================================================================

    /// Creates a new segment on the specified cell.
    ///
    /// # Arguments
    ///
    /// * `cell` - The cell to create the segment on
    /// * `max_segments_per_cell` - Optional limit on segments per cell
    ///
    /// # Returns
    ///
    /// The unique ID of the created segment.
    pub fn create_segment(
        &mut self,
        cell: CellIdx,
        max_segments_per_cell: Option<SegmentIdx>,
    ) -> Segment {
        // Enforce segment limit if specified
        if let Some(max) = max_segments_per_cell {
            while self.cells[cell as usize].segments.len() >= max as usize {
                self.prune_segment(cell);
            }
        }

        let segment = if let Some(reuse) = self.destroyed_segments.pop() {
            // Reuse destroyed segment slot
            self.segments[reuse as usize] = SegmentData::new(cell);
            reuse
        } else {
            // Allocate new segment
            let segment = self.segments.len() as Segment;
            self.segments.push(SegmentData::new(cell));
            segment
        };

        self.cells[cell as usize].segments.push(segment);
        segment
    }

    /// Destroys a segment and all its synapses.
    pub fn destroy_segment(&mut self, segment: Segment) {
        let segment_data = &self.segments[segment as usize];
        let cell = segment_data.cell;

        // Destroy all synapses on this segment
        let synapses: Vec<Synapse> = segment_data.synapses.iter().copied().collect();
        for synapse in synapses {
            self.destroy_synapse(synapse);
        }

        // Remove segment from cell
        let cell_segments = &mut self.cells[cell as usize].segments;
        if let Some(pos) = cell_segments.iter().position(|&s| s == segment) {
            cell_segments.swap_remove(pos);
        }

        self.destroyed_segments.push(segment);
        self.pruned_segments += 1;
    }

    /// Prunes the least useful segment from a cell.
    fn prune_segment(&mut self, cell: CellIdx) {
        let segments = &self.cells[cell as usize].segments;
        if segments.is_empty() {
            return;
        }

        // Find segment with fewest synapses (least useful heuristic)
        let segment_to_prune = segments
            .iter()
            .copied()
            .min_by_key(|&s| self.segments[s as usize].synapses.len())
            .unwrap();

        self.destroy_segment(segment_to_prune);
    }

    /// Gets the segments for a cell.
    #[inline]
    pub fn segments_for_cell(&self, cell: CellIdx) -> &[Segment] {
        &self.cells[cell as usize].segments
    }

    /// Gets the cell that owns a segment.
    #[inline]
    pub fn cell_for_segment(&self, segment: Segment) -> CellIdx {
        self.segments[segment as usize].cell
    }

    /// Gets the segment data.
    #[inline]
    pub fn data_for_segment(&self, segment: Segment) -> &SegmentData {
        &self.segments[segment as usize]
    }

    /// Gets mutable segment data.
    #[inline]
    pub fn data_for_segment_mut(&mut self, segment: Segment) -> &mut SegmentData {
        &mut self.segments[segment as usize]
    }

    /// Gets the segment at a specific index on a cell.
    #[inline]
    pub fn get_segment(&self, cell: CellIdx, idx: SegmentIdx) -> Segment {
        self.cells[cell as usize].segments[idx as usize]
    }

    /// Gets the index of a segment on its cell.
    pub fn idx_on_cell_for_segment(&self, segment: Segment) -> SegmentIdx {
        let cell = self.segments[segment as usize].cell;
        self.cells[cell as usize]
            .segments
            .iter()
            .position(|&s| s == segment)
            .unwrap() as SegmentIdx
    }

    // ========================================================================
    // Synapse operations
    // ========================================================================

    /// Creates a new synapse on a segment.
    ///
    /// If a synapse to the same presynaptic cell already exists, returns the
    /// existing synapse (possibly updating its permanence if the new one is higher).
    ///
    /// # Arguments
    ///
    /// * `segment` - The segment to create the synapse on
    /// * `presynaptic_cell` - The presynaptic cell to connect to
    /// * `permanence` - The initial permanence value
    pub fn create_synapse(
        &mut self,
        segment: Segment,
        presynaptic_cell: CellIdx,
        permanence: Permanence,
    ) -> Synapse {
        // Check for existing synapse to same presynaptic cell
        let existing = self.segments[segment as usize]
            .synapses
            .iter()
            .find(|&&s| self.synapses[s as usize].presynaptic_cell == presynaptic_cell)
            .copied();

        if let Some(existing_synapse) = existing {
            // Update permanence if new value is higher
            let current_perm = self.synapses[existing_synapse as usize].permanence;
            if permanence > current_perm {
                self.update_synapse_permanence(existing_synapse, permanence);
            }
            return existing_synapse;
        }

        let permanence = permanence.clamp(MIN_PERMANENCE, MAX_PERMANENCE);

        let synapse = if let Some(reuse) = self.destroyed_synapses.pop() {
            self.synapses[reuse as usize] = SynapseData::new(presynaptic_cell, permanence, segment);
            reuse
        } else {
            let synapse = self.synapses.len() as Synapse;
            self.synapses
                .push(SynapseData::new(presynaptic_cell, permanence, segment));
            synapse
        };

        // Add to segment
        self.segments[segment as usize].synapses.push(synapse);

        // Update connected count
        if permanence >= self.connected_threshold {
            self.segments[segment as usize].num_connected += 1;
        }

        // Add to presynaptic maps
        self.add_synapse_to_presynaptic_map(synapse, presynaptic_cell, permanence);

        synapse
    }

    /// Destroys a synapse.
    pub fn destroy_synapse(&mut self, synapse: Synapse) {
        let synapse_data = &self.synapses[synapse as usize];
        let segment = synapse_data.segment;
        let presynaptic_cell = synapse_data.presynaptic_cell;
        let was_connected = synapse_data.permanence >= self.connected_threshold;

        // Remove from segment
        let segment_synapses = &mut self.segments[segment as usize].synapses;
        if let Some(pos) = segment_synapses.iter().position(|&s| s == synapse) {
            segment_synapses.swap_remove(pos);
        }

        // Update connected count
        if was_connected {
            self.segments[segment as usize].num_connected =
                self.segments[segment as usize].num_connected.saturating_sub(1);
        }

        // Remove from presynaptic maps
        self.remove_synapse_from_presynaptic_map(synapse, presynaptic_cell, was_connected);

        // Mark as destroyed (use invalid permanence as marker)
        self.synapses[synapse as usize].permanence = -1.0;
        self.destroyed_synapses.push(synapse);
        self.pruned_synapses += 1;
    }

    /// Updates a synapse's permanence value.
    pub fn update_synapse_permanence(&mut self, synapse: Synapse, permanence: Permanence) {
        let permanence = permanence.clamp(MIN_PERMANENCE, MAX_PERMANENCE);
        let synapse_data = &mut self.synapses[synapse as usize];
        let old_perm = synapse_data.permanence;
        let presynaptic_cell = synapse_data.presynaptic_cell;
        let segment = synapse_data.segment;

        let was_connected = old_perm >= self.connected_threshold;
        let is_connected = permanence >= self.connected_threshold;

        synapse_data.permanence = permanence;

        // Update connected count and presynaptic maps if connectivity changed
        if was_connected != is_connected {
            if is_connected {
                self.segments[segment as usize].num_connected += 1;
                // Move from potential to connected map
                self.move_synapse_to_connected(synapse, presynaptic_cell, segment);
            } else {
                self.segments[segment as usize].num_connected =
                    self.segments[segment as usize].num_connected.saturating_sub(1);
                // Move from connected to potential map
                self.move_synapse_to_potential(synapse, presynaptic_cell, segment);
            }
        }
    }

    /// Gets the synapses on a segment.
    #[inline]
    pub fn synapses_for_segment(&self, segment: Segment) -> &[Synapse] {
        &self.segments[segment as usize].synapses
    }

    /// Gets the segment that owns a synapse.
    #[inline]
    pub fn segment_for_synapse(&self, synapse: Synapse) -> Segment {
        self.synapses[synapse as usize].segment
    }

    /// Gets the synapse data.
    #[inline]
    pub fn data_for_synapse(&self, synapse: Synapse) -> &SynapseData {
        &self.synapses[synapse as usize]
    }

    /// Gets the presynaptic cells for a segment.
    pub fn presynaptic_cells_for_segment(&self, segment: Segment) -> Vec<CellIdx> {
        self.segments[segment as usize]
            .synapses
            .iter()
            .map(|&s| self.synapses[s as usize].presynaptic_cell)
            .collect()
    }

    /// Gets all synapses from a presynaptic cell.
    pub fn synapses_for_presynaptic_cell(&self, cell: CellIdx) -> Vec<Synapse> {
        self.potential_synapses_for_presynaptic_cell
            .get(&cell)
            .cloned()
            .unwrap_or_default()
    }

    // ========================================================================
    // Presynaptic map management
    // ========================================================================

    fn add_synapse_to_presynaptic_map(
        &mut self,
        synapse: Synapse,
        presynaptic_cell: CellIdx,
        permanence: Permanence,
    ) {
        let segment = self.synapses[synapse as usize].segment;

        // Add to potential maps
        self.potential_synapses_for_presynaptic_cell
            .entry(presynaptic_cell)
            .or_default()
            .push(synapse);
        self.potential_segments_for_presynaptic_cell
            .entry(presynaptic_cell)
            .or_default()
            .push(segment);

        // Add to connected maps if connected
        if permanence >= self.connected_threshold {
            self.connected_synapses_for_presynaptic_cell
                .entry(presynaptic_cell)
                .or_default()
                .push(synapse);
            self.connected_segments_for_presynaptic_cell
                .entry(presynaptic_cell)
                .or_default()
                .push(segment);
        }
    }

    fn remove_synapse_from_presynaptic_map(
        &mut self,
        synapse: Synapse,
        presynaptic_cell: CellIdx,
        was_connected: bool,
    ) {
        let segment = self.synapses[synapse as usize].segment;

        // Remove from potential maps
        if let Some(synapses) = self.potential_synapses_for_presynaptic_cell.get_mut(&presynaptic_cell) {
            if let Some(pos) = synapses.iter().position(|&s| s == synapse) {
                synapses.swap_remove(pos);
            }
        }
        if let Some(segments) = self.potential_segments_for_presynaptic_cell.get_mut(&presynaptic_cell) {
            if let Some(pos) = segments.iter().position(|&s| s == segment) {
                segments.swap_remove(pos);
            }
        }

        // Remove from connected maps if was connected
        if was_connected {
            if let Some(synapses) = self.connected_synapses_for_presynaptic_cell.get_mut(&presynaptic_cell) {
                if let Some(pos) = synapses.iter().position(|&s| s == synapse) {
                    synapses.swap_remove(pos);
                }
            }
            if let Some(segments) = self.connected_segments_for_presynaptic_cell.get_mut(&presynaptic_cell) {
                if let Some(pos) = segments.iter().position(|&s| s == segment) {
                    segments.swap_remove(pos);
                }
            }
        }
    }

    fn move_synapse_to_connected(&mut self, synapse: Synapse, presynaptic_cell: CellIdx, segment: Segment) {
        self.connected_synapses_for_presynaptic_cell
            .entry(presynaptic_cell)
            .or_default()
            .push(synapse);
        self.connected_segments_for_presynaptic_cell
            .entry(presynaptic_cell)
            .or_default()
            .push(segment);
    }

    fn move_synapse_to_potential(&mut self, synapse: Synapse, presynaptic_cell: CellIdx, segment: Segment) {
        if let Some(synapses) = self.connected_synapses_for_presynaptic_cell.get_mut(&presynaptic_cell) {
            if let Some(pos) = synapses.iter().position(|&s| s == synapse) {
                synapses.swap_remove(pos);
            }
        }
        if let Some(segments) = self.connected_segments_for_presynaptic_cell.get_mut(&presynaptic_cell) {
            if let Some(pos) = segments.iter().position(|&s| s == segment) {
                segments.swap_remove(pos);
            }
        }
    }

    // ========================================================================
    // Activity computation
    // ========================================================================

    /// Computes segment activity given active presynaptic cells.
    ///
    /// # Arguments
    ///
    /// * `active_presynaptic_cells` - The active input cells
    /// * `learn` - Whether learning updates should be tracked
    ///
    /// # Returns
    ///
    /// A vector of connected synapse counts per segment.
    pub fn compute_activity(
        &mut self,
        active_presynaptic_cells: &[CellIdx],
        learn: bool,
    ) -> Vec<SynapseIdx> {
        self.iteration += 1;

        if self.timeseries && learn {
            std::mem::swap(&mut self.previous_updates, &mut self.current_updates);
            self.current_updates.clear();
            self.current_updates.resize(self.synapses.len(), 0.0);
        }

        let mut num_active_connected = vec![0u16; self.segments.len()];

        for &cell in active_presynaptic_cells {
            if let Some(synapses) = self.connected_synapses_for_presynaptic_cell.get(&cell) {
                for &synapse in synapses {
                    let segment = self.synapses[synapse as usize].segment;
                    num_active_connected[segment as usize] += 1;
                }
            }
        }

        num_active_connected
    }

    /// Computes both connected and potential activity.
    pub fn compute_activity_full(
        &mut self,
        active_presynaptic_cells: &[CellIdx],
        learn: bool,
    ) -> (Vec<SynapseIdx>, Vec<SynapseIdx>) {
        self.iteration += 1;

        if self.timeseries && learn {
            std::mem::swap(&mut self.previous_updates, &mut self.current_updates);
            self.current_updates.clear();
            self.current_updates.resize(self.synapses.len(), 0.0);
        }

        let mut num_active_connected = vec![0u16; self.segments.len()];
        let mut num_active_potential = vec![0u16; self.segments.len()];

        for &cell in active_presynaptic_cells {
            // Count connected
            if let Some(synapses) = self.connected_synapses_for_presynaptic_cell.get(&cell) {
                for &synapse in synapses {
                    let segment = self.synapses[synapse as usize].segment;
                    num_active_connected[segment as usize] += 1;
                }
            }

            // Count potential
            if let Some(synapses) = self.potential_synapses_for_presynaptic_cell.get(&cell) {
                for &synapse in synapses {
                    let segment = self.synapses[synapse as usize].segment;
                    num_active_potential[segment as usize] += 1;
                }
            }
        }

        (num_active_connected, num_active_potential)
    }

    // ========================================================================
    // Learning operations
    // ========================================================================

    /// Adapts a segment based on active inputs.
    ///
    /// Increases permanence for synapses connected to active inputs,
    /// decreases for those connected to inactive inputs.
    pub fn adapt_segment(
        &mut self,
        segment: Segment,
        inputs: &Sdr,
        increment: Permanence,
        decrement: Permanence,
        prune_zero_synapses: bool,
        segment_threshold: u32,
    ) {
        let active_inputs: std::collections::HashSet<ElemSparse> =
            inputs.get_sparse().into_iter().collect();

        let synapses: Vec<Synapse> = self.segments[segment as usize]
            .synapses
            .iter()
            .copied()
            .collect();

        let mut synapses_to_destroy = Vec::new();

        for synapse in synapses {
            let synapse_data = &self.synapses[synapse as usize];
            let presynaptic_cell = synapse_data.presynaptic_cell;
            let old_perm = synapse_data.permanence;

            let delta = if active_inputs.contains(&presynaptic_cell) {
                increment
            } else {
                -decrement
            };

            // Handle time-series mode
            let effective_delta = if self.timeseries
                && synapse < self.previous_updates.len() as Synapse
            {
                let prev = self.previous_updates[synapse as usize];
                if prev != 0.0 && prev.signum() == delta.signum() {
                    0.0 // Skip if same direction as previous update
                } else {
                    delta
                }
            } else {
                delta
            };

            if self.timeseries && synapse < self.current_updates.len() as Synapse {
                self.current_updates[synapse as usize] = effective_delta;
            }

            let new_perm = (old_perm + effective_delta).clamp(MIN_PERMANENCE, MAX_PERMANENCE);

            if new_perm != old_perm {
                self.update_synapse_permanence(synapse, new_perm);
            }

            // Mark for destruction if permanence is zero
            if prune_zero_synapses && new_perm <= MIN_PERMANENCE {
                synapses_to_destroy.push(synapse);
            }
        }

        // Destroy zero-permanence synapses
        for synapse in synapses_to_destroy {
            self.destroy_synapse(synapse);
        }

        // Destroy segment if too few synapses remain
        if prune_zero_synapses
            && (self.segments[segment as usize].synapses.len() as u32) < segment_threshold
        {
            self.destroy_segment(segment);
        }
    }

    /// Grows new synapses on a segment to connect to growth candidates.
    pub fn grow_synapses(
        &mut self,
        segment: Segment,
        growth_candidates: &[CellIdx],
        initial_permanence: Permanence,
        rng: &mut Random,
        max_new: Option<usize>,
        max_synapses_per_segment: Option<usize>,
    ) {
        // Find candidates not already connected
        let existing: std::collections::HashSet<_> = self.segments[segment as usize]
            .synapses
            .iter()
            .map(|&s| self.synapses[s as usize].presynaptic_cell)
            .collect();

        let mut candidates: Vec<CellIdx> = growth_candidates
            .iter()
            .copied()
            .filter(|c| !existing.contains(c))
            .collect();

        if candidates.is_empty() {
            return;
        }

        // Limit number of new synapses
        let num_new = max_new
            .map(|m| m.min(candidates.len()))
            .unwrap_or(candidates.len());

        // Subsample if needed
        if num_new < candidates.len() {
            candidates = rng.sample(candidates, num_new);
        }

        // Make room if max synapses per segment is specified
        if let Some(max) = max_synapses_per_segment {
            let current = self.segments[segment as usize].synapses.len();
            if current + candidates.len() > max {
                let to_destroy = current + candidates.len() - max;
                self.destroy_min_permanence_synapses(segment, to_destroy, &[]);
            }
        }

        // Create new synapses
        for candidate in candidates {
            self.create_synapse(segment, candidate, initial_permanence);
        }
    }

    /// Raises permanences until the segment has at least `threshold` connected synapses.
    pub fn raise_permanences_to_threshold(&mut self, segment: Segment, threshold: u32) {
        let current_connected = self.segments[segment as usize].num_connected;
        if current_connected >= threshold as SynapseIdx {
            return;
        }

        let needed = threshold as usize - current_connected as usize;

        // Get synapses sorted by permanence (highest first among unconnected)
        let mut unconnected: Vec<(Synapse, Permanence)> = self.segments[segment as usize]
            .synapses
            .iter()
            .filter_map(|&s| {
                let perm = self.synapses[s as usize].permanence;
                if perm < self.connected_threshold {
                    Some((s, perm))
                } else {
                    None
                }
            })
            .collect();

        unconnected.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Raise permanences of top candidates
        for (synapse, _) in unconnected.into_iter().take(needed) {
            self.update_synapse_permanence(synapse, self.connected_threshold);
        }
    }

    /// Uniformly adjusts all permanences on a segment.
    pub fn bump_segment(&mut self, segment: Segment, delta: Permanence) {
        let synapses: Vec<Synapse> = self.segments[segment as usize]
            .synapses
            .iter()
            .copied()
            .collect();

        for synapse in synapses {
            let old_perm = self.synapses[synapse as usize].permanence;
            let new_perm = (old_perm + delta).clamp(MIN_PERMANENCE, MAX_PERMANENCE);
            self.update_synapse_permanence(synapse, new_perm);
        }
    }

    /// Destroys synapses with the lowest permanences.
    pub fn destroy_min_permanence_synapses(
        &mut self,
        segment: Segment,
        n_destroy: usize,
        exclude_cells: &[CellIdx],
    ) {
        if n_destroy == 0 {
            return;
        }

        let exclude_set: std::collections::HashSet<_> = exclude_cells.iter().copied().collect();

        // Get synapses sorted by permanence (lowest first)
        let mut candidates: Vec<(Synapse, Permanence)> = self.segments[segment as usize]
            .synapses
            .iter()
            .filter_map(|&s| {
                let synapse_data = &self.synapses[s as usize];
                if !exclude_set.contains(&synapse_data.presynaptic_cell) {
                    Some((s, synapse_data.permanence))
                } else {
                    None
                }
            })
            .collect();

        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Destroy lowest permanence synapses
        for (synapse, _) in candidates.into_iter().take(n_destroy) {
            self.destroy_synapse(synapse);
        }
    }

    /// Adjusts permanences to control connectivity within bounds.
    pub fn synapse_competition(
        &mut self,
        segment: Segment,
        min_synapses: SynapseIdx,
        max_synapses: SynapseIdx,
    ) {
        let current = self.segments[segment as usize].num_connected;

        if current < min_synapses {
            // Need to connect more synapses
            let needed = min_synapses - current;
            let mut unconnected: Vec<(Synapse, Permanence)> = self.segments[segment as usize]
                .synapses
                .iter()
                .filter_map(|&s| {
                    let perm = self.synapses[s as usize].permanence;
                    if perm < self.connected_threshold {
                        Some((s, perm))
                    } else {
                        None
                    }
                })
                .collect();

            unconnected.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            for (synapse, _) in unconnected.into_iter().take(needed as usize) {
                self.update_synapse_permanence(synapse, self.connected_threshold);
            }
        } else if current > max_synapses {
            // Need to disconnect some synapses
            let to_remove = current - max_synapses;
            let mut connected: Vec<(Synapse, Permanence)> = self.segments[segment as usize]
                .synapses
                .iter()
                .filter_map(|&s| {
                    let perm = self.synapses[s as usize].permanence;
                    if perm >= self.connected_threshold {
                        Some((s, perm))
                    } else {
                        None
                    }
                })
                .collect();

            connected.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            for (synapse, _) in connected.into_iter().take(to_remove as usize) {
                let new_perm = self.connected_threshold - 0.001;
                self.update_synapse_permanence(synapse, new_perm);
            }
        }
    }

    /// Resets time-series state.
    pub fn reset(&mut self) {
        self.previous_updates.clear();
        self.current_updates.clear();
    }

    /// Compares two segments for ordering.
    pub fn compare_segments(&self, a: Segment, b: Segment) -> std::cmp::Ordering {
        let cell_a = self.segments[a as usize].cell;
        let cell_b = self.segments[b as usize].cell;

        cell_a.cmp(&cell_b).then_with(|| {
            let idx_a = self.idx_on_cell_for_segment(a);
            let idx_b = self.idx_on_cell_for_segment(b);
            idx_a.cmp(&idx_b)
        })
    }
}

impl PartialEq for Connections {
    fn eq(&self, other: &Self) -> bool {
        self.cells == other.cells
            && self.segments == other.segments
            && self.synapses == other.synapses
            && (self.connected_threshold - other.connected_threshold).abs() < 1e-6
    }
}

impl Eq for Connections {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_connections() {
        let conn = Connections::with_cells(100);
        assert_eq!(conn.num_cells(), 100);
        assert_eq!(conn.num_segments(), 0);
        assert_eq!(conn.num_synapses(), 0);
    }

    #[test]
    fn test_create_segment() {
        let mut conn = Connections::with_cells(100);
        let seg = conn.create_segment(10, None);

        assert_eq!(conn.num_segments(), 1);
        assert_eq!(conn.cell_for_segment(seg), 10);
        assert_eq!(conn.segments_for_cell(10).len(), 1);
    }

    #[test]
    fn test_create_synapse() {
        let mut conn = Connections::with_cells(100);
        let seg = conn.create_segment(10, None);
        let syn = conn.create_synapse(seg, 50, 0.5);

        assert_eq!(conn.num_synapses(), 1);
        assert_eq!(conn.segment_for_synapse(syn), seg);
        assert_eq!(conn.data_for_synapse(syn).presynaptic_cell, 50);
        assert!((conn.data_for_synapse(syn).permanence - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_connected_count() {
        let mut conn = Connections::new(ConnectionsParams {
            num_cells: 100,
            connected_threshold: 0.5,
            timeseries: false,
        });

        let seg = conn.create_segment(10, None);
        conn.create_synapse(seg, 50, 0.6); // Connected
        conn.create_synapse(seg, 51, 0.4); // Not connected
        conn.create_synapse(seg, 52, 0.5); // Connected (at threshold)

        assert_eq!(conn.data_for_segment(seg).num_connected, 2);
    }

    #[test]
    fn test_update_permanence() {
        let mut conn = Connections::new(ConnectionsParams {
            num_cells: 100,
            connected_threshold: 0.5,
            timeseries: false,
        });

        let seg = conn.create_segment(10, None);
        let syn = conn.create_synapse(seg, 50, 0.4);

        assert_eq!(conn.data_for_segment(seg).num_connected, 0);

        conn.update_synapse_permanence(syn, 0.6);
        assert_eq!(conn.data_for_segment(seg).num_connected, 1);

        conn.update_synapse_permanence(syn, 0.3);
        assert_eq!(conn.data_for_segment(seg).num_connected, 0);
    }

    #[test]
    fn test_destroy_synapse() {
        let mut conn = Connections::with_cells(100);
        let seg = conn.create_segment(10, None);
        let syn = conn.create_synapse(seg, 50, 0.5);

        assert_eq!(conn.num_synapses(), 1);

        conn.destroy_synapse(syn);
        assert_eq!(conn.num_synapses(), 0);
        assert_eq!(conn.num_synapses_on_segment(seg), 0);
    }

    #[test]
    fn test_destroy_segment() {
        let mut conn = Connections::with_cells(100);
        let seg = conn.create_segment(10, None);
        conn.create_synapse(seg, 50, 0.5);
        conn.create_synapse(seg, 51, 0.5);

        assert_eq!(conn.num_segments(), 1);
        assert_eq!(conn.num_synapses(), 2);

        conn.destroy_segment(seg);
        assert_eq!(conn.num_segments(), 0);
        assert_eq!(conn.num_synapses(), 0);
        assert!(conn.segments_for_cell(10).is_empty());
    }

    #[test]
    fn test_compute_activity() {
        let mut conn = Connections::new(ConnectionsParams {
            num_cells: 100,
            connected_threshold: 0.5,
            timeseries: false,
        });

        let seg1 = conn.create_segment(10, None);
        conn.create_synapse(seg1, 50, 0.6);
        conn.create_synapse(seg1, 51, 0.6);
        conn.create_synapse(seg1, 52, 0.4); // Not connected

        let seg2 = conn.create_segment(20, None);
        conn.create_synapse(seg2, 50, 0.6);

        let activity = conn.compute_activity(&[50, 51, 53], true);

        assert_eq!(activity[seg1 as usize], 2); // 50 and 51 active
        assert_eq!(activity[seg2 as usize], 1); // Only 50 active
    }

    #[test]
    fn test_adapt_segment() {
        let mut conn = Connections::new(ConnectionsParams {
            num_cells: 100,
            connected_threshold: 0.5,
            timeseries: false,
        });

        let seg = conn.create_segment(10, None);
        let syn1 = conn.create_synapse(seg, 50, 0.5);
        let syn2 = conn.create_synapse(seg, 51, 0.5);

        let mut input = Sdr::new(&[100]);
        input.set_sparse(&[50]).unwrap();

        conn.adapt_segment(seg, &input, 0.1, 0.1, false, 0);

        // syn1 should increase (presynaptic cell 50 is active)
        assert!((conn.data_for_synapse(syn1).permanence - 0.6).abs() < 1e-6);

        // syn2 should decrease (presynaptic cell 51 is inactive)
        assert!((conn.data_for_synapse(syn2).permanence - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_grow_synapses() {
        let mut conn = Connections::with_cells(100);
        let mut rng = Random::new(42);

        let seg = conn.create_segment(10, None);
        conn.create_synapse(seg, 50, 0.5);

        conn.grow_synapses(seg, &[50, 51, 52, 53], 0.21, &mut rng, Some(2), None);

        // Should have 3 synapses total (1 existing + 2 new)
        // Note: 50 is excluded because it already has a synapse
        assert_eq!(conn.num_synapses_on_segment(seg), 3);
    }

    #[test]
    fn test_raise_permanences_to_threshold() {
        let mut conn = Connections::new(ConnectionsParams {
            num_cells: 100,
            connected_threshold: 0.5,
            timeseries: false,
        });

        let seg = conn.create_segment(10, None);
        conn.create_synapse(seg, 50, 0.4);
        conn.create_synapse(seg, 51, 0.3);
        conn.create_synapse(seg, 52, 0.2);

        assert_eq!(conn.data_for_segment(seg).num_connected, 0);

        conn.raise_permanences_to_threshold(seg, 2);

        // Top 2 unconnected synapses should now be connected
        assert!(conn.data_for_segment(seg).num_connected >= 2);
    }

    #[test]
    fn test_bump_segment() {
        let mut conn = Connections::with_cells(100);
        let seg = conn.create_segment(10, None);
        conn.create_synapse(seg, 50, 0.5);
        conn.create_synapse(seg, 51, 0.5);

        conn.bump_segment(seg, 0.1);

        for &syn in conn.synapses_for_segment(seg) {
            assert!((conn.data_for_synapse(syn).permanence - 0.6).abs() < 1e-6);
        }
    }

    #[test]
    fn test_max_segments_per_cell() {
        let mut conn = Connections::with_cells(100);

        conn.create_segment(10, Some(2));
        conn.create_synapse(conn.segments_for_cell(10)[0], 50, 0.5);

        conn.create_segment(10, Some(2));
        conn.create_synapse(conn.segments_for_cell(10)[1], 51, 0.5);

        // This should trigger pruning
        conn.create_segment(10, Some(2));

        assert!(conn.segments_for_cell(10).len() <= 2);
    }

    #[test]
    fn test_duplicate_synapse_prevention() {
        let mut conn = Connections::with_cells(100);
        let seg = conn.create_segment(10, None);

        let syn1 = conn.create_synapse(seg, 50, 0.5);
        let syn2 = conn.create_synapse(seg, 50, 0.3); // Same presynaptic cell

        assert_eq!(syn1, syn2); // Should return same synapse
        assert_eq!(conn.num_synapses_on_segment(seg), 1);
        assert!((conn.data_for_synapse(syn1).permanence - 0.5).abs() < 1e-6); // Keep higher permanence
    }
}
