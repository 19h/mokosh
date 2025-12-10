//! Spatial Pooler implementation.
//!
//! The Spatial Pooler is responsible for creating sparse distributed representations
//! of the input. Given an input SDR, it computes a set of active columns and
//! simultaneously updates its permanences, duty cycles, etc.

use crate::algorithms::{Connections, ConnectionsParams};
use crate::error::{MokoshError, Result};
use crate::types::{
    CellIdx, Permanence, Real, Sdr, Segment, SynapseIdx, UInt, MAX_PERMANENCE, MIN_PERMANENCE,
};
use crate::utils::{Neighborhood, Random, Topology, WrappingMode};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Parameters for creating a Spatial Pooler.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SpatialPoolerParams {
    /// Dimensions of the input space (e.g., `vec![100]` for 100 inputs).
    pub input_dimensions: Vec<UInt>,

    /// Dimensions of the column space (e.g., `vec![2048]` for 2048 columns).
    pub column_dimensions: Vec<UInt>,

    /// The extent of input that each column can potentially connect to.
    /// Defines a receptive field radius around each column's mapped input.
    pub potential_radius: UInt,

    /// Fraction of inputs within potential radius that a column connects to (0.0-1.0).
    pub potential_pct: Real,

    /// If true, all columns compete globally. If false, local inhibition is used.
    pub global_inhibition: bool,

    /// Target density of active columns (fraction of columns that should be active).
    pub local_area_density: Real,

    /// Alternate way to control sparsity (number of active columns per inhibition area).
    /// If > 0, overrides local_area_density.
    pub num_active_columns_per_inh_area: UInt,

    /// Minimum number of connected synapses for a column to be considered for activation.
    pub stimulus_threshold: UInt,

    /// Amount to decrease permanence of inactive synapses during learning.
    pub syn_perm_inactive_dec: Permanence,

    /// Amount to increase permanence of active synapses during learning.
    pub syn_perm_active_inc: Permanence,

    /// Permanence threshold for a synapse to be considered connected.
    pub syn_perm_connected: Permanence,

    /// Minimum fraction of max overlap duty cycle for a column to avoid boosting.
    pub min_pct_overlap_duty_cycles: Real,

    /// Period (in iterations) for duty cycle computations.
    pub duty_cycle_period: UInt,

    /// Strength of boosting (0.0 = no boosting).
    pub boost_strength: Real,

    /// Random seed (-1 for random seed).
    pub seed: i64,

    /// Verbosity level (0 = silent).
    pub sp_verbosity: UInt,

    /// Whether to wrap around boundaries for topology.
    pub wrap_around: bool,
}

impl Default for SpatialPoolerParams {
    fn default() -> Self {
        Self {
            input_dimensions: vec![100],
            column_dimensions: vec![2048],
            potential_radius: 16,
            potential_pct: 0.5,
            global_inhibition: true,
            local_area_density: 0.05,
            num_active_columns_per_inh_area: 0,
            stimulus_threshold: 0,
            syn_perm_inactive_dec: 0.008,
            syn_perm_active_inc: 0.05,
            syn_perm_connected: 0.1,
            min_pct_overlap_duty_cycles: 0.001,
            duty_cycle_period: 1000,
            boost_strength: 0.0,
            seed: 1,
            sp_verbosity: 0,
            wrap_around: true,
        }
    }
}

/// The Spatial Pooler algorithm.
///
/// The Spatial Pooler creates sparse distributed representations of input patterns.
/// It learns stable representations by adjusting synaptic permanences and using
/// competitive inhibition.
///
/// # Example
///
/// ```rust
/// use mokosh::algorithms::{SpatialPooler, SpatialPoolerParams};
/// use mokosh::types::Sdr;
///
/// let mut sp = SpatialPooler::new(SpatialPoolerParams {
///     input_dimensions: vec![100],
///     column_dimensions: vec![200],
///     ..Default::default()
/// }).unwrap();
///
/// let mut input = Sdr::new(&[100]);
/// let mut output = Sdr::new(&[200]);
///
/// input.set_sparse(&[1, 5, 10, 20, 30]).unwrap();
/// sp.compute(&input, true, &mut output);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SpatialPooler {
    // Configuration
    input_dimensions: Vec<UInt>,
    column_dimensions: Vec<UInt>,
    num_inputs: usize,
    num_columns: usize,
    potential_radius: UInt,
    potential_pct: Real,
    global_inhibition: bool,
    local_area_density: Real,
    num_active_columns_per_inh_area: UInt,
    stimulus_threshold: UInt,
    inhibition_radius: UInt,
    duty_cycle_period: UInt,
    boost_strength: Real,
    wrap_around: bool,
    update_period: UInt,
    sp_verbosity: UInt,

    // Permanence parameters
    syn_perm_inactive_dec: Permanence,
    syn_perm_active_inc: Permanence,
    syn_perm_below_stimulus_inc: Permanence,
    syn_perm_connected: Permanence,
    min_pct_overlap_duty_cycles: Real,
    init_connected_pct: Real,

    // State
    boost_factors: Vec<Real>,
    overlap_duty_cycles: Vec<Real>,
    active_duty_cycles: Vec<Real>,
    min_overlap_duty_cycles: Vec<Real>,
    min_active_duty_cycles: Vec<Real>,
    boosted_overlaps: Vec<Real>,

    // Synaptic connections (one segment per column)
    connections: Connections,

    // Iteration counters
    iteration_num: UInt,
    iteration_learn_num: UInt,

    // Cached neighbor map for local inhibition
    #[cfg_attr(feature = "serde", serde(skip))]
    neighbor_map: Neighborhood,

    // RNG
    rng: Random,

    version: UInt,
}

impl SpatialPooler {
    /// Creates a new Spatial Pooler with the given parameters.
    pub fn new(params: SpatialPoolerParams) -> Result<Self> {
        // Validate parameters
        if params.input_dimensions.is_empty() {
            return Err(MokoshError::InvalidParameter {
                name: "input_dimensions",
                message: "Cannot be empty".to_string(),
            });
        }
        if params.column_dimensions.is_empty() {
            return Err(MokoshError::InvalidParameter {
                name: "column_dimensions",
                message: "Cannot be empty".to_string(),
            });
        }
        if params.potential_pct <= 0.0 || params.potential_pct > 1.0 {
            return Err(MokoshError::InvalidParameter {
                name: "potential_pct",
                message: "Must be in range (0, 1]".to_string(),
            });
        }
        if params.local_area_density <= 0.0 || params.local_area_density > 0.5 {
            return Err(MokoshError::InvalidParameter {
                name: "local_area_density",
                message: "Must be in range (0, 0.5]".to_string(),
            });
        }

        let num_inputs = Topology::num_elements(&params.input_dimensions);
        let num_columns = Topology::num_elements(&params.column_dimensions);

        let mut sp = Self {
            input_dimensions: params.input_dimensions.clone(),
            column_dimensions: params.column_dimensions.clone(),
            num_inputs,
            num_columns,
            potential_radius: params.potential_radius,
            potential_pct: params.potential_pct,
            global_inhibition: params.global_inhibition,
            local_area_density: params.local_area_density,
            num_active_columns_per_inh_area: params.num_active_columns_per_inh_area,
            stimulus_threshold: params.stimulus_threshold,
            inhibition_radius: 0,
            duty_cycle_period: params.duty_cycle_period,
            boost_strength: params.boost_strength,
            wrap_around: params.wrap_around,
            update_period: 50,
            sp_verbosity: params.sp_verbosity,

            syn_perm_inactive_dec: params.syn_perm_inactive_dec,
            syn_perm_active_inc: params.syn_perm_active_inc,
            syn_perm_below_stimulus_inc: params.syn_perm_connected / 10.0,
            syn_perm_connected: params.syn_perm_connected,
            min_pct_overlap_duty_cycles: params.min_pct_overlap_duty_cycles,
            init_connected_pct: 0.5,

            boost_factors: vec![1.0; num_columns],
            overlap_duty_cycles: vec![0.0; num_columns],
            active_duty_cycles: vec![0.0; num_columns],
            min_overlap_duty_cycles: vec![0.0; num_columns],
            min_active_duty_cycles: vec![0.0; num_columns],
            boosted_overlaps: vec![0.0; num_columns],

            connections: Connections::new(ConnectionsParams {
                num_cells: num_columns as CellIdx,
                connected_threshold: params.syn_perm_connected,
                timeseries: false,
            }),

            iteration_num: 0,
            iteration_learn_num: 0,

            neighbor_map: Neighborhood::new(),
            rng: Random::new(params.seed),
            version: 2,
        };

        // Initialize columns
        sp.initialize_columns()?;

        // Initialize inhibition radius
        sp.update_inhibition_radius();

        // Initialize neighbor map for local inhibition
        if !sp.global_inhibition {
            sp.neighbor_map = Neighborhood::compute_all(
                &sp.column_dimensions,
                sp.inhibition_radius,
                if sp.wrap_around {
                    WrappingMode::Wrap
                } else {
                    WrappingMode::NoWrap
                },
                true,
            );
        }

        Ok(sp)
    }

    /// Initializes all columns with random potential pools and permanences.
    fn initialize_columns(&mut self) -> Result<()> {
        for column in 0..self.num_columns {
            // Map column to input space
            let potential = self.init_map_potential(column as CellIdx);

            // Initialize permanences
            let permanences = self.init_permanences(&potential);

            // Create segment for this column
            let segment = self.connections.create_segment(column as CellIdx, None);

            // Create synapses
            for (&input, &perm) in potential.iter().zip(&permanences) {
                self.connections
                    .create_synapse(segment, input as CellIdx, perm);
            }
        }

        Ok(())
    }

    /// Maps a column to its potential pool of inputs.
    fn init_map_potential(&mut self, column: CellIdx) -> Vec<UInt> {
        let center_input = Topology::map_column_to_input(
            column as usize,
            &self.column_dimensions,
            &self.input_dimensions,
        );

        let wrap = if self.wrap_around {
            WrappingMode::Wrap
        } else {
            WrappingMode::NoWrap
        };

        let neighborhood = Topology::neighborhood(
            center_input,
            &self.input_dimensions,
            self.potential_radius,
            wrap,
            true,
        );

        // Sample potential_pct of the neighborhood
        let num_potential = ((neighborhood.len() as Real) * self.potential_pct).round() as usize;
        let num_potential = num_potential.max(1);

        let sampled = self.rng.sample(neighborhood, num_potential);
        sampled.into_iter().map(|i| i as UInt).collect()
    }

    /// Initializes permanences for a potential pool.
    fn init_permanences(&mut self, potential: &[UInt]) -> Vec<Permanence> {
        potential
            .iter()
            .map(|_| {
                if self.rng.get_real64() < self.init_connected_pct as f64 {
                    self.init_perm_connected()
                } else {
                    self.init_perm_non_connected()
                }
            })
            .collect()
    }

    /// Returns a random permanence for a connected synapse.
    fn init_perm_connected(&mut self) -> Permanence {
        let p = self.syn_perm_connected
            + (self.rng.get_real32() * self.syn_perm_active_inc / 4.0);
        p.min(MAX_PERMANENCE)
    }

    /// Returns a random permanence for a non-connected synapse.
    fn init_perm_non_connected(&mut self) -> Permanence {
        let p = self.syn_perm_connected * self.rng.get_real32();
        p.max(MIN_PERMANENCE)
    }

    /// The main compute method.
    ///
    /// Takes an input SDR and produces an output SDR of active columns.
    /// If learning is enabled, also updates permanences and duty cycles.
    pub fn compute(&mut self, input: &Sdr, learn: bool, output: &mut Sdr) -> Vec<SynapseIdx> {
        self.update_bookkeeping_vars(learn);

        // Calculate overlaps
        let overlaps = self.calculate_overlaps(input);

        // Apply boosting
        self.boost_overlaps(&overlaps, &mut self.boosted_overlaps.clone());
        let boosted = self.boosted_overlaps.clone();

        // Inhibition
        let active_columns = self.inhibit_columns(&boosted);

        // Set output SDR
        let sparse: Vec<u32> = active_columns.iter().map(|&c| c as u32).collect();
        output.set_sparse_unchecked(sparse);

        // Learning
        if learn {
            self.adapt_synapses(input, output);
            self.update_duty_cycles(&overlaps, output);
            self.bump_up_weak_columns();
            self.update_boost_factors();

            if self.is_update_round() {
                self.update_inhibition_radius();
                self.update_min_duty_cycles();
            }
        }

        overlaps
    }

    /// Calculates overlap scores for all columns.
    fn calculate_overlaps(&self, input: &Sdr) -> Vec<SynapseIdx> {
        let active_inputs: Vec<CellIdx> = input.get_sparse().iter().map(|&i| i as CellIdx).collect();

        let mut overlaps = vec![0u16; self.num_columns];

        for &input_idx in &active_inputs {
            // Find all synapses from this input
            let synapses = self.connections.synapses_for_presynaptic_cell(input_idx);

            for synapse in synapses {
                let synapse_data = self.connections.data_for_synapse(synapse);
                if synapse_data.permanence >= self.syn_perm_connected {
                    let segment = synapse_data.segment;
                    let column = self.connections.cell_for_segment(segment);
                    overlaps[column as usize] += 1;
                }
            }
        }

        overlaps
    }

    /// Applies boost factors to overlap scores.
    fn boost_overlaps(&self, overlaps: &[SynapseIdx], boosted: &mut Vec<Real>) {
        boosted.clear();
        boosted.extend(
            overlaps
                .iter()
                .zip(&self.boost_factors)
                .map(|(&o, &b)| o as Real * b),
        );
    }

    /// Performs inhibition to select active columns.
    fn inhibit_columns(&self, overlaps: &[Real]) -> Vec<CellIdx> {
        let density = if self.num_active_columns_per_inh_area > 0 {
            // Calculate density from num_active_columns_per_inh_area
            let inhibition_area = if self.global_inhibition {
                self.num_columns
            } else {
                let radius = self.inhibition_radius as usize;
                let mut area = 1;
                for &dim in &self.column_dimensions {
                    area *= (2 * radius + 1).min(dim as usize);
                }
                area
            };
            (self.num_active_columns_per_inh_area as Real / inhibition_area as Real)
                .min(0.5)
        } else {
            self.local_area_density
        };

        if self.global_inhibition {
            self.inhibit_columns_global(overlaps, density)
        } else {
            self.inhibit_columns_local(overlaps, density)
        }
    }

    /// Global inhibition: select top columns from entire region.
    fn inhibit_columns_global(&self, overlaps: &[Real], density: Real) -> Vec<CellIdx> {
        let num_active = ((self.num_columns as Real) * density).round() as usize;
        let num_active = num_active.max(1).min(self.num_columns);

        // Get columns sorted by overlap (descending)
        let mut columns: Vec<(CellIdx, Real)> = overlaps
            .iter()
            .enumerate()
            .map(|(i, &o)| (i as CellIdx, o))
            .collect();

        columns.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Select top columns above stimulus threshold
        columns
            .into_iter()
            .take(num_active)
            .filter(|(_, overlap)| *overlap >= self.stimulus_threshold as Real)
            .map(|(col, _)| col)
            .collect()
    }

    /// Local inhibition: compete within neighborhoods.
    fn inhibit_columns_local(&self, overlaps: &[Real], density: Real) -> Vec<CellIdx> {
        let mut active = Vec::new();

        for column in 0..self.num_columns {
            let overlap = overlaps[column];

            if overlap < self.stimulus_threshold as Real {
                continue;
            }

            // Get neighbors
            let neighbors = match self.neighbor_map.get(column) {
                Some(n) => n,
                None => continue,
            };

            // Count neighbors with higher overlap
            let num_higher: usize = neighbors
                .iter()
                .filter(|&&n| overlaps[n] > overlap)
                .count();

            // Calculate max active in neighborhood
            let max_active = ((neighbors.len() + 1) as Real * density).ceil() as usize;

            if num_higher < max_active {
                active.push(column as CellIdx);
            }
        }

        active
    }

    /// Adapts synapse permanences based on input and active columns.
    fn adapt_synapses(&mut self, input: &Sdr, active: &Sdr) {
        let active_inputs: std::collections::HashSet<CellIdx> =
            input.get_sparse().iter().map(|&i| i as CellIdx).collect();

        for &column in &active.get_sparse() {
            let segment = self.connections.segments_for_cell(column as CellIdx)[0];

            let synapses: Vec<_> = self
                .connections
                .synapses_for_segment(segment)
                .iter()
                .copied()
                .collect();

            for synapse in synapses {
                let synapse_data = self.connections.data_for_synapse(synapse);
                let presynaptic = synapse_data.presynaptic_cell;
                let old_perm = synapse_data.permanence;

                let new_perm = if active_inputs.contains(&presynaptic) {
                    (old_perm + self.syn_perm_active_inc).min(MAX_PERMANENCE)
                } else {
                    (old_perm - self.syn_perm_inactive_dec).max(MIN_PERMANENCE)
                };

                if (new_perm - old_perm).abs() > 1e-6 {
                    self.connections.update_synapse_permanence(synapse, new_perm);
                }
            }
        }
    }

    /// Updates duty cycles.
    fn update_duty_cycles(&mut self, overlaps: &[SynapseIdx], active: &Sdr) {
        let period = self.duty_cycle_period as Real;

        // Update overlap duty cycles
        for (i, &overlap) in overlaps.iter().enumerate() {
            let value = if overlap > 0 { 1.0 } else { 0.0 };
            self.overlap_duty_cycles[i] =
                ((period - 1.0) * self.overlap_duty_cycles[i] + value) / period;
        }

        // Update active duty cycles
        let active_set: std::collections::HashSet<u32> = active.get_sparse().into_iter().collect();
        for i in 0..self.num_columns {
            let value = if active_set.contains(&(i as u32)) {
                1.0
            } else {
                0.0
            };
            self.active_duty_cycles[i] =
                ((period - 1.0) * self.active_duty_cycles[i] + value) / period;
        }
    }

    /// Increases permanences for columns with low overlap duty cycle.
    fn bump_up_weak_columns(&mut self) {
        for column in 0..self.num_columns {
            if self.overlap_duty_cycles[column] < self.min_overlap_duty_cycles[column] {
                let segment = self.connections.segments_for_cell(column as CellIdx)[0];
                self.connections
                    .bump_segment(segment, self.syn_perm_below_stimulus_inc);
            }
        }
    }

    /// Updates boost factors based on active duty cycles.
    fn update_boost_factors(&mut self) {
        if self.boost_strength <= 0.0 {
            return;
        }

        let target_density = self.local_area_density;

        if self.global_inhibition {
            for i in 0..self.num_columns {
                self.boost_factors[i] =
                    (self.boost_strength * (target_density - self.active_duty_cycles[i])).exp();
            }
        } else {
            // Local boosting based on neighborhood average
            for column in 0..self.num_columns {
                let neighbors = match self.neighbor_map.get(column) {
                    Some(n) => n,
                    None => continue,
                };

                let neighbor_avg: Real = neighbors
                    .iter()
                    .map(|&n| self.active_duty_cycles[n])
                    .sum::<Real>()
                    / neighbors.len().max(1) as Real;

                self.boost_factors[column] =
                    (self.boost_strength * (neighbor_avg - self.active_duty_cycles[column])).exp();
            }
        }
    }

    /// Updates the inhibition radius based on average receptive field size.
    fn update_inhibition_radius(&mut self) {
        if self.global_inhibition {
            let max_dim = *self.column_dimensions.iter().max().unwrap_or(&1);
            self.inhibition_radius = max_dim;
            return;
        }

        // Calculate average connected span
        let mut total_span = 0.0;
        for column in 0..self.num_columns {
            total_span += self.avg_connected_span_for_column(column as CellIdx);
        }
        let avg_span = total_span / self.num_columns as Real;

        // Calculate average columns per input
        let avg_columns_per_input = self.avg_columns_per_input();

        self.inhibition_radius =
            ((avg_span * avg_columns_per_input - 1.0) / 2.0).round() as UInt;
        self.inhibition_radius = self.inhibition_radius.max(1);

        // Update neighbor map
        self.neighbor_map = Neighborhood::compute_all(
            &self.column_dimensions,
            self.inhibition_radius,
            if self.wrap_around {
                WrappingMode::Wrap
            } else {
                WrappingMode::NoWrap
            },
            true,
        );
    }

    /// Calculates average connected span for a column.
    fn avg_connected_span_for_column(&self, column: CellIdx) -> Real {
        let segment = self.connections.segments_for_cell(column)[0];
        let synapses = self.connections.synapses_for_segment(segment);

        if synapses.is_empty() {
            return 0.0;
        }

        let connected: Vec<CellIdx> = synapses
            .iter()
            .filter_map(|&s| {
                let data = self.connections.data_for_synapse(s);
                if data.permanence >= self.syn_perm_connected {
                    Some(data.presynaptic_cell)
                } else {
                    None
                }
            })
            .collect();

        if connected.is_empty() {
            return 0.0;
        }

        // Calculate span in each dimension
        let num_dims = self.input_dimensions.len();
        let mut total_span = 0.0;

        for dim in 0..num_dims {
            let dim_size = self.input_dimensions[dim] as usize;
            let coords: Vec<UInt> = connected
                .iter()
                .map(|&c| Topology::index_to_coordinates(c as usize, &self.input_dimensions)[dim])
                .collect();

            if let (Some(&min), Some(&max)) = (coords.iter().min(), coords.iter().max()) {
                total_span += (max - min + 1) as Real;
            }
        }

        total_span / num_dims as Real
    }

    /// Calculates average columns per input.
    fn avg_columns_per_input(&self) -> Real {
        let num_dims = self.column_dimensions.len().max(self.input_dimensions.len());
        let mut ratio = 1.0;

        for dim in 0..num_dims {
            let col_dim = self.column_dimensions.get(dim).copied().unwrap_or(1) as Real;
            let inp_dim = self.input_dimensions.get(dim).copied().unwrap_or(1) as Real;
            ratio *= col_dim / inp_dim;
        }

        ratio
    }

    /// Updates minimum duty cycles.
    fn update_min_duty_cycles(&mut self) {
        if self.global_inhibition {
            self.update_min_duty_cycles_global();
        } else {
            self.update_min_duty_cycles_local();
        }
    }

    fn update_min_duty_cycles_global(&mut self) {
        let max_overlap_duty = self
            .overlap_duty_cycles
            .iter()
            .copied()
            .fold(0.0_f32, Real::max);
        let min_overlap = self.min_pct_overlap_duty_cycles * max_overlap_duty;

        for i in 0..self.num_columns {
            self.min_overlap_duty_cycles[i] = min_overlap;
        }
    }

    fn update_min_duty_cycles_local(&mut self) {
        for column in 0..self.num_columns {
            let neighbors = match self.neighbor_map.get(column) {
                Some(n) => n,
                None => continue,
            };

            let max_neighbor_overlap = neighbors
                .iter()
                .map(|&n| self.overlap_duty_cycles[n])
                .fold(0.0_f32, Real::max);

            self.min_overlap_duty_cycles[column] =
                self.min_pct_overlap_duty_cycles * max_neighbor_overlap;
        }
    }

    fn update_bookkeeping_vars(&mut self, learn: bool) {
        self.iteration_num += 1;
        if learn {
            self.iteration_learn_num += 1;
        }
    }

    fn is_update_round(&self) -> bool {
        self.iteration_num % self.update_period == 0
    }

    // ========================================================================
    // Getters
    // ========================================================================

    /// Returns the input dimensions.
    pub fn input_dimensions(&self) -> &[UInt] {
        &self.input_dimensions
    }

    /// Returns the column dimensions.
    pub fn column_dimensions(&self) -> &[UInt] {
        &self.column_dimensions
    }

    /// Returns the number of inputs.
    pub fn num_inputs(&self) -> usize {
        self.num_inputs
    }

    /// Returns the number of columns.
    pub fn num_columns(&self) -> usize {
        self.num_columns
    }

    /// Returns the potential radius.
    pub fn potential_radius(&self) -> UInt {
        self.potential_radius
    }

    /// Returns the potential percent.
    pub fn potential_pct(&self) -> Real {
        self.potential_pct
    }

    /// Returns whether global inhibition is enabled.
    pub fn global_inhibition(&self) -> bool {
        self.global_inhibition
    }

    /// Returns the local area density.
    pub fn local_area_density(&self) -> Real {
        self.local_area_density
    }

    /// Returns the stimulus threshold.
    pub fn stimulus_threshold(&self) -> UInt {
        self.stimulus_threshold
    }

    /// Returns the inhibition radius.
    pub fn inhibition_radius(&self) -> UInt {
        self.inhibition_radius
    }

    /// Returns the duty cycle period.
    pub fn duty_cycle_period(&self) -> UInt {
        self.duty_cycle_period
    }

    /// Returns the boost strength.
    pub fn boost_strength(&self) -> Real {
        self.boost_strength
    }

    /// Returns the current iteration number.
    pub fn iteration_num(&self) -> UInt {
        self.iteration_num
    }

    /// Returns the current learning iteration number.
    pub fn iteration_learn_num(&self) -> UInt {
        self.iteration_learn_num
    }

    /// Returns the synapse permanence connected threshold.
    pub fn syn_perm_connected(&self) -> Permanence {
        self.syn_perm_connected
    }

    /// Returns the synapse active increment.
    pub fn syn_perm_active_inc(&self) -> Permanence {
        self.syn_perm_active_inc
    }

    /// Returns the synapse inactive decrement.
    pub fn syn_perm_inactive_dec(&self) -> Permanence {
        self.syn_perm_inactive_dec
    }

    /// Returns the boost factors.
    pub fn boost_factors(&self) -> &[Real] {
        &self.boost_factors
    }

    /// Returns the overlap duty cycles.
    pub fn overlap_duty_cycles(&self) -> &[Real] {
        &self.overlap_duty_cycles
    }

    /// Returns the active duty cycles.
    pub fn active_duty_cycles(&self) -> &[Real] {
        &self.active_duty_cycles
    }

    /// Returns a reference to the connections.
    pub fn connections(&self) -> &Connections {
        &self.connections
    }

    /// Returns the boosted overlaps from the last compute.
    pub fn boosted_overlaps(&self) -> &[Real] {
        &self.boosted_overlaps
    }

    /// Gets permanences for a column.
    pub fn get_permanences(&self, column: UInt) -> Vec<(CellIdx, Permanence)> {
        let segment = self.connections.segments_for_cell(column as CellIdx)[0];
        self.connections
            .synapses_for_segment(segment)
            .iter()
            .map(|&s| {
                let data = self.connections.data_for_synapse(s);
                (data.presynaptic_cell, data.permanence)
            })
            .collect()
    }

    /// Gets connected counts for all columns.
    pub fn connected_counts(&self) -> Vec<UInt> {
        (0..self.num_columns)
            .map(|col| {
                let segment = self.connections.segments_for_cell(col as CellIdx)[0];
                self.connections.data_for_segment(segment).num_connected as UInt
            })
            .collect()
    }
}

impl PartialEq for SpatialPooler {
    fn eq(&self, other: &Self) -> bool {
        self.input_dimensions == other.input_dimensions
            && self.column_dimensions == other.column_dimensions
            && self.potential_radius == other.potential_radius
            && (self.potential_pct - other.potential_pct).abs() < 1e-6
            && self.global_inhibition == other.global_inhibition
            && (self.local_area_density - other.local_area_density).abs() < 1e-6
            && self.stimulus_threshold == other.stimulus_threshold
            && self.iteration_num == other.iteration_num
            && self.connections == other.connections
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_spatial_pooler() {
        let sp = SpatialPooler::new(SpatialPoolerParams {
            input_dimensions: vec![100],
            column_dimensions: vec![200],
            ..Default::default()
        })
        .unwrap();

        assert_eq!(sp.num_inputs(), 100);
        assert_eq!(sp.num_columns(), 200);
    }

    #[test]
    fn test_compute_basic() {
        let mut sp = SpatialPooler::new(SpatialPoolerParams {
            input_dimensions: vec![100],
            column_dimensions: vec![200],
            potential_radius: 50,
            global_inhibition: true,
            local_area_density: 0.1,
            ..Default::default()
        })
        .unwrap();

        let mut input = Sdr::new(&[100]);
        let mut output = Sdr::new(&[200]);

        input.set_sparse(&[1, 5, 10, 20, 30]).unwrap();
        sp.compute(&input, true, &mut output);

        // Should have some active columns
        assert!(output.get_sum() > 0);
        assert!(output.get_sum() <= 20); // At most 10% of 200
    }

    #[test]
    fn test_learning_changes_permanences() {
        let mut sp = SpatialPooler::new(SpatialPoolerParams {
            input_dimensions: vec![50],
            column_dimensions: vec![100],
            potential_radius: 25,
            global_inhibition: true,
            ..Default::default()
        })
        .unwrap();

        let mut input = Sdr::new(&[50]);
        let mut output = Sdr::new(&[100]);

        input.set_sparse(&[0, 1, 2, 3, 4]).unwrap();

        // Get initial permanences for first column
        let initial_perms = sp.get_permanences(0);

        // Run several iterations
        for _ in 0..100 {
            sp.compute(&input, true, &mut output);
        }

        // Permanences should have changed
        let final_perms = sp.get_permanences(0);
        assert_ne!(initial_perms, final_perms);
    }

    #[test]
    fn test_sparsity() {
        let mut sp = SpatialPooler::new(SpatialPoolerParams {
            input_dimensions: vec![100],
            column_dimensions: vec![1000],
            potential_radius: 50,
            global_inhibition: true,
            local_area_density: 0.02,
            ..Default::default()
        })
        .unwrap();

        let mut input = Sdr::new(&[100]);
        let mut output = Sdr::new(&[1000]);

        input.set_sparse(&[10, 20, 30, 40, 50]).unwrap();
        sp.compute(&input, false, &mut output);

        // Should be approximately 2% active
        let sparsity = output.get_sparsity();
        assert!(sparsity > 0.01 && sparsity < 0.05);
    }

    #[test]
    fn test_stability() {
        let mut sp = SpatialPooler::new(SpatialPoolerParams {
            input_dimensions: vec![100],
            column_dimensions: vec![200],
            potential_radius: 50,
            global_inhibition: true,
            boost_strength: 0.0, // No boosting for stability test
            ..Default::default()
        })
        .unwrap();

        let mut input = Sdr::new(&[100]);
        let mut output1 = Sdr::new(&[200]);
        let mut output2 = Sdr::new(&[200]);

        input.set_sparse(&[10, 20, 30]).unwrap();

        // Train
        for _ in 0..100 {
            sp.compute(&input, true, &mut output1);
        }

        // Same input should produce same (or very similar) output
        sp.compute(&input, false, &mut output1);
        sp.compute(&input, false, &mut output2);

        assert_eq!(output1.get_sparse(), output2.get_sparse());
    }

    #[test]
    fn test_invalid_params() {
        let result = SpatialPooler::new(SpatialPoolerParams {
            input_dimensions: vec![],
            column_dimensions: vec![100],
            ..Default::default()
        });
        assert!(result.is_err());

        let result = SpatialPooler::new(SpatialPoolerParams {
            input_dimensions: vec![100],
            column_dimensions: vec![100],
            potential_pct: 1.5,
            ..Default::default()
        });
        assert!(result.is_err());
    }
}
