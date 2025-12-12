//! HTM Fingerprint Visualization
//!
//! Visualizes what the Hierarchical Temporal Memory actually learns and predicts.
//! Shows the internal states of Spatial Pooler and Temporal Memory as 256x256 images.
//!
//! ## Visualization Layout
//!
//! The TM has 1024 columns × 64 cells = 65536 cells = 256×256 pixels
//!
//! Layout: Grid-interleaved for visual interest
//! - 32×32 grid of "mini-columns" (each 8×8 pixels)
//! - Each mini-column shows 64 cells arranged in an 8×8 pattern
//! - Columns with similar encoder inputs cluster together spatially
//!
//! ## Color Coding for TM States
//!
//! - Black: Inactive, not predicted (quiet)
//! - Blue: Predicted but not active (expected something that didn't happen)
//! - Green: Predicted AND active (correct prediction!)
//! - Red/Orange: Active but NOT predicted (SURPRISE = ANOMALY!)
//!
//! Run with: cargo run --example htm_fingerprints --release

mod ecg_utils;

use ecg_utils::{ArrhythmiaConfig, ArrhythmiaType, EcgConfig, EcgGenerator};
use image::{Rgb, RgbImage};
use mokosh::algorithms::{TemporalMemory, TemporalMemoryParams, AnomalyMode};
use mokosh::encoders::{EcgEncoder, EcgEncoderParams, Encoder};
use mokosh::types::{Sdr, Real};
use std::collections::HashSet;
use std::fs;
use std::path::Path;

// ============================================================================
// Constants
// ============================================================================

const IMG_SIZE: usize = 256;
// Use 1024 columns × 64 cells = 65536 cells = 256×256 pixels (perfect fit!)
const NUM_COLUMNS: usize = 1024;
const CELLS_PER_COLUMN: usize = 64;
const TOTAL_CELLS: usize = NUM_COLUMNS * CELLS_PER_COLUMN; // 65536

// ============================================================================
// Cell Index Mapping
// ============================================================================

/// Maps a cell index to pixel coordinates using a grid-interleaved layout
///
/// Layout for 1024 columns × 64 cells = 65536 cells = 256×256 pixels:
/// - Columns are arranged in a 32×32 grid of "mini-columns"
/// - Each mini-column is 8×8 pixels (64 cells arranged as 8×8)
/// - This creates a more distributed, interesting visual pattern
fn cell_to_pixel(cell_index: usize) -> (usize, usize) {
    let column = cell_index / CELLS_PER_COLUMN;
    let cell_in_column = cell_index % CELLS_PER_COLUMN;

    // Arrange 1024 columns in a 32×32 grid
    let col_grid_x = column % 32;
    let col_grid_y = column / 32;

    // Each cell within column maps to 8×8 mini-grid
    let cell_x = cell_in_column % 8;
    let cell_y = cell_in_column / 8;

    // Final pixel position
    let x = col_grid_x * 8 + cell_x;
    let y = col_grid_y * 8 + cell_y;

    (x, y)
}

/// Plot a single pixel for a cell
fn plot_cell(img: &mut RgbImage, cell_index: usize, color: Rgb<u8>) {
    let (x, y) = cell_to_pixel(cell_index);
    if x < IMG_SIZE && y < IMG_SIZE {
        img.put_pixel(x as u32, y as u32, color);
    }
}

/// Maps column index to pixel X coordinate (for SP visualization)
fn column_to_x(column: usize) -> usize {
    column % 256
}

/// Maps column index to pixel Y band (for SP visualization)
fn column_to_y_band(column: usize) -> usize {
    (column / 256) * 32
}

// ============================================================================
// HTM State Capture
// ============================================================================

/// Captured state from one HTM processing step
#[derive(Clone)]
struct HtmState {
    /// Active columns from Spatial Pooler
    active_columns: Vec<u32>,
    /// Active cells from Temporal Memory
    active_cells: Vec<usize>,
    /// Predicted cells (from previous step)
    predicted_cells: Vec<usize>,
    /// Winner cells
    winner_cells: Vec<usize>,
    /// Anomaly score
    anomaly: f32,
}

/// HTM system with state capture
/// Using direct encoder-to-TM mode (bypassing SP due to SP issues)
struct HtmSystem {
    encoder: EcgEncoder,
    tm: TemporalMemory,
    window_size: usize,
    /// Last predicted cells (for next step comparison)
    last_predicted: HashSet<usize>,
    /// Encoder size for SDR
    encoder_size: usize,
}

impl HtmSystem {
    fn new(window_size: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let encoder = EcgEncoder::new(EcgEncoderParams {
            window_size,
            bits_per_sample: 15,
            active_per_sample: 3,
            stats_bits: 60,
            stats_active: 12,
            amplitude_range: (-2.5, 2.5),
        })?;

        let encoder_size = Encoder::<Vec<Real>>::size(&encoder);

        // Direct encoder-to-TM: use encoder bits as columns
        // With 65k encoder bits, map first NUM_COLUMNS to TM columns
        let tm = TemporalMemory::new(TemporalMemoryParams {
            column_dimensions: vec![NUM_COLUMNS as u32],
            cells_per_column: CELLS_PER_COLUMN as u32,
            activation_threshold: 8,       // Lower for sparser patterns
            initial_permanence: 0.21,
            connected_permanence: 0.5,
            min_threshold: 6,
            max_new_synapse_count: 15,
            permanence_increment: 0.1,
            permanence_decrement: 0.1,
            predicted_segment_decrement: 0.0,
            max_segments_per_cell: 64,
            max_synapses_per_segment: 32,
            anomaly_mode: AnomalyMode::Raw,
            seed: 42,
            ..Default::default()
        })?;

        Ok(Self {
            encoder,
            tm,
            window_size,
            last_predicted: HashSet::new(),
            encoder_size,
        })
    }

    /// Process a window and capture HTM state
    /// Maps encoder bits directly to TM columns (bypassing SP)
    fn process(&mut self, samples: &[Real], learn: bool) -> Result<HtmState, Box<dyn std::error::Error>> {
        // Encode
        let input_sdr = self.encoder.encode_to_sdr(samples)?;
        let encoder_bits = input_sdr.get_sparse();

        // Map encoder bits to columns using a spreading hash
        // This distributes encoder output more evenly across column space
        let mut active_columns: Vec<u32> = encoder_bits
            .iter()
            .map(|&bit| {
                // Simple hash to spread bits more evenly
                let h = bit.wrapping_mul(2654435761) ^ (bit >> 16);
                h % NUM_COLUMNS as u32
            })
            .collect::<HashSet<_>>()  // Dedupe
            .into_iter()
            .collect();
        active_columns.sort();  // SDR requires sorted indices

        // Create SDR for TM - dimensions must match TM's column_dimensions
        let mut active_columns_sdr = Sdr::new(&[NUM_COLUMNS as u32]);
        active_columns_sdr.set_sparse(&active_columns)?;

        // Store predicted cells from BEFORE this compute (for comparison)
        let predicted_before: Vec<usize> = self.last_predicted.iter().cloned().collect();

        // Temporal Memory
        self.tm.compute(&active_columns_sdr, learn);

        // Capture active and winner cells
        let active_cells: Vec<usize> = self.tm.active_cells().iter().map(|&c| c as usize).collect();
        let winner_cells: Vec<usize> = self.tm.winner_cells().iter().map(|&c| c as usize).collect();
        let anomaly = self.tm.anomaly();

        // Get predicted cells for NEXT step
        let predicted_cells: Vec<usize> = self.tm.predictive_cells().iter().map(|&c| c as usize).collect();
        self.last_predicted = predicted_cells.iter().cloned().collect();

        Ok(HtmState {
            active_columns,
            active_cells,
            predicted_cells: predicted_before,
            winner_cells,
            anomaly,
        })
    }

    fn reset(&mut self) {
        self.tm.reset();
        self.last_predicted.clear();
    }

    fn window_size(&self) -> usize {
        self.window_size
    }
}

// ============================================================================
// Visualization
// ============================================================================

/// Render SP column activations as a 256x256 image
fn render_sp_columns(active_columns: &[u32]) -> RgbImage {
    let mut img = RgbImage::new(IMG_SIZE as u32, IMG_SIZE as u32);

    // Dark background
    for pixel in img.pixels_mut() {
        *pixel = Rgb([15, 15, 25]);
    }

    // Mark active columns
    let active_set: HashSet<u32> = active_columns.iter().cloned().collect();

    for col in 0..NUM_COLUMNS {
        let x = column_to_x(col);
        let y_base = column_to_y_band(col);

        if active_set.contains(&(col as u32)) {
            // Active column - fill its vertical strip
            for dy in 0..32 {
                let y = y_base + dy;
                // Cyan/teal for active
                img.put_pixel(x as u32, y as u32, Rgb([50, 200, 220]));
            }
        }
    }

    img
}

/// Render TM cell states with prediction/anomaly coloring
fn render_tm_cells(state: &HtmState) -> RgbImage {
    let mut img = RgbImage::new(IMG_SIZE as u32, IMG_SIZE as u32);

    // Dark blue-ish background
    for pixel in img.pixels_mut() {
        *pixel = Rgb([8, 12, 20]);
    }

    let active_set: HashSet<usize> = state.active_cells.iter().cloned().collect();
    let predicted_set: HashSet<usize> = state.predicted_cells.iter().cloned().collect();

    // First pass: draw predicted cells (blue glow)
    for &cell in &state.predicted_cells {
        if !active_set.contains(&cell) {
            // Predicted but didn't fire - cyan/blue
            plot_cell(&mut img, cell, Rgb([40, 120, 200]));
        }
    }

    // Second pass: draw active cells (green or orange based on prediction)
    for &cell in &state.active_cells {
        let was_predicted = predicted_set.contains(&cell);
        let color = if was_predicted {
            Rgb([80, 255, 120])   // Bright green - correct prediction!
        } else {
            Rgb([255, 60, 60])    // Bright red - ANOMALY!
        };
        plot_cell(&mut img, cell, color);
    }

    img
}

/// Render anomaly intensity map (red = high anomaly cells)
fn render_anomaly_map(state: &HtmState) -> RgbImage {
    let mut img = RgbImage::new(IMG_SIZE as u32, IMG_SIZE as u32);

    // Dark background
    for pixel in img.pixels_mut() {
        *pixel = Rgb([10, 10, 15]);
    }

    let predicted_set: HashSet<usize> = state.predicted_cells.iter().cloned().collect();

    // Only highlight unpredicted active cells (the anomalies)
    for &cell in &state.active_cells {
        if !predicted_set.contains(&cell) {
            // Bright red/orange for anomaly
            plot_cell(&mut img, cell, Rgb([255, 80, 30]));
        }
    }

    img
}

/// Accumulator for HTM states
struct StateAccumulator {
    /// Count of times each cell was active
    active_counts: Vec<u32>,
    /// Count of times each cell was correctly predicted
    correct_counts: Vec<u32>,
    /// Count of times each cell was an anomaly (active but not predicted)
    anomaly_counts: Vec<u32>,
    /// Total frames
    total: u32,
}

impl StateAccumulator {
    fn new() -> Self {
        Self {
            active_counts: vec![0; TOTAL_CELLS],
            correct_counts: vec![0; TOTAL_CELLS],
            anomaly_counts: vec![0; TOTAL_CELLS],
            total: 0,
        }
    }

    fn add(&mut self, state: &HtmState) {
        let predicted_set: HashSet<usize> = state.predicted_cells.iter().cloned().collect();

        for &cell in &state.active_cells {
            self.active_counts[cell] += 1;

            if predicted_set.contains(&cell) {
                self.correct_counts[cell] += 1;
            } else {
                self.anomaly_counts[cell] += 1;
            }
        }

        self.total += 1;
    }

    fn render_activity(&self) -> RgbImage {
        let mut img = RgbImage::new(IMG_SIZE as u32, IMG_SIZE as u32);

        // Dark background
        for pixel in img.pixels_mut() {
            *pixel = Rgb([10, 10, 15]);
        }

        let max_count = self.active_counts.iter().max().cloned().unwrap_or(1).max(1);

        for cell in 0..TOTAL_CELLS {
            let intensity = self.active_counts[cell] as f32 / max_count as f32;
            if intensity > 0.01 {
                // Plasma-like colormap
                let color = intensity_to_plasma(intensity);
                plot_cell(&mut img, cell, color);
            }
        }

        img
    }

    fn render_anomaly_rate(&self) -> RgbImage {
        let mut img = RgbImage::new(IMG_SIZE as u32, IMG_SIZE as u32);

        // Dark background
        for pixel in img.pixels_mut() {
            *pixel = Rgb([10, 10, 15]);
        }

        for cell in 0..TOTAL_CELLS {
            let total_active = self.active_counts[cell];
            if total_active == 0 {
                continue;
            }

            // Anomaly rate = anomaly_counts / active_counts
            let anomaly_rate = self.anomaly_counts[cell] as f32 / total_active as f32;

            // Red intensity based on anomaly rate
            let r = (anomaly_rate * 255.0) as u8;
            let g = ((1.0 - anomaly_rate) * 100.0) as u8;
            let b = 30;

            plot_cell(&mut img, cell, Rgb([r, g, b]));
        }

        img
    }

    fn render_prediction_accuracy(&self) -> RgbImage {
        let mut img = RgbImage::new(IMG_SIZE as u32, IMG_SIZE as u32);

        // Dark background
        for pixel in img.pixels_mut() {
            *pixel = Rgb([10, 10, 15]);
        }

        for cell in 0..TOTAL_CELLS {
            let total_active = self.active_counts[cell];
            if total_active == 0 {
                continue;
            }

            // Prediction accuracy = correct_counts / active_counts
            let accuracy = self.correct_counts[cell] as f32 / total_active as f32;

            // Green for high accuracy, red for low
            let r = ((1.0 - accuracy) * 200.0) as u8;
            let g = (accuracy * 220.0) as u8;
            let b = 50;

            plot_cell(&mut img, cell, Rgb([r, g, b]));
        }

        img
    }
}

/// Vibrant plasma-style colormap: black → purple → magenta → orange → yellow
fn intensity_to_plasma(v: f32) -> Rgb<u8> {
    let v = v.clamp(0.0, 1.0);
    if v < 0.01 {
        Rgb([8, 12, 20])
    } else if v < 0.2 {
        let t = v / 0.2;
        // Black to deep purple
        Rgb([
            (8.0 + t * 50.0) as u8,
            (12.0 + t * 5.0) as u8,
            (20.0 + t * 80.0) as u8,
        ])
    } else if v < 0.4 {
        let t = (v - 0.2) / 0.2;
        // Deep purple to magenta
        Rgb([
            (58.0 + t * 140.0) as u8,
            (17.0 + t * 20.0) as u8,
            (100.0 + t * 55.0) as u8,
        ])
    } else if v < 0.6 {
        let t = (v - 0.4) / 0.2;
        // Magenta to orange-red
        Rgb([
            (198.0 + t * 55.0) as u8,
            (37.0 + t * 60.0) as u8,
            (155.0 - t * 120.0) as u8,
        ])
    } else if v < 0.8 {
        let t = (v - 0.6) / 0.2;
        // Orange-red to orange
        Rgb([
            (253.0) as u8,
            (97.0 + t * 100.0) as u8,
            (35.0 - t * 15.0) as u8,
        ])
    } else {
        let t = (v - 0.8) / 0.2;
        // Orange to bright yellow
        Rgb([
            255,
            (197.0 + t * 58.0) as u8,
            (20.0 + t * 100.0) as u8,
        ])
    }
}

/// Create a temporal heatmap showing HTM activity over time
/// X = time step, Y = column index (compressed), color = activity/anomaly
fn create_temporal_heatmap(
    states: &[HtmState],
    mode: &str,  // "activity", "anomaly", or "prediction"
) -> RgbImage {
    let width = states.len().min(256) as u32;
    let height = 256u32;
    let mut img = RgbImage::new(width, height);

    // Dark background
    for pixel in img.pixels_mut() {
        *pixel = Rgb([8, 12, 20]);
    }

    // For each time step
    for (t, state) in states.iter().enumerate() {
        if t >= 256 { break; }

        // Count activity per column band (compress 1024 cols → 256 rows)
        let mut band_activity = vec![0u32; 256];
        let mut band_anomaly = vec![0u32; 256];
        let mut band_predicted = vec![0u32; 256];

        let predicted_set: HashSet<usize> = state.predicted_cells.iter().cloned().collect();

        for &cell in &state.active_cells {
            let col = cell / CELLS_PER_COLUMN;
            let band = col * 256 / NUM_COLUMNS;
            band_activity[band] += 1;

            if !predicted_set.contains(&cell) {
                band_anomaly[band] += 1;
            }
        }

        for &cell in &state.predicted_cells {
            let col = cell / CELLS_PER_COLUMN;
            let band = col * 256 / NUM_COLUMNS;
            band_predicted[band] += 1;
        }

        // Draw column
        let max_val = match mode {
            "activity" => band_activity.iter().max().cloned().unwrap_or(1).max(1),
            "anomaly" => band_anomaly.iter().max().cloned().unwrap_or(1).max(1),
            "prediction" => band_predicted.iter().max().cloned().unwrap_or(1).max(1),
            _ => 1,
        };

        for y in 0..256 {
            let val = match mode {
                "activity" => band_activity[y],
                "anomaly" => band_anomaly[y],
                "prediction" => band_predicted[y],
                _ => 0,
            };

            let intensity = val as f32 / max_val as f32;
            let color = if mode == "anomaly" {
                // Red colormap for anomalies
                let r = (intensity * 255.0) as u8;
                let g = (intensity * 40.0) as u8;
                let b = (intensity * 20.0) as u8;
                Rgb([r, g, b])
            } else {
                intensity_to_plasma(intensity)
            };

            img.put_pixel(t as u32, y as u32, color);
        }
    }

    img
}

fn create_side_by_side(images: &[&RgbImage], gap: u32) -> RgbImage {
    let total_width: u32 = images.iter().map(|i| i.width()).sum::<u32>()
        + gap * (images.len() as u32 - 1);
    let height = images.iter().map(|i| i.height()).max().unwrap_or(256);

    let mut combined = RgbImage::new(total_width, height);
    for pixel in combined.pixels_mut() {
        *pixel = Rgb([20, 20, 20]);
    }

    let mut x_offset = 0;
    for img in images {
        for (x, y, pixel) in img.enumerate_pixels() {
            combined.put_pixel(x_offset + x, y, *pixel);
        }
        x_offset += img.width() + gap;
    }

    combined
}

// ============================================================================
// Main
// ============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("========================================");
    println!("  HTM Fingerprint Visualization");
    println!("========================================");
    println!();

    let output_dir = Path::new("fingerprints");
    if !output_dir.exists() {
        fs::create_dir_all(output_dir)?;
    }

    let window_size = 50;  // ~200ms at 250Hz

    println!("HTM Configuration:");
    println!("  Encoder window:     {} samples", window_size);
    println!("  Encoder → TM:       Direct mapping (encoder bits mod {} = columns)", NUM_COLUMNS);
    println!("  Temporal Memory:    {} columns × {} cells = {} total",
             NUM_COLUMNS, CELLS_PER_COLUMN, TOTAL_CELLS);
    println!("  Image size:         {}×{} pixels", IMG_SIZE, IMG_SIZE);
    println!();

    println!("Cell-to-pixel mapping:");
    println!("  32×32 grid of mini-columns (8×8 pixels each)");
    println!("  Each mini-column shows 64 cells in an 8×8 pattern");
    println!();

    // Create HTM system
    let mut htm = HtmSystem::new(window_size)?;

    let ecg_config = EcgConfig {
        sample_rate: 250.0,
        base_heart_rate: 72.0,
        amplitude: 1.0,
        enable_hrv: true,
        hrv_magnitude: 0.05,
        baseline_wander: 0.02,
        noise_amplitude: 0.01,
    };

    // ========================================================================
    // Phase 1: Train on Normal ECG
    // ========================================================================
    println!("Phase 1: Training HTM on normal ECG...");

    let mut normal_gen = EcgGenerator::new(ecg_config.clone(), ArrhythmiaConfig::default());
    normal_gen.rng_state = 12345;

    let training_windows = 500;
    for i in 0..training_windows {
        let window = normal_gen.generate_window(window_size);
        let _ = htm.process(&window, true)?;

        if i % 100 == 0 {
            print!("  {}...", i);
        }
    }
    println!(" done ({} windows)", training_windows);
    println!();

    // ========================================================================
    // Phase 2: Capture Normal ECG States
    // ========================================================================
    println!("Phase 2: Capturing trained HTM states on normal ECG...");

    htm.reset();
    let mut normal_gen = EcgGenerator::new(ecg_config.clone(), ArrhythmiaConfig::default());
    normal_gen.rng_state = 54321;

    let mut normal_accum = StateAccumulator::new();
    let mut normal_states: Vec<HtmState> = Vec::new();
    let mut last_normal_state: Option<HtmState> = None;

    let test_windows = 200;
    for _ in 0..test_windows {
        let window = normal_gen.generate_window(window_size);
        let state = htm.process(&window, false)?;
        normal_accum.add(&state);
        normal_states.push(state.clone());
        last_normal_state = Some(state);
    }
    println!("  Captured {} windows", test_windows);

    // Save normal visualizations
    if let Some(state) = &last_normal_state {
        let sp_img = render_sp_columns(&state.active_columns);
        sp_img.save(output_dir.join("htm_sp_normal.png"))?;

        let tm_img = render_tm_cells(state);
        tm_img.save(output_dir.join("htm_tm_normal.png"))?;

        println!("  Saved: htm_sp_normal.png, htm_tm_normal.png");
        println!("  Last anomaly score: {:.4}", state.anomaly);

        // Debug cell counts
        let predicted_set: HashSet<usize> = state.predicted_cells.iter().cloned().collect();
        let mut correct = 0;
        let mut unpredicted = 0;
        for &cell in &state.active_cells {
            if predicted_set.contains(&cell) {
                correct += 1;
            } else {
                unpredicted += 1;
            }
        }
        println!("  Active cells: {}, Predicted (prior step): {}", state.active_cells.len(), state.predicted_cells.len());
        println!("  Correctly predicted: {}, Unpredicted (anomalies): {}", correct, unpredicted);
        let first_cols: Vec<u32> = state.active_columns.iter().take(5).cloned().collect();
        println!("  Active columns: {} ({:?}...)", state.active_columns.len(), first_cols);
    }

    // Save accumulated normal patterns
    let normal_activity = normal_accum.render_activity();
    normal_activity.save(output_dir.join("htm_normal_activity.png"))?;

    let normal_accuracy = normal_accum.render_prediction_accuracy();
    normal_accuracy.save(output_dir.join("htm_normal_prediction.png"))?;

    // Save temporal heatmaps for normal ECG
    let normal_temporal = create_temporal_heatmap(&normal_states, "activity");
    normal_temporal.save(output_dir.join("htm_temporal_normal.png"))?;

    println!("  Saved: htm_normal_activity.png, htm_normal_prediction.png, htm_temporal_normal.png");
    println!();

    // ========================================================================
    // Phase 3: Capture Arrhythmia States
    // ========================================================================
    println!("Phase 3: Capturing HTM states on arrhythmias...");

    let arrhythmias = [
        (ArrhythmiaType::Pvc, "pvc", 0.8, 0.9),
        (ArrhythmiaType::StElevation, "st_elev", 0.9, 0.8),
        (ArrhythmiaType::AtrialFibrillation, "afib", 1.0, 0.8),
        (ArrhythmiaType::TWaveInversion, "t_inv", 0.8, 0.9),
        (ArrhythmiaType::Tachycardia, "tachy", 1.0, 0.5),
    ];

    for (arr_type, name, prob, severity) in &arrhythmias {
        htm.reset();

        // Run some normal first to establish predictions
        let mut warmup_gen = EcgGenerator::new(ecg_config.clone(), ArrhythmiaConfig::default());
        warmup_gen.rng_state = 11111;
        let mut warmup_states: Vec<HtmState> = Vec::new();
        for _ in 0..50 {
            let window = warmup_gen.generate_window(window_size);
            let state = htm.process(&window, false)?;
            warmup_states.push(state);
        }

        // Now run arrhythmia
        let arr_config = ArrhythmiaConfig {
            arrhythmia_type: *arr_type,
            probability: *prob,
            severity: *severity,
        };

        let mut arr_gen = EcgGenerator::new(ecg_config.clone(), arr_config);
        arr_gen.rng_state = 98765;

        let mut arr_accum = StateAccumulator::new();
        let mut arr_states: Vec<HtmState> = Vec::new();
        let mut last_arr_state: Option<HtmState> = None;
        let mut total_anomaly = 0.0f32;

        for _ in 0..test_windows {
            let window = arr_gen.generate_window(window_size);
            let state = htm.process(&window, false)?;
            total_anomaly += state.anomaly;
            arr_accum.add(&state);
            arr_states.push(state.clone());
            last_arr_state = Some(state);
        }

        let mean_anomaly = total_anomaly / test_windows as f32;

        // Debug: check cell-level anomaly for last state
        if let Some(state) = &last_arr_state {
            let predicted_set: HashSet<usize> = state.predicted_cells.iter().cloned().collect();
            let unpredicted: usize = state.active_cells.iter()
                .filter(|c| !predicted_set.contains(c))
                .count();

            // Also check which columns are active
            let first_cols: Vec<u32> = state.active_columns.iter().take(5).cloned().collect();
            println!("    Active: {}, Predicted: {}, Unpred: {}, Cols: {} ({:?}...)",
                     state.active_cells.len(), state.predicted_cells.len(), unpredicted,
                     state.active_columns.len(), first_cols);
        }

        // Save visualizations
        if let Some(state) = &last_arr_state {
            let tm_img = render_tm_cells(state);
            tm_img.save(output_dir.join(format!("htm_tm_{}.png", name)))?;

            let anomaly_img = render_anomaly_map(state);
            anomaly_img.save(output_dir.join(format!("htm_anomaly_{}.png", name)))?;
        }

        // Save accumulated
        let arr_activity = arr_accum.render_activity();
        arr_activity.save(output_dir.join(format!("htm_activity_{}.png", name)))?;

        let arr_anomaly_rate = arr_accum.render_anomaly_rate();
        arr_anomaly_rate.save(output_dir.join(format!("htm_anomaly_rate_{}.png", name)))?;

        // Save temporal heatmap showing transition from normal → arrhythmia
        let mut combined_states = warmup_states.clone();
        combined_states.extend(arr_states.iter().take(200).cloned());
        let temporal_anomaly = create_temporal_heatmap(&combined_states, "anomaly");
        temporal_anomaly.save(output_dir.join(format!("htm_temporal_{}.png", name)))?;

        println!("  {}: mean anomaly = {:.4}", arr_type.display_name(), mean_anomaly);
    }
    println!();

    // ========================================================================
    // Phase 4: Comparison Visualizations
    // ========================================================================
    println!("Phase 4: Creating comparison visualizations...");

    // Load and compare normal vs PVC
    htm.reset();

    // Get fresh normal state
    let mut normal_gen = EcgGenerator::new(ecg_config.clone(), ArrhythmiaConfig::default());
    normal_gen.rng_state = 77777;
    for _ in 0..50 {
        let window = normal_gen.generate_window(window_size);
        let _ = htm.process(&window, false)?;
    }
    let normal_window = normal_gen.generate_window(window_size);
    let normal_state = htm.process(&normal_window, false)?;

    // Get PVC state
    let pvc_config = ArrhythmiaConfig {
        arrhythmia_type: ArrhythmiaType::Pvc,
        probability: 1.0,
        severity: 0.9,
    };
    let mut pvc_gen = EcgGenerator::new(ecg_config.clone(), pvc_config.clone());
    pvc_gen.rng_state = 88888;
    let pvc_window = pvc_gen.generate_window(window_size);
    let pvc_state = htm.process(&pvc_window, false)?;

    // Create comparison: Normal TM | PVC TM | PVC Anomaly
    let normal_tm = render_tm_cells(&normal_state);
    let pvc_tm = render_tm_cells(&pvc_state);
    let pvc_anomaly = render_anomaly_map(&pvc_state);

    let comparison = create_side_by_side(&[&normal_tm, &pvc_tm, &pvc_anomaly], 4);
    comparison.save(output_dir.join("htm_compare_pvc.png"))?;
    println!("  Saved: htm_compare_pvc.png (Normal | PVC | PVC Anomalies)");

    // Normal activity vs PVC anomaly rate
    let normal_act = normal_accum.render_activity();

    // Reload PVC accumulator
    htm.reset();
    let mut pvc_accum = StateAccumulator::new();
    let mut pvc_gen = EcgGenerator::new(ecg_config.clone(), pvc_config.clone());
    pvc_gen.rng_state = 99999;

    // Warmup
    for _ in 0..30 {
        let window = pvc_gen.generate_window(window_size);
        let _ = htm.process(&window, false)?;
    }

    for _ in 0..test_windows {
        let window = pvc_gen.generate_window(window_size);
        let state = htm.process(&window, false)?;
        pvc_accum.add(&state);
    }

    let pvc_anom_rate = pvc_accum.render_anomaly_rate();
    let comparison2 = create_side_by_side(&[&normal_act, &pvc_anom_rate], 4);
    comparison2.save(output_dir.join("htm_normal_vs_pvc_anomaly.png"))?;
    println!("  Saved: htm_normal_vs_pvc_anomaly.png (Normal Activity | PVC Anomaly Rate)");

    println!();

    // ========================================================================
    // Summary
    // ========================================================================
    println!("========================================");
    println!("  Summary");
    println!("========================================");
    println!();
    println!("Generated HTM fingerprint visualizations in ./fingerprints/");
    println!();
    println!("Color coding for TM cell states:");
    println!("  Black  = Inactive, not predicted (quiet)");
    println!("  Blue   = Predicted but not active (expected but didn't happen)");
    println!("  GREEN  = Predicted AND active (CORRECT prediction!)");
    println!("  ORANGE = Active but NOT predicted (ANOMALY!)");
    println!();
    println!("Key visualizations:");
    println!("  htm_tm_normal.png      - TM state on normal ECG (mostly green = good predictions)");
    println!("  htm_tm_pvc.png         - TM state on PVC (orange = anomalies!)");
    println!("  htm_compare_pvc.png    - Side-by-side: Normal | PVC | PVC Anomalies");
    println!("  htm_anomaly_rate_*.png - Where each arrhythmia causes prediction failures");
    println!();
    println!("The orange/red cells show WHERE in the HTM's learned representation");
    println!("the arrhythmia causes unexpected activations - this is the anomaly signal!");
    println!();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cell_mapping() {
        // Cell 0 (column 0, cell 0) should be at (0, 0)
        assert_eq!(cell_to_pixel(0), (0, 0));

        // Cell 63 (column 0, cell 63) - in 8×8 grid: cell 63 = (7, 7)
        assert_eq!(cell_to_pixel(63), (7, 7));

        // Cell 64 (column 1, cell 0) - column 1 is at grid position (1, 0)
        // So pixel = (1*8 + 0, 0*8 + 0) = (8, 0)
        assert_eq!(cell_to_pixel(64), (8, 0));

        // Cell at column 32 (first column in second row of grid)
        // Grid position (0, 1), cell 0 → pixel (0, 8)
        assert_eq!(cell_to_pixel(32 * 64), (0, 8));

        // Last cell (column 1023, cell 63)
        // Column 1023: grid (31, 31), cell 63: mini (7, 7)
        // Pixel: (31*8+7, 31*8+7) = (255, 255)
        assert_eq!(cell_to_pixel(TOTAL_CELLS - 1), (255, 255));
    }

    #[test]
    fn test_htm_creation() {
        let htm = HtmSystem::new(50);
        assert!(htm.is_ok());
    }
}
