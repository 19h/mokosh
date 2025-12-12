//! ECG Fingerprint Visualization
//!
//! Generates 256x256 pixel "fingerprint" images from SDR encodings of ECG signals.
//! Uses a semantically meaningful 2D encoding where:
//! - X-axis represents time position within the cardiac cycle
//! - Y-axis represents amplitude/feature intensity
//!
//! This creates visually distinct patterns for different ECG morphologies.
//!
//! Run with: cargo run --example ecg_fingerprints --release

mod ecg_utils;

use ecg_utils::{ArrhythmiaConfig, ArrhythmiaType, EcgConfig, EcgGenerator};
use image::{Rgb, RgbImage};
use std::collections::HashMap;
use std::f32::consts::PI;
use std::fs;
use std::path::Path;

// ============================================================================
// Constants
// ============================================================================

const IMG_SIZE: usize = 256;
const TOTAL_BITS: usize = IMG_SIZE * IMG_SIZE;

// ============================================================================
// Semantic 2D Encoder
// ============================================================================

/// A semantically meaningful encoder that creates 2D fingerprints where:
/// - X-axis = temporal position (0-255 maps to window position)
/// - Y-axis = amplitude/feature value
/// - Multiple "layers" encode different aspects (raw amplitude, derivatives, frequency)
struct SemanticEncoder {
    window_size: usize,
    /// Hash seeds for feature randomization
    seeds: Vec<u64>,
}

impl SemanticEncoder {
    fn new(window_size: usize) -> Self {
        // Different seeds for different feature layers
        let seeds = vec![
            0xDEADBEEF,
            0xCAFEBABE,
            0x12345678,
            0xFEEDFACE,
            0xABCDEF01,
            0x98765432,
        ];
        Self { window_size, seeds }
    }

    /// Simple hash function for deterministic randomization
    fn hash(&self, seed: u64, x: u32, y: u32) -> u32 {
        let mut h = seed;
        h ^= x as u64;
        h = h.wrapping_mul(0x517cc1b727220a95);
        h ^= y as u64;
        h = h.wrapping_mul(0x517cc1b727220a95);
        h ^= h >> 32;
        (h & 0xFFFFFFFF) as u32
    }

    /// Encode samples into a 2D activation map
    fn encode(&self, samples: &[f32]) -> Vec<Vec<f32>> {
        let mut activation = vec![vec![0.0f32; IMG_SIZE]; IMG_SIZE];

        // Normalize samples to 0-1 range
        let min_val = samples.iter().cloned().fold(f32::MAX, f32::min);
        let max_val = samples.iter().cloned().fold(f32::MIN, f32::max);
        let range = (max_val - min_val).max(0.001);

        let normalized: Vec<f32> = samples
            .iter()
            .map(|&s| ((s - min_val) / range).clamp(0.0, 1.0))
            .collect();

        // Compute derivatives
        let mut derivatives: Vec<f32> = vec![0.0; samples.len()];
        for i in 1..samples.len() {
            derivatives[i] = samples[i] - samples[i - 1];
        }
        let deriv_max = derivatives.iter().map(|d| d.abs()).fold(0.0f32, f32::max).max(0.001);
        let norm_deriv: Vec<f32> = derivatives.iter().map(|d| (d / deriv_max + 1.0) / 2.0).collect();

        // Compute second derivatives (curvature)
        let mut curvature: Vec<f32> = vec![0.0; samples.len()];
        for i in 1..derivatives.len() {
            curvature[i] = derivatives[i] - derivatives[i - 1];
        }
        let curv_max = curvature.iter().map(|c| c.abs()).fold(0.0f32, f32::max).max(0.001);
        let norm_curv: Vec<f32> = curvature.iter().map(|c| (c / curv_max + 1.0) / 2.0).collect();

        // Compute local energy (windowed RMS)
        let energy_window = 5;
        let mut energy: Vec<f32> = vec![0.0; samples.len()];
        for i in 0..samples.len() {
            let start = i.saturating_sub(energy_window);
            let end = (i + energy_window).min(samples.len());
            let sum_sq: f32 = samples[start..end].iter().map(|s| s * s).sum();
            energy[i] = (sum_sq / (end - start) as f32).sqrt();
        }
        let energy_max = energy.iter().cloned().fold(0.0f32, f32::max).max(0.001);
        let norm_energy: Vec<f32> = energy.iter().map(|e| e / energy_max).collect();

        // Layer 1: Raw amplitude waveform (occupies bottom half of image)
        // Creates a "waveform trace" visualization
        for (i, &val) in normalized.iter().enumerate() {
            let x = (i as f32 / self.window_size as f32 * IMG_SIZE as f32) as usize;
            let x = x.min(IMG_SIZE - 1);

            // Map amplitude to Y position (inverted so high amplitude is at top)
            let y_center = ((1.0 - val) * 0.4 + 0.55) * IMG_SIZE as f32;

            // Draw a vertical brush stroke with gaussian falloff
            let brush_height = 20.0;
            for dy in -15i32..=15 {
                let y = (y_center as i32 + dy) as usize;
                if y < IMG_SIZE {
                    let dist = dy.abs() as f32 / brush_height;
                    let intensity = (-dist * dist * 2.0).exp();
                    activation[y][x] = activation[y][x].max(intensity * 0.8);
                }
            }
        }

        // Layer 2: Derivative/slope visualization (top portion)
        // High derivatives = sharp transitions (R wave, etc.)
        for (i, &val) in norm_deriv.iter().enumerate() {
            let x = (i as f32 / self.window_size as f32 * IMG_SIZE as f32) as usize;
            let x = x.min(IMG_SIZE - 1);

            // Map to top 40% of image
            let y_center = (1.0 - val) * 0.35 * IMG_SIZE as f32;

            // Intensity based on absolute derivative
            let intensity = (derivatives[i].abs() / deriv_max).min(1.0);

            for dy in -8i32..=8 {
                let y = (y_center as i32 + dy) as usize;
                if y < IMG_SIZE {
                    let dist = dy.abs() as f32 / 10.0;
                    let brush = (-dist * dist * 2.0).exp() * intensity;
                    activation[y][x] = activation[y][x].max(brush * 0.7);
                }
            }
        }

        // Layer 3: Energy bands (creates horizontal "heat" bands)
        for (i, &e) in norm_energy.iter().enumerate() {
            let x = (i as f32 / self.window_size as f32 * IMG_SIZE as f32) as usize;
            let x = x.min(IMG_SIZE - 1);

            // High energy creates activation in multiple bands
            if e > 0.3 {
                let band_y = (0.45 * IMG_SIZE as f32) as usize;
                let spread = (e * 15.0) as i32;
                for dy in -spread..=spread {
                    let y = (band_y as i32 + dy) as usize;
                    if y < IMG_SIZE {
                        let dist = dy.abs() as f32 / (spread as f32 + 1.0);
                        activation[y][x] = activation[y][x].max(e * (1.0 - dist) * 0.5);
                    }
                }
            }
        }

        // Layer 4: Curvature highlights (marks inflection points)
        for (i, &c) in norm_curv.iter().enumerate() {
            let x = (i as f32 / self.window_size as f32 * IMG_SIZE as f32) as usize;
            let x = x.min(IMG_SIZE - 1);

            let curv_val = curvature[i].abs() / curv_max;
            if curv_val > 0.4 {
                // Create small "dots" at high curvature points
                let y_base = ((1.0 - normalized[i]) * 0.4 + 0.55) * IMG_SIZE as f32;
                let radius = (curv_val * 8.0) as i32;

                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        let dist_sq = dx * dx + dy * dy;
                        if dist_sq <= radius * radius {
                            let px = (x as i32 + dx) as usize;
                            let py = (y_base as i32 + dy) as usize;
                            if px < IMG_SIZE && py < IMG_SIZE {
                                let dist = (dist_sq as f32).sqrt() / radius as f32;
                                activation[py][px] = activation[py][px].max(curv_val * (1.0 - dist) * 0.6);
                            }
                        }
                    }
                }
            }
        }

        // Layer 5: Frequency texture using simple DFT magnitudes
        // Compute local frequency content at a few scales
        let freq_bands = [4, 8, 16, 32];
        for (band_idx, &period) in freq_bands.iter().enumerate() {
            let y_band_start = (band_idx as f32 * 0.08 + 0.02) * IMG_SIZE as f32;

            for chunk_start in (0..self.window_size).step_by(period / 2) {
                let chunk_end = (chunk_start + period).min(self.window_size);
                if chunk_end - chunk_start < period / 2 {
                    continue;
                }

                // Simple magnitude estimate for this frequency
                let chunk: Vec<f32> = samples[chunk_start..chunk_end].to_vec();
                let mut mag = 0.0f32;
                for (k, _) in chunk.iter().enumerate().take(chunk.len() / 2) {
                    let mut real = 0.0f32;
                    let mut imag = 0.0f32;
                    for (n, &sample) in chunk.iter().enumerate() {
                        let angle = 2.0 * PI * k as f32 * n as f32 / chunk.len() as f32;
                        real += sample * angle.cos();
                        imag -= sample * angle.sin();
                    }
                    mag += (real * real + imag * imag).sqrt();
                }
                mag /= chunk.len() as f32;
                mag = (mag / range).min(1.0);

                let x_center = ((chunk_start + chunk_end) / 2) as f32 / self.window_size as f32 * IMG_SIZE as f32;
                let x = (x_center as usize).min(IMG_SIZE - 1);

                // Draw frequency band
                let band_height = 8;
                for dy in 0..band_height {
                    let y = (y_band_start as usize + dy).min(IMG_SIZE - 1);
                    activation[y][x] = activation[y][x].max(mag * 0.4);
                    // Spread horizontally
                    if x > 0 {
                        activation[y][x - 1] = activation[y][x - 1].max(mag * 0.3);
                    }
                    if x < IMG_SIZE - 1 {
                        activation[y][x + 1] = activation[y][x + 1].max(mag * 0.3);
                    }
                }
            }
        }

        activation
    }

    /// Convert activation map to sparse bit indices
    fn to_sparse(&self, activation: &[Vec<f32>], threshold: f32) -> Vec<u32> {
        let mut bits = Vec::new();
        for (y, row) in activation.iter().enumerate() {
            for (x, &val) in row.iter().enumerate() {
                if val > threshold {
                    bits.push((y * IMG_SIZE + x) as u32);
                }
            }
        }
        bits
    }
}

// ============================================================================
// Fingerprint Accumulator
// ============================================================================

struct FingerprintAccumulator {
    activation: Vec<Vec<f32>>,
    count: u32,
}

impl FingerprintAccumulator {
    fn new() -> Self {
        Self {
            activation: vec![vec![0.0; IMG_SIZE]; IMG_SIZE],
            count: 0,
        }
    }

    fn add(&mut self, activation: &[Vec<f32>]) {
        for (y, row) in activation.iter().enumerate() {
            for (x, &val) in row.iter().enumerate() {
                self.activation[y][x] += val;
            }
        }
        self.count += 1;
    }

    fn get_normalized(&self, x: usize, y: usize) -> f32 {
        if self.count == 0 {
            return 0.0;
        }
        self.activation[y][x] / self.count as f32
    }

    fn get_peak_normalized(&self, x: usize, y: usize) -> f32 {
        let max_val = self.activation.iter()
            .flat_map(|row| row.iter())
            .cloned()
            .fold(0.0f32, f32::max)
            .max(0.001);
        self.activation[y][x] / max_val
    }

    fn max_value(&self) -> f32 {
        self.activation.iter()
            .flat_map(|row| row.iter())
            .cloned()
            .fold(0.0f32, f32::max)
    }

    fn difference(&self, other: &FingerprintAccumulator) -> Vec<Vec<f32>> {
        let self_max = self.max_value().max(0.001);
        let other_max = other.max_value().max(0.001);

        let mut diff = vec![vec![0.0f32; IMG_SIZE]; IMG_SIZE];
        for y in 0..IMG_SIZE {
            for x in 0..IMG_SIZE {
                let self_norm = self.activation[y][x] / self_max;
                let other_norm = other.activation[y][x] / other_max;
                diff[y][x] = self_norm - other_norm;
            }
        }
        diff
    }
}

// ============================================================================
// Color Schemes
// ============================================================================

#[derive(Clone, Copy)]
enum ColorScheme {
    /// Single color: white → blue
    BlueScale,
    /// Heatmap: black → blue → cyan → yellow → white
    Plasma,
    /// Diverging: blue (negative) → white (zero) → red (positive)
    Diverging,
    /// Viridis-like: dark purple → blue → green → yellow
    Viridis,
}

impl ColorScheme {
    fn to_rgb(&self, value: f32) -> Rgb<u8> {
        match self {
            ColorScheme::BlueScale => {
                let v = value.clamp(0.0, 1.0);
                if v < 0.01 {
                    Rgb([255, 255, 255])
                } else {
                    // White → Light blue → Deep blue
                    let r = (255.0 * (1.0 - v * 0.9)) as u8;
                    let g = (255.0 * (1.0 - v * 0.6)) as u8;
                    let b = 255;
                    Rgb([r, g, b])
                }
            }
            ColorScheme::Plasma => {
                let v = value.clamp(0.0, 1.0);
                if v < 0.01 {
                    Rgb([10, 10, 20]) // Near black background
                } else if v < 0.2 {
                    let t = v / 0.2;
                    Rgb([
                        (10.0 + t * 50.0) as u8,
                        (10.0 + t * 20.0) as u8,
                        (20.0 + t * 100.0) as u8,
                    ])
                } else if v < 0.4 {
                    let t = (v - 0.2) / 0.2;
                    Rgb([
                        (60.0 + t * 40.0) as u8,
                        (30.0 + t * 100.0) as u8,
                        (120.0 + t * 60.0) as u8,
                    ])
                } else if v < 0.6 {
                    let t = (v - 0.4) / 0.2;
                    Rgb([
                        (100.0 + t * 100.0) as u8,
                        (130.0 + t * 70.0) as u8,
                        (180.0 - t * 80.0) as u8,
                    ])
                } else if v < 0.8 {
                    let t = (v - 0.6) / 0.2;
                    Rgb([
                        (200.0 + t * 55.0) as u8,
                        (200.0 + t * 30.0) as u8,
                        (100.0 - t * 50.0) as u8,
                    ])
                } else {
                    let t = (v - 0.8) / 0.2;
                    Rgb([
                        255,
                        (230.0 + t * 25.0) as u8,
                        (50.0 + t * 200.0) as u8,
                    ])
                }
            }
            ColorScheme::Diverging => {
                let v = value.clamp(-1.0, 1.0);
                if v.abs() < 0.05 {
                    Rgb([240, 240, 240])
                } else if v < 0.0 {
                    let t = (-v).min(1.0);
                    Rgb([
                        (240.0 - t * 180.0) as u8,
                        (240.0 - t * 140.0) as u8,
                        (240.0 + t * 15.0) as u8,
                    ])
                } else {
                    let t = v.min(1.0);
                    Rgb([
                        (240.0 + t * 15.0) as u8,
                        (240.0 - t * 160.0) as u8,
                        (240.0 - t * 180.0) as u8,
                    ])
                }
            }
            ColorScheme::Viridis => {
                let v = value.clamp(0.0, 1.0);
                if v < 0.01 {
                    Rgb([68, 1, 84])
                } else if v < 0.25 {
                    let t = v / 0.25;
                    Rgb([
                        (68.0 - t * 20.0) as u8,
                        (1.0 + t * 50.0) as u8,
                        (84.0 + t * 40.0) as u8,
                    ])
                } else if v < 0.5 {
                    let t = (v - 0.25) / 0.25;
                    Rgb([
                        (48.0 - t * 15.0) as u8,
                        (51.0 + t * 60.0) as u8,
                        (124.0 - t * 20.0) as u8,
                    ])
                } else if v < 0.75 {
                    let t = (v - 0.5) / 0.25;
                    Rgb([
                        (33.0 + t * 90.0) as u8,
                        (111.0 + t * 50.0) as u8,
                        (104.0 - t * 60.0) as u8,
                    ])
                } else {
                    let t = (v - 0.75) / 0.25;
                    Rgb([
                        (123.0 + t * 130.0) as u8,
                        (161.0 + t * 70.0) as u8,
                        (44.0 + t * 30.0) as u8,
                    ])
                }
            }
        }
    }
}

// ============================================================================
// Image Rendering
// ============================================================================

fn render_activation(activation: &[Vec<f32>], scheme: ColorScheme, normalize: bool) -> RgbImage {
    let mut img = RgbImage::new(IMG_SIZE as u32, IMG_SIZE as u32);

    let max_val = if normalize {
        activation.iter()
            .flat_map(|row| row.iter())
            .cloned()
            .fold(0.0f32, f32::max)
            .max(0.001)
    } else {
        1.0
    };

    for (y, row) in activation.iter().enumerate() {
        for (x, &val) in row.iter().enumerate() {
            let normalized = if normalize { val / max_val } else { val };
            img.put_pixel(x as u32, y as u32, scheme.to_rgb(normalized));
        }
    }

    img
}

fn render_accumulated(accum: &FingerprintAccumulator, scheme: ColorScheme) -> RgbImage {
    let mut img = RgbImage::new(IMG_SIZE as u32, IMG_SIZE as u32);

    for y in 0..IMG_SIZE {
        for x in 0..IMG_SIZE {
            let val = accum.get_peak_normalized(x, y);
            img.put_pixel(x as u32, y as u32, scheme.to_rgb(val));
        }
    }

    img
}

fn render_difference(diff: &[Vec<f32>], scheme: ColorScheme) -> RgbImage {
    let mut img = RgbImage::new(IMG_SIZE as u32, IMG_SIZE as u32);

    // Find max absolute value for normalization
    let max_abs = diff.iter()
        .flat_map(|row| row.iter())
        .map(|v| v.abs())
        .fold(0.0f32, f32::max)
        .max(0.001);

    for (y, row) in diff.iter().enumerate() {
        for (x, &val) in row.iter().enumerate() {
            let normalized = val / max_abs;
            img.put_pixel(x as u32, y as u32, scheme.to_rgb(normalized));
        }
    }

    img
}

fn create_side_by_side(images: &[&RgbImage], gap: u32) -> RgbImage {
    let total_width: u32 = images.iter().map(|i| i.width()).sum::<u32>() + gap * (images.len() as u32 - 1);
    let height = images.iter().map(|i| i.height()).max().unwrap_or(256);

    let mut combined = RgbImage::new(total_width, height);

    // Dark gray background
    for pixel in combined.pixels_mut() {
        *pixel = Rgb([30, 30, 30]);
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
    println!("  ECG Fingerprint Visualization v2");
    println!("========================================");
    println!();

    let output_dir = Path::new("fingerprints");
    if !output_dir.exists() {
        fs::create_dir_all(output_dir)?;
    }

    // Use a longer window to capture full cardiac cycle
    let window_size = 256;
    let encoder = SemanticEncoder::new(window_size);

    println!("Semantic Encoder:");
    println!("  Window size: {} samples (~1 second at 250Hz)", window_size);
    println!("  Image size:  {}x{} pixels", IMG_SIZE, IMG_SIZE);
    println!("  Layers: amplitude, derivatives, energy, curvature, frequency");
    println!();

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
    // 1. Single Fingerprints
    // ========================================================================
    println!("1. Generating Single Fingerprints...");

    // Normal ECG
    let mut normal_gen = EcgGenerator::new(ecg_config.clone(), ArrhythmiaConfig::default());
    normal_gen.rng_state = 12345;
    let normal_window = normal_gen.generate_window(window_size);
    let normal_activation = encoder.encode(&normal_window);

    let single_normal = render_activation(&normal_activation, ColorScheme::Plasma, true);
    single_normal.save(output_dir.join("single_normal.png"))?;
    println!("  single_normal.png");

    // PVC
    let pvc_config = ArrhythmiaConfig {
        arrhythmia_type: ArrhythmiaType::Pvc,
        probability: 1.0,
        severity: 0.9,
    };
    let mut pvc_gen = EcgGenerator::new(ecg_config.clone(), pvc_config);
    pvc_gen.rng_state = 12345;
    let pvc_window = pvc_gen.generate_window(window_size);
    let pvc_activation = encoder.encode(&pvc_window);

    let single_pvc = render_activation(&pvc_activation, ColorScheme::Plasma, true);
    single_pvc.save(output_dir.join("single_pvc.png"))?;
    println!("  single_pvc.png");

    // ST Elevation
    let st_config = ArrhythmiaConfig {
        arrhythmia_type: ArrhythmiaType::StElevation,
        probability: 1.0,
        severity: 0.8,
    };
    let mut st_gen = EcgGenerator::new(ecg_config.clone(), st_config);
    st_gen.rng_state = 12345;
    let st_window = st_gen.generate_window(window_size);
    let st_activation = encoder.encode(&st_window);

    let single_st = render_activation(&st_activation, ColorScheme::Plasma, true);
    single_st.save(output_dir.join("single_st_elevation.png"))?;
    println!("  single_st_elevation.png");

    // AF
    let af_config = ArrhythmiaConfig {
        arrhythmia_type: ArrhythmiaType::AtrialFibrillation,
        probability: 1.0,
        severity: 0.8,
    };
    let mut af_gen = EcgGenerator::new(ecg_config.clone(), af_config);
    af_gen.rng_state = 12345;
    let af_window = af_gen.generate_window(window_size);
    let af_activation = encoder.encode(&af_window);

    let single_af = render_activation(&af_activation, ColorScheme::Plasma, true);
    single_af.save(output_dir.join("single_afib.png"))?;
    println!("  single_afib.png");

    // Comparison strip
    let comparison = create_side_by_side(&[&single_normal, &single_pvc, &single_st, &single_af], 4);
    comparison.save(output_dir.join("comparison_singles.png"))?;
    println!("  comparison_singles.png (Normal | PVC | ST Elev | AFib)");
    println!();

    // ========================================================================
    // 2. Accumulated Fingerprints
    // ========================================================================
    println!("2. Generating Accumulated Fingerprints...");
    let num_accumulate = 200;

    // Normal consensus
    let mut normal_accum = FingerprintAccumulator::new();
    let mut normal_gen = EcgGenerator::new(ecg_config.clone(), ArrhythmiaConfig::default());
    normal_gen.rng_state = 54321;

    for _ in 0..num_accumulate {
        let window = normal_gen.generate_window(window_size);
        let activation = encoder.encode(&window);
        normal_accum.add(&activation);
    }

    let accum_normal = render_accumulated(&normal_accum, ColorScheme::Plasma);
    accum_normal.save(output_dir.join("accumulated_normal.png"))?;
    println!("  accumulated_normal.png ({} patterns)", num_accumulate);

    // Also save with different color schemes
    let accum_normal_viridis = render_accumulated(&normal_accum, ColorScheme::Viridis);
    accum_normal_viridis.save(output_dir.join("accumulated_normal_viridis.png"))?;

    let accum_normal_blue = render_accumulated(&normal_accum, ColorScheme::BlueScale);
    accum_normal_blue.save(output_dir.join("accumulated_normal_blue.png"))?;
    println!("  accumulated_normal_viridis.png, accumulated_normal_blue.png");

    // ========================================================================
    // 3. All Arrhythmia Types
    // ========================================================================
    println!();
    println!("3. Generating Arrhythmia Fingerprints...");

    let arrhythmia_configs: Vec<(ArrhythmiaType, f32, f32)> = vec![
        (ArrhythmiaType::Pvc, 0.6, 0.85),
        (ArrhythmiaType::Pac, 0.6, 0.85),
        (ArrhythmiaType::StElevation, 0.7, 0.8),
        (ArrhythmiaType::StDepression, 0.7, 0.8),
        (ArrhythmiaType::TWaveInversion, 0.6, 0.9),
        (ArrhythmiaType::TWavePeaking, 0.6, 0.85),
        (ArrhythmiaType::AtrialFibrillation, 1.0, 0.8),
        (ArrhythmiaType::Bradycardia, 1.0, 0.5),
        (ArrhythmiaType::Tachycardia, 1.0, 0.5),
    ];

    let mut arrhythmia_accums: HashMap<ArrhythmiaType, FingerprintAccumulator> = HashMap::new();

    for (arr_type, probability, severity) in &arrhythmia_configs {
        let arr_config = ArrhythmiaConfig {
            arrhythmia_type: *arr_type,
            probability: *probability,
            severity: *severity,
        };

        let mut arr_accum = FingerprintAccumulator::new();
        let mut arr_gen = EcgGenerator::new(ecg_config.clone(), arr_config);
        arr_gen.rng_state = 98765;

        for _ in 0..num_accumulate {
            let window = arr_gen.generate_window(window_size);
            let activation = encoder.encode(&window);
            arr_accum.add(&activation);
        }

        // Save accumulated
        let img = render_accumulated(&arr_accum, ColorScheme::Plasma);
        let filename = format!("arr_{}.png", arr_type.short_name());
        img.save(output_dir.join(&filename))?;

        // Save difference from normal
        let diff = arr_accum.difference(&normal_accum);
        let diff_img = render_difference(&diff, ColorScheme::Diverging);
        let diff_filename = format!("diff_{}.png", arr_type.short_name());
        diff_img.save(output_dir.join(&diff_filename))?;

        println!("  {} -> {}, {}", arr_type.display_name(), filename, diff_filename);

        arrhythmia_accums.insert(*arr_type, arr_accum);
    }

    // ========================================================================
    // 4. Grid Views
    // ========================================================================
    println!();
    println!("4. Generating Grid Views...");

    // All types grid (3x4 = 12 cells, we have 10 patterns)
    let cell_size = 128u32;
    let grid_cols = 5u32;
    let grid_rows = 2u32;
    let grid_gap = 2u32;

    let grid_width = grid_cols * cell_size + (grid_cols - 1) * grid_gap;
    let grid_height = grid_rows * cell_size + (grid_rows - 1) * grid_gap;

    let mut grid = RgbImage::new(grid_width, grid_height);
    for pixel in grid.pixels_mut() {
        *pixel = Rgb([20, 20, 20]);
    }

    // Place normal first
    let normal_small = image::imageops::resize(&accum_normal, cell_size, cell_size, image::imageops::FilterType::Lanczos3);
    image::imageops::overlay(&mut grid, &normal_small, 0, 0);

    // Place arrhythmias
    let mut idx = 1usize;
    for arr_type in ArrhythmiaType::all_types() {
        if let Some(accum) = arrhythmia_accums.get(arr_type) {
            let img = render_accumulated(accum, ColorScheme::Plasma);
            let small = image::imageops::resize(&img, cell_size, cell_size, image::imageops::FilterType::Lanczos3);

            let col = (idx % grid_cols as usize) as i64;
            let row = (idx / grid_cols as usize) as i64;
            let x = col * (cell_size + grid_gap) as i64;
            let y = row * (cell_size + grid_gap) as i64;

            image::imageops::overlay(&mut grid, &small, x, y);
            idx += 1;
            if idx >= (grid_cols * grid_rows) as usize {
                break;
            }
        }
    }

    grid.save(output_dir.join("grid_all.png"))?;
    println!("  grid_all.png");

    // Difference grid
    let mut diff_grid = RgbImage::new(grid_width, grid_height);
    for pixel in diff_grid.pixels_mut() {
        *pixel = Rgb([128, 128, 128]);
    }

    // First cell: normal (neutral gray in diverging)
    image::imageops::overlay(&mut diff_grid, &normal_small, 0, 0);

    idx = 1;
    for arr_type in ArrhythmiaType::all_types() {
        if let Some(accum) = arrhythmia_accums.get(arr_type) {
            let diff = accum.difference(&normal_accum);
            let img = render_difference(&diff, ColorScheme::Diverging);
            let small = image::imageops::resize(&img, cell_size, cell_size, image::imageops::FilterType::Lanczos3);

            let col = (idx % grid_cols as usize) as i64;
            let row = (idx / grid_cols as usize) as i64;
            let x = col * (cell_size + grid_gap) as i64;
            let y = row * (cell_size + grid_gap) as i64;

            image::imageops::overlay(&mut diff_grid, &small, x, y);
            idx += 1;
            if idx >= (grid_cols * grid_rows) as usize {
                break;
            }
        }
    }

    diff_grid.save(output_dir.join("grid_differences.png"))?;
    println!("  grid_differences.png");

    // ========================================================================
    // 5. Detailed Comparisons
    // ========================================================================
    println!();
    println!("5. Generating Detailed Comparisons...");

    // PVC detailed
    if let Some(pvc_accum) = arrhythmia_accums.get(&ArrhythmiaType::Pvc) {
        let normal_img = render_accumulated(&normal_accum, ColorScheme::Plasma);
        let pvc_img = render_accumulated(pvc_accum, ColorScheme::Plasma);
        let diff = pvc_accum.difference(&normal_accum);
        let diff_img = render_difference(&diff, ColorScheme::Diverging);

        let detail = create_side_by_side(&[&normal_img, &pvc_img, &diff_img], 4);
        detail.save(output_dir.join("detail_pvc.png"))?;
        println!("  detail_pvc.png (Normal | PVC | Difference)");
    }

    // ST Elevation detailed
    if let Some(st_accum) = arrhythmia_accums.get(&ArrhythmiaType::StElevation) {
        let normal_img = render_accumulated(&normal_accum, ColorScheme::Plasma);
        let st_img = render_accumulated(st_accum, ColorScheme::Plasma);
        let diff = st_accum.difference(&normal_accum);
        let diff_img = render_difference(&diff, ColorScheme::Diverging);

        let detail = create_side_by_side(&[&normal_img, &st_img, &diff_img], 4);
        detail.save(output_dir.join("detail_st_elevation.png"))?;
        println!("  detail_st_elevation.png");
    }

    // AFib detailed
    if let Some(af_accum) = arrhythmia_accums.get(&ArrhythmiaType::AtrialFibrillation) {
        let normal_img = render_accumulated(&normal_accum, ColorScheme::Plasma);
        let af_img = render_accumulated(af_accum, ColorScheme::Plasma);
        let diff = af_accum.difference(&normal_accum);
        let diff_img = render_difference(&diff, ColorScheme::Diverging);

        let detail = create_side_by_side(&[&normal_img, &af_img, &diff_img], 4);
        detail.save(output_dir.join("detail_afib.png"))?;
        println!("  detail_afib.png");
    }

    // T Wave abnormalities
    if let (Some(t_inv), Some(t_peak)) = (
        arrhythmia_accums.get(&ArrhythmiaType::TWaveInversion),
        arrhythmia_accums.get(&ArrhythmiaType::TWavePeaking),
    ) {
        let normal_img = render_accumulated(&normal_accum, ColorScheme::Plasma);
        let inv_img = render_accumulated(t_inv, ColorScheme::Plasma);
        let peak_img = render_accumulated(t_peak, ColorScheme::Plasma);

        let detail = create_side_by_side(&[&normal_img, &inv_img, &peak_img], 4);
        detail.save(output_dir.join("detail_t_waves.png"))?;
        println!("  detail_t_waves.png (Normal | T Inversion | T Peaking)");
    }

    // ========================================================================
    // Summary
    // ========================================================================
    println!();
    println!("========================================");
    println!("  Summary");
    println!("========================================");
    println!();
    println!("Generated semantic fingerprint visualizations in ./fingerprints/");
    println!();
    println!("The 2D encoding maps:");
    println!("  - X-axis: Time position in cardiac cycle");
    println!("  - Y-axis: Multiple feature layers");
    println!("    * Top band: Derivative/slope intensity");
    println!("    * Middle: Frequency content bands");
    println!("    * Lower: Waveform amplitude trace");
    println!("    * Dots: High curvature inflection points");
    println!();
    println!("Color schemes:");
    println!("  - Plasma: Dark → Blue → Cyan → Yellow → White (intensity)");
    println!("  - Diverging: Blue (less) → White (same) → Red (more)");
    println!();
    println!("Key visualizations:");
    println!("  - grid_all.png: All 10 patterns at a glance");
    println!("  - grid_differences.png: How each arrhythmia differs");
    println!("  - detail_*.png: Side-by-side comparisons");
    println!();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder() {
        let encoder = SemanticEncoder::new(256);
        let samples: Vec<f32> = (0..256).map(|i| (i as f32 / 256.0 * 6.28).sin()).collect();
        let activation = encoder.encode(&samples);

        assert_eq!(activation.len(), IMG_SIZE);
        assert_eq!(activation[0].len(), IMG_SIZE);

        // Should have some non-zero activations
        let total: f32 = activation.iter().flat_map(|r| r.iter()).sum();
        assert!(total > 0.0);
    }

    #[test]
    fn test_accumulator() {
        let mut accum = FingerprintAccumulator::new();
        let mut test_activation = vec![vec![0.0; IMG_SIZE]; IMG_SIZE];
        test_activation[0][0] = 1.0;
        test_activation[100][100] = 0.5;

        accum.add(&test_activation);
        accum.add(&test_activation);

        assert_eq!(accum.count, 2);
        assert!(accum.get_normalized(0, 0) > 0.0);
    }

    #[test]
    fn test_color_schemes() {
        for scheme in [ColorScheme::Plasma, ColorScheme::Diverging, ColorScheme::Viridis, ColorScheme::BlueScale] {
            let low = scheme.to_rgb(0.0);
            let high = scheme.to_rgb(1.0);
            // Just verify they don't panic and produce different colors
            assert!(low != high || matches!(scheme, ColorScheme::BlueScale));
        }
    }
}
