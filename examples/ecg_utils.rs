//! ECG Signal Generation Utilities
//!
//! Shared utilities for generating realistic synthetic ECG signals with
//! physiologically-accurate PQRST wave morphology and natural heart rate
//! variability. Supports injection of various arrhythmia types.
//!
//! This module is used by multiple examples:
//! - `ecg_anomaly_detection.rs` - HTM-based anomaly detection
//! - `ecg_fingerprints.rs` - SDR fingerprint visualization

use std::f32::consts::PI;

// ============================================================================
// Configuration Types
// ============================================================================

/// Configuration for ECG signal generation.
#[derive(Debug, Clone)]
pub struct EcgConfig {
    /// Sampling rate in Hz (e.g., 250, 500, 1000).
    pub sample_rate: f32,
    /// Base heart rate in BPM.
    pub base_heart_rate: f32,
    /// Amplitude scaling factor (mV).
    pub amplitude: f32,
    /// Enable heart rate variability.
    pub enable_hrv: bool,
    /// HRV magnitude (0.0-1.0, fraction of RR interval variation).
    pub hrv_magnitude: f32,
    /// Baseline wander amplitude (mV).
    pub baseline_wander: f32,
    /// High-frequency noise amplitude (mV).
    pub noise_amplitude: f32,
}

impl Default for EcgConfig {
    fn default() -> Self {
        Self {
            sample_rate: 250.0,
            base_heart_rate: 72.0,
            amplitude: 1.0,
            enable_hrv: true,
            hrv_magnitude: 0.05,
            baseline_wander: 0.05,
            noise_amplitude: 0.02,
        }
    }
}

/// Types of arrhythmias that can be injected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArrhythmiaType {
    /// No arrhythmia - normal sinus rhythm.
    None,
    /// Premature Ventricular Contraction - wide, abnormal QRS.
    Pvc,
    /// Premature Atrial Contraction - early beat, normal QRS.
    Pac,
    /// ST segment elevation (simulating ischemia).
    StElevation,
    /// ST segment depression.
    StDepression,
    /// T wave inversion.
    TWaveInversion,
    /// T wave peaking (hyperkalemia-like).
    TWavePeaking,
    /// Atrial fibrillation - irregular rhythm, no P waves.
    AtrialFibrillation,
    /// Bradycardia - slow heart rate.
    Bradycardia,
    /// Tachycardia - fast heart rate.
    Tachycardia,
}

impl ArrhythmiaType {
    /// Returns all arrhythmia types (excluding None).
    pub fn all_types() -> &'static [ArrhythmiaType] {
        &[
            ArrhythmiaType::Pvc,
            ArrhythmiaType::Pac,
            ArrhythmiaType::StElevation,
            ArrhythmiaType::StDepression,
            ArrhythmiaType::TWaveInversion,
            ArrhythmiaType::TWavePeaking,
            ArrhythmiaType::AtrialFibrillation,
            ArrhythmiaType::Bradycardia,
            ArrhythmiaType::Tachycardia,
        ]
    }

    /// Returns a short name for the arrhythmia type.
    pub fn short_name(&self) -> &'static str {
        match self {
            ArrhythmiaType::None => "normal",
            ArrhythmiaType::Pvc => "pvc",
            ArrhythmiaType::Pac => "pac",
            ArrhythmiaType::StElevation => "st_elev",
            ArrhythmiaType::StDepression => "st_dep",
            ArrhythmiaType::TWaveInversion => "t_inv",
            ArrhythmiaType::TWavePeaking => "t_peak",
            ArrhythmiaType::AtrialFibrillation => "afib",
            ArrhythmiaType::Bradycardia => "brady",
            ArrhythmiaType::Tachycardia => "tachy",
        }
    }

    /// Returns a human-readable name for the arrhythmia type.
    pub fn display_name(&self) -> &'static str {
        match self {
            ArrhythmiaType::None => "Normal Sinus Rhythm",
            ArrhythmiaType::Pvc => "Premature Ventricular Contraction",
            ArrhythmiaType::Pac => "Premature Atrial Contraction",
            ArrhythmiaType::StElevation => "ST Elevation",
            ArrhythmiaType::StDepression => "ST Depression",
            ArrhythmiaType::TWaveInversion => "T Wave Inversion",
            ArrhythmiaType::TWavePeaking => "T Wave Peaking",
            ArrhythmiaType::AtrialFibrillation => "Atrial Fibrillation",
            ArrhythmiaType::Bradycardia => "Bradycardia",
            ArrhythmiaType::Tachycardia => "Tachycardia",
        }
    }
}

/// Configuration for arrhythmia injection.
#[derive(Debug, Clone)]
pub struct ArrhythmiaConfig {
    /// Type of arrhythmia.
    pub arrhythmia_type: ArrhythmiaType,
    /// Probability of occurrence per beat (0.0-1.0).
    pub probability: f32,
    /// Severity of the arrhythmia (0.0-1.0).
    pub severity: f32,
}

impl Default for ArrhythmiaConfig {
    fn default() -> Self {
        Self {
            arrhythmia_type: ArrhythmiaType::None,
            probability: 0.0,
            severity: 0.5,
        }
    }
}

// ============================================================================
// Internal Types
// ============================================================================

/// PQRST wave parameters for a single heartbeat.
#[derive(Debug, Clone)]
struct PqrstParams {
    // P wave
    p_amplitude: f32,
    p_duration: f32,
    p_position: f32, // relative to R peak

    // Q wave
    q_amplitude: f32,
    q_duration: f32,
    q_position: f32,

    // R wave
    r_amplitude: f32,
    r_duration: f32,

    // S wave
    s_amplitude: f32,
    s_duration: f32,
    s_position: f32,

    // T wave
    t_amplitude: f32,
    t_duration: f32,
    t_position: f32,

    // ST segment
    st_level: f32,
}

impl Default for PqrstParams {
    fn default() -> Self {
        Self {
            // P wave: ~80ms before QRS, small positive deflection
            p_amplitude: 0.15,
            p_duration: 0.08,
            p_position: -0.16,

            // Q wave: small negative deflection just before R
            q_amplitude: -0.1,
            q_duration: 0.02,
            q_position: -0.02,

            // R wave: main spike
            r_amplitude: 1.0,
            r_duration: 0.04,

            // S wave: negative deflection after R
            s_amplitude: -0.2,
            s_duration: 0.025,
            s_position: 0.03,

            // T wave: repolarization hump
            t_amplitude: 0.25,
            t_duration: 0.16,
            t_position: 0.25,

            // ST segment baseline
            st_level: 0.0,
        }
    }
}

// ============================================================================
// ECG Generator
// ============================================================================

/// Generates realistic synthetic ECG signals.
pub struct EcgGenerator {
    config: EcgConfig,
    arrhythmia: ArrhythmiaConfig,
    pub rng_state: u64,
    phase: f32,
    sample_count: usize,
    last_rr_interval: f32,
    hrv_phase_vlf: f32,
    hrv_phase_lf: f32,
    hrv_phase_hf: f32,
    // Track current beat's arrhythmia status
    current_beat_is_arrhythmia: bool,
    current_beat_params: Option<PqrstParams>,
}

impl EcgGenerator {
    /// Creates a new ECG generator with the given configuration.
    pub fn new(config: EcgConfig, arrhythmia: ArrhythmiaConfig) -> Self {
        Self {
            config,
            arrhythmia,
            rng_state: 12345,
            phase: 0.0,
            sample_count: 0,
            last_rr_interval: 0.0,
            hrv_phase_vlf: 0.0,
            hrv_phase_lf: 0.0,
            hrv_phase_hf: 0.0,
            current_beat_is_arrhythmia: false,
            current_beat_params: None,
        }
    }

    /// Simple pseudo-random number generator (xorshift).
    fn rand(&mut self) -> f32 {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 17;
        self.rng_state ^= self.rng_state << 5;
        self.rng_state as f32 / u64::MAX as f32
    }

    /// Gaussian random number (Box-Muller).
    fn rand_gaussian(&mut self) -> f32 {
        let u1 = self.rand().max(1e-10);
        let u2 = self.rand();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }

    /// Calculate current RR interval with HRV.
    fn get_rr_interval(&mut self) -> f32 {
        let base_rr = 60.0 / self.config.base_heart_rate;

        if !self.config.enable_hrv {
            return base_rr;
        }

        // Apply arrhythmia effects on RR interval
        let arrhythmia_factor = match self.arrhythmia.arrhythmia_type {
            ArrhythmiaType::Bradycardia => 1.5,
            ArrhythmiaType::Tachycardia => 0.6,
            ArrhythmiaType::AtrialFibrillation => {
                // Highly irregular RR intervals
                0.7 + 0.6 * self.rand()
            }
            _ => 1.0,
        };

        // Multi-component HRV model
        let dt = 1.0 / self.config.sample_rate;

        // VLF: ~0.02 Hz (thermoregulation, very slow drift)
        self.hrv_phase_vlf += 0.02 * 2.0 * PI * dt;
        let vlf = 0.02 * self.hrv_phase_vlf.sin();

        // LF: ~0.1 Hz (baroreflex, Mayer waves)
        self.hrv_phase_lf += 0.1 * 2.0 * PI * dt;
        let lf = 0.03 * self.hrv_phase_lf.sin();

        // HF: ~0.25 Hz (respiratory sinus arrhythmia)
        self.hrv_phase_hf += 0.25 * 2.0 * PI * dt;
        let hf = 0.04 * self.hrv_phase_hf.sin();

        // Combine HRV components
        let hrv_modulation = 1.0 + self.config.hrv_magnitude * (vlf + lf + hf);

        base_rr * hrv_modulation * arrhythmia_factor
    }

    /// Generate a single Gaussian pulse for wave components.
    fn gaussian_pulse(t: f32, center: f32, width: f32, amplitude: f32) -> f32 {
        let x = (t - center) / width;
        amplitude * (-0.5 * x * x).exp()
    }

    /// Generate PQRST waveform at a given phase within the cardiac cycle.
    fn generate_pqrst(&mut self, phase: f32, params: &PqrstParams) -> f32 {
        // phase is 0-1 representing one cardiac cycle
        // Center of R wave is at phase 0.5
        let t = phase - 0.5;

        let mut signal = 0.0;

        // P wave
        signal += Self::gaussian_pulse(
            t,
            params.p_position,
            params.p_duration / 2.0,
            params.p_amplitude,
        );

        // Q wave
        signal += Self::gaussian_pulse(
            t,
            params.q_position,
            params.q_duration / 2.0,
            params.q_amplitude,
        );

        // R wave (main spike)
        signal += Self::gaussian_pulse(t, 0.0, params.r_duration / 2.0, params.r_amplitude);

        // S wave
        signal += Self::gaussian_pulse(
            t,
            params.s_position,
            params.s_duration / 2.0,
            params.s_amplitude,
        );

        // ST segment contribution
        if t > 0.04 && t < 0.15 {
            signal += params.st_level;
        }

        // T wave
        signal += Self::gaussian_pulse(
            t,
            params.t_position,
            params.t_duration / 2.0,
            params.t_amplitude,
        );

        signal
    }

    /// Get PQRST parameters, potentially modified by arrhythmia.
    fn get_beat_params(&mut self, is_arrhythmia_beat: bool) -> PqrstParams {
        let mut params = PqrstParams::default();

        // Add natural beat-to-beat variation
        params.r_amplitude *= 1.0 + 0.03 * self.rand_gaussian();
        params.t_amplitude *= 1.0 + 0.05 * self.rand_gaussian();
        params.p_amplitude *= 1.0 + 0.05 * self.rand_gaussian();

        // Slight timing variations
        params.t_position += 0.01 * self.rand_gaussian();

        if !is_arrhythmia_beat {
            return params;
        }

        // Apply arrhythmia modifications - STRONG signatures for detection
        let severity = self.arrhythmia.severity;

        match self.arrhythmia.arrhythmia_type {
            ArrhythmiaType::Pvc => {
                // PVC: Wide, bizarre QRS complex - very distinctive
                params.r_amplitude *= 1.8 + 1.0 * severity;
                params.r_duration *= 2.5 + 1.5 * severity;
                params.q_amplitude *= 2.5 + severity;
                params.q_duration *= 2.0;
                params.s_amplitude *= 2.5 + severity;
                params.s_duration *= 2.0;
                params.t_amplitude *= -(1.0 + 0.5 * severity);
                params.t_position = 0.35;
                params.p_amplitude = 0.0;
                params.s_position = 0.06 + 0.02 * severity;
            }
            ArrhythmiaType::Pac => {
                // PAC: Premature beat with abnormal P wave and altered QRS
                params.p_amplitude *= 0.2 + 0.3 * self.rand_gaussian().abs();
                params.p_duration *= 0.5;
                params.p_position = -0.10;
                params.r_amplitude *= 0.7 + 0.2 * severity;
                params.s_amplitude *= 1.8 + 0.5 * severity;
                params.q_amplitude *= 1.5;
                params.t_amplitude *= 0.7;
            }
            ArrhythmiaType::StElevation => {
                // ST elevation (acute MI pattern)
                params.st_level = 0.4 + 0.3 * severity;
                params.t_amplitude *= 2.0 + severity;
                params.t_duration *= 1.3;
                params.s_amplitude *= 0.5;
            }
            ArrhythmiaType::StDepression => {
                // ST depression (ischemia/strain pattern)
                params.st_level = -(0.3 + 0.2 * severity);
                params.t_amplitude *= -(0.5 + 0.5 * severity);
                params.s_amplitude *= 1.5;
            }
            ArrhythmiaType::TWaveInversion => {
                // Deep T wave inversion (ischemia/Wellens)
                params.t_amplitude *= -(1.5 + severity);
                params.t_duration *= 1.2;
                params.t_position = 0.28;
            }
            ArrhythmiaType::TWavePeaking => {
                // Peaked T waves (hyperkalemia)
                params.t_amplitude *= 3.0 + 1.5 * severity;
                params.t_duration *= 0.4;
                params.t_position = 0.22;
                params.r_duration *= 1.2;
            }
            ArrhythmiaType::AtrialFibrillation => {
                // AF: Absent P waves, fibrillatory baseline
                params.p_amplitude = 0.05 * self.rand_gaussian();
                params.p_duration *= 0.3;
                params.r_amplitude *= 0.8 + 0.4 * self.rand().abs();
                params.t_amplitude *= 0.7 + 0.5 * self.rand();
            }
            ArrhythmiaType::Bradycardia => {
                // Rate change handled in get_rr_interval
            }
            ArrhythmiaType::Tachycardia => {
                params.t_amplitude *= 0.8;
                params.t_position = 0.20;
            }
            _ => {}
        }

        params
    }

    /// Generate the next sample of ECG signal.
    pub fn next_sample(&mut self) -> f32 {
        let dt = 1.0 / self.config.sample_rate;
        self.sample_count += 1;

        // Get current RR interval (updates HRV phases)
        let rr_interval = self.get_rr_interval();
        self.last_rr_interval = rr_interval;

        // Advance phase (0-1 over one cardiac cycle)
        self.phase += dt / rr_interval;

        // Check if we're starting a new beat
        let is_new_beat = self.phase >= 1.0;
        if is_new_beat {
            self.phase -= 1.0;
            // Determine if this NEW beat should be an arrhythmia
            self.current_beat_is_arrhythmia = self.rand() < self.arrhythmia.probability;
            // Generate and cache beat parameters for the entire beat
            self.current_beat_params = Some(self.get_beat_params(self.current_beat_is_arrhythmia));
        }

        // Initialize beat params on first sample if needed
        if self.current_beat_params.is_none() {
            self.current_beat_params = Some(self.get_beat_params(false));
        }

        // Use cached beat parameters for consistent morphology throughout beat
        let params = self.current_beat_params.clone().unwrap();

        // Generate PQRST waveform
        let mut signal = self.generate_pqrst(self.phase, &params);

        // Apply amplitude scaling
        signal *= self.config.amplitude;

        // Add baseline wander (very low frequency)
        let baseline = self.config.baseline_wander
            * (0.1 * self.sample_count as f32 / self.config.sample_rate).sin();
        signal += baseline;

        // Add high-frequency noise
        signal += self.config.noise_amplitude * self.rand_gaussian();

        signal
    }

    /// Generate a window of ECG samples.
    pub fn generate_window(&mut self, size: usize) -> Vec<f32> {
        (0..size).map(|_| self.next_sample()).collect()
    }

    /// Reset the generator state.
    pub fn reset(&mut self) {
        self.phase = 0.0;
        self.sample_count = 0;
        self.hrv_phase_vlf = 0.0;
        self.hrv_phase_lf = 0.0;
        self.hrv_phase_hf = 0.0;
        self.current_beat_is_arrhythmia = false;
        self.current_beat_params = None;
    }

    /// Check if current beat is an arrhythmia beat.
    pub fn is_current_beat_arrhythmia(&self) -> bool {
        self.current_beat_is_arrhythmia
    }

    /// Get the current configuration.
    pub fn config(&self) -> &EcgConfig {
        &self.config
    }

    /// Get the current arrhythmia configuration.
    pub fn arrhythmia_config(&self) -> &ArrhythmiaConfig {
        &self.arrhythmia
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ecg_generator_produces_valid_samples() {
        let config = EcgConfig::default();
        let mut generator = EcgGenerator::new(config, ArrhythmiaConfig::default());

        let samples: Vec<f32> = (0..1000).map(|_| generator.next_sample()).collect();

        // Check samples are in reasonable range
        for sample in &samples {
            assert!(sample.abs() < 5.0, "Sample out of range: {}", sample);
        }

        // Check we have variation
        let min = samples.iter().cloned().fold(f32::MAX, f32::min);
        let max = samples.iter().cloned().fold(f32::MIN, f32::max);
        assert!(max - min > 0.5, "Insufficient variation in samples");
    }

    #[test]
    fn test_ecg_generator_with_arrhythmia() {
        let config = EcgConfig::default();
        let arrhythmia = ArrhythmiaConfig {
            arrhythmia_type: ArrhythmiaType::Pvc,
            probability: 0.5,
            severity: 0.7,
        };
        let mut generator = EcgGenerator::new(config, arrhythmia);

        let samples: Vec<f32> = (0..500).map(|_| generator.next_sample()).collect();
        assert_eq!(samples.len(), 500);
    }

    #[test]
    fn test_arrhythmia_types() {
        assert_eq!(ArrhythmiaType::all_types().len(), 9);
        assert_eq!(ArrhythmiaType::Pvc.short_name(), "pvc");
        assert_eq!(
            ArrhythmiaType::AtrialFibrillation.display_name(),
            "Atrial Fibrillation"
        );
    }
}
