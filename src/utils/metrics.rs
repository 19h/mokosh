//! Streaming SDR metrics corresponding to htm.core's SdrMetrics classes.
use crate::types::{Real, Sdr, UInt};

#[derive(Debug, Clone)]
struct Summary {
    period: u64,
    samples: u64,
    value: Real,
    minimum: Real,
    maximum: Real,
    mean: Real,
    variance: Real,
}
impl Summary {
    fn new(period: UInt) -> Self {
        assert!(period > 0, "Metric period must be positive");
        Self {
            period: period as u64,
            samples: 0,
            value: Real::NAN,
            minimum: Real::INFINITY,
            maximum: Real::NEG_INFINITY,
            mean: 1234.567,
            variance: 1234.567,
        }
    }
    fn update(&mut self, value: Real) {
        self.samples += 1;
        self.value = value;
        self.minimum = self.minimum.min(value);
        self.maximum = self.maximum.max(value);
        let alpha = 1.0 / self.period.min(self.samples) as Real;
        let difference = value - self.mean;
        let increment = alpha * difference;
        self.mean += increment;
        self.variance = (1.0 - alpha) * (self.variance + difference * increment);
    }
}

/// Exponentially weighted sparsity, with ordinary-mean warmup.
#[derive(Debug, Clone)]
pub struct Sparsity {
    dimensions: Vec<UInt>,
    state: Summary,
}
impl Sparsity {
    /// Creates a metric for a fixed SDR shape and positive period.
    pub fn new(dimensions: &[UInt], period: UInt) -> Self {
        let _ = Sdr::new(dimensions);
        Self {
            dimensions: dimensions.to_vec(),
            state: Summary::new(period),
        }
    }
    /// Adds one observation.
    pub fn add_data(&mut self, data: &Sdr) {
        assert_eq!(data.dimensions(), self.dimensions);
        self.state.update(data.get_sparsity());
    }
    /// Most recent sparsity, or NaN before the first observation.
    pub fn sparsity(&self) -> Real {
        self.state.value
    }
    /// Smallest observed sparsity.
    pub fn min(&self) -> Real {
        self.state.minimum
    }
    /// Largest observed sparsity.
    pub fn max(&self) -> Real {
        self.state.maximum
    }
    /// Streaming mean.
    pub fn mean(&self) -> Real {
        self.state.mean
    }
    /// Streaming population standard deviation.
    pub fn std(&self) -> Real {
        self.state.variance.max(0.0).sqrt()
    }
}

/// Per-bit exponential activation frequency and normalized binary entropy.
#[derive(Debug, Clone)]
pub struct ActivationFrequency {
    dimensions: Vec<UInt>,
    period: u64,
    samples: u64,
    frequencies: Vec<Real>,
    always_exponential: bool,
}
impl ActivationFrequency {
    /// Creates a metric with ordinary-mean warmup.
    pub fn new(dimensions: &[UInt], period: UInt) -> Self {
        Self::with_initial_value(dimensions, period, None)
    }
    /// Seeds every bit's frequency, optionally bypassing warmup.
    pub fn with_initial_value(dimensions: &[UInt], period: UInt, initial: Option<Real>) -> Self {
        assert!(period > 0, "Metric period must be positive");
        if let Some(v) = initial {
            assert!(v.is_finite() && (0.0..=1.0).contains(&v));
        }
        let size = Sdr::new(dimensions).size();
        assert!(size > 0, "Activation metrics require a nonempty shape");
        Self {
            dimensions: dimensions.to_vec(),
            period: period as u64,
            samples: 0,
            frequencies: vec![initial.unwrap_or(1234.567); size],
            always_exponential: initial.is_some(),
        }
    }
    /// Adds one observation.
    pub fn add_data(&mut self, data: &Sdr) {
        assert_eq!(data.dimensions(), self.dimensions);
        self.samples += 1;
        let alpha = 1.0
            / if self.always_exponential {
                self.period
            } else {
                self.period.min(self.samples)
            } as Real;
        for f in &mut self.frequencies {
            *f *= 1.0 - alpha;
        }
        for i in data.get_sparse() {
            self.frequencies[i as usize] += alpha;
        }
    }
    /// Per-bit frequencies.
    pub fn frequencies(&self) -> &[Real] {
        &self.frequencies
    }
    /// Minimum frequency.
    pub fn min(&self) -> Real {
        self.frequencies
            .iter()
            .copied()
            .fold(Real::INFINITY, Real::min)
    }
    /// Maximum frequency.
    pub fn max(&self) -> Real {
        self.frequencies
            .iter()
            .copied()
            .fold(Real::NEG_INFINITY, Real::max)
    }
    /// Mean activation frequency.
    pub fn mean(&self) -> Real {
        self.frequencies.iter().sum::<Real>() / self.frequencies.len() as Real
    }
    /// Population standard deviation across bits.
    pub fn std(&self) -> Real {
        let mean = self.mean();
        (self
            .frequencies
            .iter()
            .map(|f| (f - mean).powi(2))
            .sum::<Real>()
            / self.frequencies.len() as Real)
            .sqrt()
    }
    /// Binary entropy normalized by the maximum at the observed mean sparsity.
    pub fn entropy(&self) -> Real {
        fn binary(p: Real) -> Real {
            if p <= 0.0 || p >= 1.0 {
                0.0
            } else {
                -p * p.log2() - (1.0 - p) * (1.0 - p).log2()
            }
        }
        let maximum = binary(self.mean());
        if maximum == 0.0 {
            0.0
        } else {
            self.frequencies.iter().map(|&p| binary(p)).sum::<Real>()
                / self.frequencies.len() as Real
                / maximum
        }
    }
}

/// Streaming overlap normalized by the larger active-bit count.
#[derive(Debug, Clone)]
pub struct Overlap {
    dimensions: Vec<UInt>,
    state: Summary,
    previous: Option<Sdr>,
}
impl Overlap {
    /// Creates a metric for a fixed shape and period.
    pub fn new(dimensions: &[UInt], period: UInt) -> Self {
        let _ = Sdr::new(dimensions);
        Self {
            dimensions: dimensions.to_vec(),
            state: Summary::new(period),
            previous: None,
        }
    }
    /// Starts a new sequence while preserving accumulated statistics.
    pub fn reset(&mut self) {
        self.previous = None;
        self.state.value = Real::NAN;
    }
    /// Adds one observation. The first observation of each sequence establishes context.
    pub fn add_data(&mut self, data: &Sdr) {
        assert_eq!(data.dimensions(), self.dimensions);
        if let Some(previous) = &self.previous {
            let size = previous.get_sum().max(data.get_sum());
            self.state.update(if size == 0 {
                1.0
            } else {
                previous.get_overlap(data) as Real / size as Real
            });
        }
        self.previous = Some(data.clone());
    }
    /// Most recent normalized overlap.
    pub fn overlap(&self) -> Real {
        self.state.value
    }
    /// Minimum overlap.
    pub fn min(&self) -> Real {
        self.state.minimum
    }
    /// Maximum overlap.
    pub fn max(&self) -> Real {
        self.state.maximum
    }
    /// Streaming mean.
    pub fn mean(&self) -> Real {
        self.state.mean
    }
    /// Population standard deviation.
    pub fn std(&self) -> Real {
        self.state.variance.max(0.0).sqrt()
    }
}

/// The three htm.core streaming metrics over one SDR shape.
#[derive(Debug, Clone)]
pub struct Metrics {
    /// Sparsity statistics.
    pub sparsity: Sparsity,
    /// Bit-frequency statistics.
    pub activation_frequency: ActivationFrequency,
    /// Consecutive-overlap statistics.
    pub overlap: Overlap,
}
impl Metrics {
    /// Creates all three metrics with the same shape and period.
    pub fn new(dimensions: &[UInt], period: UInt) -> Self {
        Self {
            sparsity: Sparsity::new(dimensions, period),
            activation_frequency: ActivationFrequency::new(dimensions, period),
            overlap: Overlap::new(dimensions, period),
        }
    }
    /// Adds one observation to all metrics.
    pub fn add_data(&mut self, data: &Sdr) {
        self.sparsity.add_data(data);
        self.activation_frequency.add_data(data);
        self.overlap.add_data(data);
    }
    /// Starts a new overlap sequence; frequency and sparsity history are retained.
    pub fn reset(&mut self) {
        self.overlap.reset();
    }
}

/// Automatically updated metrics whose observer disconnects when dropped.
pub struct AttachedMetrics {
    state: std::sync::Arc<std::sync::Mutex<Metrics>>,
    _subscription: crate::types::SdrSubscription,
}
impl AttachedMetrics {
    /// Returns a consistent snapshot of the current metrics.
    pub fn snapshot(&self) -> Metrics {
        self.state.lock().expect("Metric state poisoned").clone()
    }
}
impl Metrics {
    /// Attaches all metrics to assignments to an SDR, with RAII detachment.
    pub fn attach(data: &Sdr, period: UInt) -> AttachedMetrics {
        let state =
            std::sync::Arc::new(std::sync::Mutex::new(Self::new(data.dimensions(), period)));
        let observer = state.clone();
        let subscription = data.subscribe(move |data| {
            observer
                .lock()
                .expect("Metric state poisoned")
                .add_data(data)
        });
        AttachedMetrics {
            state,
            _subscription: subscription,
        }
    }
}
