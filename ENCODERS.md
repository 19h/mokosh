# Mokosh Encoders Reference

Encoders convert raw data into Sparse Distributed Representations (SDRs) suitable for HTM processing. This document provides comprehensive documentation for all 39 encoders in Mokosh.

## Table of Contents

- [Encoder Trait](#encoder-trait)
- [Scalar Encoders](#scalar-encoders)
- [Categorical Encoders](#categorical-encoders)
- [Temporal Encoders](#temporal-encoders)
- [Spatial Encoders](#spatial-encoders)
- [Text & NLP Encoders](#text--nlp-encoders)
- [Audio Encoders](#audio-encoders)
- [Vision Encoders](#vision-encoders)
- [Network Encoders](#network-encoders)
- [Biometric Encoders](#biometric-encoders)
- [Financial Encoders](#financial-encoders)
- [Probabilistic Encoders](#probabilistic-encoders)
- [Composite Encoders](#composite-encoders)

---

## Encoder Trait

All encoders implement the `Encoder<T>` trait:

```rust
pub trait Encoder<T> {
    fn dimensions(&self) -> &[UInt];
    fn size(&self) -> usize;
    fn encode(&self, value: T, output: &mut Sdr) -> Result<()>;
    fn encode_to_sdr(&self, value: T) -> Result<Sdr>;  // convenience method
}
```

Common patterns:
- Each encoder has a corresponding `*Params` struct for configuration
- All params structs implement `Default`
- Constructors return `Result<Self>` with validation errors
- Encodings are deterministic (same input → same output)
- Similar inputs produce overlapping SDRs (semantic similarity preservation)

---

## Scalar Encoders

### ScalarEncoder

Encodes numeric values as contiguous blocks of active bits. Adjacent values have overlapping representations.

```rust
use mokosh::encoders::{ScalarEncoder, ScalarEncoderParams, Encoder};

let encoder = ScalarEncoder::new(ScalarEncoderParams {
    minimum: 0.0,
    maximum: 100.0,
    size: 400,
    active_bits: 21,
    clip_input: true,   // clip out-of-range values (vs error)
    periodic: false,    // true for circular values like angles
    ..Default::default()
})?;

let sdr = encoder.encode_to_sdr(42.0)?;
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `minimum` | `Real` | 0.0 | Minimum input value |
| `maximum` | `Real` | 100.0 | Maximum input value |
| `size` | `UInt` | 400 | Total bits in output SDR |
| `active_bits` | `UInt` | 21 | Number of active bits |
| `radius` | `Real` | 0.0 | Distance for non-overlapping (auto-computes size) |
| `clip_input` | `bool` | true | Clip values outside range |
| `periodic` | `bool` | false | Wrap-around for circular values |
| `category` | `bool` | false | Non-overlapping discrete categories |

**Input types:** `Real`

---

### RandomDistributedScalarEncoder (RDSE)

Hash-based encoding that doesn't require knowing min/max in advance. Values within `resolution` of each other share bits.

```rust
use mokosh::encoders::{RandomDistributedScalarEncoder, RdseParams, Encoder};

let encoder = RandomDistributedScalarEncoder::new(RdseParams {
    size: 1000,
    active_bits: 40,
    resolution: 1.0,  // values within 1.0 share most bits
    seed: 42,
    ..Default::default()
})?;

let sdr = encoder.encode_to_sdr(42.5)?;
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `size` | `UInt` | 1000 | Total bits in output SDR |
| `active_bits` | `UInt` | 40 | Number of active bits (mutually exclusive with `sparsity`) |
| `sparsity` | `Real` | 0.0 | Fraction of active bits |
| `resolution` | `Real` | 1.0 | Granularity (mutually exclusive with `radius`/`category`) |
| `radius` | `Real` | 0.0 | Distance for non-overlapping encodings |
| `category` | `bool` | false | Treat inputs as discrete categories |
| `seed` | `u32` | 42 | Random seed for reproducibility |

**Input types:** `Real`

---

### LogEncoder

Encodes positive values on a logarithmic scale. Equal ratios produce equal spacing.

```rust
use mokosh::encoders::{LogEncoder, LogEncoderParams, Encoder};

let encoder = LogEncoder::new(LogEncoderParams {
    minimum: 1.0,      // must be > 0
    maximum: 10000.0,
    size: 400,
    active_bits: 21,
    clip_input: true,
})?;

// 10 and 100 are as different as 100 and 1000
let sdr = encoder.encode_to_sdr(100.0)?;
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `minimum` | `Real` | 1.0 | Minimum value (must be > 0) |
| `maximum` | `Real` | 1000.0 | Maximum value |
| `size` | `UInt` | 400 | Total bits |
| `active_bits` | `UInt` | 21 | Active bits |
| `clip_input` | `bool` | true | Clip out-of-range values |

**Input types:** `Real`

---

### DeltaEncoder

Stateful encoder that encodes the rate of change between consecutive values.

```rust
use mokosh::encoders::{DeltaEncoder, DeltaEncoderParams, Encoder};

let mut encoder = DeltaEncoder::new(DeltaEncoderParams {
    max_delta: 10.0,
    size: 200,
    active_bits: 11,
    periodic: false,
    clip_input: true,
})?;

encoder.encode_to_sdr(100.0)?;  // first value establishes baseline
let delta_sdr = encoder.encode_to_sdr(105.0)?;  // encodes +5.0 change
encoder.reset();  // reset for new sequence
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_delta` | `Real` | 10.0 | Maximum expected change magnitude |
| `size` | `UInt` | 200 | Total bits |
| `active_bits` | `UInt` | 11 | Active bits |
| `periodic` | `bool` | false | Wrap large deltas |
| `clip_input` | `bool` | true | Clip deltas outside range |

**Input types:** `Real`

---

## Categorical Encoders

### CategoryEncoder

Encodes discrete categories with non-overlapping bit patterns.

```rust
use mokosh::encoders::{CategoryEncoder, CategoryEncoderParams, Encoder};

let encoder = CategoryEncoder::new(CategoryEncoderParams {
    categories: vec!["red".into(), "green".into(), "blue".into()],
    active_bits: 10,
    size: 0,  // auto-computed: categories.len() * active_bits
})?;

let sdr = encoder.encode_to_sdr("green")?;
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `categories` | `Vec<String>` | `[]` | List of category names |
| `active_bits` | `UInt` | 10 | Bits per category |
| `size` | `UInt` | 0 | Total bits (0 = auto-compute) |

**Input types:** `&str`, `String`, `usize` (index)

---

### BooleanEncoder

Encodes true/false with non-overlapping patterns.

```rust
use mokosh::encoders::{BooleanEncoder, BooleanEncoderParams, Encoder};

let encoder = BooleanEncoder::new(BooleanEncoderParams {
    active_bits: 10,
})?;

let sdr_true = encoder.encode_to_sdr(true)?;
let sdr_false = encoder.encode_to_sdr(false)?;
assert_eq!(sdr_true.get_overlap(&sdr_false), 0);  // no overlap
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `active_bits` | `UInt` | 10 | Bits per state (total size = 2 × active_bits) |

**Input types:** `bool`, `u8`, `i32`

---

### OrdinalEncoder

Encodes ordered categories where adjacent values share bits.

```rust
use mokosh::encoders::{OrdinalEncoder, OrdinalEncoderParams, Encoder};

let encoder = OrdinalEncoder::new(OrdinalEncoderParams {
    categories: vec!["cold".into(), "cool".into(), "warm".into(), "hot".into()],
    size: 100,
    active_bits: 10,
})?;

// "cool" and "warm" have overlapping SDRs (adjacent)
// "cold" and "hot" have minimal overlap (far apart)
let sdr = encoder.encode_to_sdr("warm")?;
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `categories` | `Vec<String>` | `[]` | Ordered category names |
| `size` | `UInt` | 100 | Total bits |
| `active_bits` | `UInt` | 10 | Active bits |

**Input types:** `&str`, `String`, `usize` (index)

---

### HierarchicalCategoryEncoder

Encodes categories in a taxonomy where parent categories share bits with children.

```rust
use mokosh::encoders::{HierarchicalCategoryEncoder, HierarchicalCategoryEncoderParams, Encoder};

let encoder = HierarchicalCategoryEncoder::new(HierarchicalCategoryEncoderParams {
    separator: "/".into(),
    bits_per_level: 50,
    active_per_level: 5,
    max_depth: 4,
})?;

// "animal/mammal/dog" shares bits with "animal/mammal/cat"
// Both share bits with "animal/mammal"
let sdr = encoder.encode_to_sdr("animal/mammal/dog")?;
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `separator` | `String` | "/" | Hierarchy level separator |
| `bits_per_level` | `UInt` | 50 | Bits per hierarchy level |
| `active_per_level` | `UInt` | 5 | Active bits per level |
| `max_depth` | `usize` | 4 | Maximum hierarchy depth |

**Input types:** `&str`, `String`, `Vec<&str>`

---

### SetEncoder

Encodes variable-size sets of items. Order-invariant.

```rust
use mokosh::encoders::{SetEncoder, SetEncoderParams, Encoder};

let encoder = SetEncoder::new(SetEncoderParams {
    size: 500,
    bits_per_element: 10,
    max_elements: 20,
})?;

// Order doesn't matter
let sdr1 = encoder.encode_to_sdr(vec!["apple", "banana"])?;
let sdr2 = encoder.encode_to_sdr(vec!["banana", "apple"])?;
assert_eq!(sdr1.get_sparse(), sdr2.get_sparse());
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `size` | `UInt` | 500 | Total bits |
| `bits_per_element` | `UInt` | 10 | Bits per set member |
| `max_elements` | `usize` | 20 | Maximum elements to encode |

**Input types:** `Vec<&str>`, `Vec<String>`, `HashSet<String>`, `&[&str]`

---

## Temporal Encoders

### DateEncoder

Multi-attribute temporal encoding with configurable components.

```rust
use mokosh::encoders::{DateEncoder, DateEncoderParams, DateTime, Encoder};

let encoder = DateEncoder::new(DateEncoderParams {
    season_width: 3,        // day-of-year (seasonal patterns)
    day_of_week_width: 1,   // Mon-Sun
    weekend_width: 3,       // weekend flag
    time_of_day_width: 5,   // hour of day
    holiday_width: 0,       // disabled
    ..Default::default()
})?;

let dt = DateTime {
    year: 2024, month: 6, day: 15,
    hour: 14, minute: 30, second: 0,
};
let sdr = encoder.encode_to_sdr(dt)?;
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `season_width` | `UInt` | 3 | Bits for day-of-year (0 = disable) |
| `season_radius` | `Real` | 91.5 | Days for seasonal granularity |
| `day_of_week_width` | `UInt` | 1 | Bits for day of week |
| `day_of_week_radius` | `Real` | 1.0 | Day radius |
| `weekend_width` | `UInt` | 3 | Bits for weekend flag |
| `holiday_width` | `UInt` | 0 | Bits for holidays |
| `holiday_dates` | `Vec<Holiday>` | `[]` | Holiday definitions |
| `time_of_day_width` | `UInt` | 5 | Bits for time |
| `time_of_day_radius` | `Real` | 4.0 | Hours per bucket |
| `custom_width` | `UInt` | 0 | Bits for custom days |
| `custom_days` | `Vec<String>` | `[]` | Custom day specs |

**Input types:** `DateTime`, `i64` (Unix timestamp)

---

## Spatial Encoders

### CoordinateEncoder

Encodes N-dimensional coordinates with locality-sensitive hashing.

```rust
use mokosh::encoders::{CoordinateEncoder, CoordinateEncoderParams, Encoder};

let encoder = CoordinateEncoder::new(CoordinateEncoderParams {
    num_dimensions: 2,
    size: 500,
    active_bits: 25,
    radius: 1.0,  // distance that shares ~1 bit
})?;

// Nearby coordinates have overlapping SDRs
let sdr1 = encoder.encode_to_sdr((10.0, 20.0))?;
let sdr2 = encoder.encode_to_sdr((10.5, 20.5))?;
assert!(sdr1.get_overlap(&sdr2) > 0);
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_dimensions` | `usize` | 2 | Coordinate space dimensionality |
| `size` | `UInt` | 500 | Total bits |
| `active_bits` | `UInt` | 25 | Active bits |
| `radius` | `Real` | 1.0 | Distance where overlap decreases ~1 bit |

**Input types:** `Vec<Real>`, `&[Real]`, `(Real, Real)`, `(Real, Real, Real)`

---

### GeospatialEncoder

Encodes GPS coordinates (latitude/longitude) with multi-scale locality.

```rust
use mokosh::encoders::{GeospatialEncoder, GeospatialEncoderParams, GpsCoordinate, Encoder};

let encoder = GeospatialEncoder::new(GeospatialEncoderParams {
    size: 1000,
    active_bits: 50,
    scale: 1000.0,     // meters for base locality
    num_scales: 3,     // multi-resolution levels
})?;

let coord = GpsCoordinate { latitude: 40.7128, longitude: -74.0060 };
let sdr = encoder.encode_to_sdr(coord)?;
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `size` | `UInt` | 1000 | Total bits |
| `active_bits` | `UInt` | 50 | Active bits |
| `scale` | `Real` | 1000.0 | Base scale in meters |
| `num_scales` | `usize` | 3 | Multi-resolution levels |

**Input types:** `GpsCoordinate`, `(Real, Real)` (lat, lon)

---

### GridCellEncoder

Biologically-inspired encoder mimicking entorhinal cortex grid cells.

```rust
use mokosh::encoders::{GridCellEncoder, GridCellEncoderParams, Encoder};

let encoder = GridCellEncoder::new(GridCellEncoderParams {
    num_modules: 4,
    cells_per_module: 64,  // must be perfect square
    base_scale: 1.0,
    scale_ratio: 1.5,      // each module 1.5x larger
    orientation_offset: 0.1,
})?;

let sdr = encoder.encode_to_sdr((10.0, 20.0))?;
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_modules` | `usize` | 4 | Number of grid modules |
| `cells_per_module` | `UInt` | 64 | Cells per module (perfect square) |
| `base_scale` | `Real` | 1.0 | Smallest grid period |
| `scale_ratio` | `Real` | 1.5 | Scale multiplier between modules |
| `orientation_offset` | `Real` | 0.1 | Rotation offset in radians |

**Input types:** `(Real, Real)`, `[Real; 2]`, `Vec<Real>`

---

## Text & NLP Encoders

### SimHashDocumentEncoder

Encodes documents/text with SimHash for similarity preservation.

```rust
use mokosh::encoders::{SimHashDocumentEncoder, SimHashDocumentEncoderParams, Encoder};

let encoder = SimHashDocumentEncoder::new(SimHashDocumentEncoderParams {
    size: 2048,
    active_bits: 41,
    case_sensitivity: false,
    token_similarity: true,    // character-level similarity
    frequency_ceiling: 5,      // max token repetitions
    ..Default::default()
})?;

let sdr = encoder.encode_to_sdr("The quick brown fox jumps over the lazy dog")?;
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `size` | `UInt` | 2048 | Total bits |
| `active_bits` | `UInt` | 41 | Active bits |
| `sparsity` | `Real` | 0.0 | Alternative to active_bits |
| `case_sensitivity` | `bool` | false | Case-sensitive encoding |
| `encode_orphans` | `bool` | true | Encode unknown tokens |
| `excludes` | `Vec<String>` | `[]` | Tokens to exclude |
| `frequency_ceiling` | `UInt` | 0 | Max token repetitions (0 = no limit) |
| `frequency_floor` | `UInt` | 0 | Min token occurrences |
| `token_similarity` | `bool` | false | Character-level similarity |
| `vocabulary` | `HashMap<String, UInt>` | `{}` | Token weights |

**Input types:** `&str`, `String`, `Vec<String>`

---

### WordEmbeddingEncoder

Converts pre-trained word vectors (word2vec, GloVe) to SDRs. **Requires fixed dimension.**

```rust
use mokosh::encoders::{WordEmbeddingEncoder, WordEmbeddingEncoderParams, Encoder};

let encoder = WordEmbeddingEncoder::new(WordEmbeddingEncoderParams {
    embedding_dim: 300,  // word2vec/GloVe dimension
    size: 2048,
    active_bits: 41,
    num_hyperplanes: 128,
})?;

let word_vector: Vec<f32> = load_word2vec("cat");  // your embedding
let sdr = encoder.encode_to_sdr(word_vector)?;
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedding_dim` | `usize` | 300 | **Fixed** input embedding dimension |
| `size` | `UInt` | 2048 | Total output bits |
| `active_bits` | `UInt` | 41 | Active bits |
| `num_hyperplanes` | `usize` | 128 | LSH hyperplanes |

**Input types:** `Vec<Real>`, `&[Real]`

---

### LlmEmbeddingEncoder

Converts LLM embeddings of **arbitrary dimensions** (384, 768, 1536, 3072, etc.) to SDRs.

```rust
use mokosh::encoders::{LlmEmbeddingEncoder, LlmEmbeddingEncoderParams, Encoder};

let encoder = LlmEmbeddingEncoder::new(LlmEmbeddingEncoderParams {
    size: 2048,
    active_bits: 41,
    bits_per_dimension: 2,
    normalize: true,
    mix_embedding_length: true,  // different dims → different hashes
})?;

// Works with ANY embedding dimension
let ada_002 = get_openai_embedding();     // 1536-dim
let minilm = get_sentence_transformer();  // 384-dim
let bert = get_bert_embedding();          // 768-dim

let sdr1 = encoder.encode_to_sdr(ada_002.as_slice())?;
let sdr2 = encoder.encode_to_sdr(minilm.as_slice())?;
let sdr3 = encoder.encode_to_sdr(bert.as_slice())?;
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `size` | `UInt` | 2048 | Total bits (must be divisible by `bits_per_dimension`) |
| `active_bits` | `UInt` | 41 | Active bits |
| `bits_per_dimension` | `UInt` | 2 | Bits per projected dimension (≥2) |
| `normalize` | `bool` | true | L2-normalize embeddings |
| `mix_embedding_length` | `bool` | true | Mix length into hash (prevents cross-model comparison) |

**Input types:** `&[Real]`, `Vec<Real>`

**Key features:**
- Handles arbitrary embedding dimensions without pre-configuration
- Hash-based random projection with deterministic output
- Similar embeddings → overlapping SDRs
- Optional length mixing prevents comparing embeddings from different models

---

### CharacterEncoder

Character-level encoding with optional semantic similarity for adjacent characters.

```rust
use mokosh::encoders::{CharacterEncoder, CharacterEncoderParams, Encoder};

let encoder = CharacterEncoder::new(CharacterEncoderParams {
    charset: None,  // default: ASCII printable
    active_bits: 10,
    include_unknown: true,
    semantic_similarity: true,  // 'a' and 'b' share bits
    overlap_bits: 3,
})?;

let sdr = encoder.encode_to_sdr('x')?;
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `charset` | `Option<Vec<char>>` | `None` | Custom character set |
| `active_bits` | `UInt` | 10 | Bits per character |
| `include_unknown` | `bool` | true | Handle unknown characters |
| `semantic_similarity` | `bool` | false | Adjacent characters share bits |
| `overlap_bits` | `UInt` | 3 | Overlap for similarity |

**Input types:** `char`, `u8`

---

### NGramEncoder

Encodes character or word n-grams preserving sequence information.

```rust
use mokosh::encoders::{NGramEncoder, NGramEncoderParams, Encoder};

let encoder = NGramEncoder::new(NGramEncoderParams {
    n: 3,  // trigrams
    vocabulary_size: 1000,
    size: 500,
    active_bits: 25,
})?;

let tokens = vec!["the", "quick", "brown", "fox"];
let sdr = encoder.encode_to_sdr(tokens)?;
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n` | `usize` | 3 | N-gram size |
| `vocabulary_size` | `usize` | 1000 | Unique token count |
| `size` | `UInt` | 500 | Total bits |
| `active_bits` | `UInt` | 25 | Active bits per n-gram |

**Input types:** `Vec<&str>`, `Vec<String>`

---

## Audio Encoders

### SpectrogramEncoder

Encodes audio frequency spectra (FFT/mel spectrogram bins).

```rust
use mokosh::encoders::{SpectrogramEncoder, SpectrogramEncoderParams, Encoder};

let encoder = SpectrogramEncoder::new(SpectrogramEncoderParams {
    num_bins: 64,
    bits_per_bin: 10,
    active_bits_per_bin: 2,
    min_magnitude: 0.0,
    max_magnitude: 1.0,
    log_scale: true,  // perceptual scaling
})?;

let spectrum: Vec<f32> = compute_fft(audio_frame);
let sdr = encoder.encode_to_sdr(spectrum)?;
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_bins` | `usize` | 64 | Frequency bins |
| `bits_per_bin` | `UInt` | 10 | Bits allocated per bin |
| `active_bits_per_bin` | `UInt` | 2 | Active bits when bin activated |
| `min_magnitude` | `Real` | 0.0 | Silence threshold |
| `max_magnitude` | `Real` | 1.0 | Maximum expected magnitude |
| `log_scale` | `bool` | true | Log scale for perceptual accuracy |

**Input types:** `Vec<Real>`, `&[Real]`

---

### WaveformEncoder

Encodes raw audio samples directly.

```rust
use mokosh::encoders::{WaveformEncoder, WaveformEncoderParams, Encoder};

let encoder = WaveformEncoder::new(WaveformEncoderParams {
    window_size: 256,
    bits_per_sample: 8,
    active_bits_per_sample: 2,
    min_value: -1.0,
    max_value: 1.0,
})?;

let audio_window: Vec<f32> = get_audio_samples(256);
let sdr = encoder.encode_to_sdr(audio_window)?;
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `window_size` | `usize` | 256 | Samples per window |
| `bits_per_sample` | `UInt` | 8 | Bits per sample |
| `active_bits_per_sample` | `UInt` | 2 | Active bits per sample |
| `min_value` | `Real` | -1.0 | Minimum sample value |
| `max_value` | `Real` | 1.0 | Maximum sample value |

**Input types:** `Vec<Real>`, `&[Real]`

---

### PitchEncoder

Encodes musical pitch with separate octave and pitch class components.

```rust
use mokosh::encoders::{PitchEncoder, PitchEncoderParams, Pitch, Encoder};

let encoder = PitchEncoder::new(PitchEncoderParams {
    pitch_class_bits: 100,
    pitch_class_active: 10,
    octave_bits: 50,
    octave_active: 5,
    min_octave: 0,
    max_octave: 8,
    cents_resolution: 10.0,
})?;

// From frequency
let sdr = encoder.encode_to_sdr(440.0)?;  // A4

// From Pitch struct
let pitch = Pitch { midi_note: 69, cents: 0.0 };
let sdr = encoder.encode_to_sdr(pitch)?;
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pitch_class_bits` | `UInt` | 100 | Bits for pitch class (0-11, periodic) |
| `pitch_class_active` | `UInt` | 10 | Active bits for pitch class |
| `octave_bits` | `UInt` | 50 | Bits for octave (linear) |
| `octave_active` | `UInt` | 5 | Active bits for octave |
| `min_octave` | `i32` | 0 | Minimum octave |
| `max_octave` | `i32` | 8 | Maximum octave |
| `cents_resolution` | `Real` | 10.0 | Sub-semitone precision |

**Input types:** `Pitch`, `Real` (frequency in Hz)

---

## Vision Encoders

### PatchEncoder

Encodes image patches using random projections (Vision Transformer-like).

```rust
use mokosh::encoders::{PatchEncoder, PatchEncoderParams, ImagePatch, Encoder};

let encoder = PatchEncoder::new(PatchEncoderParams {
    patch_width: 16,
    patch_height: 16,
    channels: 3,  // RGB
    size: 2048,
    active_bits: 41,
    num_projections: 128,
})?;

let patch = ImagePatch {
    width: 16, height: 16, channels: 3,
    pixels: image_data,  // Vec<Real> of pixel values
};
let sdr = encoder.encode_to_sdr(patch)?;
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `patch_width` | `usize` | 16 | Patch width in pixels |
| `patch_height` | `usize` | 16 | Patch height in pixels |
| `channels` | `usize` | 3 | Color channels (1=gray, 3=RGB) |
| `size` | `UInt` | 2048 | Total bits |
| `active_bits` | `UInt` | 41 | Active bits |
| `num_projections` | `usize` | 128 | Random projection count |

**Input types:** `ImagePatch`, `Vec<Real>` (raw pixels)

---

### ColorEncoder

Encodes colors in various color spaces.

```rust
use mokosh::encoders::{ColorEncoder, ColorEncoderParams, RgbColor, Encoder};

let encoder = ColorEncoder::new(ColorEncoderParams {
    size: 300,
    active_bits: 15,
    channel_separation: true,  // encode channels separately
})?;

let color = RgbColor { r: 255, g: 128, b: 64 };
let sdr = encoder.encode_to_sdr(color)?;
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `size` | `UInt` | 300 | Total bits |
| `active_bits` | `UInt` | 15 | Active bits |
| `channel_separation` | `bool` | true | Separate channel encoding |

**Input types:** `RgbColor`, `HsvColor`, `(u8, u8, u8)`

---

### EdgeOrientationEncoder

Encodes oriented edge features (gradient directions).

```rust
use mokosh::encoders::{EdgeOrientationEncoder, EdgeOrientationEncoderParams, Encoder};

let encoder = EdgeOrientationEncoder::new(EdgeOrientationEncoderParams {
    num_orientations: 8,  // 8 directions (45° apart)
    size: 200,
    active_bits: 10,
})?;

// From angle in radians
let sdr = encoder.encode_to_sdr(std::f32::consts::PI / 4.0)?;

// From gradient (dx, dy)
let sdr = encoder.encode_to_sdr((1.0, 1.0))?;
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_orientations` | `usize` | 8 | Quantization levels (4, 8, 16, etc.) |
| `size` | `UInt` | 200 | Total bits |
| `active_bits` | `UInt` | 10 | Active bits |

**Input types:** `Real` (angle), `(Real, Real)` (dx, dy), `(Real, Real, Real)` (dx, dy, strength)

---

## Network Encoders

### IpAddressEncoder

Encodes IPv4/IPv6 addresses with subnet hierarchy awareness.

```rust
use mokosh::encoders::{IpAddressEncoder, IpAddressEncoderParams, Encoder};
use std::net::Ipv4Addr;

let encoder = IpAddressEncoder::new(IpAddressEncoderParams {
    bits_per_segment: 32,
    active_per_segment: 4,
    support_ipv6: false,
})?;

let sdr = encoder.encode_to_sdr(Ipv4Addr::new(192, 168, 1, 100))?;
// Or from string
let sdr = encoder.encode_to_sdr("192.168.1.100")?;
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bits_per_segment` | `UInt` | 32 | Bits per octet/segment |
| `active_per_segment` | `UInt` | 4 | Active bits per segment |
| `support_ipv6` | `bool` | false | IPv6 support (4 vs 8 segments) |

**Input types:** `Ipv4Addr`, `Ipv6Addr`, `IpAddr`, `&str`

---

### MacAddressEncoder

Encodes MAC addresses with OUI (vendor) awareness.

```rust
use mokosh::encoders::{MacAddressEncoder, MacAddressEncoderParams, MacAddress, Encoder};

let encoder = MacAddressEncoder::new(MacAddressEncoderParams {
    bits_per_octet: 32,
    active_per_octet: 4,
    separate_oui: true,  // vendor prefix gets separate encoding
})?;

let mac = MacAddress([0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF]);
let sdr = encoder.encode_to_sdr(mac)?;
// Or from string
let sdr = encoder.encode_to_sdr("AA:BB:CC:DD:EE:FF")?;
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bits_per_octet` | `UInt` | 32 | Bits per octet |
| `active_per_octet` | `UInt` | 4 | Active bits per octet |
| `separate_oui` | `bool` | true | OUI-separate encoding |

**Input types:** `MacAddress`, `&str`

---

### GraphNodeEncoder

Encodes graph node embeddings using LSH.

```rust
use mokosh::encoders::{GraphNodeEncoder, GraphNodeEncoderParams, GraphNode, Encoder};

let encoder = GraphNodeEncoder::new(GraphNodeEncoderParams {
    embedding_dim: 64,
    size: 1000,
    active_bits: 50,
    num_hyperplanes: 64,
})?;

let node = GraphNode {
    embedding: vec![0.1; 64],
    degree: Some(5),
    clustering_coefficient: Some(0.3),
};
let sdr = encoder.encode_to_sdr(node)?;
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedding_dim` | `usize` | 64 | Node embedding dimensions |
| `size` | `UInt` | 1000 | Total bits |
| `active_bits` | `UInt` | 50 | Active bits |
| `num_hyperplanes` | `usize` | 64 | LSH hyperplanes |

**Input types:** `GraphNode`, `Vec<Real>`

---

## Biometric Encoders

### HrvEncoder

Encodes heart rate variability metrics.

```rust
use mokosh::encoders::{HrvEncoder, HrvEncoderParams, HrvMetrics, Encoder};

let encoder = HrvEncoder::new(HrvEncoderParams {
    mean_rr_bits: 50,
    mean_rr_active: 5,
    sdnn_bits: 50,
    sdnn_active: 5,
    rmssd_bits: 50,
    rmssd_active: 5,
    pnn50_bits: 50,
    pnn50_active: 5,
    mean_rr_range: (400.0, 1200.0),  // ms
    sdnn_range: (0.0, 200.0),
    rmssd_range: (0.0, 100.0),
})?;

// From pre-computed metrics
let metrics = HrvMetrics {
    mean_rr: 800.0,
    sdnn: 50.0,
    rmssd: 30.0,
    pnn50: 15.0,
};
let sdr = encoder.encode_to_sdr(metrics)?;

// Or from raw RR intervals (auto-computes metrics)
let rr_intervals: Vec<f32> = vec![800.0, 810.0, 790.0, 820.0];
let sdr = encoder.encode_to_sdr(rr_intervals)?;
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mean_rr_bits/active` | `UInt` | 50/5 | Mean RR interval encoding |
| `sdnn_bits/active` | `UInt` | 50/5 | Standard deviation encoding |
| `rmssd_bits/active` | `UInt` | 50/5 | Root mean square of differences |
| `pnn50_bits/active` | `UInt` | 50/5 | Percentage of NN50 |
| `mean_rr_range` | `(Real, Real)` | (400, 1200) | Mean RR physiological range (ms) |
| `sdnn_range` | `(Real, Real)` | (0, 200) | SDNN range (ms) |
| `rmssd_range` | `(Real, Real)` | (0, 100) | RMSSD range (ms) |

**Input types:** `HrvMetrics`, `Vec<Real>` (RR intervals)

---

### EcgEncoder

Encodes ECG waveform morphology and statistics.

```rust
use mokosh::encoders::{EcgEncoder, EcgEncoderParams, Encoder};

let encoder = EcgEncoder::new(EcgEncoderParams {
    window_size: 256,
    bits_per_sample: 8,
    active_per_sample: 2,
    stats_bits: 100,
    stats_active: 10,
    amplitude_range: (-1.0, 1.0),
})?;

let ecg_window: Vec<f32> = get_ecg_samples(256);
let sdr = encoder.encode_to_sdr(ecg_window)?;
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `window_size` | `usize` | 256 | Samples per window |
| `bits_per_sample` | `UInt` | 8 | Bits per sample |
| `active_per_sample` | `UInt` | 2 | Active bits per sample |
| `stats_bits` | `UInt` | 100 | Statistics encoding bits |
| `stats_active` | `UInt` | 10 | Statistics active bits |
| `amplitude_range` | `(Real, Real)` | (-1, 1) | Expected amplitude range |

**Input types:** `Vec<Real>`, `&[Real]`

---

### AccelerometerEncoder

Encodes 3-axis accelerometer data with derived features.

```rust
use mokosh::encoders::{AccelerometerEncoder, AccelerometerEncoderParams, AccelerometerReading, Encoder};

let encoder = AccelerometerEncoder::new(AccelerometerEncoderParams {
    axis_bits: 50,
    axis_active: 5,
    magnitude_bits: 50,
    magnitude_active: 5,
    orientation_bits: 50,
    orientation_active: 5,
    max_acceleration: 16.0,  // g
})?;

let reading = AccelerometerReading { x: 0.1, y: 0.2, z: 9.8 };
let sdr = encoder.encode_to_sdr(reading)?;

// Or from tuple
let sdr = encoder.encode_to_sdr((0.1, 0.2, 9.8))?;
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `axis_bits/active` | `UInt` | 50/5 | Per-axis (X, Y, Z) encoding |
| `magnitude_bits/active` | `UInt` | 50/5 | Total acceleration magnitude |
| `orientation_bits/active` | `UInt` | 50/5 | Pitch/roll orientation |
| `max_acceleration` | `Real` | 16.0 | Maximum expected (g) |

**Input types:** `AccelerometerReading`, `(Real, Real, Real)`, `[Real; 3]`

---

## Financial Encoders

### PriceEncoder

Encodes financial prices with optional log-scale and price change.

```rust
use mokosh::encoders::{PriceEncoder, PriceEncoderParams, PriceData, Encoder};

let encoder = PriceEncoder::new(PriceEncoderParams {
    min_price: 0.01,
    max_price: 10000.0,
    price_bits: 200,
    price_active: 10,
    change_bits: 100,
    change_active: 5,
    max_percent_change: 10.0,
    log_scale: true,
})?;

let price_data = PriceData {
    price: 150.0,
    previous_price: Some(145.0),
};
let sdr = encoder.encode_to_sdr(price_data)?;

// Or just price
let sdr = encoder.encode_to_sdr(150.0)?;
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_price/max_price` | `Real` | 0.01/10000 | Price range |
| `price_bits/active` | `UInt` | 200/10 | Price encoding |
| `change_bits/active` | `UInt` | 100/5 | Price change encoding |
| `max_percent_change` | `Real` | 10.0 | Max change for normalization |
| `log_scale` | `bool` | true | Log scale for prices |

**Input types:** `PriceData`, `Real`

---

### CurrencyPairEncoder

Encodes forex currency pairs. Pairs sharing currencies have overlapping encodings.

```rust
use mokosh::encoders::{CurrencyPairEncoder, CurrencyPairEncoderParams, CurrencyPair, Encoder};

let encoder = CurrencyPairEncoder::new(CurrencyPairEncoderParams {
    base_bits: 100,
    base_active: 10,
    quote_bits: 100,
    quote_active: 10,
    rate_bits: 200,
    rate_active: 10,
    rate_range: (0.1, 10.0),
    log_scale: true,
})?;

let pair = CurrencyPair {
    base: "EUR".into(),
    quote: "USD".into(),
    rate: 1.08,
};
let sdr = encoder.encode_to_sdr(pair)?;

// Or from string
let sdr = encoder.encode_to_sdr("EUR/USD")?;
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_bits/active` | `UInt` | 100/10 | Base currency encoding |
| `quote_bits/active` | `UInt` | 100/10 | Quote currency encoding |
| `rate_bits/active` | `UInt` | 200/10 | Exchange rate encoding |
| `rate_range` | `(Real, Real)` | (0.1, 10) | Rate bounds |
| `log_scale` | `bool` | true | Log scale for rate |

**Input types:** `CurrencyPair`, `&str` (e.g., "EUR/USD")

---

### OrderBookEncoder

Encodes order book depth with spread and imbalance features.

```rust
use mokosh::encoders::{OrderBookEncoder, OrderBookEncoderParams, OrderBook, PriceLevel, Encoder};

let encoder = OrderBookEncoder::new(OrderBookEncoderParams {
    depth: 5,
    bits_per_level: 50,
    active_per_level: 5,
    spread_bits: 50,
    spread_active: 5,
    imbalance_bits: 50,
    imbalance_active: 5,
    max_volume: 10000.0,
    max_spread_pct: 1.0,
})?;

let book = OrderBook {
    bids: vec![
        PriceLevel { price: 99.9, volume: 1000.0 },
        PriceLevel { price: 99.8, volume: 2000.0 },
    ],
    asks: vec![
        PriceLevel { price: 100.1, volume: 500.0 },
        PriceLevel { price: 100.2, volume: 1500.0 },
    ],
};
let sdr = encoder.encode_to_sdr(book)?;
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `depth` | `usize` | 5 | Price levels to encode |
| `bits_per_level/active` | `UInt` | 50/5 | Per-level encoding |
| `spread_bits/active` | `UInt` | 50/5 | Spread encoding |
| `imbalance_bits/active` | `UInt` | 50/5 | Order imbalance encoding |
| `max_volume` | `Real` | 10000 | Volume normalization |
| `max_spread_pct` | `Real` | 1.0 | Spread normalization |

**Input types:** `OrderBook`

---

## Probabilistic Encoders

### DistributionEncoder

Encodes full probability distributions (not just point estimates).

```rust
use mokosh::encoders::{DistributionEncoder, DistributionEncoderParams, Distribution, Encoder};

let encoder = DistributionEncoder::new(DistributionEncoderParams {
    num_bins: 20,
    bits_per_bin: 10,
    min_probability: 0.01,
    value_range: (0.0, 100.0),
})?;

let dist = Distribution {
    probabilities: vec![0.1, 0.3, 0.4, 0.15, 0.05],
    min_value: 0.0,
    max_value: 100.0,
};
let sdr = encoder.encode_to_sdr(dist)?;

// Or from raw probabilities
let sdr = encoder.encode_to_sdr(vec![0.1, 0.3, 0.4, 0.15, 0.05])?;
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_bins` | `usize` | 20 | Discretization bins |
| `bits_per_bin` | `UInt` | 10 | Bits per bin |
| `min_probability` | `Real` | 0.01 | Threshold for activation |
| `value_range` | `(Real, Real)` | (0, 100) | Range for normalization |

**Input types:** `Distribution`, `Vec<Real>` (probabilities)

---

### ConfidenceIntervalEncoder

Encodes values with uncertainty (both center and range).

```rust
use mokosh::encoders::{ConfidenceIntervalEncoder, ConfidenceIntervalEncoderParams, ConfidenceInterval, Encoder};

let encoder = ConfidenceIntervalEncoder::new(ConfidenceIntervalEncoderParams {
    min_value: 0.0,
    max_value: 100.0,
    center_bits: 200,
    center_active: 10,
    width_bits: 100,
    width_active: 5,
    max_width_fraction: 0.5,
})?;

let ci = ConfidenceInterval {
    lower: 45.0,
    upper: 55.0,
};
let sdr = encoder.encode_to_sdr(ci)?;

// From tuple (lower, upper)
let sdr = encoder.encode_to_sdr((45.0, 55.0))?;

// Point estimate (zero uncertainty)
let sdr = encoder.encode_to_sdr(50.0)?;
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_value/max_value` | `Real` | 0/100 | Value range |
| `center_bits/active` | `UInt` | 200/10 | Center estimate encoding |
| `width_bits/active` | `UInt` | 100/5 | Uncertainty range encoding |
| `max_width_fraction` | `Real` | 0.5 | Max width as fraction of range |

**Input types:** `ConfidenceInterval`, `(Real, Real)` (bounds), `Real` (point)

---

## Composite Encoders

### MultiEncoder

Combines multiple encoders for the same input type via concatenation.

```rust
use mokosh::encoders::{MultiEncoderBuilder, ScalarEncoder, ScalarEncoderParams, Encoder};

let temp_encoder = ScalarEncoder::new(ScalarEncoderParams {
    minimum: -20.0, maximum: 50.0, size: 200, active_bits: 10,
    ..Default::default()
})?;

let humidity_encoder = ScalarEncoder::new(ScalarEncoderParams {
    minimum: 0.0, maximum: 100.0, size: 200, active_bits: 10,
    ..Default::default()
})?;

let multi = MultiEncoderBuilder::new()
    .add("temperature", temp_encoder)
    .add("humidity", humidity_encoder)
    .build()?;

// Encodes same value through both encoders
let sdr = multi.encode_to_sdr(25.0)?;  // interprets as both temp and humidity
```

---

### VecMultiEncoder

Combines encoders for different values in a vector.

```rust
use mokosh::encoders::{VecMultiEncoderBuilder, ScalarEncoder, ScalarEncoderParams, BooleanEncoder, BooleanEncoderParams, Encoder};

let temp_encoder = ScalarEncoder::new(ScalarEncoderParams {
    minimum: -20.0, maximum: 50.0, size: 200, active_bits: 10,
    ..Default::default()
})?;

let humidity_encoder = ScalarEncoder::new(ScalarEncoderParams {
    minimum: 0.0, maximum: 100.0, size: 200, active_bits: 10,
    ..Default::default()
})?;

// For heterogeneous inputs, use separate encoding calls
// MultiEncoder concatenates outputs from multiple encoders
```

---

### PassThroughEncoder

Passes pre-encoded sparse indices through. Useful for integrating external encodings.

```rust
use mokosh::encoders::{PassThroughEncoder, PassThroughEncoderParams, Encoder};

let encoder = PassThroughEncoder::new(PassThroughEncoderParams {
    size: 1000,
    active_bits: Some(50),  // optional validation
})?;

let pre_computed: Vec<u32> = vec![10, 42, 100, 500, 999];
let sdr = encoder.encode_to_sdr(pre_computed)?;
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `size` | `UInt` | 1000 | Output SDR size |
| `active_bits` | `Option<UInt>` | `None` | Optional validation |

**Input types:** `Vec<UInt>`, `&[UInt]`, `Sdr`

---

## Encoder Selection Guide

| Use Case | Recommended Encoder |
|----------|---------------------|
| Known numeric range | `ScalarEncoder` |
| Unknown numeric range | `RandomDistributedScalarEncoder` |
| Values spanning orders of magnitude | `LogEncoder` |
| Rate of change / derivatives | `DeltaEncoder` |
| Fixed categories | `CategoryEncoder` |
| Ordered categories | `OrdinalEncoder` |
| Hierarchical taxonomy | `HierarchicalCategoryEncoder` |
| Sets of items | `SetEncoder` |
| Timestamps / dates | `DateEncoder` |
| 2D/3D coordinates | `CoordinateEncoder` or `GridCellEncoder` |
| GPS locations | `GeospatialEncoder` |
| Documents / text | `SimHashDocumentEncoder` |
| Word vectors (fixed dim) | `WordEmbeddingEncoder` |
| LLM embeddings (any dim) | `LlmEmbeddingEncoder` |
| Audio spectrum | `SpectrogramEncoder` |
| Raw audio | `WaveformEncoder` |
| Musical notes | `PitchEncoder` |
| Image patches | `PatchEncoder` |
| Colors | `ColorEncoder` |
| IP addresses | `IpAddressEncoder` |
| Stock prices | `PriceEncoder` |
| Probability distributions | `DistributionEncoder` |
| Values with uncertainty | `ConfidenceIntervalEncoder` |
| Multiple features | `MultiEncoder` / `VecMultiEncoder` |
| External pre-encoded data | `PassThroughEncoder` |
