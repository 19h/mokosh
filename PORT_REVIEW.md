# Mokosh port review

Review baseline: Mokosh `d030a358df4efbea95c5a5f9686064411fad1c64` against local
htm.core `d86999487036c36c3b242ee46b9ba9b8f566087d`, 2026-10-22.
The C++ reference checkout was left unchanged. Findings below refer to these
commits, not to an unspecified current upstream version.

## Assumption register

| ID | Assumption and dependent results | Stress test / falsification probe |
| --- | --- | --- |
| A1 | The supplied htm.core commit is the behavioral reference. All parity findings depend on A1. | Build the pinned C++ source and regenerate the checked-in oracle; a different commit requires another comparison. |
| A2 | Scope is the shared core algorithms, four shared C++ encoder families, SDR/topology/RNG utilities and persistence. The previously excluded network/region/REST frameworks are separate products. Coverage conclusions depend on A2. | Inventory public C++ components against Rust modules; the table below lists remaining omissions instead of calling the port complete. |
| A3 | Rust's existing f32 numeric API and u32 indices remain the public representation. Numeric parity depends on A3. | Exact integer and RNG comparisons; absolute tolerances of 1e-7 for duty cycles and 1e-6 for boost/classifier results. Inputs requiring C++ encoder f64 precision remain outside exact parity. |
| A4 | Correctness takes precedence over reproducing demonstrable reference defects. Deliberate differences below depend on A4. | Check mathematical neighborhood sets, current-sample likelihood response, finite endpoint logarithms and digest bit bounds independently of C++ output. |
| A5 | Validation is native macOS arm64. Platform coverage depends on A5. | Debug/release Rust tests and a native C++ oracle are exercised. AVX2 machine-code execution, Windows/Linux runtime behavior and exhaustive state-space equivalence are unknown. |

## Confirmed defects corrected

Nine initial public regressions failed on the unmodified Rust baseline before
repairs. Additional focused regressions and actual C++ oracle comparisons cover
the learning and boundary cases below.

| Impact | Component | Defect and implemented correction |
| --- | --- | --- |
| High | Spatial Pooler | Boosted overlaps were computed into a discarded clone, so inhibition used zeros. The real buffer now drives winner selection. |
| High | Spatial Pooler | Global winners could violate sorted SDR invariants; ties and active counts differed. Canonical sparse output and reference ranking/count rules are restored. |
| High | Spatial Pooler | Local overlap ties activated too many columns. Already-selected equal-overlap neighbors now participate in inhibition. |
| High | Spatial Pooler | Duty-cycle warmup divided by the full configured period immediately. It now uses min(period, iteration). |
| High | Spatial Pooler | Connected-permanence initialization used a narrow interval; stimulus thresholds were not restored during initialization/learning. Reference range and uniform threshold raising are implemented. |
| Medium | Spatial Pooler | Multidimensional column/input ratios used a product; local boost/minimum-duty calculations mishandled the center. Arithmetic mean and reference neighborhoods are restored. |
| Medium | Spatial Pooler | Boost target ignored active-column count configuration, and approximate exponentials changed reference factors. Count-derived density and exact exponentials are used for model learning. |
| High | Temporal Memory | Adaptation reinforced winner cells instead of all previous active cells; correct predictions never grew synapses. Both reference learning rules are restored. |
| High | Temporal Memory | False predictions punished inactive presynaptic cells too. Only previously active synapses are decremented. |
| High | Temporal Memory | Zero-permanence synapses and unusable segments survived. Adaptation now prunes them and avoids regrowing destroyed segments. |
| Medium | Temporal Memory | Repeated bursts discarded the previous winner; zero growth could create empty segments. Winner reuse and nonempty growth guards are implemented. |
| High | Temporal Memory | External-context parameters were accepted but unused, and Likelihood silently returned raw anomaly. External active/winner input, likelihood state and log-likelihood mode are implemented. |
| Medium | Temporal Memory | Public predictions described the preceding context. Rust compute now refreshes next-step predictions, equivalent to C++ compute followed by activateDendrites(false). Reset restores the reference anomaly sentinel. |
| High | Connections | Growth exceeded synapse capacity and could evict protected growth candidates. Candidates are unique; protected cells survive; growth is bounded after pruning. |
| Medium | Connections | Segment eviction counted synapses instead of summing squared permanences. The reference strength heuristic is restored. |
| Medium | Connections | Threshold raising and synapse competition altered selected weights individually. Both now shift the potential pool uniformly. Connectedness includes the reference 1e-6 epsilon. |
| Medium | Connections | Time-series updates alternated between applying and suppressing identical updates and conflated different magnitudes. Requested deltas are remembered exactly; inference does not increment learning iterations. |
| High | RNG/persistence | ChaCha was advertised as reference-compatible MT19937. Replay counted API calls rather than consumed words. MT19937, reference sampling/draws, exact word accounting and an algorithm-tagged checkpoint are implemented. |
| High | Persistence | Local SP neighborhoods disappeared after loading; buffered write failures were discarded. Neighborhoods persist, and save explicitly flushes and propagates errors. |
| High | SDR | Deserialization bypassed sparse validation. Invalid dimensions, overflow, unsorted/duplicate/out-of-range indices now return errors. |
| Medium | SDR | Concatenation ignored its axis; coordinate caches retained duplicate points. Axis-aware mapping and canonical coordinates preserve representation invariants. Noise preserves active count when inactive capacity is limited. |
| Medium | Scalar encoder | Bucket rounding/resolution and category layout differed; NaN encoded a value; invalid ranges were accepted. Reference rounding, disjoint categories, missing-value handling and numeric validation are implemented. |
| High | RDSE | Negative floats saturated to u32 zero and lost all distinctions. Explicit signed-domain wrapping preserves neighborhoods; nonfinite inputs and invalid density parameters are checked. |
| High | SimHash | The implementation used FNV while claiming SHAKE256. It now uses SHAKE256 with MSB-first digest bits and wider weight accumulation. |
| Medium | Date | Negative POSIX timestamps were divided incorrectly and year conversion took time linear in distance from 1970. Euclidean division and bounded binary search handle the signed year domain. Recurring holidays cross year boundaries and reject invalid calendar dates. |
| High | SIMD API | Safe functions used debug-only buffer-length checks before unsafe writes. Length validation now also runs in optimized builds. The simd feature controls SIMD dispatch. |
| Medium | Classifier | Untrained models invented a bucket; horizons shared incorrect category counts; record gaps and multilabel targets were unavailable. Per-horizon state, monotonic record IDs and normalized multilabel updates are implemented. |
| Medium | Build/API claims | Cargo auto-discovered a helper as an example binary and failed; one ordinal doctest asserted overlap for disjoint bands. Example discovery and that fixture are corrected. The tested dependency floor is Rust 1.85; unsupported numeric feature branches and the false no_std declaration are removed. |

## Coverage and remaining differences

| Surface | Established coverage | Remaining difference / limit |
| --- | --- | --- |
| RNG | 1,000 exact MT19937 words, exact f64 draws, mixed-draw checkpoint continuation | u64 draws and negative-seed convenience are Rust extensions. Existing ChaCha checkpoints are incompatible and rejected. |
| SP | C++ global/local learning; 60-step wrapped 2D learning with boosting and iteration-50 radius update | No universal equivalence claim for all parameters. Rust permits count configuration to override the default density and ensures at least one global winner in tiny models. |
| TM | Reference sequence/winner/prediction/anomaly outputs; targeted adaptation, growth, punishment, pruning, external-context and restore tests | Public next-step predictions refresh automatically; C++ exposes a separate phase. External context supplies the context for the current compute call. |
| Likelihood | Probation, aggregation, bounded history, periodic estimation, endpoint-log and mode tests | C++ and Python upstream likelihood implementations differ. Rust follows the C++ rolling-model structure but evaluates the current averaged sample and retains monotonic probation instead of copying C++ timestamp-reset/oldest-sample defects. No exact likelihood parity claim. |
| Encoders | Compiled C++ Scalar/RDSE/SimHash/UTC Date fixtures; calendar regressions | Rust scalar input/configuration remains f32, C++ uses f64. Periodic scalar wrapping is a Rust convenience. Rust dates use UTC and timestamp zero is the POSIX epoch; upstream local-time timestamp zero means now. RDSE negative wrapping explicitly defines behavior where the reference cast is not portable. SimHash corrects final-byte bounds and keeps inclusive frequency ceilings and Unicode token handling as existing Rust behavior. |
| SDR/utilities | Shape validation, multidimensional concatenation, canonical representation, bounded wrapping, persistence | Callbacks/metrics interfaces are Rust-specific; the metrics accumulator is not a full port of C++ SdrMetrics classes. |
| Serialization | Binary/JSON continuation and I/O errors | Rust bincode/serde files are not Cereal files. Existing model files require retraining/re-encoding after RNG/encoder changes; no legacy model migration is implemented. |
| Frameworks | Explicitly inventoried | Network/Region/REST, C++/Python bindings and several engine-specific helper APIs remain outside the Rust library. |
| Features | Default, all-features, no-default-features and optimized regressions | std is required. rayon remains a compatibility/dependency flag; algorithms do not dispatch parallel work. Hardware-specific speedups are not measured here. |

The checked C++ non-wrapping neighborhood contains [22,22,23,30] for center 31,
shape [4,8], radius 1 and excluded center. The iterator skips the final center,
advances to end, then dereferences the reset position. Rust returns [22,23,30].
This is a reproduced upstream defect, not a port regression. Duplicating the
neighbor changes local boost averages; the divergence is tested explicitly.

## Primary source index

All sources are from the pinned local C++ checkout above. The generated fixture
and regression tests are separate evidence for exercised behavior.

| Source | Contracts inspected |
| --- | --- |
| [SpatialPooler.cpp](../htm.core/src/htm/algorithms/SpatialPooler.cpp) | initialization, inhibition, boosting, duty cycles, radius updates |
| [TemporalMemory.cpp](../htm.core/src/htm/algorithms/TemporalMemory.cpp) | activation phases, learning, punishment, winner reuse and external context |
| [Connections.cpp](../htm.core/src/htm/algorithms/Connections.cpp) | thresholds, activity, adaptation, growth, pruning and competition |
| [Random.hpp](../htm.core/src/htm/utils/Random.hpp), [Random.cpp](../htm.core/src/htm/utils/Random.cpp) | MT19937 initialization, draws, shuffle and checkpoint steps |
| [ScalarEncoder.cpp](../htm.core/src/htm/encoders/ScalarEncoder.cpp), [RandomDistributedScalarEncoder.cpp](../htm.core/src/htm/encoders/RandomDistributedScalarEncoder.cpp) | resolution, rounding, hash inputs and missing values |
| [DateEncoder.cpp](../htm.core/src/htm/encoders/DateEncoder.cpp), [SimHashDocumentEncoder.cpp](../htm.core/src/htm/encoders/SimHashDocumentEncoder.cpp) | composite layouts, calendar/time handling, SHAKE256 and bit order |
| [SDRClassifier.cpp](../htm.core/src/htm/algorithms/SDRClassifier.cpp) | per-horizon classifiers, record-number history and multilabel targets |
| [Topology.cpp](../htm.core/src/htm/utils/Topology.cpp), [AnomalyLikelihood.cpp](../htm.core/src/htm/algorithms/AnomalyLikelihood.cpp) | neighborhood enumeration and explicitly divergent likelihood behavior |

## Reproducible verification

- `cargo test --all-features --locked`: library, integration and documentation tests.
- `cargo test --locked`: default feature suite.
- `cargo check --no-default-features --locked`: supported std-based build without compatibility flags.
- `cargo test --release --all-features --test port_regressions --test reference_parity --locked`: optimized boundary and differential tests.
- `cargo +1.85.0 test --all-features --locked`: declared minimum toolchain suite.
- [C++ fixture generation](tests/reference/README.md) and [probe source](tests/reference/htm_core_probe.cpp).

Final results on macOS arm64:

| Check | Result |
| --- | --- |
| All features | 559 passed, 0 failed, 1 ignored documentation example |
| Default features | 549 passed, 0 failed |
| No default features | Check passed |
| Optimized regression/reference tests | 34 passed, 0 failed |
| Rust 1.85, all features | 559 passed, 0 failed, 1 ignored documentation example |
| C++ fixture repeat | Byte-identical on regeneration |
| Reference checkout and whitespace | Reference clean; git diff --check passed |

The ignored example is the pre-existing illustrative serialization snippet.
Equivalent binary/JSON operations and continuation are covered by executable
tests. Existing unused-code and example warnings remain; no warning-free claim
is made.

The oracle checks numerical outputs, not only successful execution. Dimensionless
permanences/probabilities use absolute error bounds stated above; integer indices
and RNG words are exact. POSIX timestamps are in seconds, civil offsets in days,
and encoder radii in the input's units.

Algorithm costs: MT19937 draws are amortized O(1), with 624 u32 state words;
wrapped neighborhoods enumerate O(K*d) coordinates, where K is at most the
number of cells and d is rank. SDR concatenation is O(k*d + k log k) time and
O(k+d) additional space. Uniform permanence adjustment uses expected O(s)
selection plus O(s) updates for s synapses. Classifier history is bounded by the
largest requested lag in records; learning retains the existing sparse weight
representation. Civil-date inversion uses at most 32 binary-search iterations
plus 12 month steps. Likelihood updates are O(1), with O(H) reestimation every R
samples and O(H+A) storage for history H and averaging window A.

## Bounded extensions and quality gates

- High impact: model outputs and checkpoint layouts change. Old learned artifacts
  are not evidence of corrected-model behavior; retraining and fresh artifacts
  are required for downstream validation.
- High impact: the C++ reference itself has observable edge defects. Keeping an
  explicit divergence fixture prevents later reintroduction through blind copying.
- Medium impact: checked-in C++ vectors allow Rust-only CI to detect semantic
  regressions without downloading/building the reference on every run.
- Low impact: APIs named after upstream classes do not establish byte-format,
  full feature, floating-point or platform equivalence.

QG1: the work requires no normative judgments. QG2: assumptions and falsification
probes are registered. QG3: the bounded shared-core inventory is covered and
omissions are explicit. QG4: units, calculations, tolerances and regeneration
commands are stated. QG5: reference defects and API/numeric differences are
separated from parity assertions. QG6: provenance is local primary source at an
exact commit and compiled outputs. QG7: bounded extensions are listed above.
