# Core review closure and verification

This extends the initial review at Mokosh `b1d35e5` against the same pinned
htm.core commit `d86999487036c36c3b242ee46b9ba9b8f566087d`.

## Assumptions and falsification probes

| ID | Assumption | Probe / dependent result |
| --- | --- | --- |
| F1 | The supplied C++ revision defines the reference behavior for ordinary inputs. | Recompile the included C++ probes; direct graph comparison and encoder/metric results depend on this revision. |
| F2 | Existing core APIs and their supporting utilities are the implementation target; previously excluded Network/Region/REST and bindings remain separately scoped. | Their absence is listed explicitly below. The core test results do not establish completion of those separate products. |
| F3 | Exact equality is claimed only for the tested discrete states, not every possible input or execution. | The live campaign compares full graph records, without relying on digest collision assumptions. Exhaustive topology checks are bounded to the stated domains. |
| F4 | Correct mathematical invariants take precedence over reproducing known upstream defects. | The documented non-wrapping-neighbor and likelihood divergences remain explicit. |
| F5 | Translation/emulation validates the executed code paths but is not native-hardware validation. | macOS x86 tests use Rosetta; Linux AVX2/FMA tests use QEMU with an explicit instruction-availability assertion. Windows compilation is not Windows runtime proof. |

## Additional port defects corrected

The first small sequence fixture was insufficient. A longer C++ campaign found
new differences, which were reproduced before correction:

| Impact | Defect | Correction |
| --- | --- | --- |
| High | Least-used-cell selection omitted C++ shuffle draws, changing later synapse sampling. | Match reference RNG consumption, including draws whose immediate winner is deterministic. |
| High | Growth sampled a filtered candidate list, changing reference selection and capacity behavior. | Match the reference candidate, capacity, shuffle and duplicate-handling order. |
| High | Equal-permanence pruning depended on mutable vector order. | Break ties by synapse ID, as in C++. |
| High | Segment comparison and best-match ties used position within a cell instead of segment ID. | Match cell/segment-ID ordering. |
| High | Punishment happened after all active-column growth, changing free-list reuse. | Interleave inactive-column punishment and active-column growth in ascending column order. |
| High | Deserializers trusted redundant graph indexes and model caches. | Validate graph ownership, free lists, synapse counts, reverse indexes, model dimensions, temporal state, scalar geometry and classifier/likelihood state. |
| Medium | RNG loading replayed an unbounded number of draws. | Persist 624 engine words and the current index; validate and restore directly. Prior MT JSON import is explicitly bounded. |
| Medium | Existing f32 encoder interfaces could not represent C++ f64 inputs. | Add ScalarEncoder64/ScalarEncoderParams64 and RDSE encode_f64/encode_f64_to_sdr without narrowing input. |
| Medium | C++ streaming sparsity, activation frequency, normalized entropy and overlap metrics were absent. | Add the matching streaming metrics and automatic SDR observation with RAII disconnection. Preserve the older cumulative SdrMetrics API. |
| Medium | The rayon flag performed no work. | Add parallel large-column boost updates, checked for bit-identical results with one and four worker threads. |
| Medium | Derived encoder resolution could underflow to zero despite valid raw arguments. | Validate derived constructor state. Reject unrepresentable f32 boost exponent ranges. |
| Low | The serialization documentation example was skipped. | Make it compile as a no-run example; executable tests continue covering persistence and continuation. |

## Evidence

### Exact sequence comparison

`tests/reference/tm_campaign.cpp` drives 12 seeds through 300 steps each. The
campaign uses resets, learning/inference alternation, bursts, matching, growth,
pruning and capacity pressure. At every step it compares active, winner and
predictive cells. The live mode also compares every graph record: cell, segment,
presynaptic endpoint and exact f32 permanence bits.

Run the full comparison with:

```sh
python3 tests/reference/run_campaign.py \
  --reference-root /tmp/mokosh-reference-src \
  --reference-build /tmp/mokosh-reference-build
```

This completed with 3,600 matching states. The checked-in campaign fixture uses
compact digests for ordinary Rust-only CI; the live mode removes the digest
assumption by comparing the complete records.

### Independent invariants

`tests/state_invariants.rs` checks 128 generated mutation streams of up to 149
operations, validating graph ownership and reverse indexes after each mutation
and comparing activity counts against a separate graph scan. It also exhausts
2,700 combinations of two-dimensional shape, center, radius and wrapping mode:
axis sizes 1 through 5, every center, radii 0, 1, 2, 5, 20 and u32::MAX, with and
without wrapping. Shape output is compared against the coordinate-distance
definition, not the implementation's recursion.

Malformed checkpoint tests cover redundant-index corruption, free-list
corruption, dimensions, scalar state and invalid RNG state. The new engine-state
format loads in O(624), independently of the historical draw count; no unbounded
legacy replay is accepted.

### Numeric and metric coverage

`tests/reference/followup_probe.cpp` and `tests/followup_parity.rs` compare f64
inputs which f32 would merge and compare streaming metric values with C++ over
warmup and post-warmup updates. Parallel boost tests require exact equality with
one and four workers. Numeric probabilities/statistics use an absolute 1e-6
comparison tolerance; the campaign's permanence bits are compared exactly.
All SDR measures are dimensionless; scalar resolutions retain input units.

### Platform validation

The local validation includes macOS ARM64 execution, macOS x86-64 execution
through Rosetta, Linux ARM64 container execution, and Linux x86-64 through QEMU.
The QEMU run sets MOKOSH_REQUIRE_AVX2=1: a passing result requires the CPU-feature
checks and execution of the AVX2/FMA routines, rather than accepting scalar
fallback. Windows GNU-target compilation is checked separately.

The repository now includes a native Windows/Linux/macOS CI matrix for Rust
1.85 and stable. Creating that workflow is not evidence that remote CI has run.

## Final observed results

| Check | Result |
| --- | --- |
| macOS ARM64, all features | 571 passed, 0 failed, 0 ignored |
| Rust 1.85, all features | 571 passed, 0 failed, 0 ignored |
| Default features | 557 passed, 0 failed, 0 ignored |
| Optimized all-feature tests | 524 passed, 0 failed, 0 ignored |
| Linux ARM64, all features | 571 passed, 0 failed, 0 ignored |
| macOS x86-64 through Rosetta | 525 passed, 0 failed, 0 ignored |
| Linux x86-64 through QEMU, AVX2/FMA required | 525 passed, 0 failed, 0 ignored |
| Windows GNU-target, all-feature tests | Compilation passed; no native runtime claim |
| Full C++ graph campaign | 3,600 complete graph/state comparisons passed |
| Whitespace/reference isolation | git diff --check passed; htm.core checkout unchanged |

The x86 totals cover executable tests; the ARM64 all-feature totals additionally
include documentation tests. The optimized total excludes documentation tests.
The zero ignored tests reflect converting the former serialization illustration
into a checked compilation example.

Reproduction commands for the host checks:

```sh
cargo test --all-features --locked
cargo +1.85.0 test --all-features --locked
cargo test --locked
cargo test --release --all-features --tests --locked
cargo test --target x86_64-apple-darwin --all-features --tests --locked
cargo check --target x86_64-pc-windows-gnu --all-features --tests --locked
```

The Linux emulator environment uses a Rust Debian image with `qemu-user` and
`gcc-x86-64-linux-gnu`, the x86 linker `x86_64-linux-gnu-gcc`, and the runner
`qemu-x86_64 -L /usr/x86_64-linux-gnu -cpu max`. Set
`MOKOSH_REQUIRE_AVX2=1` and run all-feature x86 Linux tests to require instruction
execution. The native Linux ARM64 run uses the same source snapshot without the
cross-target override. No repository mutation or privileged container is needed.

## Limits which cannot be converted into proof by wording

- Finite testing, even with exact graph comparison, does not establish universal
  semantic equivalence or absence of all defects. No such claim is made.
- The original ChaCha checkpoint stored seed plus API-call count. A u32 draw and
  a u64 draw each incremented that count once but consumed different engine
  state. Identical stored metadata can therefore correspond to different future
  streams. Exact recovery needs missing draw history or an original live state;
  guessing a migration would fabricate information.
- Cereal and Rust serde/bincode formats are distinct. No general Cereal import
  or export converter was implemented.
- Native Windows execution, physical x86 AVX2 hardware validation, and the
  previously excluded Network/Region/REST and language bindings are not claimed
  complete by the core test results.

The first review's claim that its bounded inventory passed quality gates did not
establish full port completion. This follow-up separates corrected defects,
newly implemented core gaps, executed evidence and the remaining scope limits.
