# Pinned htm.core oracle

`htm_core_d869994.txt` is stdout from `htm_core_probe.cpp`, linked against an
unmodified C++ htm.core build at commit
`d86999487036c36c3b242ee46b9ba9b8f566087d`.

The fixture was generated on macOS arm64 with AppleClang 21.0.0, C++17, Release.
Date fixtures set TZ=UTC and use the tm overload, because the upstream timestamp
overload interprets zero as the current time rather than the POSIX epoch.
Fixture SHA-256: `8674ecdb5e0002d9f8fe0482aca4644e0acdd78bd50acc65cfd5f8876de3e805`.

The Rust tests parse integers exactly, RNG doubles exactly, classifier
probabilities within 1e-6 absolute error, and SP duty cycles within 1e-7.
The SP geometry case checks boosts within 1e-6. All values are dimensionless.

To regenerate on a POSIX system with CMake and a C++17 compiler, from Mokosh:

```sh
mkdir -p /tmp/mokosh-reference-src
git -C /Users/int/dev/htm.core archive d86999487036c36c3b242ee46b9ba9b8f566087d | tar -x -C /tmp/mokosh-reference-src
cmake -S /tmp/mokosh-reference-src -B /tmp/mokosh-reference-build -DBINDING_BUILD=CPP_Only -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/mokosh-reference-build --target htm_core -j 8
c++ -std=c++17 -O2 \
  -I/tmp/mokosh-reference-src/src \
  -I/tmp/mokosh-reference-build/cereal-src/include \
  -I/tmp/mokosh-reference-build/eigen-src \
  -I/tmp/mokosh-reference-src/external/common/include \
  tests/reference/htm_core_probe.cpp \
  /tmp/mokosh-reference-build/src/libhtm_core.a \
  -o /tmp/mokosh-htm-core-probe
/tmp/mokosh-htm-core-probe > /tmp/htm_core_actual.txt
cmp tests/reference/htm_core_d869994.txt /tmp/htm_core_actual.txt
cargo test --all-features --test reference_parity
```

The upstream build downloads its declared dependencies. The source archive is
isolated because upstream CMake writes a VERSION file into its source tree.

`reference_bug.nonwrapping_neighbors` intentionally captures an upstream defect:
for center 31 of dimensions [4,8], radius 1, no wrapping and excluding the center,
the C++ iterator emits [22,22,23,30]. Mokosh's expected answer is [22,23,30].
`nonwrapping_reference_duplicate_is_not_reproduced` makes this divergence explicit.
The wrapped 60-step SP fixture compares normal learning through update iteration
50, rather than propagating that duplicate into boost statistics.
