# WeakLibReader

GPU-friendly C++ reimplementation of WeakLib’s equation-of-state and opacity interpolators. The library mirrors the original Fortran routines under `ref/weaklib/`, provides AMReX-ready device functions, and ships with a lightweight regression suite.

## Requirements

- CMake ≥ 3.18
- C++17-capable compiler
- AMReX headers (point `AMREX_ROOT` to your installation)
- OpenMP runtime if AMReX was built with OpenMP (e.g. `libomp` on macOS)
- HDF5 C library (for table loader + unit tests)

## Configure & Build

```bash
# Configure (fill in your AMReX path)
cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DAMREX_ROOT=/path/to/amrex \
      -DOpenMP_CXX_FLAGS='-Xpreprocessor -fopenmp' \
      -DOpenMP_CXX_LIB_NAMES=omp \
      -DOpenMP_omp_LIBRARY=$(brew --prefix libomp)/lib/libomp.dylib

# Build library + tests
cmake --build build -j
```

## Tests

```bash
ctest --test-dir build -j
```

The regression suite runs via a bundled Catch2-style shim. It exercises:

- 2D log interpolation vs. bilinear expectation
- FillNaN out-of-range policy
- Aligned 2D plane symmetry and weighted sums
- Derivative wrappers (`LogInterpolateDifferentiateSingleVariable*`)
- HDF5 loader round-trip into `amrex::TableData`

## Fortran Parity

Original routines live under `ref/weaklib/`. The C++ API mirrors the Fortran functions (indices, weights, log handling) and targets ≤ 1e-12 relative agreement. Use the regression suite as a starting point when adding additional parity checks.

## Notes

- The AMReX CUDA demo is a placeholder; integrate once the GPU kernels are ready.
- Headers include `AMReX_GpuQualifiers.H` and `AMReX_Extension.H`; ensure your include path covers `${AMREX_ROOT}/include`.
- HDF5 tables load into `amrex::TableData<double,4>`; for 5D datasets the final two axes are flattened when allocating the table storage while preserving the raw row-major layout for interpolation.
