# WeakLibReader

![CI](https://github.com/lwJi/WeakLibReader/actions/workflows/ci.yml/badge.svg)

GPU-friendly C++ reimplementation of WeakLib's equation-of-state and opacity interpolators. The library mirrors the original Fortran routines under `ref/weaklib/`, provides AMReX-ready device functions, and ships with a lightweight regression suite.

## Features

- 1D-5D log-space interpolation with compile-time dimension dispatch
- GPU-ready kernels (`AMREX_GPU_HOST_DEVICE` qualified)
- Configurable out-of-range policies: `Clamp`, `Error`, `FillNaN`
- Derivative computation for 2D-4D interpolation
- HDF5 table loading with MPI broadcast support
- Numerical parity with Fortran reference (≤1e-12 relative error)

## Requirements

- CMake ≥ 3.18
- C++17-capable compiler
- AMReX (point `AMREX_ROOT` to your installation)
- OpenMP runtime
- HDF5 C library

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DAMREX_ROOT=/path/to/amrex
cmake --build build -j
```

**macOS note:** If using Homebrew's libomp, add OpenMP flags:
```bash
-DOpenMP_CXX_FLAGS='-Xpreprocessor -fopenmp' \
-DOpenMP_CXX_LIB_NAMES=omp \
-DOpenMP_omp_LIBRARY=$(brew --prefix libomp)/lib/libomp.dylib
```

## Test

```bash
ctest --test-dir build --output-on-failure
```

The suite exercises 2D-5D interpolation, out-of-range policies, derivatives, symmetric plane helpers, and HDF5 round-trips.

## Usage Example

```cpp
#include "Hdf5Loader.hpp"
#include "LogInterpolate.hpp"

using namespace WeakLibReader;

// Load table from HDF5
Hdf5Table table;
auto status = LoadHdf5Table("opacity.h5", table);
if (status != Hdf5LoadStatus::Success) { /* handle error */ }

// Get table view
TableView view = table.View();

// 3D interpolation at a single point
double d = 1.0e10, t = 0.5, y = 300.0;
InterpConfig cfg{OutOfRangePolicy::Clamp};
double result = LogInterpolateSingleVariable3DCustomPoint(
    d, t, y,
    view.axes[0].grid, view.axes[0].n,
    view.axes[1].grid, view.axes[1].n,
    view.axes[2].grid, view.axes[2].n,
    view.data, 0.0, cfg);
```

## API Overview

### Core Types

| Type | Description |
|------|-------------|
| `Axis` | Grid metadata: pointer, size, scale (Linear/Log10) |
| `Layout` | Row-major strides for N-D data |
| `InterpConfig` | Out-of-range policy configuration |
| `Hdf5Table` | Host-side table storage (owns data + axes) |
| `TableView` | Read-only view into loaded table |
| `TableDevice` | Device-side table copy |

### Key Functions

| Function | Description |
|----------|-------------|
| `LoadHdf5Table()` | Load HDF5 into `amrex::TableData` |
| `LoadHdf5TableParallel()` | MPI-aware loader (rank 0 reads, broadcasts) |
| `MakeDeviceCopy()` | Copy host table to GPU device |
| `LogInterpolateSingleVariable*DCustomPoint()` | Single-point N-D interpolation |
| `LogInterpolateSingleVariable*DCustom()` | Batch N-D interpolation |
| `LogInterpolateDifferentiateSingleVariable*()` | Interpolation with derivatives |

### HDF5 File Format

```
/values              # N-D array (row-major layout)
/axis0               # 1D array with "scale" attribute ("linear" or "log10")
/axis1               # ...
```

## Project Structure

```
src/                 # Header-only library (8 headers)
ref/weaklib/         # Fortran reference implementation
test/                # Regression tests (39 test cases)
```

## Fortran Parity

Original routines live under `ref/weaklib/`. The C++ API mirrors Fortran functions (indices, weights, log handling) and targets ≤1e-12 relative agreement.

## Contributing

PRs require CI pass. See `CLAUDE.md` for code conventions and design principles.
