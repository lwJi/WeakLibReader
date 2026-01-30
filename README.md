# WeakLibReader

![CI](https://github.com/lwJi/WeakLibReader/actions/workflows/ci.yml/badge.svg)

GPU-friendly C++ reimplementation of WeakLib's equation-of-state and opacity interpolators. The library mirrors the original Fortran routines under `ref/weaklib/`, provides AMReX-ready device functions, and ships with a lightweight regression suite.

## Features

- 1D-5D log-space interpolation with compile-time dimension dispatch
- GPU-ready kernels (`AMREX_GPU_HOST_DEVICE` qualified)
- Out-of-range coordinates extrapolate naturally (matches Fortran)
- Derivative computation for 2D-4D interpolation
- HDF5 table loading with MPI broadcast support
- Native WeakLib EOS table loading (full and single-variable)
- Cactus framework integration (thorns provided)
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

The suite exercises 2D-5D interpolation, derivatives, symmetric plane helpers, and HDF5 round-trips.

## Usage Example

```cpp
#include "WeakLibReader_Hdf5Loader.hpp"
#include "WeakLibReader_LogInterpolate.hpp"

using namespace WeakLibReader;

// Load table from HDF5
Hdf5Table table;
auto status = LoadHdf5Table("opacity.h5", table);
if (status != Hdf5LoadStatus::Success) { /* handle error */ }

// Get table view
TableView view = table.View();

// 3D interpolation at a single point
double d = 1.0e10, t = 0.5, y = 300.0;
double result = LogInterpolateSingleVariable3DCustomPoint(
    d, t, y,
    view.axes[0].grid, view.axes[0].n,
    view.axes[1].grid, view.axes[1].n,
    view.axes[2].grid, view.axes[2].n,
    view.data, 0.0);
```

## API Overview

### Core Types

| Type | Description |
|------|-------------|
| `Axis` | Grid metadata: pointer, size, scale (Linear/Log10) |
| `Layout` | Row-major strides for N-D data |
| `Hdf5Table` | Host-side table storage (owns data + axes) |
| `TableView` | Read-only view into loaded table |
| `TableDevice` | Device-side table copy |
| `WeakLibEosTable` | Full EOS table storage with all variables and metadata |
| `WeakLibEosTableDevice` | Device copy of full EOS table |
| `WeakLibEosIndices` | Index mappings for EOS dependent variables |

### Key Functions

| Function | Description |
|----------|-------------|
| `LoadHdf5Table()` | Load HDF5 into `amrex::TableData` |
| `LoadHdf5TableParallel()` | MPI-aware loader (rank 0 reads, broadcasts) |
| `LoadWeakLibEosTable()` | Load single variable from native WeakLib format |
| `LoadWeakLibEosTableFull()` | Load complete EOS table with all metadata |
| `MakeDeviceCopy()` | Copy host table to GPU device |
| `LogInterpolateSingleVariable*DCustomPoint()` | Single-point N-D interpolation |
| `LogInterpolateSingleVariable*DCustom()` | Batch N-D interpolation |
| `LogInterpolateDifferentiateSingleVariable*()` | Interpolation with derivatives |

### HDF5 File Format

#### Simple Format (Generic Tables)

```
/values              # N-D array (row-major layout)
/axis0               # 1D array with "scale" attribute ("linear" or "log10")
/axis1               # ...
```

#### Native WeakLib EOS Format

```
/ThermoState/
  Dimensions[3]                     # Grid dimensions [nRho, nT, nYe]
  Density[nRho]                     # Density axis (log10 scale)
  Temperature[nT]                   # Temperature axis (log10 scale)
  Electron Fraction[nYe]            # Ye axis (linear scale)
/DependentVariables/
  nVariables                        # Number of dependent variables
  Names[], Units[], Offsets[]       # Variable metadata
  iPressure, iEntropyPerBaryon, ... # Index mappings for each variable
  {variable_name}[nYe,nT,nRho]      # Variable data arrays
```

## Project Structure

```
src/                 # Header-only library
ref/weaklib/         # Fortran reference implementation (incl. wlIOModuleHDF.F90)
test/                # Regression tests
cactus_interface/    # Cactus thorn integration (WeakLibReader, TestWeakLibReader)
```

## Fortran Parity

Original routines live under `ref/weaklib/`. The C++ API mirrors Fortran functions (indices, weights, log handling) and targets ≤1e-12 relative agreement.

## Contributing

PRs require CI pass. See `CLAUDE.md` for code conventions and design principles.
