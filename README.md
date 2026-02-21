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
- Native WeakLib opacity table loading (EmAb, Iso, NES, Pair, Brem)
- EOS temperature inversion (root-finding from energy/pressure/entropy)
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
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DAMREX_ROOT=/path/to/amrex -DHDF5_ROOT=/path/to/hdf5
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

The suite exercises 2D-5D interpolation, derivatives, symmetric plane helpers, HDF5 round-trips, opacity table loading, EOS inversion, and error handling.

## Usage Example

```cpp
#include "WeakLibReader_Hdf5Loader.hpp"
#include "WeakLibReader_LogInterpolate.hpp"

using namespace WeakLibReader;

// Load table from HDF5
Hdf5Table table;
auto status = LoadHdf5Table("opacity.h5", table);
if (status != Hdf5LoadStatus::Success) { /* handle error */ }

// 3D interpolation at a single point
double d = 1.0e10, t = 0.5, y = 300.0;
double result = LogInterpolateSingleVariable3DCustomPoint(
    d, t, y,
    table.axes,
    table.DataPtr(), 0.0);
```

## API Overview

### Core Types

| Type | Description |
|------|-------------|
| `Axis` | Grid metadata: pointer, size, scale (Linear/Log10) |
| `Layout` | Column-major strides for N-D data |
| `Hdf5Table` | Host-side table storage (owns data + axes) |
| `TableDevice` | Device-side table copy |
| `WeakLibEosTable` | Full EOS table storage with all variables and metadata |
| `WeakLibEosTableDevice` | Device copy of full EOS table |
| `WeakLibEosIndices` | Index mappings for EOS dependent variables |
| `WeakLibOpacityTable` | Full opacity table storage (EmAb, Iso, NES, Pair, Brem) |
| `WeakLibOpacityTableDevice` | Device copy of full opacity table |
| `EosInversionBounds` | Min/max bounds for EOS inversion input validation |

### Key Functions

| Function | Description |
|----------|-------------|
| `LoadHdf5Table()` | Load HDF5 into `amrex::TableData` |
| `LoadHdf5TableParallel()` | MPI-aware loader (rank 0 reads, broadcasts) |
| `LoadWeakLibEosTableFull()` | Load complete EOS table with all metadata |
| `MakeDeviceCopy()` | Copy host table to GPU device |
| `LogInterpolateSingleVariable{2,3,4}DCustomPoint()` | Single-point N-D interpolation (GPU) |
| `LogInterpolateSingleVariable{2,3,4}DCustom()` | Batch N-D interpolation |
| `LogInterpolateSingleVariable1D3DCustomPoint()` | 1D energy sweep over 3D table |
| `LogInterpolateSingleVariable2D2DCustomPoint()` | 2D energy sweep over 2D table (NES/Pair) |
| `SumLogInterpolateSingleVariable2D2DCustomAligned()` | Density-weighted sum interpolation (Brem) |
| `PreAlignScatteringKernelMoment()` | Pre-align scattering kernel to energy grid |
| `LogInterpolateDifferentiateSingleVariable3DCustomPoint()` | 3D interpolation with derivatives |
| `LogInterpolateDifferentiateSingleVariable2D2DCustom*()` | 2D×2D interpolation with derivatives (NES/Pair) |
| `LoadWeakLibOpacityTableFull()` | Load opacity tables from up to 5 HDF5 files |
| `LoadWeakLibOpacityTableFullParallel()` | MPI-aware opacity loader (rank 0 reads, broadcasts) |
| `LoadWeakLibEosTableFullParallel()` | MPI-aware EOS loader (rank 0 reads, broadcasts) |
| `ExtractIsoMomentSlice4D()` | Extract 4D moment slice from 5D Iso kernel |
| `InitializeEosInversionBounds()` | Compute EOS inversion bounds from table data |
| `ComputeTemperatureFromEnergy()` | Temperature root-finding from internal energy |
| `ComputeTemperatureFromPressure()` | Temperature root-finding from pressure |
| `ComputeTemperatureFromEntropy()` | Temperature root-finding from entropy |

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

#### Native WeakLib Opacity Format

Each opacity type is stored in its own HDF5 file. All files share the same grid layout:

```
/EnergyGrid/
  Values[nE]                        # Energy grid points
/EtaGrid/
  Values[nEta]                      # Eta grid (NES/Pair only)
/ThermoState/
  Dimensions[3], Density[], Temperature[], Electron Fraction[]
/OpacityType/                       # e.g. EmAb, Iso, NES, Pair, Brem
  Opacity[nYe,nT,nRho,nE,...]      # Opacity data (4D or 5D)
  Units, Offsets
```

## Project Structure

```
src/                 # Header-only library
ref/weaklib/         # Fortran reference implementation (incl. wlIOModuleHDF.F90)
test/                # Regression tests
cactus_interface/    # Cactus thorn integration (WeakLibReader, TestWeakLibReader)
scripts/             # Build, test, and check convenience scripts
specs/               # Detailed subsystem documentation (HDF5 formats, opacity, Cactus)
```

## Fortran Parity

Original routines live under `ref/weaklib/`. The C++ API mirrors Fortran functions (indices, weights, log handling) and targets ≤1e-12 relative agreement.

## Contributing

PRs require CI pass. See `CLAUDE.md` for code conventions and design principles.
