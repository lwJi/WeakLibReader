# Opacity Tables

## Overview

Five neutrino opacity table types, loaded from WeakLib HDF5 files and interpolated on GPU. All values stored as log10 with per-variable offsets.

## Table Types and Dimensions

| Type | Dimensions | Axes | Species |
|------|-----------|------|---------|
| **EmAb** | 4D `[nE, nRho, nT, nYe]` | Energy, Density, Temperature, Ye | 2 (nue, anue) |
| **Iso** | 5D `[nE, nMom, nRho, nT, nYe]` | Energy, Moments, Density, Temperature, Ye | 2 (nue, anue) |
| **NES** | 5D `[nE_in, nE_out, nMom, nT, nEta]` | Energy×2, Moments, Temperature, Eta | 1 kernel |
| **Pair** | 5D `[nE_in, nE_out, nMom, nT, nEta]` | Energy×2, Moments, Temperature, Eta | 1 kernel |
| **Brem** | 5D `[nE_in, nE_out, nMom, nRho, nT]` | Energy×2, Moments, Density, Temperature | 1 kernel |

Shared grids: `EnergyGrid`, `ThermoState` (Rho, T, Ye), `EtaGrid` (NES/Pair only).

## Interpolation API by Table Type

### EmAb — Direct 4D interpolation

```cpp
// 4D log-interpolation: [E, Rho, T, Ye]
const Axis axes[4] = {energyAxis, rhoAxis, tempAxis, yeAxis};
double opacity = LogInterpolateSingleVariable4DCustomPoint(
    E, rho, T, ye, axes, opacityData, offset);
```

### Iso — Extract moment slice, then 4D interpolation

```cpp
// Host: extract contiguous 4D slice at fixed moment
auto slice = ExtractIsoMomentSlice4D(kernel5d, dims, /*iMom=*/0);

// Device: same 4D interpolation as EmAb
double opacity = LogInterpolateSingleVariable4DCustomPoint(
    E, rho, T, ye, axes, sliceData, offset);
```

### NES / Pair — Pre-align energy grid, then 2D aligned interpolation

```cpp
// Host: pre-align 5D kernel to 4D [nAlignedE, nAlignedE, nT, nEta]
PreAlignScatteringKernelMoment(rawKernel, rawLayout, energyAxis,
    iMom, nT, nEta, alignedE, nAlignedE, offset, output);

// Device: compute indices manually, then 2D aligned interp
detail::IndexAndDelta(tempAxis, T, idxT, fracT);
detail::IndexAndDelta(etaAxis, eta, idxEta, fracEta);
double opacity = LinearInterp2D4DArray2DAlignedPoint(
    iE_in, iE_out, idxT, idxEta, fracT, fracEta,
    offset, alignedData, alignedLayout);
```

### Brem — Pre-align + weighted sum over density terms

```cpp
// Host: same PreAlignScatteringKernelMoment as NES/Pair
// but with nDim3=nRho, nDim4=nT

// Device: sum over 3 density-weighted terms [rho*Xp, rho*Xn, rho*sqrt(Xp*Xn)]
const double alpha[3] = {1.0, 1.0, 28.0/3.0};
double sum = 0.0;
for (int l = 0; l < 3; ++l) {
  detail::IndexAndDelta(rhoAxis, dxVals[l], idxD, fracD);
  sum += alpha[l] * LinearInterp2D4DArray2DAlignedPoint(
      iE_in, iE_out, idxD, idxT, fracD, fracT,
      offset, bremData, bremLayout);
}
```

## Key API Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `LogInterpolateSingleVariable4DCustomPoint()` | `detail/LogInterpolatePoint.hpp` | Full 4D log-interpolation (EmAb, Iso slice) |
| `LinearInterp2D4DArray2DAlignedPoint()` | `detail/InterpLogTableSlice.hpp` | 2D interp in 4D array with fixed leading dims (NES/Pair/Brem) |
| `detail::IndexAndDelta()` | `WeakLibReader_IndexDelta.hpp` | Convert raw coordinate to grid index + fraction |
| `PreAlignScatteringKernelMoment()` | `detail/LogInterpolateSweep.hpp` | Host-side: interpolate 5D kernel to aligned 4D energy grid |
| `ExtractIsoMomentSlice4D()` | `WeakLibReader_Hdf5Loader.hpp` | Host-side: extract contiguous 4D slice from 5D Iso kernel |

## Device Copy and Pre-alignment Patterns

### Loading and Device Transfer

```cpp
// 1. Load all opacity tables from HDF5
WeakLibOpacityTable opacity_table;
LoadWeakLibOpacityTableFull(opacity_table, fileEmAb, fileIso, fileNES, filePair, fileBrem);

// 2. Copy to device (handles all subtables)
auto device_table = MakeDeviceCopy(opacity_table);
```

### Pre-alignment for Scattering Kernels (NES/Pair/Brem)

Scattering kernels are 5D with energy×energy leading dimensions. For device use, pre-align onto a common energy grid to get contiguous 4D arrays:

```cpp
// Per-moment pre-alignment
for (int iMom = 0; iMom < nMom; ++iMom) {
  std::vector<double> hostBuf(alignedSize);
  PreAlignScatteringKernelMoment(rawKernel, rawLayout, energyAxis,
      iMom, nDim3, nDim4, alignedEnergyValues, nAlignedE,
      offset, hostBuf.data());

  // Copy to device
  deviceVec.resize(alignedSize);
  amrex::Gpu::copy(amrex::Gpu::hostToDevice,
      hostBuf.begin(), hostBuf.end(), deviceVec.begin());
}
```

### Offset Handling

- **EmAb**: `offsets[species]` — 1D array indexed by species
- **Iso/NES/Pair/Brem**: `OffsetValue(species, moment)` — 2D column-major table

## Usage Example

See `cactus_interface/TestWeakLibReader/src/testweaklibreader_eos.cxx` and `testweaklibreader_opacity.cxx` for a complete working example showing all five table types loaded, pre-aligned, device-copied, and interpolated in CarpetX device loops.
