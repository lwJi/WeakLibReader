# Interpolation Subsystem

## Overview

Table data is stored as `log10(value + offset)` and interpolated linearly in
log-space; results are converted back via `Pow10(result) - offset`. All public
functions live in the `WeakLibReader` namespace. Device-callable functions are
`AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE`.

## Core Types

### `Axis` — `src/base/WeakLibReader_AxisTypes.hpp:14`

```cpp
struct Axis { const double* grid; int n; AxisScale scale; };
```

Passed by value. `detail::IndexAndDelta(axis, x, idx, frac)` (line 29)
dispatches to `IndexAndDeltaLin` or `IndexAndDeltaLog10` based on `axis.scale`.

### `Layout` — `src/base/WeakLibReader_Layout.hpp:10`

Column-major descriptor for up to 5 dimensions. `stride[0]=1` always.
`Offset()` has 1-D through 5-D overloads. `SliceLeading(layout, drop)` (line 76)
strips leading dimensions for aligned-slice access.

### `IndexDelta` — `src/base/WeakLibReader_IndexDelta.hpp`

- `IndexAndDeltaLin` (line 27): linear axis coordinate → `(index, fraction)`
- `IndexAndDeltaLog10` (line 54): log10 axis coordinate → `(index, fraction)`

Both clamp out-of-range to `[0, n-2]` for natural extrapolation.

## Basis Functions — `src/base/WeakLibReader_InterpBasis.hpp`

All operate in log-space on pre-transformed corner values.

| Function | Dims | Corners | Line |
|----------|------|---------|------|
| `Linear` | 1-D | 2 | 9 |
| `BiLinear` | 2-D | 4 | 16 |
| `TriLinear` | 3-D | 8 | 41 |
| `TetraLinear` | 4-D | 16 | 78 |
| `PentaLinear` | 5-D | 32 | 153 |

Each has derivative variants (e.g., `BiLinearDerivativeX1/X2`) returning
the log-space per-fraction derivative.

## ND Dispatch

### `LinearInterpPointDirect<ND>` — `src/interp/WeakLibReader_InterpLogTablePoint.hpp:160`

Uses `if constexpr` to call `LinearInterp{1..5}DPoint`. Each variant reads
2^ND corners via `Layout::Offset`, calls the matching basis function, returns
`Pow10(result) - offset`.

### `LogInterpolatedValueDirect<ND>` — `src/interp/WeakLibReader_LogInterpolateCore.hpp:16`

Central dispatch: computes index/fraction for each axis, then calls
`LinearInterpPointDirect<ND>`.

### `LogInterpolatedDerivativeDirect<ND>` — `src/interp/WeakLibReader_LogInterpolateCore.hpp:64`

Same structure plus axis scale factors via `ComputeAxisScale()` (line 46) for
chain-rule conversion from log-space to physical-space derivatives.

## Public API

### Point Functions — `src/interp/WeakLibReader_LogInterpolatePoint.hpp`

GPU-callable. Return `NaN` if any pointer is null.

| Function | Line | Dims |
|----------|------|------|
| `LogInterpolateSingleVariable2DCustomPoint` | 12 | 2-D |
| `LogInterpolateSingleVariable3DCustomPoint` | 49 | 3-D |
| `LogInterpolateSingleVariable4DCustomPoint` | 89 | 4-D |

Batch (host-only) variants: `LogInterpolateSingleVariable{2,3,4}DCustom`.

### Energy Sweep — `src/interp/WeakLibReader_LogInterpolateSweep.hpp`

| Function | Line | Pattern |
|----------|------|---------|
| `LogInterpolateSingleVariable1D3DCustomPoint` | 12 | Sweep E over fixed (D,T,Ye) in 4D table |
| `LogInterpolateSingleVariable2D2DCustomPoint` | 77 | Symmetric E×E sweep in 4D table (NES/Pair) |
| `LogInterpolateSingleVariable2D2DCustomAlignedPoint` | 148 | Same on pre-aligned 4D layout |
| `PreAlignScatteringKernelMoment` | 227 | Host: 5D raw kernel → 4D aligned table for one moment |

### Derivatives — `src/interp/WeakLibReader_LogInterpolateDeriv.hpp`

| Function | Line | Dims |
|----------|------|------|
| `LogInterpolateDifferentiateSingleVariable3DCustomPoint` | 10 | 3-D value + 3 derivatives |
| `LogInterpolateDifferentiateSingleVariable2D2DCustomAlignedPoint` | 173 | Aligned 2D-in-4D value + T/X derivatives |

### Aligned Slice Helpers — `src/interp/WeakLibReader_InterpLogTableSlice.hpp`

For tables with dense integer leading dimensions (pre-aligned energy bins):

| Function | Line | Fixed dims | Remaining interp |
|----------|------|------------|-----------------|
| `LinearInterp2D4DArray2DAlignedPoint` | 24 | 2 | 2-D |
| `LinearInterpDeriv2D4DArray2DAlignedPoint` | 76 | 2 | 2-D deriv |

### Sum — `src/interp/WeakLibReader_LogInterpolateSum.hpp:11`

`SumLogInterpolateSingleVariable2D2DCustomAligned` — density-weighted sum
for Bremsstrahlung (`alpha[l] * interp` over quadrature nodes, symmetric
upper-triangular). Host-only.

## Call Chain (Standard 3D Point)

```
LogInterpolateSingleVariable3DCustomPoint      (LogInterpolatePoint.hpp:49)
  └─ detail::LogInterpolatedValueDirect<3>     (LogInterpolateCore.hpp:17)
       ├─ detail::IndexAndDelta × 3            (AxisTypes.hpp:29)
       └─ LinearInterpPointDirect<3>           (InterpLogTablePoint.hpp:160)
            └─ LinearInterp3DPoint             (InterpLogTablePoint.hpp:41)
                 ├─ TriLinear(8 corners)       (InterpBasis.hpp:41)
                 └─ Pow10(result) - offset     (Math.hpp:34)
```

## EOS Inversion — `src/interp/WeakLibReader_EosInversion.hpp`

Recovers temperature from a thermodynamic variable (energy, pressure, or
entropy) via bisection in the T-grid, matching Fortran
`wlEOSInversionModule.F90`.

### Key Types

- `EosInversionBounds` (line 15): 96-byte struct with axis/variable min/max
- `EosInversionError` (line 28): enum — `Success`, `DensityOutOfRange`,
  `VariableOutOfRange`, `ElectronFractionOutOfRange`, `NaNInput`, `NoRootFound`

### Key Functions

| Function | Line | GPU | Purpose |
|----------|------|-----|---------|
| `InitializeEosInversionBounds` | 42 | No | Scan table for physical-space min/max |
| `CheckInversionInputError` | 90 | Yes | Range/NaN checks before inversion |
| `ComputeTemperatureWithDxyGuess` | 163 | Yes | Bisection with initial T guess |
| `ComputeTemperatureFromEnergy` | 466 | Yes | Public wrapper (two overloads: ±guess) |
| `ComputeTemperatureFromPressure` | 497 | Yes | Public wrapper (two overloads: ±guess) |
| `ComputeTemperatureFromEntropy` | 528 | Yes | Public wrapper (two overloads: ±guess) |

### Bisection Algorithm (with guess)

1. Pre-compute D and Y index/fraction once (T varies during search)
2. **Phase 1**: Test guess vicinity `[iT, iT+1]` — early exit if bracket found
3. **Phase 2**: Evaluate at full T-range endpoints `[0, nT-1]`
4. **Phase 3**: If endpoints bracket root, bisect biased toward guess
5. **Phase 4**: If not, scan all T indices for nearest bracket to guess
6. Final sub-cell refinement via `InverseLogInterp` (log-space linear, line 109)

## File Index

| File | Contents |
|------|----------|
| `src/base/WeakLibReader_Math.hpp` | `Floor`, `Log10`, `Pow10`, `Ln10` |
| `src/base/WeakLibReader_IndexDelta.hpp` | `IndexAndDeltaLin/Log10`, `ClampIndex` |
| `src/base/WeakLibReader_AxisTypes.hpp` | `AxisScale`, `Axis`, `MakeAxis`, `IndexAndDelta` |
| `src/base/WeakLibReader_Layout.hpp` | `Layout`, `MakeLayout`, `SliceLeading` |
| `src/base/WeakLibReader_InterpBasis.hpp` | All basis functions and derivatives |
| `src/interp/WeakLibReader_InterpLogTablePoint.hpp` | `LinearInterp{1..5}DPoint`, `LinearInterpPointDirect<ND>` |
| `src/interp/WeakLibReader_InterpLogTableDeriv.hpp` | `LinearInterpDeriv{2..4}DPoint`, `LinearInterpDerivPointDirect<ND>` |
| `src/interp/WeakLibReader_InterpLogTableSlice.hpp` | Aligned slice helpers |
| `src/interp/WeakLibReader_LogInterpolateCore.hpp` | `LogInterpolatedValueDirect<ND>`, `LogInterpolatedDerivativeDirect<ND>`, `ComputeAxisScale`, `StoreSymmetric` |
| `src/interp/WeakLibReader_LogInterpolatePoint.hpp` | Public point and batch API |
| `src/interp/WeakLibReader_LogInterpolateSweep.hpp` | Energy sweep, aligned, `PreAlignScatteringKernelMoment` |
| `src/interp/WeakLibReader_LogInterpolateDeriv.hpp` | Derivative API (3D, aligned 2D-in-4D) |
| `src/interp/WeakLibReader_LogInterpolateSum.hpp` | `SumLogInterpolateSingleVariable2D2DCustomAligned` |
| `src/interp/WeakLibReader_EosInversion.hpp` | EOS temperature inversion (bisection) |
