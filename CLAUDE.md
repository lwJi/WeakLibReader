# AGENTS Instructions

## Project Goal

Translate WeakLib's EOS & opacity **interpolators** from Fortran into **GPU-friendly C++** that integrates with **AMReX**. Target numerical parity with Fortran (≤1e-12 relative error).

## Repository Structure

```
WeakLibReader/
  WeakLibReader/src/          # Core headers (public API + device helpers)
    AxisTypes.hpp             # Core types: Axis, InterpConfig, AxisScale, OutOfRangePolicy
    IndexDelta.hpp            # Linear/log10 indexing helpers
    InterpBasis.hpp           # Linear/bi-/tri-/tetra-/penta-linear basis routines
    InterpLogTable.hpp        # Log-space point kernels and aligned slices
    Hdf5Loader.hpp            # Host-side HDF5 reader via amrex::TableData
    LogInterpolate.hpp        # Log wrappers, derivatives, weighted sums
    Layout.hpp                # Row-major stride helpers
    Math.hpp                  # Minimal math utilities (log10, pow10, etc.)
ref/weaklib/                  # Fortran reference implementation
test/
  test_log_interpolate.cpp    # Interpolation kernel tests
  test_hdf5_loader.cpp        # HDF5 loader round-trip tests
```

## Code Style & Conventions

### Naming (Strictly Enforced)
- **Namespace:** `WeakLibReader`
- **Types/Structs/Enums/Classes:** `PascalCase` (e.g., `Axis`, `Layout`, `InterpConfig`)
- **Functions:** `PascalCase` (e.g., `IndexAndDeltaLin`, `LogInterpolateSingleVariable3DCustomPoint`)
- **Variables/Parameters/Members:** `lowerCamelCase` (e.g., `outOfRange`, `rowStride`)
- **Constants/Enumerators:** `PascalCase` (e.g., `Clamp`, `Linear`, `Log10`)
- **Standard:** C++17+

### Design Principles
1. Row-major layout with precomputed strides
2. No STL containers in device code; no dynamic allocations in kernels
3. Pass small structs by value (`Axis`, `Layout`, `InterpConfig`)
4. Out-of-range default: `Clamp` (matches Fortran)
5. All interpolation kernels: `AMREX_GPU_HOST_DEVICE`, `noexcept`, inline
6. Precision: double throughout
7. Table I/O: HDF5 only
8. GPU backend: CUDA first, HIP later

## Build & Test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DAMREX_ROOT=/path/to/amrex
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## When Modifying Code

1. Inspect existing files before changes
2. Add/update tests for all changes
3. Verify interpolation changes against `ref/weaklib/*.F90`
4. Use `AMREX_GPU_HOST_DEVICE` for device functions

## Integration Notes

- **AMReX:** Tables use `amrex::TableData<double,4>`; 5D datasets flatten last two axes
- **HDF5:** Axes at `/axis1`, `/axis2`...; values at `/values`; scale attrs `"linear"` or `"log10"`
- **Fortran:** Consult `ref/weaklib/wlInterpolationModule.F90` before implementing new logic

## Do's and Don'ts

**Do:**
- Preserve numerical behavior at boundaries and under mixed Linear/Log10 axes
- Keep device code free of STL containers and dynamic allocations
- Provide small, `constexpr` helpers; keep functions `AMREX_GPU_HOST_DEVICE`
- Ensure HDF5 loader retains axis storage backing raw pointers in `Axis`

**Don't:**
- Add alternate table I/O formats beyond HDF5
- Add OpenACC or non-AMReX GPU pragmas
- Skip Fortran reference verification for interpolation changes
