# CLAUDE.md

## Project Goal

Translate WeakLib's EOS & opacity **interpolators** from Fortran into **GPU-friendly C++** that integrates with **AMReX**. Target: numerical parity with Fortran (≤1e-12 relative error), CUDA first (HIP later).

## Repository Structure

```
src/                          # Core headers (public API + device helpers)
  AxisTypes.hpp               # Axis, AxisScale
  IndexDelta.hpp              # Linear/log10 indexing helpers
  InterpBasis.hpp             # Linear through penta-linear basis routines
  InterpLogTable.hpp          # Log-space point kernels and aligned slices
  LogInterpolate.hpp          # High-level API: interpolation, derivatives, sweeps
  Layout.hpp                  # Row-major stride helpers
  Math.hpp                    # GPU math utilities (Log10, Pow10)
  Hdf5Loader.hpp              # HDF5 reader via amrex::TableData
ref/weaklib/                  # Fortran reference implementation
test/
  test_log_interpolate.cpp    # Interpolation tests (1D-5D, derivatives, sweeps)
  test_hdf5_loader.cpp        # 11 HDF5 loader tests
```

## Naming Conventions

- **Namespace:** `WeakLibReader`
- **Types/Functions:** `PascalCase` (e.g., `Axis`, `IndexAndDeltaLin`)
- **Variables:** `lowerCamelCase` (e.g., `fracT`, `rowStride`)
- **Standard:** C++17+

## Key Design Principles

1. **Row-major layout** with precomputed strides
2. **No STL containers** in device code; no dynamic allocations in kernels
3. **Pass by value** for small structs (`Axis`, `Layout`)
4. **Out-of-range handling:** Extrapolation (matches Fortran behavior)
5. **Strict monotonicity** for axis grids (validated at load time)
6. **Device functions** must be `noexcept`, inline, `AMREX_GPU_HOST_DEVICE`

## Build & Test

```bash
scripts/build.sh   # Build only
scripts/test.sh    # Run tests only
scripts/check.sh   # Build and test together
```

Set `VERBOSE=1` for full test output (e.g., `VERBOSE=1 scripts/test.sh`).

## HDF5 Table Format

```
/values              # N-D array (row-major)
/axis0, /axis1, ...  # 1D arrays with "scale" attribute ("linear" or "log10")
```

Validation: monotonic ascending, positive values for Log10 axes.

## Fortran Reference

Located in `ref/weaklib/`. Key modules: `wlInterpolationModule.F90`, `wlInterpolationUtilitiesModule.F90`. Always consult before implementing new interpolation logic.

## Architectural Decisions (Locked)

1. **Out-of-range behavior:** Natural extrapolation (matches Fortran)
2. **Precision:** Double throughout
3. **Table I/O:** HDF5 only
4. **GPU backend:** CUDA first, HIP/DPCPP later
5. **Memory layout:** Explicit row-major with precomputed strides

## Do's and Don'ts

**Do**
- Read existing files before proposing changes
- Add tests for new features; verify against `ref/weaklib/` for interpolation changes
- Preserve numerical behavior at boundaries and mixed Linear/Log10 axes
- Use `Layout::Offset` for bounds-safe data access
- Ensure HDF5 loader retains axis storage backing `Axis` pointers

**Don't**
- Add I/O formats beyond HDF5
- Use STL containers or dynamic allocations in device code
- Add OpenACC or non-AMReX GPU pragmas
- Skip host-side validation (monotonicity, Log10 positivity)
