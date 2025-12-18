# AGENTS Instructions

## Project Goal

Translate WeakLib’s EOS & opacity **interpolators** from Fortran into **GPU‑friendly C++** that integrates cleanly with **AMReX**. Achieve tight numerical agreement with the Fortran reference at interior points and boundaries, with predictable performance on **CUDA** first (HIP later).

## Repository Structure

```
WeakLibReader/
  src/                        # Core headers (public API + device helpers)
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
  include/catch2/             # Minimal Catch2-compatible shim
  test_log_interpolate.cpp    # Regression tests (aligned planes, derivatives, etc.)
  test_hdf5_loader.cpp        # HDF5 loader round-trip coverage
```

## Code Style & Conventions

### Naming (Strictly Enforced)
- **Namespace:** `WeakLibReader`
- **Types/Structs/Enums/Classes:** `PascalCase` (e.g., `Axis`, `Layout`, `InterpConfig`, `AxisScale`, `OutOfRangePolicy`).
- **Functions:** `PascalCase` (e.g., `IndexAndDeltaLin`, `LogInterpolateSingleVariable3DCustomPoint`)
- **Variables/Parameters/Data members:** `lowerCamelCase` (e.g., `outOfRange`, `rowStride`)
- **Constants/Enumerators:** `PascalCase` (e.g., `Clamp`, `Linear`, `Log10`)
- **Standard:** C++17+ required

### Key Design Principles
1. **Row-major layout** with precomputed strides
2. **No STL containers** in device code
3. **Pass by value** for small structs (`Axis`, `Layout`, `InterpConfig`)
4. **Out-of-range handling** via policy (default: `Clamp`)
5. **Strict monotonicity** for axis grids (validated at load time)
6. **Numerical parity** with Fortran reference (≤1e-12 relative error)

## Development Workflows

### Building
```bash
# Configure (set AMREX_ROOT to your AMReX installation)
cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DAMREX_ROOT=/path/to/amrex

# Build
cmake --build build -j

# Run tests
ctest --test-dir build --output-on-failure
```

### Testing Strategy
- **Unit tests:** `test/test_log_interpolate.cpp` (interpolation kernels, policies, derivatives)
- **Integration tests:** `test/test_hdf5_loader.cpp` (HDF5 → TableData round-trip)
- **Coverage areas:**
  - 2D/3D/4D log interpolation vs. analytical expectations
  - Out-of-range policies (Clamp, FillNaN, Error)
  - Symmetric plane helpers and weighted sums
  - Derivative wrappers (`LogInterpolateDifferentiateSingleVariable*`)
  - HDF5 axis metadata validation (monotonicity, scale attributes)

### Git Practices
- **Main branch:** Protected; requires PR + CI pass
- **Commit messages:** Concise, imperative mood (e.g., "Add 4D log interpolation")
- **CI:** All PRs must pass `ci.yml` (builds + tests on Ubuntu + AMReX development branch)

### When Modifying Code
1. **Read before writing:** Always inspect existing files before proposing changes
2. **Test coverage:** Add tests for new features; update existing tests for changes
3. **Fortran parity:** If touching interpolation logic, verify against `ref/weaklib/` behavior
4. **Device safety:** Ensure new functions are `noexcept`, inline, and use `AMREX_GPU_HOST_DEVICE`
5. **Documentation:** Update `AGENTS.md` if changing scope/design; update `README.md` for user-facing changes

## Integration Points

### AMReX
- Headers include `AMReX_GpuQualifiers.H`, `AMReX_Extension.H`
- Tables stored as `amrex::TableData<double,4>`
- 5D datasets: last two axes flattened into TableData, but full 5D layout preserved for interpolation
- Use `LoadHdf5TableParallel` for MPI runs (rank 0 reads, broadcasts)

### HDF5
- Loader reads datasets + axis metadata from HDF5 files
- Required HDF5 structure:
  - Axis datasets: `/axis1`, `/axis2`, ... (1D arrays)
  - Value dataset: `/values` (N-D array matching axis dimensions)
  - Scale attributes: `"linear"` or `"log10"` per axis
- Validation: monotonic ascending, positive values for Log10 axes

### Fortran Reference
- Located in `ref/weaklib/`
- Key modules:
  - `wlKindModule.f90`: Precision definitions
  - `wlInterpolationModule.F90`: Main interpolation routines
  - `wlInterpolationUtilitiesModule.F90`: Helper functions
- **Always consult Fortran code** before implementing new interpolation logic

## Key Architectural Decisions (Locked)

1. **Out-of-range policy:** Default is `Clamp` (matches Fortran implicit behavior)
2. **Precision:** Double throughout (templatable later if needed)
3. **Table I/O:** HDF5 only (no NetCDF, no discovery frameworks)
4. **GPU backend:** CUDA first, HIP/DPCPP later
5. **Memory layout:** Explicit row-major with precomputed strides
6. **Device qualification:** All interpolation kernels are `AMREX_GPU_HOST_DEVICE`

## Current Implementation Snapshot

- **Layout & Strides:** `WeakLibReader/src/Layout.hpp` provides `Layout` and `MakeLayout` helpers that precompute row-major strides and sub-slice utilities (`SliceLeading`), keeping device functions inline and `noexcept`.
- **Index Lookup:** `WeakLibReader/src/IndexDelta.hpp` implements `IndexAndDeltaLin/Log10`, returning clamped cell indices and interpolation fractions plus an out-of-range flag to honor the configured policy.
- **Interpolation Kernels:** `WeakLibReader/src/InterpBasis.hpp` supplies linear through penta-linear blending and partial derivatives. `WeakLibReader/src/AxisTypes.hpp` provides axis metadata (`Axis`, `MakeAxis`), out-of-range policy (`InterpConfig`), and internal helpers used by higher-level APIs.
- **Log-Wrapped APIs:** `WeakLibReader/src/InterpLogTable.hpp` and `WeakLibReader/src/LogInterpolate.hpp` layer pow10/offset handling, symmetric plane helpers, and derivative evaluators that replicate the Fortran log-table entry points.
- **HDF5 Loader:** `WeakLibReader/src/Hdf5Loader.hpp` reads axis metadata + value datasets, validates monotonicity, and materializes tables into `amrex::TableData<double,4>` while preserving axis storage for interpolation.
- **Build & Tests:** `CMakeLists.txt` exposes the headers as an INTERFACE target with AMReX/OpenMP/HDF5 includes; `test/test_log_interpolate.cpp` and `test/test_hdf5_loader.cpp` cover interpolation kernels, policies, symmetry helpers, weighted sums, derivatives, and HDF5 round-trips via the bundled Catch shim.

## Performance Notes

- Precompute strides; pass compact axis/layout structs by value.
- Avoid branching in weight calc; coalesce reads; no dynamic allocations in kernels.
- Keep device functions `noexcept`; return status flags when needed.

## Security & Quality

- **Validation:** Host-side checks for axis monotonicity, Log10 domain positivity
- **Error handling:** Out-of-range policy configurable; device code returns status flags
- **No unsafe operations:** No raw pointer arithmetic without bounds (use `Layout::Offset`)

## Getting Help

- **Fortran reference:** Inspect `ref/weaklib/*.F90` for original implementation

## Do’s and Don’ts for Agents

**Do**

- Preserve numerical behavior at boundaries and under mixed Linear/Log10 axes.
- Keep device code free of STL containers and dynamic allocations.
- Provide small, `constexpr` helpers; keep functions `AMREX_GPU_HOST_DEVICE`.
- Ensure HDF5 loader retains axis storage backing the raw pointers returned in `Axis`.

**Don’t**

- Don’t add alternate table I/O formats beyond the sanctioned HDF5 loader.
- Don’t add OpenACC or non‑AMReX GPU pragmas.
