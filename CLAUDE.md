# WeakLibReader: AI Assistant Guide

## Quick Start

This is **WeakLibReader**, a GPU-friendly C++ reimplementation of WeakLib's equation-of-state and opacity interpolators for AMReX. This document provides essential guidance for AI assistants working on this codebase.

**📋 For detailed specifications, see [`AGENTS.md`](AGENTS.md)**

## Project Status (v1 - Active Development)

### ✅ Completed
- **Core N-D interpolation** (1D–5D) with Linear/Log10 axis scaling
- **Device-ready kernels** (`AMREX_GPU_HOST_DEVICE` qualified)
- **HDF5 table loader** with `amrex::TableData<double,4>` integration
- **Log-space interpolation** with derivatives and symmetric plane helpers
- **Comprehensive test suite** with 2D/3D/4D coverage
- **CI/CD pipeline** (GitHub Actions)
- **Out-of-range policies** (Clamp, Error, FillNaN)
- **Fortran parity** for numerical accuracy (target: ≤1e-12 relative error)

### 🚧 Pending
- **AMReX CUDA demo** (`examples/amrex/` - Phase 3, marked as TBD)
- **HIP/DPCPP backend validation** (post-v1)
- **Comprehensive Fortran parity harness** (random point validation)

## Repository Structure

```
WeakLibReader/
├── CLAUDE.md                    # This file (AI assistant guide)
├── AGENTS.md                    # Detailed project specifications
├── README.md                    # User-facing documentation
├── CMakeLists.txt               # Build configuration
├── .github/workflows/           # CI/CD workflows
│   ├── ci.yml                   # Main CI pipeline
│   ├── claude.yml               # Claude Code integration
│   └── claude-code-review.yml   # Automated code review
├── WeakLibReader/src/           # Core library (header-only)
│   ├── WeakLibReader.hpp        # Main API, N-D interpolation
│   ├── IndexDelta.hpp           # Linear/log10 indexing
│   ├── InterpBasis.hpp          # Basis functions (linear→penta-linear)
│   ├── InterpLogTable.hpp       # Log-space interpolation kernels
│   ├── LogInterpolate.hpp       # Fortran-parity log API
│   ├── Hdf5Loader.hpp           # HDF5 → amrex::TableData loader
│   ├── Layout.hpp               # Row-major stride computation
│   └── Math.hpp                 # Minimal math utilities
├── ref/weaklib/                 # Fortran reference implementation
│   ├── wlKindModule.f90
│   ├── wlInterpolationModule.F90
│   └── wlInterpolationUtilitiesModule.F90
└── test/                        # Test suite (Catch2-style)
    ├── test_log_interpolate.cpp
    ├── test_hdf5_loader.cpp
    └── include/catch2/          # Bundled test framework shim
```

## Code Style & Conventions

### Naming (Strictly Enforced)
- **Namespace:** `WeakLibReader`
- **Types/Structs/Enums:** `PascalCase` (e.g., `Axis`, `InterpConfig`, `OutOfRangePolicy`)
- **Functions:** `PascalCase` (e.g., `IndexAndDeltaLin`, `InterpLinearND`)
- **Variables/Parameters:** `lowerCamelCase` (e.g., `outOfRange`, `rowStride`)
- **Constants/Enumerators:** `PascalCase` (e.g., `Clamp`, `Linear`, `Log10`)

### C++ Standards
- **Standard:** C++17+ required
- **Qualifiers:** Use `AMREX_GPU_HOST_DEVICE` for device-callable functions
- **Style:** Keep device code `noexcept`, inline, and allocation-free
- **Headers:** Include `<AMReX_GpuQualifiers.H>` for GPU macros

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
- **Feature branches:** Named descriptively (e.g., `feature/add-5d-interp`)
- **Commit messages:** Concise, imperative mood (e.g., "Add 4D log interpolation")
- **CI:** All PRs must pass `ci.yml` (builds + tests on Ubuntu + AMReX development branch)

### When Modifying Code
1. **Read before writing:** Always inspect existing files before proposing changes
2. **Test coverage:** Add tests for new features; update existing tests for changes
3. **Fortran parity:** If touching interpolation logic, verify against `ref/weaklib/` behavior
4. **Device safety:** Ensure new functions are `noexcept`, inline, and use `AMREX_GPU_HOST_DEVICE`
5. **Documentation:** Update `AGENTS.md` if changing scope/design; update `README.md` for user-facing changes

## Common AI Assistant Tasks

### Adding New Interpolation Functionality
1. Check `ref/weaklib/*.F90` for reference implementation
2. Implement in appropriate header (`InterpBasis.hpp`, `InterpLogTable.hpp`, etc.)
3. Add `AMREX_GPU_HOST_DEVICE` qualifier and `noexcept`
4. Write unit test in `test/test_log_interpolate.cpp`
5. Validate numerical parity with Fortran (if applicable)

### Adding New HDF5 Loader Features
1. Modify `Hdf5Loader.hpp` (host-side only)
2. Maintain `amrex::TableData<double,4>` materialization
3. Validate axis monotonicity and scale attributes
4. Add test case to `test/test_hdf5_loader.cpp`

### Debugging Numerical Differences
1. Check axis scale (`Linear` vs `Log10`) consistency
2. Verify index clamping behavior (Fortran uses 1-based, C++ uses 0-based)
3. Confirm out-of-range policy matches Fortran's implicit clamping
4. Inspect `IndexAndDeltaLin`/`IndexAndDeltaLog10` for edge cases
5. Compare with Fortran reference output (captured TSV recommended)

### Performance Optimization
1. Keep functions inline and `noexcept`
2. Avoid branching in inner loops (use compile-time dispatch where possible)
3. Coalesce memory reads (row-major layout assumed)
4. Precompute strides in `Layout` struct
5. Profile on target GPU (CUDA first, then HIP)

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
3. **Table I/O:** HDF5 only (no NetCDF, no discovery frameworks in v1)
4. **GPU backend:** CUDA first, HIP/DPCPP later
5. **Memory layout:** Explicit row-major with precomputed strides
6. **Device qualification:** All interpolation kernels are `AMREX_GPU_HOST_DEVICE`

## Security & Quality

- **Validation:** Host-side checks for axis monotonicity, Log10 domain positivity
- **Error handling:** Out-of-range policy configurable; device code returns status flags
- **No unsafe operations:** No raw pointer arithmetic without bounds (use `Layout::Offset`)
- **Build system:** CMake 3.18+; AMReX found via config or manual path

## Recent Activity (Last 5 Commits)

1. **#38:** Enhanced `.gitignore` with comprehensive patterns
2. **#36:** Respect error policy in log interpolation
3. **#35:** Added more test coverage
4. **#34:** Added 4D log interpolation coverage
5. **#32:** Improved AMReX discovery in CMake

## Non-Goals (v1)

- Additional table formats (NetCDF, custom binary)
- HIP/DPCPP backend validation (deferred to v2)
- Higher-order interpolation (cubic, spline)
- Runtime axis scale detection (must be specified at load time)
- Dynamic dimensionality (N-D dispatch is compile-time via overloads)

## Getting Help

- **Project spec:** See [`AGENTS.md`](AGENTS.md) for detailed design
- **User guide:** See [`README.md`](README.md) for build/test instructions
- **Fortran reference:** Inspect `ref/weaklib/*.F90` for original implementation
- **CI logs:** Check `.github/workflows/ci.yml` output for build/test failures

## Summary for AI Assistants

**When working on this codebase:**
1. ✅ **Do** read `AGENTS.md` for full context before starting
2. ✅ **Do** consult Fortran reference in `ref/weaklib/` for numerical parity
3. ✅ **Do** add tests for all new functionality
4. ✅ **Do** use PascalCase for functions, lowerCamelCase for variables
5. ✅ **Do** keep device code `noexcept`, inline, and allocation-free
6. ❌ **Don't** add new table I/O formats (HDF5 only in v1)
7. ❌ **Don't** change AMReX build options outside `examples/amrex/`
8. ❌ **Don't** use OpenACC or non-AMReX GPU pragmas
9. ❌ **Don't** modify code without reading existing implementation first

---

**Last updated:** 2025-12-03
**Project version:** v0.1.0
**Status:** Active development (v1 nearing completion)
