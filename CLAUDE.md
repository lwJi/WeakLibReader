# CLAUDE.md

WeakLibReader translates WeakLib's EOS & opacity interpolators from Fortran into GPU-friendly C++ integrated with AMReX. Target: numerical parity with Fortran (≤1e-12 relative error), CUDA first (HIP later).

## Tech Stack

- **C++17 header-only library** (all code in `src/` headers)
- **AMReX** for GPU portability (`AMREX_GPU_HOST_DEVICE`, `Gpu::DeviceVector`, `Gpu::PinnedVector`)
- **HDF5** for table I/O (only supported format)
- **Catch2-compatible stub** for testing (`test/` — custom lightweight implementation, not real Catch2)
- **CMake 3.18+** build system

## Project Structure

```
src/
  base/             # Core types: Axis, Layout, IndexDelta, InterpBasis, Math
  interp/           # Interpolation: LogInterpolate*, InterpLogTable* (2D–5D), EosInversion
  hdf5/             # HDF5 loaders: tables, EOS, opacity, device copies, MPI broadcast
test/               # Regression tests (EOS, opacity, interpolation, HDF5)
ref/weaklib/        # Fortran reference (consult before new interp logic)
cactus_interface/   # Cactus thorns: WeakLibReader (library) + TestWeakLibReader (example)
scripts/            # build.sh, test.sh, check.sh
specs/              # Detailed subsystem docs (see below)
```

## Public API

Three umbrella headers cover the full API:
- `WeakLibReader_Hdf5Loader.hpp` — All HDF5 loaders, device copy, MPI broadcast
- `WeakLibReader_LogInterpolate.hpp` — All interpolation functions (point, sweep, deriv)
- `WeakLibReader_EosInversion.hpp` — EOS temperature inversion (bisection)

## Build & Test

```bash
scripts/build.sh   # Build only
scripts/test.sh    # Run tests only
scripts/check.sh   # Build and test together
```

Set `VERBOSE=1` for full test output (e.g., `VERBOSE=1 scripts/test.sh`). The env var is read by the test binary, not the scripts themselves.

## Architecture

- **Column-major layout** — `stride[0]=1`, first dimension varies fastest; precomputed strides via `MakeLayout()`
- **No STL containers in device code** — no dynamic allocations in kernels
- **Pass by value** for small structs (`Axis`, `Layout`)
- **Out-of-range:** natural extrapolation (matches Fortran)
- **Double precision** throughout
- **Device functions** must be `noexcept`, inline, `AMREX_GPU_HOST_DEVICE`
- **Strict monotonicity** for axis grids (validated at load time)
- **GPU backend:** CUDA first, HIP/DPCPP later

## Do's and Don'ts

**Do**
- Add tests for new features; verify against `ref/weaklib/` for interpolation changes
- Preserve numerical behavior at boundaries and mixed Linear/Log10 axes
- Use `Layout::Offset` for bounds-safe data access
- Use imperative present-tense commit messages (e.g., "Add feature", "Fix bug")

**Don't**
- Add I/O formats beyond HDF5
- Use STL containers or dynamic allocations in device code
- Add OpenACC or non-AMReX GPU pragmas
- Skip host-side validation (monotonicity, Log10 positivity)

## Specs

Detailed reference material — consult when working on specific subsystems:

- **`specs/interpolation.md`** — Interpolation subsystem: log-space math, ND dispatch, sweep/deriv variants, EOS inversion
- **`specs/testing.md`** — Test structure, helpers, how to add new tests
- **`specs/hdf5_formats.md`** — HDF5 table schemas (simple, EOS, opacity) and loader function reference
- **`specs/opacity_tables.md`** — Opacity table types, dimensions, interpolation API, device patterns
- **`specs/cactus_integration.md`** — Cactus thorn CCL patterns, device loops, parameter sharing
