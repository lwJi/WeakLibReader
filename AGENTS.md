# AGENTS.md

## Project Goal

Translate WeakLib's EOS & opacity **interpolators** from Fortran into **GPU-friendly C++** that integrates with **AMReX**. Target: numerical parity with Fortran (≤1e-12 relative error), CUDA first (HIP later).

## Tech Stack

- **C++17 header-only library** (all code in `src/` headers)
- **AMReX** for GPU portability (`AMREX_GPU_HOST_DEVICE`, `Gpu::DeviceVector`, `Gpu::PinnedVector`)
- **HDF5** for table I/O (only supported format)
- **Catch2** test framework (`test/`)
- **CMake 3.18+** build system

## Project Structure

```
src/
  base/             # Core types: Axis, Layout, IndexDelta, InterpBasis, Math
  interp/           # Interpolation: LogInterpolate*, InterpLogTable* (2D–5D)
  hdf5/             # HDF5 loaders: tables, EOS, opacity, device copies, MPI broadcast
test/               # Catch2 regression tests (EOS, opacity, interpolation, HDF5)
ref/weaklib/        # Fortran reference (consult before new interp logic)
cactus_interface/   # Cactus thorns: WeakLibReader (library) + TestWeakLibReader (example)
scripts/            # build.sh, test.sh, check.sh
agent_docs/         # Detailed subsystem docs (see below)
```

## Build & Test

```bash
scripts/build.sh   # Build only
scripts/test.sh    # Run tests only
scripts/check.sh   # Build and test together
```

Set `VERBOSE=1` for full test output (e.g., `VERBOSE=1 scripts/test.sh`).

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

**Don't**
- Add I/O formats beyond HDF5
- Use STL containers or dynamic allocations in device code
- Add OpenACC or non-AMReX GPU pragmas
- Skip host-side validation (monotonicity, Log10 positivity)

## Fortran Reference

Located in `ref/weaklib/`. Key modules: `wlInterpolationModule.F90`, `wlInterpolationUtilitiesModule.F90`. Always consult before implementing new interpolation logic.

## Agent Docs

Detailed reference material in `agent_docs/` — consult when working on specific subsystems:

- **`hdf5_formats.md`** — HDF5 table schemas (simple, EOS, opacity) and loader function reference
- **`opacity_tables.md`** — Opacity table types, dimensions, interpolation API, device patterns
- **`cactus_integration.md`** — Cactus thorn CCL patterns, device loops, parameter sharing
