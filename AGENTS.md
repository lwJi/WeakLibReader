# AGENTS.md

## Project Goal

Translate WeakLib’s EOS & opacity **interpolators** from Fortran into **GPU‑friendly C++** that integrates cleanly with **AMReX**. Achieve tight numerical agreement with the Fortran reference at interior points and boundaries, with predictable performance on **CUDA** first (HIP later).

## Locked Decisions (Approved)

1. **Out‑of‑range policy:** default **Clamp**.
2. **Precision:** default **double** throughout (templatable later).
3. **v1 scope:** **No HDF5 loader** (table loading deferred to a later phase).
4. **Backends to validate:** **CUDA first** (HIP/DPCPP later).

## Scope (v1)

* Generic **N‑D linear interpolation** (1D–5D) with per‑axis **Linear/Log10** spacing.
* **C++ API** usable in device code (`AMREX_GPU_HOST_DEVICE`) with thin host wrappers.
* Explicit **row‑major** data layout with precomputed strides and axis metadata.
* **No** table I/O in v1; assume arrays are already in memory.

## Naming & Style (CamelCase)

* **Namespace:** `WeakLibReader`
* **Types/structs/enums/classes:** `PascalCase` (e.g., `Axis`, `Layout`, `InterpConfig`, `AxisScale`, `OutOfRangePolicy`).
* **Functions (public & device):** `PascalCase` (e.g., `IndexAndDeltaLin`, `InterpLinearND`).
* **Variables/parameters/data members:** `lowerCamelCase` (e.g., `outOfRange`, `rowStride`).
* **Constants/enumerators:** `PascalCase` (e.g., `Clamp`, `Error`, `FillNaN`).
* **C++ standard:** C++17+.

## Directory Layout (v1)

```
WeakLibReader/
  src/                        # Core headers (public API + device helpers)
    WeakLibReader.hpp         # Axis metadata, layout, N-D interpolation entrypoints
    IndexDelta.hpp            # Linear/log10 indexing helpers
    InterpBasis.hpp           # Linear/bi-/tri-/tetra-/penta-linear basis routines
    InterpLogTable.hpp        # Log-space point kernels and aligned slices
    LogInterpolate.hpp        # Log wrappers, derivatives, weighted sums
    Layout.hpp                # Row-major stride helpers
    Math.hpp                  # Minimal math utilities (log10, pow10, etc.)
ref/weaklib/                  # Fortran reference implementation
test/
  include/catch2/             # Minimal Catch2-compatible shim
  test_log_interpolate.cpp    # Regression tests (aligned planes, derivatives, etc.)
examples/amrex/               # CUDA/AMReX demo scaffold (TBD)
```

> No HDF5 loader files in v1.

## Setup & Build (CUDA first)

Refer to `README.md` for build and test commands (including the required `AMREX_ROOT` and OpenMP flags).

## Public API (Target)

```cpp
#pragma once
#include <cstddef>
#include <AMReX_GpuQualifiers.H>

namespace WeakLibReader {

enum class AxisScale : uint8_t { Linear, Log10 };
struct Axis {
  const double* grid;  // length n
  int n;
  AxisScale scale;
};

struct Layout {
  int nd;                  // 1..5
  int n[5];                // extents
  std::size_t stride[5];   // row-major strides (data[k] * stride[k])
};

enum class OutOfRangePolicy : uint8_t { Clamp, Error, FillNaN };

struct InterpConfig {
  OutOfRangePolicy outOfRange = OutOfRangePolicy::Clamp;
};

// Index and interpolation fraction for linearly spaced axis
AMREX_GPU_HOST_DEVICE
bool IndexAndDeltaLin(double x, const double* grid, int n,
                      int& i, double& t) noexcept;

// Index and interpolation fraction for log10-spaced axis
AMREX_GPU_HOST_DEVICE
bool IndexAndDeltaLog10(double x, const double* grid, int n,
                        int& i, double& t) noexcept;

// N-D linear interpolation (row-major data, up to 5D)
AMREX_GPU_HOST_DEVICE
double InterpLinearND(const double* data, const Layout& layout,
                      const Axis axes[5], const double x[5],
                      const InterpConfig& cfg, int nd) noexcept;

// Optional 1D..5D convenience overloads may forward to InterpLinearND(...)

} // namespace WeakLibReader
```

## Current Implementation Snapshot

- **Layout & Strides:** `WeakLibReader/src/Layout.hpp` provides `Layout` and `MakeLayout` helpers that precompute row-major strides and sub-slice utilities (`SliceLeading`), keeping device functions inline and `noexcept`.
- **Index Lookup:** `WeakLibReader/src/IndexDelta.hpp` implements `IndexAndDeltaLin/Log10`, returning clamped cell indices and interpolation fractions plus an out-of-range flag to honor the configured policy.
- **Interpolation Kernels:** `WeakLibReader/src/InterpBasis.hpp` supplies linear through penta-linear blending and partial derivatives, which `WeakLibReader/src/WeakLibReader.hpp` composes in `InterpLinearND` and its 1D–5D overloads.
- **Log-Wrapped APIs:** `WeakLibReader/src/InterpLogTable.hpp` and `WeakLibReader/src/LogInterpolate.hpp` layer pow10/offset handling, symmetric plane helpers, and derivative evaluators that replicate the Fortran log-table entry points.
- **Build & Tests:** `CMakeLists.txt` exposes the headers as an INTERFACE target with AMReX/OpenMP includes; `test/test_log_interpolate.cpp` exercises 2D log interpolation, FillNaN policy, symmetric plane fills, and weighted-sum helpers via a bundled Catch-style runner.

## Behavior Details

* **Axes:** strictly monotone ascending; `Log10` axes require `grid[k] > 0`.
* **Out‑of‑range:** default **Clamp** to domain; other policies via `InterpConfig`.
* **Precision:** default `double`; can template later if required.
* **Parity target:** match Fortran edge handling for indices/weights (lower/upper bounds).

## AMReX Usage Example (Sketch)

```cpp
using namespace WeakLibReader;

amrex::ParallelFor(mf.boxArray(), mf.DistributionMap(), mf.nComp(),
[=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept {
  double X[3] = {/* rho, T, Ye, etc. */};
  double value = InterpLinearND(deviceDataPtr, layout, axes, X, InterpConfig{}, 3);
  mf.array(i,j,k,n) = value;
});
```

## Agent Task Plan / Phases (v1)

**Phase 0 – Reference Mapping**

* Extract exact interpolation steps (indexing, delta, lin/log) from Fortran.
* **AC:** Design notes + test matrix of edge cases.

**Phase 1 – Core N‑D Interpolation**

* Implement `IndexAndDelta*` and `InterpLinearND` (1D–5D).
* **AC:** Unit tests pass on synthetic grids; O(2^N) loads only.

**Phase 2 – Fortran Parity Tests**

* Build a small harness to evaluate random points vs Fortran outputs (captured TSV).
* **AC:** Relative error ≤ 1e‑12 (double) on representative domains.

**Phase 3 – AMReX Demo (CUDA)**

* Example fills a `MultiFab` by interpolating table values on GPU.
* **AC:** Runs on CUDA; smoke tests pass.

> **Deferred (post‑v1):** HDF5 table loader and HIP/DPCPP validation.

## Tests

* Catch-style unit tests (bundled shim) for:

  * 2D log interpolation parity with bilinear expectation.
  * FillNaN out-of-range behavior.
  * Symmetric plane helpers and weighted-sum accumulators.
  * Derivative wrappers for log-stored tables (3D cases).

## Performance Notes

* Precompute strides; pass compact axis/layout structs by value.
* Avoid branching in weight calc; coalesce reads; no dynamic allocations in kernels.
* Keep device functions `noexcept`; return status flags when needed.

## Security & Quality Guardrails

* Host‑side validation for axis monotonicity and log domain.
* Device side honors `outOfRange` policy (Clamp by default).
* No OpenACC; only AMReX GPU constructs.

## Supplying Fortran Sources to the Agent

* Place Fortran reference files under `ref/weaklib/` (e.g., `wlInterpolationModule.F90`, `wlInterpolationUtilitiesModule.F90`).
* Agents should **read all Fortran files** in this directory **before** implementing the C++ API.
* If your tooling supports globbing, use patterns like `ref/weaklib/**/*.F90`.
* If globbing is unavailable, list files explicitly in the task invocation.

## Do’s and Don’ts for Agents

**Do**

* Preserve numerical behavior at boundaries and under mixed Linear/Log10 axes.
* Keep device code free of STL containers and dynamic allocations.
* Provide small, `constexpr` helpers; keep functions `AMREX_GPU_HOST_DEVICE`.

**Don’t**

* Don’t introduce I/O or HDF5 in v1.
* Don’t change AMReX build options outside `examples/amrex/`.
* Don’t add OpenACC or non‑AMReX GPU pragmas.

## Deliverables (v1)

* `WeakLibReader/src/` headers (API + kernels + helpers).
* `test/` regression suite (Catch-style shim + coverage for log/derivative paths).
* `ref/weaklib/` Fortran references and notes.
* `examples/amrex/` CUDA demo scaffold (fleshed out in Phase 3).
* `README.md` documenting build/test steps and dependencies.

## Completion Criteria (v1)

* All tests pass on CPU and CUDA.
* Parity harness meets error tolerance (≤ 1e‑12 relative in double).
* AMReX demo runs on CUDA and writes expected field values.

## Non‑Goals (v1)

* HDF5/NetCDF readers, table discovery, or I/O pipelines.
* HIP/DPCPP backend validation.
* Higher‑order interpolation (e.g., cubic).
