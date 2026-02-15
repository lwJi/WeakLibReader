# Refactor Opportunities (Codebase Survey)

This note captures high-impact refactor opportunities identified from a quick survey of the current headers. The focus is reducing maintenance cost while preserving the current numerical behavior and GPU portability constraints.

## 1) Consolidate repeated N-D interpolation wrappers (2D/3D/4D)

### Current state
`WeakLibReader_LogInterpolatePoint.hpp` and `WeakLibReader_LogInterpolateSweep.hpp` contain many near-identical wrappers that differ only by dimensionality and argument packing.

Examples:
- `LogInterpolateSingleVariable2DCustomPoint`, `3D...Point`, `4D...Point`
- batch variants that repeat the same null checks, `MakeLayout` setup, and loop structure

### Refactor idea
Introduce small template helpers for:
- validating axis/data pointers
- constructing extents/layout from `Axis[ND]`
- evaluating one point from `coords[ND]`

Then keep the public API functions as thin, stable wrappers.

### Why
- De-duplicates null-check + layout boilerplate.
- Reduces risk of behavior drift between dimensions when bug-fixing.
- Keeps generated code efficient (compile-time ND remains available).

## 2) Unify symmetric plane sweep logic with derivative and non-derivative paths

### Current state
2D×2D sweep and aligned-sweep implementations are repeated in both:
- `WeakLibReader_LogInterpolateSweep.hpp`
- `WeakLibReader_LogInterpolateDeriv.hpp`

The nested `(j, i<=j)` loops and `StoreSymmetric(...)` write pattern are duplicated, with only the compute kernel differing.

### Refactor idea
Create a shared internal symmetric traversal helper (host/device inline) that:
- iterates upper triangle
- calls a supplied functor/lambda to compute value(s)
- applies symmetric storage

Use it in both derivative and non-derivative APIs.

### Why
- Makes indexing/symmetry rules single-source.
- Easier to reason about correctness at matrix boundaries.
- Reduces copy/paste maintenance for future kernels.

## 3) Reduce HDF5 file-open repetition in opacity top-level loader

### Current state
`LoadWeakLibOpacityTableFull` repeatedly performs:
1. `H5Fopen` + validity check
2. one specific table load call
3. immediate status return on failure

This pattern appears for EmAb/Iso/NES/Pair/Brem plus shared-grid reads.

### Refactor idea
Introduce an internal helper:
- `WithOpenedFile(path, callable)` returning `Hdf5LoadStatus`

Then each section becomes a one-liner invoking the appropriate loader.

### Why
- Less repetitive control flow.
- Centralized file-open error handling.
- Easier to extend if more table families are ever added.

## 4) Clarify and centralize data-layout terminology

### Current state
Core code and project docs consistently describe column-major traversal, but the README API table still describes `Layout` as row-major.

### Refactor idea
- Correct README wording to column-major.
- Consider adding a short `Layout` contract block in one location (README/spec) and linking to it from interpolation and HDF5 docs.

### Why
- Prevents user confusion when interoperating with external arrays.
- Reduces onboarding mistakes for contributors touching interpolation offsets/strides.

## 5) Add micro-regression tests around shared helpers before structural refactors

### Current state
Interpolation coverage is strong, but helper-level behavior (pointer validation, symmetry fill invariants, layout assembly from axes) is mostly exercised indirectly.

### Refactor idea
Before larger refactors, add focused unit tests for:
- pointer/null guard return codes on wrapper entry points
- symmetry invariants (`out[i,j] == out[j,i]`) for sweep and derivative paths
- layout stride expectations from common extents

### Why
- De-risks internal refactor by locking expected wrapper behavior.
- Faster diagnosis when refactor affects only scaffolding, not math kernels.

## Suggested execution order

1. Add helper-focused tests (Opportunity 5).
2. Refactor interpolation wrapper boilerplate (Opportunity 1).
3. Refactor symmetric traversal helper (Opportunity 2).
4. Refactor HDF5 open/load flow (Opportunity 3).
5. Keep terminology/docs synchronized continuously (Opportunity 4).
