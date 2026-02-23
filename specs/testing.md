# Testing Infrastructure

## Framework

The project ships a bundled Catch2-compatible stub at
`test/include/catch2/catch_test_macros.hpp` (not the real Catch2). It provides
`TEST_CASE`, `CHECK`, `CHECK_FALSE`, `REQUIRE`, `REQUIRE_FALSE`, `Catch::Approx`, and a simple test runner.

## Test Structure

- `test/test_main.cpp` — Provides `main()` (includes the stub without
  `SIMPLE_CATCH_NO_MAIN`, which emits `main()`)
- All other test files start with:
  ```cpp
  #define SIMPLE_CATCH_NO_MAIN
  #include <catch2/catch_test_macros.hpp>
  ```
- All 13 `.cpp` files compile into one executable: `WeakLibReaderTests`
- One CTest target: `WeakLibReader.LogInterpolation` (runs all tests)

## Test Helpers — `test/include/`

### `test_constants.hpp`

```cpp
constexpr double Tol = 1.0e-12;  // the parity target
```

### `test_amrex_guard.hpp`

RAII singleton for AMReX init/finalize. Use in any test touching AMReX:

```cpp
AmrexGuard amrex{};  // at top of TEST_CASE
```

`GetGlobalAmrexGuard()` ensures `amrex::Initialize` is called exactly once.

### `test_hdf5_helpers.hpp`

All in `namespace test_helpers`. Used to build synthetic HDF5 files in tests:

| Function | Purpose |
|----------|---------|
| `WriteStringAttribute` | Fixed-length string attribute on HDF5 object |
| `WriteIntArrayDataset` | 1-D int dataset |
| `WriteDoubleArrayDataset` | 1-D double dataset |
| `WriteStringArrayDataset` | 1-D fixed-length string dataset |
| `WriteDoubleNdDataset<N>` | N-dimensional double dataset |
| `WriteIntNdDataset<N>` | N-dimensional int dataset |
| `CreateAxisDataset` | 1-D double dataset + `"scale"` attribute |
| `CopyDeviceToHost<T>` | `DeviceVector` → host `std::vector` |
| `VerifyDeviceRoundTrip<T>` | Copy back from device and check element-by-element |

## Test Categories

| File | Coverage |
|------|----------|
| `test_index_delta.cpp` | `IndexAndDeltaLin`, `IndexAndDeltaLog10`, clamping |
| `test_log_interpolate_basis.cpp` | All basis functions and derivatives, finite-difference checks |
| `test_log_interpolate_2d.cpp` | 2D point/batch interpolation, extrapolation |
| `test_log_interpolate_3d.cpp` | 3D point/batch, mixed axis scales |
| `test_log_interpolate_4d5d.cpp` | 4D/5D point, 1D-in-3D sweep |
| `test_log_interpolate_sweep.cpp` | 2D-in-4D aligned sweep, pre-alignment, Brem sum |
| `test_log_interpolate_deriv.cpp` | 3D and aligned 2D-in-4D derivatives |
| `test_log_interpolate_kernel.cpp` | Low-level kernel functions, aligned slice helpers |
| `test_weaklib_eos_loader.cpp` | Synthetic HDF5 EOS: load, device copy, parallel, errors |
| `test_weaklib_opacity_loader.cpp` | Synthetic HDF5 opacity: all 5 types, device copy, parallel |
| `test_eos_inversion.cpp` | Inversion bounds, round-trip T recovery, error codes |
| `test_extract_iso_slice.cpp` | `ExtractIsoMomentSlice4D` correctness |

## How to Add a New Test

### 1. Create the source file

```cpp
// test/test_my_feature.cpp

#define SIMPLE_CATCH_NO_MAIN
#include <catch2/catch_test_macros.hpp>

#include "WeakLibReader_SomeThing.hpp"
#include "test_constants.hpp"

using test_constants::Tol;

TEST_CASE("My feature does X", "[myfeature]")
{
    const double result = MyFunction(input);
    CHECK(result == Catch::Approx(expected).margin(Tol));
}
```

Include `test_amrex_guard.hpp` and declare `AmrexGuard amrex{};` if the test
uses AMReX or GPU vectors. Include `test_hdf5_helpers.hpp` for synthetic HDF5.

### 2. Register in CMake

Add the filename to `TEST_SOURCES` in `test/CMakeLists.txt`. No other changes
needed.

## Key Patterns

### Synthetic HDF5 data

Tests never read pre-existing table files. They create files with `H5Fcreate`,
populate with helpers from `test_hdf5_helpers.hpp`, load through the library,
assert, then delete. See `test_weaklib_eos_loader.cpp` for the pattern (helper functions define the file layout; `TEST_CASE` bodies perform load/assert/delete).

### Floating-point comparison

- **Absolute tolerance**: `CHECK(result == Catch::Approx(expected).margin(Tol))`
  with `Tol = 1e-12`
- **Relative tolerance**: `CHECK(a == Catch::Approx(b).epsilon(1e-6))` for
  derivative tests comparing analytical vs finite-difference
- **Exact equality**: `CHECK(a == b)` for integer/enum values and device
  round-trip (bitwise copy)

### Device round-trip

`VerifyDeviceRoundTrip<T, HostContainer>(deviceVec, hostVec)` copies back from GPU and checks
exact element-by-element equality. Used in all `MakeDeviceCopy` tests.

### EOS inversion round-trip

Forward-interpolate at known T to get observable (E/P/S), invert back, check
relative error: `|T_recovered - T_known| / T_known < 1e-10`.
