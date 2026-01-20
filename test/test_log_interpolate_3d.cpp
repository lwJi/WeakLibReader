#define SIMPLE_CATCH_NO_MAIN
#include <catch2/catch_test_macros.hpp>

#include "LogInterpolate.hpp"
#include "Layout.hpp"
#include "AxisTypes.hpp"

#include <array>
#include <cmath>
#include <limits>

namespace {

constexpr double kTol = 1.0e-12;

} // namespace

TEST_CASE("3D log interpolation matches trilinear expectation", "[loginterp][3d]")
{
  using namespace WeakLibReader;

  const std::array<double, 2> gridX{1.0, 2.0};
  const std::array<double, 2> gridY{1.0, 3.0};
  const std::array<double, 2> gridZ{0.0, 1.0};

  // Build a 2x2x2 table with known values
  const int extents[3] = {2, 2, 2};
  const Layout layout = MakeLayout(extents, 3);
  std::array<double, 8> table{};
  auto actual = [](double x, double y, double z) {
    return 2.0 + 0.5 * x + 0.3 * y + 0.2 * z;
  };
  for (int ix = 0; ix < 2; ++ix) {
    for (int iy = 0; iy < 2; ++iy) {
      for (int iz = 0; iz < 2; ++iz) {
        table[layout.Offset(ix, iy, iz)] =
            std::log10(actual(gridX[ix], gridY[iy], gridZ[iz]));
      }
    }
  }

  const double x = 1.5;
  const double y = 2.0;
  const double z = 0.4;
  const double result = LogInterpolateSingleVariable3DCustomPoint(
      x, y, z,
      gridX.data(), 2,
      gridY.data(), 2,
      gridZ.data(), 2,
      table.data(), 0.0);

  // Compute expected via trilinear interpolation in log space
  // X and Y axes use Log10 scale, Z axis uses Linear scale
  const double dX = (std::log10(x) - std::log10(gridX[0])) /
                    (std::log10(gridX[1]) - std::log10(gridX[0]));
  const double dY = (std::log10(y) - std::log10(gridY[0])) /
                    (std::log10(gridY[1]) - std::log10(gridY[0]));
  const double dZ = (z - gridZ[0]) / (gridZ[1] - gridZ[0]);

  const double c000 = table[layout.Offset(0, 0, 0)];
  const double c100 = table[layout.Offset(1, 0, 0)];
  const double c010 = table[layout.Offset(0, 1, 0)];
  const double c110 = table[layout.Offset(1, 1, 0)];
  const double c001 = table[layout.Offset(0, 0, 1)];
  const double c101 = table[layout.Offset(1, 0, 1)];
  const double c011 = table[layout.Offset(0, 1, 1)];
  const double c111 = table[layout.Offset(1, 1, 1)];

  const double c00 = (1 - dX) * c000 + dX * c100;
  const double c10 = (1 - dX) * c010 + dX * c110;
  const double c01 = (1 - dX) * c001 + dX * c101;
  const double c11 = (1 - dX) * c011 + dX * c111;
  const double c0 = (1 - dY) * c00 + dY * c10;
  const double c1 = (1 - dY) * c01 + dY * c11;
  const double logExpected = (1 - dZ) * c0 + dZ * c1;
  const double expected = std::pow(10.0, logExpected);

  CHECK(result == Catch::Approx(expected).margin(kTol));
}

TEST_CASE("Batch 3D log interpolation matches point wrapper", "[loginterp][3d][batch]")
{
  using namespace WeakLibReader;

  const std::array<double, 2> gridX{1.0, 2.0};
  const std::array<double, 2> gridY{1.0, 3.0};
  const std::array<double, 2> gridZ{0.0, 1.0};

  const int extents[3] = {2, 2, 2};
  const Layout layout = MakeLayout(extents, 3);
  std::array<double, 8> table{};
  for (int ix = 0; ix < 2; ++ix) {
    for (int iy = 0; iy < 2; ++iy) {
      for (int iz = 0; iz < 2; ++iz) {
        table[layout.Offset(ix, iy, iz)] =
            std::log10(2.0 + 0.5 * gridX[ix] + 0.3 * gridY[iy] + 0.2 * gridZ[iz]);
      }
    }
  }

  std::array<double, 3> x0{1.0, 1.5, 2.0};
  std::array<double, 3> x1{1.0, 2.0, 3.0};
  std::array<double, 3> x2{0.0, 0.5, 1.0};
  std::array<double, 3> out{};

  const int rc = LogInterpolateSingleVariable3DCustom(
      x0.data(), x1.data(), x2.data(), x0.size(),
      gridX.data(), 2,
      gridY.data(), 2,
      gridZ.data(), 2,
      table.data(),
      0.0,
      out.data());
  REQUIRE(rc == 0);

  for (std::size_t i = 0; i < x0.size(); ++i) {
    const double point = LogInterpolateSingleVariable3DCustomPoint(
        x0[i], x1[i], x2[i],
        gridX.data(), 2,
        gridY.data(), 2,
        gridZ.data(), 2,
        table.data(),
        0.0);
    CHECK(out[i] == Catch::Approx(point).margin(kTol));
  }
}

// Note: The test "Invalid log coordinate triggers error code" was removed as part
// of Issue #67. The defensive validation in ComputeAxisScale was removed because:
// 1. HDF5 loader guarantees valid axes at load time
// 2. IndexAndDelta validates coordinates for Log10 axes
// 3. Invalid coordinates are programming errors caught by AMREX_ASSERT in debug builds
// 4. The Fortran reference has no such runtime validation

// Note: The test "Zero-span axis yields NaN under FillNaN policy" was removed as part
// of Issue #70. OutOfRangePolicy and InterpConfig have been removed to match Fortran
// behavior where out-of-range coordinates simply extrapolate.
