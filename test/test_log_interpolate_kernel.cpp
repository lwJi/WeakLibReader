#define SIMPLE_CATCH_NO_MAIN
#include <catch2/catch_test_macros.hpp>

#include "WeakLibReader_InterpLogTable.hpp"
#include "WeakLibReader_InterpBasis.hpp"
#include "WeakLibReader_Layout.hpp"
#include "WeakLibReader_Math.hpp"

#include <array>
#include <cmath>
#include <vector>

namespace {

constexpr double kTol = 1.0e-12;

} // namespace

// =============================================================================
// 1D Interpolation and 2D Derivative Kernel Tests (InterpLogTable.hpp)
// =============================================================================

TEST_CASE("LinearInterp1DPoint matches linear expectation", "[interp][1d]")
{
  using namespace WeakLibReader;

  // 1D table with log10 values
  const std::array<double, 3> table{std::log10(2.0), std::log10(4.0), std::log10(8.0)};
  const int extents[1] = {3};
  const Layout layout = MakeLayout(extents, 1);

  // Test interpolation at cell 0, fraction 0.5
  const int i0 = 0;
  const double d0 = 0.5;
  const double os = 0.0;

  const double result = LinearInterp1DPoint(i0, d0, os, table.data(), layout);

  // Expected: 10^(Linear(log10(2), log10(4), 0.5)) = 10^(0.5*log10(2) + 0.5*log10(4))
  const double logExpected = Linear(table[0], table[1], d0);
  const double expected = std::pow(10.0, logExpected);

  CHECK(result == Catch::Approx(expected).margin(kTol));
}

TEST_CASE("LinearInterp1DPoint with offset", "[interp][1d]")
{
  using namespace WeakLibReader;

  const std::array<double, 2> table{std::log10(5.0), std::log10(10.0)};
  const int extents[1] = {2};
  const Layout layout = MakeLayout(extents, 1);

  const double os = 1.0;
  const double d0 = 0.5;

  const double result = LinearInterp1DPoint(0, d0, os, table.data(), layout);

  const double logExpected = Linear(table[0], table[1], d0);
  const double expected = std::pow(10.0, logExpected) - os;

  CHECK(result == Catch::Approx(expected).margin(kTol));
}

TEST_CASE("LinearInterp1DPoint at boundaries", "[interp][1d]")
{
  using namespace WeakLibReader;

  const std::array<double, 2> table{std::log10(3.0), std::log10(6.0)};
  const int extents[1] = {2};
  const Layout layout = MakeLayout(extents, 1);

  // At d0=0, should return first value
  CHECK(LinearInterp1DPoint(0, 0.0, 0.0, table.data(), layout) ==
        Catch::Approx(3.0).margin(kTol));

  // At d0=1, should return second value
  CHECK(LinearInterp1DPoint(0, 1.0, 0.0, table.data(), layout) ==
        Catch::Approx(6.0).margin(kTol));
}

TEST_CASE("LinearInterpDeriv2DPoint returns correct interpolant", "[interp][deriv][2d]")
{
  using namespace WeakLibReader;

  const std::array<double, 4> table{
      std::log10(2.0), std::log10(3.0),
      std::log10(4.0), std::log10(5.0)};
  const int extents[2] = {2, 2};
  const Layout layout = MakeLayout(extents, 2);

  const int i0 = 0, i1 = 0;
  const double d0 = 0.5, d1 = 0.5;
  const double a0 = 1.0, a1 = 1.0;  // axis scale factors
  const double os = 0.0;

  double interpolant, dIdX0, dIdX1;
  LinearInterpDeriv2DPoint(i0, i1, d0, d1, a0, a1, os, table.data(), layout,
                           interpolant, dIdX0, dIdX1);

  // Verify interpolant matches LinearInterp2DPoint
  const double expected = LinearInterp2DPoint(i0, i1, d0, d1, os, table.data(), layout);
  CHECK(interpolant == Catch::Approx(expected).margin(kTol));
}

TEST_CASE("LinearInterpDeriv2DPoint derivatives match numerical", "[interp][deriv][2d][numerical]")
{
  using namespace WeakLibReader;

  const std::array<double, 4> table{
      std::log10(2.0), std::log10(3.0),
      std::log10(4.0), std::log10(5.0)};
  const int extents[2] = {2, 2};
  const Layout layout = MakeLayout(extents, 2);

  const double d0 = 0.4, d1 = 0.6;
  // a0, a1 encode axis scale factors; use Ln10 to get derivative w.r.t. fractional coordinate
  // since d(10^logValue)/d(d0) = 10^logValue * ln(10) * d(logValue)/d(d0)
  const double a0 = math::Ln10, a1 = math::Ln10;
  const double os = 0.0;
  const double h = 1.0e-7;

  double interpolant, dIdX0, dIdX1;
  LinearInterpDeriv2DPoint(0, 0, d0, d1, a0, a1, os, table.data(), layout,
                           interpolant, dIdX0, dIdX1);

  // Numerical derivative w.r.t. d0
  const double fPlusX0 = LinearInterp2DPoint(0, 0, d0 + h, d1, os, table.data(), layout);
  const double fMinusX0 = LinearInterp2DPoint(0, 0, d0 - h, d1, os, table.data(), layout);
  const double numericalX0 = (fPlusX0 - fMinusX0) / (2.0 * h);

  // Numerical derivative w.r.t. d1
  const double fPlusX1 = LinearInterp2DPoint(0, 0, d0, d1 + h, os, table.data(), layout);
  const double fMinusX1 = LinearInterp2DPoint(0, 0, d0, d1 - h, os, table.data(), layout);
  const double numericalX1 = (fPlusX1 - fMinusX1) / (2.0 * h);

  CHECK(dIdX0 == Catch::Approx(numericalX0).epsilon(1.0e-6));
  CHECK(dIdX1 == Catch::Approx(numericalX1).epsilon(1.0e-6));
}

TEST_CASE("LinearInterpDeriv2DPoint with offset", "[interp][deriv][2d]")
{
  using namespace WeakLibReader;

  const std::array<double, 4> table{
      std::log10(10.0), std::log10(20.0),
      std::log10(30.0), std::log10(40.0)};
  const int extents[2] = {2, 2};
  const Layout layout = MakeLayout(extents, 2);

  const double os = 5.0;
  const double d0 = 0.3, d1 = 0.7;
  // Use Ln10 as base scale, then multiply by arbitrary factors to test scaling
  const double scale0 = 2.0, scale1 = 0.5;
  const double a0 = math::Ln10 * scale0, a1 = math::Ln10 * scale1;

  double interpolant, dIdX0, dIdX1;
  LinearInterpDeriv2DPoint(0, 0, d0, d1, a0, a1, os, table.data(), layout,
                           interpolant, dIdX0, dIdX1);

  // Interpolant should match with offset
  const double expectedInterp = LinearInterp2DPoint(0, 0, d0, d1, os, table.data(), layout);
  CHECK(interpolant == Catch::Approx(expectedInterp).margin(kTol));

  // Compute numerical derivatives (w.r.t. fractional coordinates)
  const double h = 1.0e-7;
  const double fPlusX0 = LinearInterp2DPoint(0, 0, d0 + h, d1, os, table.data(), layout);
  const double fMinusX0 = LinearInterp2DPoint(0, 0, d0 - h, d1, os, table.data(), layout);
  const double numericalX0 = (fPlusX0 - fMinusX0) / (2.0 * h);

  const double fPlusX1 = LinearInterp2DPoint(0, 0, d0, d1 + h, os, table.data(), layout);
  const double fMinusX1 = LinearInterp2DPoint(0, 0, d0, d1 - h, os, table.data(), layout);
  const double numericalX1 = (fPlusX1 - fMinusX1) / (2.0 * h);

  // dIdX0 should be numericalX0 * scale0 (since a0 = Ln10 * scale0)
  CHECK(dIdX0 == Catch::Approx(numericalX0 * scale0).epsilon(1.0e-5));
  CHECK(dIdX1 == Catch::Approx(numericalX1 * scale1).epsilon(1.0e-5));
}

// =============================================================================
// Aligned Array Interpolation Tests (InterpLogTable.hpp)
// =============================================================================

TEST_CASE("LinearInterp2D3DArray1DAlignedPoint matches slice extraction", "[interp][aligned][3d]")
{
  using namespace WeakLibReader;

  // 3D array: 2 x 3 x 4 (iFixed x i0 x i1)
  const int extents[3] = {2, 3, 4};
  const Layout layout = MakeLayout(extents, 3);
  const std::size_t totalSize = 2 * 3 * 4;
  std::array<double, 24> table;

  // Fill with distinguishable log10 values
  for (std::size_t k = 0; k < totalSize; ++k) {
    table[k] = std::log10(static_cast<double>(k + 1));
  }

  const double d0 = 0.4, d1 = 0.6;
  const double os = 0.0;

  // Test at iFixed = 0
  {
    const int iFixed = 0;
    const int i0 = 1, i1 = 2;
    const double result = LinearInterp2D3DArray1DAlignedPoint(
        iFixed, i0, i1, d0, d1, os, table.data(), layout);

    // Manual slice: start at offset for iFixed=0, use 2D layout (3x4)
    const double* slice = table.data() + layout.Offset(iFixed, 0, 0);
    const Layout sliceLayout = SliceLeading(layout, 1);
    const double expected = LinearInterp2DPoint(i0, i1, d0, d1, os, slice, sliceLayout);

    CHECK(result == Catch::Approx(expected).margin(kTol));
  }

  // Test at iFixed = 1
  {
    const int iFixed = 1;
    const int i0 = 0, i1 = 1;
    const double result = LinearInterp2D3DArray1DAlignedPoint(
        iFixed, i0, i1, d0, d1, os, table.data(), layout);

    const double* slice = table.data() + layout.Offset(iFixed, 0, 0);
    const Layout sliceLayout = SliceLeading(layout, 1);
    const double expected = LinearInterp2DPoint(i0, i1, d0, d1, os, slice, sliceLayout);

    CHECK(result == Catch::Approx(expected).margin(kTol));
  }
}

TEST_CASE("LinearInterp3D4DArray1DAlignedPoint matches slice extraction", "[interp][aligned][4d]")
{
  using namespace WeakLibReader;

  // 4D array: 2 x 3 x 3 x 3 (iFixed x i0 x i1 x i2)
  const int extents[4] = {2, 3, 3, 3};
  const Layout layout = MakeLayout(extents, 4);
  const std::size_t totalSize = 2 * 3 * 3 * 3;
  std::vector<double> table(totalSize);

  for (std::size_t k = 0; k < totalSize; ++k) {
    table[k] = std::log10(static_cast<double>(k + 1));
  }

  const double d0 = 0.3, d1 = 0.5, d2 = 0.7;
  const double os = 0.0;

  for (int iFixed = 0; iFixed < 2; ++iFixed) {
    const int i0 = 1, i1 = 1, i2 = 1;
    const double result = LinearInterp3D4DArray1DAlignedPoint(
        iFixed, i0, i1, i2, d0, d1, d2, os, table.data(), layout);

    const double* slice = table.data() + layout.Offset(iFixed, 0, 0, 0);
    const Layout sliceLayout = SliceLeading(layout, 1);
    const double expected = LinearInterp3DPoint(i0, i1, i2, d0, d1, d2, os, slice, sliceLayout);

    CHECK(result == Catch::Approx(expected).margin(kTol));
  }
}

TEST_CASE("LinearInterp3D5DArray2DAlignedPoint matches slice extraction", "[interp][aligned][5d]")
{
  using namespace WeakLibReader;

  // 5D array: 2 x 2 x 3 x 3 x 3 (iFixed0 x iFixed1 x i0 x i1 x i2)
  const int extents[5] = {2, 2, 3, 3, 3};
  const Layout layout = MakeLayout(extents, 5);
  const std::size_t totalSize = 2 * 2 * 3 * 3 * 3;
  std::vector<double> table(totalSize);

  for (std::size_t k = 0; k < totalSize; ++k) {
    table[k] = std::log10(static_cast<double>(k + 1));
  }

  const double d0 = 0.25, d1 = 0.5, d2 = 0.75;
  const double os = 0.0;

  // Test all combinations of fixed indices
  for (int iFixed0 = 0; iFixed0 < 2; ++iFixed0) {
    for (int iFixed1 = 0; iFixed1 < 2; ++iFixed1) {
      const int i0 = 1, i1 = 1, i2 = 1;
      const double result = LinearInterp3D5DArray2DAlignedPoint(
          iFixed0, iFixed1, i0, i1, i2, d0, d1, d2, os, table.data(), layout);

      const double* slice = table.data() + layout.Offset(iFixed0, iFixed1, 0, 0, 0);
      const Layout sliceLayout = SliceLeading(layout, 2);
      const double expected = LinearInterp3DPoint(i0, i1, i2, d0, d1, d2, os, slice, sliceLayout);

      CHECK(result == Catch::Approx(expected).margin(kTol));
    }
  }
}

TEST_CASE("LinearInterp4D5DArray1DAlignedPoint matches slice extraction", "[interp][aligned][5d]")
{
  using namespace WeakLibReader;

  // 5D array: 2 x 3 x 3 x 3 x 3 (iFixed x i0 x i1 x i2 x i3)
  const int extents[5] = {2, 3, 3, 3, 3};
  const Layout layout = MakeLayout(extents, 5);
  const std::size_t totalSize = 2 * 3 * 3 * 3 * 3;
  std::vector<double> table(totalSize);

  for (std::size_t k = 0; k < totalSize; ++k) {
    table[k] = std::log10(static_cast<double>(k + 1));
  }

  const double d0 = 0.2, d1 = 0.4, d2 = 0.6, d3 = 0.8;
  const double os = 0.0;

  for (int iFixed = 0; iFixed < 2; ++iFixed) {
    const int i0 = 1, i1 = 1, i2 = 1, i3 = 1;
    const double result = LinearInterp4D5DArray1DAlignedPoint(
        iFixed, i0, i1, i2, i3, d0, d1, d2, d3, os, table.data(), layout);

    const double* slice = table.data() + layout.Offset(iFixed, 0, 0, 0, 0);
    const Layout sliceLayout = SliceLeading(layout, 1);
    const double expected = LinearInterp4DPoint(i0, i1, i2, i3, d0, d1, d2, d3, os, slice, sliceLayout);

    CHECK(result == Catch::Approx(expected).margin(kTol));
  }
}
