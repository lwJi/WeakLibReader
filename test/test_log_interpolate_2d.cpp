#define SIMPLE_CATCH_NO_MAIN
#include <catch2/catch_test_macros.hpp>

#include "interp/WeakLibReader_LogInterpolate.hpp"
#include "interp/WeakLibReader_InterpLogTable.hpp"
#include "base/WeakLibReader_Layout.hpp"
#include "base/WeakLibReader_AxisTypes.hpp"
#include "test_constants.hpp"

#include <array>
#include <cmath>

using test_constants::Tol;

TEST_CASE("2D log interpolation matches bilinear expectation", "[loginterp][2d]")
{
  using namespace WeakLibReader;

  const std::array<double, 2> gridX{1.0, 2.0};
  const std::array<double, 2> gridY{1.0, 3.0};
  const std::array<double, 4> table{
      std::log10(2.0),
      std::log10(3.0),
      std::log10(4.0),
      std::log10(5.0)};

  Axis axes[2] = {
      MakeAxis(gridX.data(), 2, AxisScale::Linear),
      MakeAxis(gridY.data(), 2, AxisScale::Linear)};

  const double x = 1.5;
  const double y = 2.0;
  const double result = LogInterpolateSingleVariable2DCustomPoint(
      x, y, axes, table.data(), 0.0);

  const double dX = (x - gridX[0]) / (gridX[1] - gridX[0]);
  const double dY = (y - gridY[0]) / (gridY[1] - gridY[0]);
  const double p00 = table[0];
  const double p10 = table[1];
  const double p01 = table[2];
  const double p11 = table[3];
  const double logExpected = (1.0 - dY) * ((1.0 - dX) * p00 + dX * p10) +
                             dY * ((1.0 - dX) * p01 + dX * p11);
  const double expected = std::pow(10.0, logExpected);

  CHECK(result == Catch::Approx(expected).margin(Tol));
}

TEST_CASE("Batch 2D log interpolation matches point wrapper", "[loginterp][2d][batch]")
{
  using namespace WeakLibReader;

  const std::array<double, 2> gridX{1.0, 2.0};
  const std::array<double, 2> gridY{1.0, 3.0};
  const std::array<double, 4> table{
      std::log10(2.0),
      std::log10(3.0),
      std::log10(4.0),
      std::log10(5.0)};

  Axis axes[2] = {
      MakeAxis(gridX.data(), 2, AxisScale::Linear),
      MakeAxis(gridY.data(), 2, AxisScale::Linear)};

  std::array<double, 3> x0{1.0, 1.5, 2.0};
  std::array<double, 3> x1{1.0, 2.0, 3.0};
  std::array<double, 3> out{};

  const int rc = LogInterpolateSingleVariable2DCustom(
      x0.data(), x1.data(), x0.size(),
      axes,
      table.data(),
      0.0,
      out.data());
  REQUIRE(rc == 0);

  for (std::size_t i = 0; i < x0.size(); ++i) {
    const double point = LogInterpolateSingleVariable2DCustomPoint(
        x0[i], x1[i], axes, table.data(), 0.0);
    CHECK(out[i] == Catch::Approx(point).margin(Tol));
  }
}

TEST_CASE("Out-of-range on mixed axes extrapolates correctly", "[loginterp][policy]")
{
  using namespace WeakLibReader;

  const std::array<double, 2> gridLog{1.0, 10.0};
  const std::array<double, 2> gridLin{2.0, 4.0};

  const int extents[2] = {2, 2};
  const Layout layout = MakeLayout(extents, 2);
  const double offset = 0.3;

  std::array<double, 4> table{};
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      const double actual = 1.0 + 0.2 * i + 0.4 * j;
      table[layout.Offset(i, j)] = std::log10(actual + offset);
    }
  }

  Axis axes[2] = {
      MakeAxis(gridLog.data(), 2, AxisScale::Log10),
      MakeAxis(gridLin.data(), 2, AxisScale::Linear)};

  double coords[2] = {0.5, 1.5};  // below both grids but positive

  int idxLog = 0;
  int idxLin = 0;
  double fracLog = 0.0;
  double fracLin = 0.0;
  IndexAndDeltaLog10(coords[0], gridLog.data(), 2, idxLog, fracLog);
  IndexAndDeltaLin(coords[1], gridLin.data(), 2, idxLin, fracLin);

  // Extrapolation should work with natural fractions (can be < 0 or > 1)
  const double aLog = 1.0 / (coords[0] * WeakLibReader::math::Log10(gridLog[1] / gridLog[0]));
  const double aLin = WeakLibReader::math::Ln10 / (gridLin[1] - gridLin[0]);

  double expectedInterp = 0.0;
  double expectedDLog = 0.0;
  double expectedDLin = 0.0;
  LinearInterpDeriv2DPoint(
      idxLog, idxLin,
      fracLog, fracLin,
      aLog, aLin,
      offset, table.data(), layout,
      expectedInterp, expectedDLog, expectedDLin);

  double interpolant = 0.0;
  double deriv[2] = {0.0, 0.0};
  detail::LogInterpolatedDerivativeDirect<2>(
      table.data(), layout, axes, coords, offset, interpolant, deriv);

  CHECK(interpolant == Catch::Approx(expectedInterp).margin(Tol));
  CHECK(deriv[0] == Catch::Approx(expectedDLog).margin(Tol));
  CHECK(deriv[1] == Catch::Approx(expectedDLin).margin(Tol));
}

TEST_CASE("Template instantiation compiles for all dimensions", "[compile-time]")
{
  using namespace WeakLibReader;

  if (false) {  // Never executed, but forces template instantiation
    double dummy_data[32] = {};
    int dummy_indices[5] = {};
    double dummy_fractions[5] = {};
    double dummy_scales[5] = {};
    double dummy_offset = 0.0;
    Layout dummy_layout{};

    // Test LinearInterpPointDirect for 1D-5D
    (void)LinearInterpPointDirect<1>(dummy_indices, dummy_fractions, dummy_offset,
                                     dummy_data, dummy_layout);
    (void)LinearInterpPointDirect<2>(dummy_indices, dummy_fractions, dummy_offset,
                                     dummy_data, dummy_layout);
    (void)LinearInterpPointDirect<3>(dummy_indices, dummy_fractions, dummy_offset,
                                     dummy_data, dummy_layout);
    (void)LinearInterpPointDirect<4>(dummy_indices, dummy_fractions, dummy_offset,
                                     dummy_data, dummy_layout);
    (void)LinearInterpPointDirect<5>(dummy_indices, dummy_fractions, dummy_offset,
                                     dummy_data, dummy_layout);

    // Test LinearInterpDerivPointDirect for 2D-4D (derivatives only)
    double dummy_interp = 0.0;
    double dummy_derivs[4] = {};
    LinearInterpDerivPointDirect<2>(dummy_indices, dummy_fractions, dummy_scales,
                                    dummy_offset, dummy_data, dummy_layout,
                                    dummy_interp, dummy_derivs);
    LinearInterpDerivPointDirect<3>(dummy_indices, dummy_fractions, dummy_scales,
                                    dummy_offset, dummy_data, dummy_layout,
                                    dummy_interp, dummy_derivs);
    LinearInterpDerivPointDirect<4>(dummy_indices, dummy_fractions, dummy_scales,
                                    dummy_offset, dummy_data, dummy_layout,
                                    dummy_interp, dummy_derivs);
  }

  // If we get here, all templates instantiated successfully
  CHECK(true);
}
