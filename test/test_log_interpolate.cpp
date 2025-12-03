#include <catch2/catch_test_macros.hpp>

#include "LogInterpolate.hpp"
#include "InterpLogTable.hpp"
#include "Layout.hpp"
#include "WeakLibReader.hpp"

#include <array>
#include <cmath>
#include <limits>

namespace {

constexpr double kTol = 1.0e-12;

} // namespace

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

  const int extents[2] = {2, 2};
  const Layout layout = MakeLayout(extents, 2);

  Axis axes[2] = {
      MakeAxis(gridX.data(), 2, AxisScale::Linear),
      MakeAxis(gridY.data(), 2, AxisScale::Linear)};

  const double x = 1.5;
  const double y = 2.0;
  const double result = LogInterpolateSingleVariable2DCustomPoint(
      x, y,
      gridX.data(), 2,
      gridY.data(), 2,
      table.data(), 0.0);

  const double dX = (x - gridX[0]) / (gridX[1] - gridX[0]);
  const double dY = (y - gridY[0]) / (gridY[1] - gridY[0]);
  const double p00 = table[0];
  const double p10 = table[1];
  const double p01 = table[2];
  const double p11 = table[3];
  const double logExpected = (1.0 - dY) * ((1.0 - dX) * p00 + dX * p10) +
                             dY * ((1.0 - dX) * p01 + dX * p11);
  const double expected = std::pow(10.0, logExpected);

  CHECK(result == Catch::Approx(expected).margin(kTol));
}

TEST_CASE("Out-of-range clamp FillNaN policy returns NaN", "[loginterp][2d][nan]")
{
  using namespace WeakLibReader;

  const std::array<double, 2> gridX{1.0, 2.0};
  const std::array<double, 2> gridY{1.0, 3.0};
  const std::array<double, 4> table{
      std::log10(2.0),
      std::log10(3.0),
      std::log10(4.0),
      std::log10(5.0)};

  const int extents[2] = {2, 2};
  const Layout layout = MakeLayout(extents, 2);

  Axis axes[2] = {
      MakeAxis(gridX.data(), 2, AxisScale::Linear),
      MakeAxis(gridY.data(), 2, AxisScale::Linear)};

  InterpConfig cfg;
  cfg.outOfRange = OutOfRangePolicy::FillNaN;

  const double value = LogInterpolateSingleVariable2DCustomPoint(
      0.5, 2.0,
      gridX.data(), 2,
      gridY.data(), 2,
      table.data(), 0.0, cfg);

  CHECK(std::isnan(value));
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

  std::array<double, 3> x0{1.0, 1.5, 2.0};
  std::array<double, 3> x1{1.0, 2.0, 3.0};
  std::array<double, 3> out{};

  const int rc = LogInterpolateSingleVariable2DCustom(
      x0.data(), x1.data(), x0.size(),
      gridX.data(), 2,
      gridY.data(), 2,
      table.data(),
      0.0,
      out.data());
  REQUIRE(rc == 0);

  for (std::size_t i = 0; i < x0.size(); ++i) {
    const double point = LogInterpolateSingleVariable2DCustomPoint(
        x0[i], x1[i],
        gridX.data(), 2,
        gridY.data(), 2,
        table.data(),
        0.0);
    CHECK(out[i] == Catch::Approx(point).margin(kTol));
  }
}

TEST_CASE("Upper-endpoint queries are treated as in-range", "[interp][boundary]")
{
  using namespace WeakLibReader;

  const std::array<double, 2> grid{1.0, 2.0};
  const std::array<double, 2> values{10.0, 20.0};

  const int extents[1] = {2};
  const Layout layout = MakeLayout(extents, 1);

  Axis axes[5] = {};
  axes[0] = MakeAxis(grid.data(), 2, AxisScale::Linear);

  InterpConfig cfg;
  cfg.outOfRange = OutOfRangePolicy::FillNaN;

  double coords[5] = {2.0, 0.0, 0.0, 0.0, 0.0};
  const double result = InterpLinearND(values.data(), layout, axes, coords, cfg, 1);

  CHECK(result == Catch::Approx(values[1]).margin(kTol));
}

TEST_CASE("Aligned 2D plane interpolation mirrors underlying kernel", "[loginterp][2d2d]")
{
  using namespace WeakLibReader;

  constexpr std::size_t sizeE = 2;
  const std::array<double, 2> gridT{1.0, 2.0};
  const std::array<double, 2> gridX{1.0, 3.0};

  std::array<double, sizeE * sizeE * 2 * 2> table{};
  std::size_t idx = 0;
  for (std::size_t e0 = 0; e0 < sizeE; ++e0) {
    for (std::size_t e1 = 0; e1 < sizeE; ++e1) {
      for (int t = 0; t < 2; ++t) {
        for (int x = 0; x < 2; ++x) {
          const double actual = 1.0 + 0.05 * static_cast<double>(e0) +
                                0.07 * static_cast<double>(e1) +
                                0.3 * static_cast<double>(t) +
                                0.4 * static_cast<double>(x);
          table[idx++] = std::log10(actual);
        }
      }
    }
  }

  const int extents[4] = {static_cast<int>(sizeE), static_cast<int>(sizeE), 2, 2};
  const Layout layout = MakeLayout(extents, 4);

  Axis axes[2] = {
      MakeAxis(gridT.data(), 2, AxisScale::Linear),
      MakeAxis(gridX.data(), 2, AxisScale::Linear)};

  std::array<double, sizeE * sizeE> plane{};
  const double logT = 1.5;
  const double logX = 2.0;

  const int rc = LogInterpolateSingleVariable2D2DCustomAlignedPoint(
      sizeE, logT, logX,
      gridT.data(), 2,
      gridX.data(), 2,
      table.data(), 0.0, plane.data());
  REQUIRE(rc == 0);

  int idxT = 0;
  double fracT = 0.0;
  REQUIRE_FALSE(IndexAndDeltaLin(logT, gridT.data(), 2, idxT, fracT));
  int idxX = 0;
  double fracX = 0.0;
  REQUIRE_FALSE(IndexAndDeltaLin(logX, gridX.data(), 2, idxX, fracX));

  for (std::size_t j = 0; j < sizeE; ++j) {
    for (std::size_t i = 0; i <= j; ++i) {
      const double expected = LinearInterp2D4DArray2DAlignedPoint(
          static_cast<int>(i), static_cast<int>(j),
          idxT, idxX, fracT, fracX, 0.0,
          table.data(), layout);
      const std::size_t lower = j * sizeE + i;
      const std::size_t upper = i * sizeE + j;
      CHECK(plane[lower] == Catch::Approx(expected).margin(kTol));
      CHECK(plane[upper] == Catch::Approx(expected).margin(kTol));
    }
  }
}

TEST_CASE("Weighted sum aligned helper reproduces manual accumulation", "[loginterp][2d2d][weighted]")
{
  using namespace WeakLibReader;

  constexpr std::size_t sizeE = 2;
  constexpr std::size_t nAlpha = 2;
  constexpr std::size_t count = 1;

  const std::array<double, 2> gridD{1.0, 3.0};
  const std::array<double, 2> gridT{1.0, 2.0};

  std::array<double, sizeE * sizeE * 2 * 2> table{};
  std::size_t idx = 0;
  for (std::size_t e0 = 0; e0 < sizeE; ++e0) {
    for (std::size_t e1 = 0; e1 < sizeE; ++e1) {
      for (int d = 0; d < 2; ++d) {
        for (int t = 0; t < 2; ++t) {
          const double actual = 1.0 + 0.05 * static_cast<double>(e0) +
                                0.07 * static_cast<double>(e1) +
                                0.2 * static_cast<double>(d) +
                                0.3 * static_cast<double>(t);
          table[idx++] = std::log10(actual);
        }
      }
    }
  }

  const int extents[4] = {static_cast<int>(sizeE), static_cast<int>(sizeE), 2, 2};
  const Layout layout = MakeLayout(extents, 4);

  Axis axes[2] = {
      MakeAxis(gridD.data(), 2, AxisScale::Linear),
      MakeAxis(gridT.data(), 2, AxisScale::Linear)};

  const std::array<double, nAlpha> alpha{0.6, 1.1};
  const std::array<double, nAlpha * count> logD{
      1.5,
      2.4};
  const std::array<double, count> logT{1.5};

  std::array<double, sizeE * sizeE * count> out{};

  const int rc = SumLogInterpolateSingleVariable2D2DCustomAligned(
      sizeE,
      logD.data(), nAlpha,
      logT.data(), count,
      gridD.data(), 2,
      gridT.data(), 2,
      alpha.data(),
      table.data(),
      0.0,
      out.data());
  REQUIRE(rc == 0);

  int idxT = 0;
  double fracT = 0.0;
  REQUIRE_FALSE(IndexAndDeltaLin(logT[0], gridT.data(), 2, idxT, fracT));

  for (std::size_t j = 0; j < sizeE; ++j) {
    for (std::size_t i = 0; i <= j; ++i) {
      double expected = 0.0;
      for (std::size_t l = 0; l < nAlpha; ++l) {
        int idxD = 0;
        double fracD = 0.0;
        REQUIRE_FALSE(IndexAndDeltaLin(logD[l], gridD.data(), 2, idxD, fracD));
        const double interp = LinearInterp2D4DArray2DAlignedPoint(
            static_cast<int>(i), static_cast<int>(j),
            idxD, idxT, fracD, fracT,
            0.0,
            table.data(), layout);
        expected += alpha[l] * interp;
      }
      const std::size_t lower = j * sizeE + i;
      const std::size_t upper = i * sizeE + j;
      CHECK(out[lower] == Catch::Approx(expected).margin(kTol));
      CHECK(out[upper] == Catch::Approx(expected).margin(kTol));
    }
  }
}

TEST_CASE("1D3D sweep batch matches direct interpolation", "[loginterp][4d][batch]")
{
  using namespace WeakLibReader;

  constexpr std::size_t sizeE = 2;
  constexpr std::size_t count = 2;

  const std::array<double, 2> gridE{1.0, 2.0};
  const std::array<double, 2> gridD{1.0, 3.0};
  const std::array<double, 2> gridT{10.0, 20.0};
  const std::array<double, 2> gridY{0.0, 1.0};

  const int extents[4] = {2, 2, 2, 2};
  const Layout layout = MakeLayout(extents, 4);

  std::array<double, 16> table{};
  auto actual = [](double e, double d, double t, double y) {
    return 1.0 + 0.2 * e + 0.3 * d + 0.4 * t + 0.5 * y;
  };

  for (int ie = 0; ie < 2; ++ie) {
    for (int id = 0; id < 2; ++id) {
      for (int it = 0; it < 2; ++it) {
        for (int iy = 0; iy < 2; ++iy) {
          table[layout.Offset(ie, id, it, iy)] =
              std::log10(actual(gridE[ie], gridD[id], gridT[it], gridY[iy]));
        }
      }
    }
  }

  std::array<double, sizeE> logE{1.2, 1.8};
  std::array<double, count> logD{1.1, 2.5};
  std::array<double, count> logT{12.0, 18.0};
  std::array<double, count> y{0.25, 0.75};
  std::array<double, sizeE * count> out{};

  const int rc = LogInterpolateSingleVariable1D3DCustom(
      logE.data(), sizeE,
      logD.data(), logT.data(), y.data(), count,
      gridE.data(), 2,
      gridD.data(), 2,
      gridT.data(), 2,
      gridY.data(), 2,
      table.data(),
      0.0,
      out.data());
  REQUIRE(rc == 0);

  Axis axes[4] = {
      MakeAxis(gridE.data(), 2, AxisScale::Linear),
      MakeAxis(gridD.data(), 2, AxisScale::Linear),
      MakeAxis(gridT.data(), 2, AxisScale::Linear),
      MakeAxis(gridY.data(), 2, AxisScale::Linear)};

  for (std::size_t j = 0; j < count; ++j) {
    for (std::size_t i = 0; i < sizeE; ++i) {
      double coords[4] = {logE[i], logD[j], logT[j], y[j]};
      const double expected = detail::LogInterpolatedValueDirect<4>(
          table.data(), layout, axes, coords, 0.0, InterpConfig{});
      const std::size_t idx = j * sizeE + i;
      CHECK(out[idx] == Catch::Approx(expected).margin(kTol));
    }
  }
}

TEST_CASE("Invalid log inputs return errors/NaN as expected", "[loginterp][invalid]")
{
  using namespace WeakLibReader;

  const std::array<double, 2> gridD{1.0, 10.0};
  const std::array<double, 2> gridT{1.0, 100.0};
  const std::array<double, 2> gridY{0.0, 1.0};

  const int extents[3] = {2, 2, 2};
  const Layout layout = MakeLayout(extents, 3);

  std::array<double, 8> table{};
  for (int id = 0; id < 2; ++id) {
    for (int it = 0; it < 2; ++it) {
      for (int iy = 0; iy < 2; ++iy) {
        table[layout.Offset(id, it, iy)] = std::log10(1.0 + 0.1 * id + 0.2 * it + 0.3 * iy);
      }
    }
  }

  SECTION("Negative log coordinate triggers error code")
  {
    double interpolant = 0.0;
    double deriv[3] = {0.0, 0.0, 0.0};
    const int rc = LogInterpolateDifferentiateSingleVariable3DCustomPoint(
        -1.0, 2.0, 0.3,   // invalid log axis coord
        gridD.data(), 2,
        gridT.data(), 2,
        gridY.data(), 2,
        table.data(),
        0.0,
        interpolant, deriv);
    CHECK(rc == 4);
    CHECK(std::isnan(interpolant));
    CHECK(std::isnan(deriv[0]));
    CHECK(std::isnan(deriv[1]));
    CHECK(std::isnan(deriv[2]));
  }

  SECTION("Zero-span linear axis yields NaN under FillNaN policy")
  {
    const std::array<double, 2> badGridY{0.5, 0.5}; // degenerate span

    double interpolant = 0.0;
    double deriv[3] = {0.0, 0.0, 0.0};
    InterpConfig cfg;
    cfg.outOfRange = OutOfRangePolicy::FillNaN;
    const int rc = LogInterpolateDifferentiateSingleVariable3DCustomPoint(
        2.0, 2.0, 0.4,
        gridD.data(), 2,
        gridT.data(), 2,
        badGridY.data(), 2,
        table.data(),
        0.0,
        interpolant, deriv, cfg);
    CHECK(rc == 0);
    CHECK(std::isnan(interpolant));
    CHECK(std::isnan(deriv[0]));
    CHECK(std::isnan(deriv[1]));
    CHECK(std::isnan(deriv[2]));
  }
}

TEST_CASE("Log derivative wrapper matches direct kernel for 3D tables", "[loginterp][derivative][3d]")
{
  using namespace WeakLibReader;

  const std::array<double, 2> gridD{1.0, 10.0};
  const std::array<double, 2> gridT{1.0, 100.0};
  const std::array<double, 2> gridY{0.0, 1.0};

  const int extents[3] = {2, 2, 2};
  const Layout layout = MakeLayout(extents, 3);

  std::array<double, 8> table{};
  auto actual = [](double d, double t, double y) {
    return 1.0 + 0.5 * d + 0.25 * t + 0.1 * y;
  };

  for (int id = 0; id < 2; ++id) {
    const double dVal = gridD[id];
    for (int it = 0; it < 2; ++it) {
      const double tVal = gridT[it];
      for (int iy = 0; iy < 2; ++iy) {
        const double yVal = gridY[iy];
        table[layout.Offset(id, it, iy)] = std::log10(actual(dVal, tVal, yVal));
      }
    }
  }

  Axis axes[3] = {
      MakeAxis(gridD.data(), 2, AxisScale::Log10),
      MakeAxis(gridT.data(), 2, AxisScale::Log10),
      MakeAxis(gridY.data(), 2, AxisScale::Linear)};

  const double dCoord = 3.0;
  const double tCoord = 6.0;
  const double yCoord = 0.4;

  int idxD = 0;
  double fracD = 0.0;
  REQUIRE_FALSE(IndexAndDeltaLog10(dCoord, gridD.data(), 2, idxD, fracD));
  int idxT = 0;
  double fracT = 0.0;
  REQUIRE_FALSE(IndexAndDeltaLog10(tCoord, gridT.data(), 2, idxT, fracT));
  int idxY = 0;
  double fracY = 0.0;
  REQUIRE_FALSE(IndexAndDeltaLin(yCoord, gridY.data(), 2, idxY, fracY));

  const double spanLogD = std::log10(gridD[1] / gridD[0]);
  const double spanLogT = std::log10(gridT[1] / gridT[0]);
  const double aD = 1.0 / (dCoord * spanLogD);
  const double aT = 1.0 / (tCoord * spanLogT);
  const double aY = WeakLibReader::math::Ln10 / (gridY[1] - gridY[0]);

  double expectedInterp = 0.0;
  double expectedDD = 0.0;
  double expectedDT = 0.0;
  double expectedDY = 0.0;
  LinearInterpDeriv3DPoint(idxD, idxT, idxY,
                           fracD, fracT, fracY,
                           aD, aT, aY,
                           0.0, table.data(), layout,
                           expectedInterp, expectedDD, expectedDT, expectedDY);

  double interpolated = 0.0;
  double deriv[3] = {0.0, 0.0, 0.0};
  const int rc = LogInterpolateDifferentiateSingleVariable3DCustomPoint(
      dCoord, tCoord, yCoord,
      gridD.data(), 2,
      gridT.data(), 2,
      gridY.data(), 2,
      table.data(),
      0.0, interpolated, deriv);
  REQUIRE(rc == 0);

  CHECK(interpolated == Catch::Approx(expectedInterp).margin(kTol));
  CHECK(deriv[0] == Catch::Approx(expectedDD).margin(kTol));
  CHECK(deriv[1] == Catch::Approx(expectedDT).margin(kTol));
  CHECK(deriv[2] == Catch::Approx(expectedDY).margin(kTol));

  InterpConfig cfg;
  cfg.outOfRange = OutOfRangePolicy::FillNaN;
  double nanInterp = 0.0;
  double nanDeriv[3] = {0.0, 0.0, 0.0};
  const int nanRc = LogInterpolateDifferentiateSingleVariable3DCustomPoint(
      0.1, tCoord, yCoord,
      gridD.data(), 2,
      gridT.data(), 2,
      gridY.data(), 2,
      table.data(),
      0.0, nanInterp, nanDeriv, cfg);
  REQUIRE(nanRc == 0);
  CHECK(std::isnan(nanInterp));
  CHECK(std::isnan(nanDeriv[0]));
  CHECK(std::isnan(nanDeriv[1]));
  CHECK(std::isnan(nanDeriv[2]));
}

TEST_CASE("Aligned derivative wrapper mirrors kernel output", "[loginterp][derivative][2d2d]")
{
  using namespace WeakLibReader;

  constexpr std::size_t sizeE = 2;
  const std::array<double, 2> gridT{1.0, 2.0};
  const std::array<double, 2> gridX{1.0, 3.0};

  const int extents[4] = {static_cast<int>(sizeE), static_cast<int>(sizeE), 2, 2};
  const Layout layout = MakeLayout(extents, 4);

  std::array<double, sizeE * sizeE * 2 * 2> table{};
  auto value = [](int i, int j, double t, double x) {
    return 2.0 + 0.1 * static_cast<double>(i) +
           0.2 * static_cast<double>(j) +
           0.3 * t +
           0.4 * x;
  };

  for (std::size_t j = 0; j < sizeE; ++j) {
    for (std::size_t i = 0; i < sizeE; ++i) {
      for (int it = 0; it < 2; ++it) {
        for (int ix = 0; ix < 2; ++ix) {
          table[layout.Offset(static_cast<int>(i), static_cast<int>(j), it, ix)] =
              std::log10(value(static_cast<int>(i), static_cast<int>(j), gridT[it], gridX[ix]));
        }
      }
    }
  }

  Axis axes[2] = {
      MakeAxis(gridT.data(), 2, AxisScale::Linear),
      MakeAxis(gridX.data(), 2, AxisScale::Linear)};

  const double logTCoord = 1.4;
  const double logXCoord = 2.4;

  int idxT = 0;
  double fracT = 0.0;
  REQUIRE_FALSE(IndexAndDeltaLin(logTCoord, gridT.data(), 2, idxT, fracT));
  int idxX = 0;
  double fracX = 0.0;
  REQUIRE_FALSE(IndexAndDeltaLin(logXCoord, gridX.data(), 2, idxX, fracX));

  const double spanT = gridT[idxT + 1] - gridT[idxT];
  const double spanX = gridX[idxX + 1] - gridX[idxX];
  const double aT = 1.0 / (spanT * WeakLibReader::math::Pow10(logTCoord));
  const double aX = 1.0 / (spanX * WeakLibReader::math::Pow10(logXCoord));

  std::array<double, sizeE * sizeE> planeInterp{};
  std::array<double, sizeE * sizeE> planeDerivT{};
  std::array<double, sizeE * sizeE> planeDerivX{};

  const int rc = LogInterpolateDifferentiateSingleVariable2D2DCustomAlignedPoint(
      sizeE, logTCoord, logXCoord,
      gridT.data(), 2,
      gridX.data(), 2,
      table.data(),
      0.0,
      planeInterp.data(),
      planeDerivT.data(),
      planeDerivX.data(),
      InterpConfig{});
  REQUIRE(rc == 0);

  for (std::size_t j = 0; j < sizeE; ++j) {
    for (std::size_t i = 0; i <= j; ++i) {
      double interpExpected = 0.0;
      double dTExpected = 0.0;
      double dXExpected = 0.0;
      LinearInterpDeriv2D4DArray2DAlignedPoint(
          static_cast<int>(i), static_cast<int>(j),
          idxT, idxX,
          fracT, fracX,
          aT, aX,
          0.0,
          table.data(), layout,
          interpExpected, dTExpected, dXExpected);

      const std::size_t lower = j * sizeE + i;
      const std::size_t upper = i * sizeE + j;
      CHECK(planeInterp[lower] == Catch::Approx(interpExpected).margin(kTol));
      CHECK(planeDerivT[lower] == Catch::Approx(dTExpected).margin(kTol));
      CHECK(planeDerivX[lower] == Catch::Approx(dXExpected).margin(kTol));
      CHECK(planeInterp[upper] == Catch::Approx(interpExpected).margin(kTol));
      CHECK(planeDerivT[upper] == Catch::Approx(dTExpected).margin(kTol));
      CHECK(planeDerivX[upper] == Catch::Approx(dXExpected).margin(kTol));
    }
  }
}

TEST_CASE("4D mixed axes log interpolation respects offset", "[loginterp][4d][offset]")
{
  using namespace WeakLibReader;

  const std::array<double, 2> gridA{1.0, 10.0};   // log10 axis
  const std::array<double, 2> gridB{2.0, 20.0};   // log10 axis
  const std::array<double, 2> gridC{0.0, 1.0};    // linear axis
  const std::array<double, 2> gridD{5.0, 9.0};    // linear axis

  const int extents[4] = {2, 2, 2, 2};
  const Layout layout = MakeLayout(extents, 4);

  const double offset = 1.5;
  std::array<double, 16> table{};
  for (int ia = 0; ia < 2; ++ia) {
    for (int ib = 0; ib < 2; ++ib) {
      for (int ic = 0; ic < 2; ++ic) {
        for (int id = 0; id < 2; ++id) {
          const double actual = 2.0 + 0.1 * ia + 0.2 * ib + 0.3 * ic + 0.4 * id;
          table[layout.Offset(ia, ib, ic, id)] = std::log10(actual + offset);
        }
      }
    }
  }

  Axis axes[4] = {
      MakeAxis(gridA.data(), 2, AxisScale::Log10),
      MakeAxis(gridB.data(), 2, AxisScale::Log10),
      MakeAxis(gridC.data(), 2, AxisScale::Linear),
      MakeAxis(gridD.data(), 2, AxisScale::Linear)};

  double coords[4] = {3.0, 5.0, 0.25, 6.0};

  int idxA = 0, idxB = 0, idxC = 0, idxD = 0;
  double fracA = 0.0, fracB = 0.0, fracC = 0.0, fracD = 0.0;
  REQUIRE_FALSE(IndexAndDeltaLog10(coords[0], gridA.data(), 2, idxA, fracA));
  REQUIRE_FALSE(IndexAndDeltaLog10(coords[1], gridB.data(), 2, idxB, fracB));
  REQUIRE_FALSE(IndexAndDeltaLin(coords[2], gridC.data(), 2, idxC, fracC));
  REQUIRE_FALSE(IndexAndDeltaLin(coords[3], gridD.data(), 2, idxD, fracD));

  const double expected = LinearInterp4DPoint(
      idxA, idxB, idxC, idxD,
      fracA, fracB, fracC, fracD,
      offset, table.data(), layout);

  const double result = detail::LogInterpolatedValueDirect<4>(
      table.data(), layout, axes, coords, offset, InterpConfig{});

  CHECK(result == Catch::Approx(expected).margin(kTol));
}

TEST_CASE("4D mixed axes derivative matches kernel", "[loginterp][4d][derivative]")
{
  using namespace WeakLibReader;

  const std::array<double, 2> gridA{1.0, 10.0};   // log10 axis
  const std::array<double, 2> gridB{2.0, 20.0};   // log10 axis
  const std::array<double, 2> gridC{0.0, 1.0};    // linear axis
  const std::array<double, 2> gridD{5.0, 9.0};    // linear axis

  const int extents[4] = {2, 2, 2, 2};
  const Layout layout = MakeLayout(extents, 4);

  const double offset = 0.75;
  std::array<double, 16> table{};
  for (int ia = 0; ia < 2; ++ia) {
    for (int ib = 0; ib < 2; ++ib) {
      for (int ic = 0; ic < 2; ++ic) {
        for (int id = 0; id < 2; ++id) {
          const double actual = 1.5 + 0.15 * ia + 0.25 * ib + 0.35 * ic + 0.45 * id;
          table[layout.Offset(ia, ib, ic, id)] = std::log10(actual + offset);
        }
      }
    }
  }

  Axis axes[4] = {
      MakeAxis(gridA.data(), 2, AxisScale::Log10),
      MakeAxis(gridB.data(), 2, AxisScale::Log10),
      MakeAxis(gridC.data(), 2, AxisScale::Linear),
      MakeAxis(gridD.data(), 2, AxisScale::Linear)};

  double coords[4] = {4.0, 6.0, 0.4, 7.0};

  int idxA = 0, idxB = 0, idxC = 0, idxD = 0;
  double fracA = 0.0, fracB = 0.0, fracC = 0.0, fracD = 0.0;
  REQUIRE_FALSE(IndexAndDeltaLog10(coords[0], gridA.data(), 2, idxA, fracA));
  REQUIRE_FALSE(IndexAndDeltaLog10(coords[1], gridB.data(), 2, idxB, fracB));
  REQUIRE_FALSE(IndexAndDeltaLin(coords[2], gridC.data(), 2, idxC, fracC));
  REQUIRE_FALSE(IndexAndDeltaLin(coords[3], gridD.data(), 2, idxD, fracD));

  const double aA = 1.0 / (coords[0] * WeakLibReader::math::Log10(gridA[1] / gridA[0]));
  const double aB = 1.0 / (coords[1] * WeakLibReader::math::Log10(gridB[1] / gridB[0]));
  const double aC = WeakLibReader::math::Ln10 / (gridC[1] - gridC[0]);
  const double aD = WeakLibReader::math::Ln10 / (gridD[1] - gridD[0]);

  double expectedInterp = 0.0;
  double expectedDA = 0.0;
  double expectedDB = 0.0;
  double expectedDC = 0.0;
  double expectedDD = 0.0;
  LinearInterpDeriv4DPoint(
      idxA, idxB, idxC, idxD,
      fracA, fracB, fracC, fracD,
      aA, aB, aC, aD,
      offset, table.data(), layout,
      expectedInterp, expectedDA, expectedDB, expectedDC, expectedDD);

  double interpolant = 0.0;
  double deriv[4] = {0.0, 0.0, 0.0, 0.0};
  const bool success = detail::LogInterpolatedDerivativeDirect<4>(
      table.data(), layout, axes, coords, offset, InterpConfig{}, interpolant, deriv);
  REQUIRE(success);

  CHECK(interpolant == Catch::Approx(expectedInterp).margin(kTol));
  CHECK(deriv[0] == Catch::Approx(expectedDA).margin(kTol));
  CHECK(deriv[1] == Catch::Approx(expectedDB).margin(kTol));
  CHECK(deriv[2] == Catch::Approx(expectedDC).margin(kTol));
  CHECK(deriv[3] == Catch::Approx(expectedDD).margin(kTol));
}

TEST_CASE("Out-of-range policies on mixed axes", "[loginterp][policy]")
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
  const bool outLog = IndexAndDeltaLog10(coords[0], gridLog.data(), 2, idxLog, fracLog);
  const bool outLin = IndexAndDeltaLin(coords[1], gridLin.data(), 2, idxLin, fracLin);
  REQUIRE(outLog);
  REQUIRE(outLin);

  // Clamp policy should return clamped interpolation/derivatives
  {
    const double aLog = 1.0 / (coords[0] * WeakLibReader::math::Log10(gridLog[1] / gridLog[0]));
    const double aLin = WeakLibReader::math::Ln10 / (gridLin[1] - gridLin[0]);

    double expectedInterp = 0.0;
    double expectedDLog = 0.0;
    double expectedDLin = 0.0;
    LinearInterpDeriv2DPoint(
        idxLog, idxLin,
        detail::Clamp01(fracLog), detail::Clamp01(fracLin),
        aLog, aLin,
        offset, table.data(), layout,
        expectedInterp, expectedDLog, expectedDLin);

    double interpolant = 0.0;
    double deriv[2] = {0.0, 0.0};
    const bool ok = detail::LogInterpolatedDerivativeDirect<2>(
        table.data(), layout, axes, coords, offset, InterpConfig{}, interpolant, deriv);
    REQUIRE(ok);

    CHECK(interpolant == Catch::Approx(expectedInterp).margin(kTol));
    CHECK(deriv[0] == Catch::Approx(expectedDLog).margin(kTol));
    CHECK(deriv[1] == Catch::Approx(expectedDLin).margin(kTol));
  }

  // FillNaN policy should propagate NaN
  {
    InterpConfig cfg;
    cfg.outOfRange = OutOfRangePolicy::FillNaN;
    double interpolant = 0.0;
    double deriv[2] = {0.0, 0.0};
    const bool ok = detail::LogInterpolatedDerivativeDirect<2>(
        table.data(), layout, axes, coords, offset, cfg, interpolant, deriv);
    REQUIRE(ok);
    CHECK(std::isnan(interpolant));
    CHECK(std::isnan(deriv[0]));
    CHECK(std::isnan(deriv[1]));
  }

  // Error policy should report failure
  {
    InterpConfig cfg;
    cfg.outOfRange = OutOfRangePolicy::Error;
    double interpolant = 0.0;
    double deriv[2] = {0.0, 0.0};
    const bool ok = detail::LogInterpolatedDerivativeDirect<2>(
        table.data(), layout, axes, coords, offset, cfg, interpolant, deriv);
    CHECK_FALSE(ok);
  }
}

TEST_CASE("Template instantiation compiles for all dimensions", "[compile-time]")
{
  using namespace WeakLibReader;

  // This test verifies that our templated functions instantiate correctly
  // for all supported dimensions. The actual function calls don't execute
  // (they're in unreachable code), but the compiler still instantiates the
  // templates, ensuring they compile without errors.

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
