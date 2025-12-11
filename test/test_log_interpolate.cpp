#include <catch2/catch_test_macros.hpp>

#include "LogInterpolate.hpp"
#include "InterpLogTable.hpp"
#include "Layout.hpp"
#include "AxisTypes.hpp"

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

TEST_CASE("Out-of-range error policy returns NaN", "[loginterp][2d][nan][error]")
{
  using namespace WeakLibReader;

  const std::array<double, 2> gridX{1.0, 2.0};
  const std::array<double, 2> gridY{1.0, 3.0};
  const std::array<double, 4> table{
      std::log10(2.0),
      std::log10(3.0),
      std::log10(4.0),
      std::log10(5.0)};

  InterpConfig cfg;
  cfg.outOfRange = OutOfRangePolicy::Error;

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

TEST_CASE("Invalid log coordinate triggers error code", "[loginterp][invalid]")
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

TEST_CASE("Zero-span axis yields NaN under FillNaN policy", "[loginterp][invalid]")
{
  using namespace WeakLibReader;

  const std::array<double, 2> gridD{1.0, 10.0};
  const std::array<double, 2> gridT{1.0, 100.0};
  const std::array<double, 2> badGridY{0.5, 0.5}; // degenerate span

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
  const double dX = (x - gridX[0]) / (gridX[1] - gridX[0]);
  const double dY = (y - gridY[0]) / (gridY[1] - gridY[0]);
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

TEST_CASE("4D log interpolation matches quadrilinear expectation", "[loginterp][4d]")
{
  using namespace WeakLibReader;

  const std::array<double, 2> gridA{1.0, 2.0};
  const std::array<double, 2> gridB{1.0, 3.0};
  const std::array<double, 2> gridC{0.0, 1.0};
  const std::array<double, 2> gridD{5.0, 9.0};

  const int extents[4] = {2, 2, 2, 2};
  const Layout layout = MakeLayout(extents, 4);
  std::array<double, 16> table{};
  auto actual = [](double a, double b, double c, double d) {
    return 1.0 + 0.2 * a + 0.3 * b + 0.4 * c + 0.1 * d;
  };
  for (int ia = 0; ia < 2; ++ia) {
    for (int ib = 0; ib < 2; ++ib) {
      for (int ic = 0; ic < 2; ++ic) {
        for (int id = 0; id < 2; ++id) {
          table[layout.Offset(ia, ib, ic, id)] =
              std::log10(actual(gridA[ia], gridB[ib], gridC[ic], gridD[id]));
        }
      }
    }
  }

  const double a = 1.5;
  const double b = 2.0;
  const double c = 0.4;
  const double d = 7.0;
  const double result = LogInterpolateSingleVariable4DCustomPoint(
      a, b, c, d,
      gridA.data(), 2,
      gridB.data(), 2,
      gridC.data(), 2,
      gridD.data(), 2,
      table.data(), 0.0);

  // Verify by computing via 4D wrapper
  Axis axes[4] = {
      MakeAxis(gridA.data(), 2, AxisScale::Linear),
      MakeAxis(gridB.data(), 2, AxisScale::Linear),
      MakeAxis(gridC.data(), 2, AxisScale::Linear),
      MakeAxis(gridD.data(), 2, AxisScale::Linear)};
  double coords[4] = {a, b, c, d};
  const double expected = detail::LogInterpolatedValueDirect<4>(
      table.data(), layout, axes, coords, 0.0, InterpConfig{});

  CHECK(result == Catch::Approx(expected).margin(kTol));
}

TEST_CASE("Batch 4D log interpolation matches point wrapper", "[loginterp][4d][batch]")
{
  using namespace WeakLibReader;

  const std::array<double, 2> gridA{1.0, 2.0};
  const std::array<double, 2> gridB{1.0, 3.0};
  const std::array<double, 2> gridC{0.0, 1.0};
  const std::array<double, 2> gridD{5.0, 9.0};

  const int extents[4] = {2, 2, 2, 2};
  const Layout layout = MakeLayout(extents, 4);
  std::array<double, 16> table{};
  for (int ia = 0; ia < 2; ++ia) {
    for (int ib = 0; ib < 2; ++ib) {
      for (int ic = 0; ic < 2; ++ic) {
        for (int id = 0; id < 2; ++id) {
          table[layout.Offset(ia, ib, ic, id)] =
              std::log10(1.0 + 0.2 * gridA[ia] + 0.3 * gridB[ib] +
                         0.4 * gridC[ic] + 0.1 * gridD[id]);
        }
      }
    }
  }

  std::array<double, 3> x0{1.0, 1.5, 2.0};
  std::array<double, 3> x1{1.0, 2.0, 3.0};
  std::array<double, 3> x2{0.0, 0.5, 1.0};
  std::array<double, 3> x3{5.0, 7.0, 9.0};
  std::array<double, 3> out{};

  const int rc = LogInterpolateSingleVariable4DCustom(
      x0.data(), x1.data(), x2.data(), x3.data(), x0.size(),
      gridA.data(), 2,
      gridB.data(), 2,
      gridC.data(), 2,
      gridD.data(), 2,
      table.data(),
      0.0,
      out.data());
  REQUIRE(rc == 0);

  for (std::size_t i = 0; i < x0.size(); ++i) {
    const double point = LogInterpolateSingleVariable4DCustomPoint(
        x0[i], x1[i], x2[i], x3[i],
        gridA.data(), 2,
        gridB.data(), 2,
        gridC.data(), 2,
        gridD.data(), 2,
        table.data(),
        0.0);
    CHECK(out[i] == Catch::Approx(point).margin(kTol));
  }
}

TEST_CASE("5D log interpolation via LinearInterp5DPoint", "[loginterp][5d]")
{
  using namespace WeakLibReader;

  const std::array<double, 2> grid0{1.0, 2.0};
  const std::array<double, 2> grid1{1.0, 3.0};
  const std::array<double, 2> grid2{0.0, 1.0};
  const std::array<double, 2> grid3{5.0, 9.0};
  const std::array<double, 2> grid4{10.0, 20.0};

  const int extents[5] = {2, 2, 2, 2, 2};
  const Layout layout = MakeLayout(extents, 5);
  std::array<double, 32> table{};
  auto actual = [](double a, double b, double c, double d, double e) {
    return 1.0 + 0.1 * a + 0.2 * b + 0.3 * c + 0.05 * d + 0.02 * e;
  };
  for (int i0 = 0; i0 < 2; ++i0) {
    for (int i1 = 0; i1 < 2; ++i1) {
      for (int i2 = 0; i2 < 2; ++i2) {
        for (int i3 = 0; i3 < 2; ++i3) {
          for (int i4 = 0; i4 < 2; ++i4) {
            table[layout.Offset(i0, i1, i2, i3, i4)] =
                std::log10(actual(grid0[i0], grid1[i1], grid2[i2],
                                  grid3[i3], grid4[i4]));
          }
        }
      }
    }
  }

  // Test point
  const double x0 = 1.5;
  const double x1 = 2.0;
  const double x2 = 0.4;
  const double x3 = 7.0;
  const double x4 = 15.0;

  // Compute fractions
  const double d0 = (x0 - grid0[0]) / (grid0[1] - grid0[0]);
  const double d1 = (x1 - grid1[0]) / (grid1[1] - grid1[0]);
  const double d2 = (x2 - grid2[0]) / (grid2[1] - grid2[0]);
  const double d3 = (x3 - grid3[0]) / (grid3[1] - grid3[0]);
  const double d4 = (x4 - grid4[0]) / (grid4[1] - grid4[0]);

  const double result = LinearInterp5DPoint(
      0, 0, 0, 0, 0,
      d0, d1, d2, d3, d4,
      0.0,
      table.data(),
      layout);

  // Compute expected using PentaLinear directly
  const double p00000 = table[layout.Offset(0, 0, 0, 0, 0)];
  const double p10000 = table[layout.Offset(1, 0, 0, 0, 0)];
  const double p01000 = table[layout.Offset(0, 1, 0, 0, 0)];
  const double p11000 = table[layout.Offset(1, 1, 0, 0, 0)];
  const double p00100 = table[layout.Offset(0, 0, 1, 0, 0)];
  const double p10100 = table[layout.Offset(1, 0, 1, 0, 0)];
  const double p01100 = table[layout.Offset(0, 1, 1, 0, 0)];
  const double p11100 = table[layout.Offset(1, 1, 1, 0, 0)];
  const double p00010 = table[layout.Offset(0, 0, 0, 1, 0)];
  const double p10010 = table[layout.Offset(1, 0, 0, 1, 0)];
  const double p01010 = table[layout.Offset(0, 1, 0, 1, 0)];
  const double p11010 = table[layout.Offset(1, 1, 0, 1, 0)];
  const double p00110 = table[layout.Offset(0, 0, 1, 1, 0)];
  const double p10110 = table[layout.Offset(1, 0, 1, 1, 0)];
  const double p01110 = table[layout.Offset(0, 1, 1, 1, 0)];
  const double p11110 = table[layout.Offset(1, 1, 1, 1, 0)];
  const double p00001 = table[layout.Offset(0, 0, 0, 0, 1)];
  const double p10001 = table[layout.Offset(1, 0, 0, 0, 1)];
  const double p01001 = table[layout.Offset(0, 1, 0, 0, 1)];
  const double p11001 = table[layout.Offset(1, 1, 0, 0, 1)];
  const double p00101 = table[layout.Offset(0, 0, 1, 0, 1)];
  const double p10101 = table[layout.Offset(1, 0, 1, 0, 1)];
  const double p01101 = table[layout.Offset(0, 1, 1, 0, 1)];
  const double p11101 = table[layout.Offset(1, 1, 1, 0, 1)];
  const double p00011 = table[layout.Offset(0, 0, 0, 1, 1)];
  const double p10011 = table[layout.Offset(1, 0, 0, 1, 1)];
  const double p01011 = table[layout.Offset(0, 1, 0, 1, 1)];
  const double p11011 = table[layout.Offset(1, 1, 0, 1, 1)];
  const double p00111 = table[layout.Offset(0, 0, 1, 1, 1)];
  const double p10111 = table[layout.Offset(1, 0, 1, 1, 1)];
  const double p01111 = table[layout.Offset(0, 1, 1, 1, 1)];
  const double p11111 = table[layout.Offset(1, 1, 1, 1, 1)];

  const double logExpected = PentaLinear(
      p00000, p10000, p01000, p11000,
      p00100, p10100, p01100, p11100,
      p00010, p10010, p01010, p11010,
      p00110, p10110, p01110, p11110,
      p00001, p10001, p01001, p11001,
      p00101, p10101, p01101, p11101,
      p00011, p10011, p01011, p11011,
      p00111, p10111, p01111, p11111,
      d0, d1, d2, d3, d4);
  const double expected = std::pow(10.0, logExpected);

  CHECK(result == Catch::Approx(expected).margin(kTol));
}

TEST_CASE("PentaLinear reduces to TetraLinear at boundaries", "[loginterp][5d][pentalinear]")
{
  using namespace WeakLibReader;

  // When dX5 = 0, PentaLinear should equal the "lo" TetraLinear
  const double p00000 = 0.1, p10000 = 0.2, p01000 = 0.15, p11000 = 0.25;
  const double p00100 = 0.12, p10100 = 0.22, p01100 = 0.17, p11100 = 0.27;
  const double p00010 = 0.11, p10010 = 0.21, p01010 = 0.16, p11010 = 0.26;
  const double p00110 = 0.13, p10110 = 0.23, p01110 = 0.18, p11110 = 0.28;
  const double p00001 = 0.3, p10001 = 0.4, p01001 = 0.35, p11001 = 0.45;
  const double p00101 = 0.32, p10101 = 0.42, p01101 = 0.37, p11101 = 0.47;
  const double p00011 = 0.31, p10011 = 0.41, p01011 = 0.36, p11011 = 0.46;
  const double p00111 = 0.33, p10111 = 0.43, p01111 = 0.38, p11111 = 0.48;

  const double dX1 = 0.3, dX2 = 0.4, dX3 = 0.5, dX4 = 0.6;

  // dX5 = 0 should give the "lo" tetralinear result
  const double penta0 = PentaLinear(
      p00000, p10000, p01000, p11000,
      p00100, p10100, p01100, p11100,
      p00010, p10010, p01010, p11010,
      p00110, p10110, p01110, p11110,
      p00001, p10001, p01001, p11001,
      p00101, p10101, p01101, p11101,
      p00011, p10011, p01011, p11011,
      p00111, p10111, p01111, p11111,
      dX1, dX2, dX3, dX4, 0.0);

  const double tetraLo = TetraLinear(
      p00000, p10000, p01000, p11000,
      p00100, p10100, p01100, p11100,
      p00010, p10010, p01010, p11010,
      p00110, p10110, p01110, p11110,
      dX1, dX2, dX3, dX4);

  CHECK(penta0 == Catch::Approx(tetraLo).margin(kTol));

  // dX5 = 1 should give the "hi" tetralinear result
  const double penta1 = PentaLinear(
      p00000, p10000, p01000, p11000,
      p00100, p10100, p01100, p11100,
      p00010, p10010, p01010, p11010,
      p00110, p10110, p01110, p11110,
      p00001, p10001, p01001, p11001,
      p00101, p10101, p01101, p11101,
      p00011, p10011, p01011, p11011,
      p00111, p10111, p01111, p11111,
      dX1, dX2, dX3, dX4, 1.0);

  const double tetraHi = TetraLinear(
      p00001, p10001, p01001, p11001,
      p00101, p10101, p01101, p11101,
      p00011, p10011, p01011, p11011,
      p00111, p10111, p01111, p11111,
      dX1, dX2, dX3, dX4);

  CHECK(penta1 == Catch::Approx(tetraHi).margin(kTol));
}

TEST_CASE("Non-aligned 2D2D single point interpolation", "[loginterp][2d2d][nonaligned]")
{
  using namespace WeakLibReader;

  constexpr std::size_t sizeE = 2;
  const std::array<double, sizeE> gridE{1.0, 2.0};
  const std::array<double, 2> gridT{1.0, 2.0};
  const std::array<double, 2> gridX{1.0, 3.0};

  const int extents[4] = {static_cast<int>(sizeE), static_cast<int>(sizeE), 2, 2};
  const Layout layout = MakeLayout(extents, 4);
  std::array<double, sizeE * sizeE * 2 * 2> table{};
  for (std::size_t e0 = 0; e0 < sizeE; ++e0) {
    for (std::size_t e1 = 0; e1 < sizeE; ++e1) {
      for (int t = 0; t < 2; ++t) {
        for (int x = 0; x < 2; ++x) {
          const double actual = 1.0 + 0.1 * gridE[e0] + 0.2 * gridE[e1] +
                                0.3 * gridT[t] + 0.4 * gridX[x];
          table[layout.Offset(static_cast<int>(e0), static_cast<int>(e1), t, x)] =
              std::log10(actual);
        }
      }
    }
  }

  const double logT = 1.4;
  const double logX = 2.0;
  std::array<double, sizeE * sizeE> out{};

  const int rc = LogInterpolateSingleVariable2D2DCustomPoint(
      gridE.data(), sizeE, logT, logX,
      gridE.data(), static_cast<int>(sizeE),
      gridT.data(), 2,
      gridX.data(), 2,
      table.data(), 0.0, out.data());
  REQUIRE(rc == 0);

  // Verify output against direct 4D interpolation
  // The function uses symmetric storage: out[i,j] = out[j,i] = interp(E[i], E[j], T, X)
  Axis axes[4] = {
      MakeAxis(gridE.data(), static_cast<int>(sizeE), AxisScale::Linear),
      MakeAxis(gridE.data(), static_cast<int>(sizeE), AxisScale::Linear),
      MakeAxis(gridT.data(), 2, AxisScale::Linear),
      MakeAxis(gridX.data(), 2, AxisScale::Linear)};

  for (std::size_t j = 0; j < sizeE; ++j) {
    for (std::size_t i = 0; i <= j; ++i) {
      // For i <= j, coords are (E[i], E[j], T, X)
      double coords[4] = {gridE[i], gridE[j], logT, logX};
      const double expected = detail::LogInterpolatedValueDirect<4>(
          table.data(), layout, axes, coords, 0.0, InterpConfig{});
      // Check both symmetric positions
      const std::size_t lower = j * sizeE + i;
      const std::size_t upper = i * sizeE + j;
      CHECK(out[lower] == Catch::Approx(expected).margin(kTol));
      CHECK(out[upper] == Catch::Approx(expected).margin(kTol));
    }
  }
}

TEST_CASE("Non-aligned 2D2D batch interpolation", "[loginterp][2d2d][nonaligned][batch]")
{
  using namespace WeakLibReader;

  constexpr std::size_t sizeE = 2;
  constexpr std::size_t count = 2;
  const std::array<double, sizeE> gridE{1.0, 2.0};
  const std::array<double, 2> gridT{1.0, 2.0};
  const std::array<double, 2> gridX{1.0, 3.0};

  const int extents[4] = {static_cast<int>(sizeE), static_cast<int>(sizeE), 2, 2};
  const Layout layout = MakeLayout(extents, 4);
  std::array<double, sizeE * sizeE * 2 * 2> table{};
  for (std::size_t e0 = 0; e0 < sizeE; ++e0) {
    for (std::size_t e1 = 0; e1 < sizeE; ++e1) {
      for (int t = 0; t < 2; ++t) {
        for (int x = 0; x < 2; ++x) {
          const double actual = 1.0 + 0.1 * gridE[e0] + 0.2 * gridE[e1] +
                                0.3 * gridT[t] + 0.4 * gridX[x];
          table[layout.Offset(static_cast<int>(e0), static_cast<int>(e1), t, x)] =
              std::log10(actual);
        }
      }
    }
  }

  const std::array<double, count> logT{1.2, 1.8};
  const std::array<double, count> logX{1.5, 2.5};
  std::array<double, sizeE * sizeE * count> out{};

  const int rc = LogInterpolateSingleVariable2D2DCustom(
      gridE.data(), sizeE,
      logT.data(), logX.data(), count,
      gridE.data(), static_cast<int>(sizeE),
      gridT.data(), 2,
      gridX.data(), 2,
      table.data(), 0.0, out.data());
  REQUIRE(rc == 0);

  // Verify against single point version
  for (std::size_t l = 0; l < count; ++l) {
    std::array<double, sizeE * sizeE> plane{};
    const int rcPoint = LogInterpolateSingleVariable2D2DCustomPoint(
        gridE.data(), sizeE, logT[l], logX[l],
        gridE.data(), static_cast<int>(sizeE),
        gridT.data(), 2,
        gridX.data(), 2,
        table.data(), 0.0, plane.data());
    REQUIRE(rcPoint == 0);

    for (std::size_t k = 0; k < sizeE * sizeE; ++k) {
      CHECK(out[l * sizeE * sizeE + k] == Catch::Approx(plane[k]).margin(kTol));
    }
  }
}

TEST_CASE("1D3D single point matches batch with count=1", "[loginterp][4d][1d3d]")
{
  using namespace WeakLibReader;

  constexpr std::size_t sizeE = 3;

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

  std::array<double, sizeE> logE{1.0, 1.5, 2.0};
  const double logD = 2.0;
  const double logT = 15.0;
  const double y = 0.5;

  std::array<double, sizeE> outPoint{};
  const int rcPoint = LogInterpolateSingleVariable1D3DCustomPoint(
      logE.data(), sizeE,
      logD, logT, y,
      gridE.data(), 2,
      gridD.data(), 2,
      gridT.data(), 2,
      gridY.data(), 2,
      table.data(),
      0.0,
      outPoint.data());
  REQUIRE(rcPoint == 0);

  // Compare against batch version with count=1
  std::array<double, 1> logDArr{logD};
  std::array<double, 1> logTArr{logT};
  std::array<double, 1> yArr{y};
  std::array<double, sizeE> outBatch{};

  const int rcBatch = LogInterpolateSingleVariable1D3DCustom(
      logE.data(), sizeE,
      logDArr.data(), logTArr.data(), yArr.data(), 1,
      gridE.data(), 2,
      gridD.data(), 2,
      gridT.data(), 2,
      gridY.data(), 2,
      table.data(),
      0.0,
      outBatch.data());
  REQUIRE(rcBatch == 0);

  for (std::size_t i = 0; i < sizeE; ++i) {
    CHECK(outPoint[i] == Catch::Approx(outBatch[i]).margin(kTol));
  }
}

TEST_CASE("Batch aligned 2D2D matches point version", "[loginterp][2d2d][aligned][batch]")
{
  using namespace WeakLibReader;

  constexpr std::size_t sizeE = 2;
  constexpr std::size_t count = 2;

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

  const std::array<double, count> logT{1.3, 1.7};
  const std::array<double, count> logX{1.5, 2.5};
  const std::size_t planeSize = sizeE * sizeE;
  std::array<double, planeSize * count> outBatch{};

  const int rcBatch = LogInterpolateSingleVariable2D2DCustomAligned(
      sizeE,
      logT.data(), logX.data(), count,
      gridT.data(), 2,
      gridX.data(), 2,
      table.data(),
      0.0,
      outBatch.data());
  REQUIRE(rcBatch == 0);

  for (std::size_t k = 0; k < count; ++k) {
    std::array<double, planeSize> outPoint{};
    const int rcPoint = LogInterpolateSingleVariable2D2DCustomAlignedPoint(
        sizeE, logT[k], logX[k],
        gridT.data(), 2,
        gridX.data(), 2,
        table.data(),
        0.0,
        outPoint.data());
    REQUIRE(rcPoint == 0);

    for (std::size_t i = 0; i < planeSize; ++i) {
      CHECK(outBatch[k * planeSize + i] == Catch::Approx(outPoint[i]).margin(kTol));
    }
  }
}

TEST_CASE("Batch 3D derivative matches point version", "[loginterp][3d][derivative][batch]")
{
  using namespace WeakLibReader;

  constexpr std::size_t count = 3;

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
    for (int it = 0; it < 2; ++it) {
      for (int iy = 0; iy < 2; ++iy) {
        table[layout.Offset(id, it, iy)] =
            std::log10(actual(gridD[id], gridT[it], gridY[iy]));
      }
    }
  }

  std::array<double, count> dCoord{2.0, 5.0, 8.0};
  std::array<double, count> tCoord{10.0, 50.0, 80.0};
  std::array<double, count> yCoord{0.2, 0.5, 0.8};

  std::array<double, count> interpBatch{};
  std::array<double, count * 3> derivBatch{};

  const int rcBatch = LogInterpolateDifferentiateSingleVariable3DCustom(
      dCoord.data(), tCoord.data(), yCoord.data(), count,
      gridD.data(), 2,
      gridT.data(), 2,
      gridY.data(), 2,
      table.data(),
      0.0,
      interpBatch.data(),
      derivBatch.data());
  REQUIRE(rcBatch == 0);

  for (std::size_t i = 0; i < count; ++i) {
    double interpPoint = 0.0;
    double derivPoint[3] = {0.0, 0.0, 0.0};

    const int rcPoint = LogInterpolateDifferentiateSingleVariable3DCustomPoint(
        dCoord[i], tCoord[i], yCoord[i],
        gridD.data(), 2,
        gridT.data(), 2,
        gridY.data(), 2,
        table.data(),
        0.0,
        interpPoint, derivPoint);
    REQUIRE(rcPoint == 0);

    CHECK(interpBatch[i] == Catch::Approx(interpPoint).margin(kTol));
    CHECK(derivBatch[i * 3 + 0] == Catch::Approx(derivPoint[0]).margin(kTol));
    CHECK(derivBatch[i * 3 + 1] == Catch::Approx(derivPoint[1]).margin(kTol));
    CHECK(derivBatch[i * 3 + 2] == Catch::Approx(derivPoint[2]).margin(kTol));
  }
}

TEST_CASE("Batch 2D2D derivative matches point version", "[loginterp][2d2d][derivative][batch]")
{
  using namespace WeakLibReader;

  constexpr std::size_t sizeE = 2;
  constexpr std::size_t count = 2;

  const std::array<double, sizeE> gridE{1.0, 2.0};
  const std::array<double, 2> gridT{1.0, 2.0};
  const std::array<double, 2> gridX{1.0, 3.0};

  const int extents[4] = {static_cast<int>(sizeE), static_cast<int>(sizeE), 2, 2};
  const Layout layout = MakeLayout(extents, 4);

  std::array<double, sizeE * sizeE * 2 * 2> table{};
  for (std::size_t e0 = 0; e0 < sizeE; ++e0) {
    for (std::size_t e1 = 0; e1 < sizeE; ++e1) {
      for (int t = 0; t < 2; ++t) {
        for (int x = 0; x < 2; ++x) {
          const double actual = 1.0 + 0.1 * gridE[e0] + 0.2 * gridE[e1] +
                                0.3 * gridT[t] + 0.4 * gridX[x];
          table[layout.Offset(static_cast<int>(e0), static_cast<int>(e1), t, x)] =
              std::log10(actual);
        }
      }
    }
  }

  const std::array<double, count> logT{1.3, 1.7};
  const std::array<double, count> logX{1.5, 2.5};
  const std::size_t planeSize = sizeE * sizeE;

  std::array<double, planeSize * count> interpBatch{};
  std::array<double, planeSize * count> derivTBatch{};
  std::array<double, planeSize * count> derivXBatch{};

  const int rcBatch = LogInterpolateDifferentiateSingleVariable2D2DCustom(
      gridE.data(), sizeE,
      logT.data(), logX.data(), count,
      gridE.data(), static_cast<int>(sizeE),
      gridT.data(), 2,
      gridX.data(), 2,
      table.data(),
      0.0,
      interpBatch.data(),
      derivTBatch.data(),
      derivXBatch.data());
  REQUIRE(rcBatch == 0);

  for (std::size_t k = 0; k < count; ++k) {
    std::array<double, planeSize> interpPoint{};
    std::array<double, planeSize> derivTPoint{};
    std::array<double, planeSize> derivXPoint{};

    const int rcPoint = LogInterpolateDifferentiateSingleVariable2D2DCustomPoint(
        gridE.data(), sizeE,
        logT[k], logX[k],
        gridE.data(), static_cast<int>(sizeE),
        gridT.data(), 2,
        gridX.data(), 2,
        table.data(),
        0.0,
        interpPoint.data(),
        derivTPoint.data(),
        derivXPoint.data());
    REQUIRE(rcPoint == 0);

    for (std::size_t i = 0; i < planeSize; ++i) {
      CHECK(interpBatch[k * planeSize + i] == Catch::Approx(interpPoint[i]).margin(kTol));
      CHECK(derivTBatch[k * planeSize + i] == Catch::Approx(derivTPoint[i]).margin(kTol));
      CHECK(derivXBatch[k * planeSize + i] == Catch::Approx(derivXPoint[i]).margin(kTol));
    }
  }
}

TEST_CASE("Batch aligned 2D2D derivative matches point version", "[loginterp][2d2d][aligned][derivative][batch]")
{
  using namespace WeakLibReader;

  constexpr std::size_t sizeE = 2;
  constexpr std::size_t count = 2;

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

  const std::array<double, count> logT{1.3, 1.7};
  const std::array<double, count> logX{1.5, 2.5};
  const std::size_t planeSize = sizeE * sizeE;

  std::array<double, planeSize * count> interpBatch{};
  std::array<double, planeSize * count> derivTBatch{};
  std::array<double, planeSize * count> derivXBatch{};

  const int rcBatch = LogInterpolateDifferentiateSingleVariable2D2DCustomAligned(
      sizeE,
      logT.data(), logX.data(), count,
      gridT.data(), 2,
      gridX.data(), 2,
      table.data(),
      0.0,
      interpBatch.data(),
      derivTBatch.data(),
      derivXBatch.data());
  REQUIRE(rcBatch == 0);

  for (std::size_t k = 0; k < count; ++k) {
    std::array<double, planeSize> interpPoint{};
    std::array<double, planeSize> derivTPoint{};
    std::array<double, planeSize> derivXPoint{};

    const int rcPoint = LogInterpolateDifferentiateSingleVariable2D2DCustomAlignedPoint(
        sizeE,
        logT[k], logX[k],
        gridT.data(), 2,
        gridX.data(), 2,
        table.data(),
        0.0,
        interpPoint.data(),
        derivTPoint.data(),
        derivXPoint.data());
    REQUIRE(rcPoint == 0);

    for (std::size_t i = 0; i < planeSize; ++i) {
      CHECK(interpBatch[k * planeSize + i] == Catch::Approx(interpPoint[i]).margin(kTol));
      CHECK(derivTBatch[k * planeSize + i] == Catch::Approx(derivTPoint[i]).margin(kTol));
      CHECK(derivXBatch[k * planeSize + i] == Catch::Approx(derivXPoint[i]).margin(kTol));
    }
  }
}
