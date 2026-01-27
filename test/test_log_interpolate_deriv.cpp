#define SIMPLE_CATCH_NO_MAIN
#include <catch2/catch_test_macros.hpp>

#include "WeakLibReader_LogInterpolate.hpp"
#include "WeakLibReader_InterpLogTable.hpp"
#include "WeakLibReader_Layout.hpp"
#include "WeakLibReader_AxisTypes.hpp"

#include <array>
#include <cmath>
#include <limits>

namespace {

constexpr double kTol = 1.0e-12;

} // namespace

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
  IndexAndDeltaLog10(dCoord, gridD.data(), 2, idxD, fracD);
  int idxT = 0;
  double fracT = 0.0;
  IndexAndDeltaLog10(tCoord, gridT.data(), 2, idxT, fracT);
  int idxY = 0;
  double fracY = 0.0;
  IndexAndDeltaLin(yCoord, gridY.data(), 2, idxY, fracY);

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
  IndexAndDeltaLin(logTCoord, gridT.data(), 2, idxT, fracT);
  int idxX = 0;
  double fracX = 0.0;
  IndexAndDeltaLin(logXCoord, gridX.data(), 2, idxX, fracX);

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
      planeDerivX.data());
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
