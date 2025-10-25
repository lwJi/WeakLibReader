#include "WeakLibReader/LogInterpolate.hpp"
#include "WeakLibReader/InterpLogTable.hpp"
#include "WeakLibReader/Layout.hpp"
#include "WeakLibReader/WeakLibReader.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

namespace {

constexpr double kTol = 1.0e-12;

void TestBasic2DInterpolation()
{
  using namespace WeakLibReader;

  const std::array<double, 2> gridX{1.0, 2.0};
  const std::array<double, 2> gridY{1.0, 3.0};

  // Table entries are log10(actual_value) so that Pow10(...) recreates the value.
  const std::array<double, 4> table{
      std::log10(2.0), // f(1,1)
      std::log10(3.0), // f(2,1)
      std::log10(4.0), // f(1,3)
      std::log10(5.0)  // f(2,3)
  };

  const int extents[2] = {2, 2};
  const WeakLibReader::Layout layout = WeakLibReader::MakeLayout(extents, 2);

  WeakLibReader::Axis axes[2] = {
      WeakLibReader::MakeAxis(gridX.data(), 2, WeakLibReader::AxisScale::Linear),
      WeakLibReader::MakeAxis(gridY.data(), 2, WeakLibReader::AxisScale::Linear)};

  const double x = 1.5;
  const double y = 2.0;
  const double result = WeakLibReader::LogInterpolateSingleVariable2DCustomPoint(
      table.data(), layout, axes, x, y, 0.0);

  const double dX = (x - gridX[0]) / (gridX[1] - gridX[0]);
  const double dY = (y - gridY[0]) / (gridY[1] - gridY[0]);
  const double p00 = table[0];
  const double p10 = table[1];
  const double p01 = table[2];
  const double p11 = table[3];
  const double logExpected = (1.0 - dY) * ((1.0 - dX) * p00 + dX * p10) +
                             dY * ((1.0 - dX) * p01 + dX * p11);
  const double expected = std::pow(10.0, logExpected);

  assert(std::abs(result - expected) < kTol);
}

void TestOutOfRangeFillNaN()
{
  using namespace WeakLibReader;

  const std::array<double, 2> gridX{1.0, 2.0};
  const std::array<double, 2> gridY{1.0, 3.0};
  const std::array<double, 4> table{
      std::log10(2.0), std::log10(3.0), std::log10(4.0), std::log10(5.0)};

  const int extents[2] = {2, 2};
  const WeakLibReader::Layout layout = WeakLibReader::MakeLayout(extents, 2);

  WeakLibReader::Axis axes[2] = {
      WeakLibReader::MakeAxis(gridX.data(), 2, WeakLibReader::AxisScale::Linear),
      WeakLibReader::MakeAxis(gridY.data(), 2, WeakLibReader::AxisScale::Linear)};

  WeakLibReader::InterpConfig cfg;
  cfg.outOfRange = WeakLibReader::OutOfRangePolicy::FillNaN;

  const double result = WeakLibReader::LogInterpolateSingleVariable2DCustomPoint(
      table.data(), layout, axes, 0.5, 2.0, 0.0, cfg);

  assert(std::isnan(result));
}

void TestAlignedPlaneInterpolation()
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
          const double actual = 1.0 + 0.1 * static_cast<double>(e0) +
                                0.2 * static_cast<double>(e1) +
                                0.3 * static_cast<double>(t) +
                                0.4 * static_cast<double>(x);
          table[idx++] = std::log10(actual);
        }
      }
    }
  }

  const int extents[4] = {static_cast<int>(sizeE), static_cast<int>(sizeE), 2, 2};
  const WeakLibReader::Layout layout = WeakLibReader::MakeLayout(extents, 4);

  WeakLibReader::Axis axes[2] = {
      WeakLibReader::MakeAxis(gridT.data(), 2, WeakLibReader::AxisScale::Linear),
      WeakLibReader::MakeAxis(gridX.data(), 2, WeakLibReader::AxisScale::Linear)};

  std::array<double, sizeE * sizeE> plane{};
  const double logT = 1.5;
  const double logX = 2.0;

  const int rc = WeakLibReader::LogInterpolateSingleVariable2D2DCustomAlignedPoint(
      sizeE, logT, logX, table.data(), layout, axes, 0.0, plane.data());
  assert(rc == 0);

  const double dT = (logT - gridT[0]) / (gridT[1] - gridT[0]);
  const double dX = (logX - gridX[0]) / (gridX[1] - gridX[0]);
  const int iT = 0;
  const int iX = 0;

  const auto planeValue = [&](int i, int j) {
    return plane[static_cast<std::size_t>(j) * sizeE + static_cast<std::size_t>(i)];
  };

  const double expected00 = WeakLibReader::LinearInterp2D4DArray2DAlignedPoint(
      0, 0, iT, iX, dT, dX, 0.0, table.data(), layout);
  const double expected01 = WeakLibReader::LinearInterp2D4DArray2DAlignedPoint(
      0, 1, iT, iX, dT, dX, 0.0, table.data(), layout);
  const double expected11 = WeakLibReader::LinearInterp2D4DArray2DAlignedPoint(
      1, 1, iT, iX, dT, dX, 0.0, table.data(), layout);

  assert(std::abs(planeValue(0, 0) - expected00) < kTol);
  assert(std::abs(planeValue(1, 1) - expected11) < kTol);
  assert(std::abs(planeValue(0, 1) - expected01) < kTol);
  assert(std::abs(planeValue(1, 0) - expected01) < kTol);
}

void TestWeightedSumAligned()
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
  const WeakLibReader::Layout layout = WeakLibReader::MakeLayout(extents, 4);

  WeakLibReader::Axis axes[2] = {
      WeakLibReader::MakeAxis(gridD.data(), 2, WeakLibReader::AxisScale::Linear),
      WeakLibReader::MakeAxis(gridT.data(), 2, WeakLibReader::AxisScale::Linear)};

  const std::array<double, nAlpha> alpha{0.6, 1.1};
  const std::array<double, nAlpha * count> logD{
      1.5, // between gridD[0] and gridD[1]
      2.4  // also between grid points
  };
  const std::array<double, count> logT{1.5};

  std::array<double, sizeE * sizeE * count> out{};

  const int rc = WeakLibReader::SumLogInterpolateSingleVariable2D2DCustomAligned(
      sizeE,
      logD.data(), nAlpha,
      logT.data(), count,
      table.data(), layout, axes,
      alpha.data(),
      0.0,
      out.data());
  assert(rc == 0);

  const double dT = (logT[0] - gridT[0]) / (gridT[1] - gridT[0]);
  const int iT = 0;

  const auto outValue = [&](int i, int j) {
    return out[static_cast<std::size_t>(j) * sizeE + static_cast<std::size_t>(i)];
  };

  for (std::size_t i = 0; i < sizeE; ++i) {
    for (std::size_t j = i; j < sizeE; ++j) {
      double expected = 0.0;
      for (std::size_t a = 0; a < nAlpha; ++a) {
        const double fracD = (logD[a] - gridD[0]) / (gridD[1] - gridD[0]);
        const int iD = 0;
        const double interp = WeakLibReader::LinearInterp2D4DArray2DAlignedPoint(
            static_cast<int>(i), static_cast<int>(j),
            iD, iT,
            fracD, dT,
            0.0,
            table.data(), layout);
        expected += alpha[a] * interp;
      }
      const double value = outValue(static_cast<int>(i), static_cast<int>(j));
      const double symmetric = outValue(static_cast<int>(j), static_cast<int>(i));
      assert(std::abs(value - expected) < kTol);
  assert(std::abs(symmetric - expected) < kTol);
    }
  }
}

void TestDerivative3DLogWrapper()
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
        const std::size_t offset = layout.Offset(id, it, iy);
        table[offset] = std::log10(actual(dVal, tVal, yVal));
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
  const bool outD = IndexAndDeltaLog10(dCoord, gridD.data(), 2, idxD, fracD);
  assert(!outD);

  int idxT = 0;
  double fracT = 0.0;
  const bool outT = IndexAndDeltaLog10(tCoord, gridT.data(), 2, idxT, fracT);
  assert(!outT);

  int idxY = 0;
  double fracY = 0.0;
  const bool outY = IndexAndDeltaLin(yCoord, gridY.data(), 2, idxY, fracY);
  assert(!outY);

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
      table.data(), layout, axes,
      0.0, interpolated, deriv);

  assert(rc == 0);
  assert(std::abs(interpolated - expectedInterp) < kTol);
  assert(std::abs(deriv[0] - expectedDD) < kTol);
  assert(std::abs(deriv[1] - expectedDT) < kTol);
  assert(std::abs(deriv[2] - expectedDY) < kTol);

  InterpConfig cfg;
  cfg.outOfRange = OutOfRangePolicy::FillNaN;
  double nanInterp = 0.0;
  double nanDeriv[3] = {0.0, 0.0, 0.0};
  const int nanRc = LogInterpolateDifferentiateSingleVariable3DCustomPoint(
      0.1, tCoord, yCoord,
      table.data(), layout, axes,
      0.0, nanInterp, nanDeriv, cfg);
  assert(nanRc == 0);
  assert(std::isnan(nanInterp));
  assert(std::isnan(nanDeriv[0]));
  assert(std::isnan(nanDeriv[1]));
  assert(std::isnan(nanDeriv[2]));
}

void TestDerivative2DAlignedWrapper()
{
  using namespace WeakLibReader;

  constexpr std::size_t sizeE = 2;
  const std::array<double, 2> gridT{1.0, 2.0};
  const std::array<double, 2> gridX{1.0, 3.0};

  const int extents[4] = {static_cast<int>(sizeE), static_cast<int>(sizeE), 2, 2};
  const Layout layout = MakeLayout(extents, 4);

  std::array<double, sizeE * sizeE * 2 * 2> table{};
  auto value = [](int i, int j, double t, double x) {
    return 2.0 + 0.1 * static_cast<double>(i) + 0.2 * static_cast<double>(j) + 0.3 * t + 0.4 * x;
  };

  for (std::size_t j = 0; j < sizeE; ++j) {
    for (std::size_t i = 0; i < sizeE; ++i) {
      for (int it = 0; it < 2; ++it) {
        for (int ix = 0; ix < 2; ++ix) {
          const double actual = value(static_cast<int>(i), static_cast<int>(j), gridT[it], gridX[ix]);
          const std::size_t offset = layout.Offset(static_cast<int>(i), static_cast<int>(j), it, ix);
          table[offset] = std::log10(actual);
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
  const bool outT = IndexAndDeltaLin(logTCoord, gridT.data(), 2, idxT, fracT);
  assert(!outT);

  int idxX = 0;
  double fracX = 0.0;
  const bool outX = IndexAndDeltaLin(logXCoord, gridX.data(), 2, idxX, fracX);
  assert(!outX);

  const double spanT = gridT[idxT + 1] - gridT[idxT];
  const double spanX = gridX[idxX + 1] - gridX[idxX];
  const double aT = 1.0 / (spanT * WeakLibReader::math::Pow10(logTCoord));
  const double aX = 1.0 / (spanX * WeakLibReader::math::Pow10(logXCoord));

  std::array<double, sizeE * sizeE> planeInterp{};
  std::array<double, sizeE * sizeE> planeDerivT{};
  std::array<double, sizeE * sizeE> planeDerivX{};

  const int rc = LogInterpolateDifferentiateSingleVariable2D2DCustomAlignedPoint(
      sizeE, logTCoord, logXCoord,
      table.data(), layout, axes,
      0.0,
      planeInterp.data(),
      planeDerivT.data(),
      planeDerivX.data(),
      InterpConfig{});
  assert(rc == 0);

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

      const std::size_t index = j * sizeE + i;
      const std::size_t mirrorIndex = i * sizeE + j;
      assert(std::abs(planeInterp[index] - interpExpected) < kTol);
      assert(std::abs(planeDerivT[index] - dTExpected) < kTol);
      assert(std::abs(planeDerivX[index] - dXExpected) < kTol);
      assert(std::abs(planeInterp[mirrorIndex] - interpExpected) < kTol);
      assert(std::abs(planeDerivT[mirrorIndex] - dTExpected) < kTol);
      assert(std::abs(planeDerivX[mirrorIndex] - dXExpected) < kTol);
    }
  }
}

} // namespace

int main()
{
  TestBasic2DInterpolation();
  TestOutOfRangeFillNaN();
  TestAlignedPlaneInterpolation();
  TestWeightedSumAligned();
  TestDerivative3DLogWrapper();
  TestDerivative2DAlignedWrapper();

  std::cout << "All WeakLibReader interpolation tests passed.\n";
  return 0;
}
