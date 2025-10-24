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

} // namespace

int main()
{
  TestBasic2DInterpolation();
  TestOutOfRangeFillNaN();
  TestAlignedPlaneInterpolation();
  TestWeightedSumAligned();

  std::cout << "All WeakLibReader interpolation tests passed.\n";
  return 0;
}

