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
      axes,
      table.data(), 0.0, plane.data());
  REQUIRE(rc == 0);

  int idxT = 0;
  double fracT = 0.0;
  IndexAndDeltaLin(logT, gridT.data(), 2, idxT, fracT);
  int idxX = 0;
  double fracX = 0.0;
  IndexAndDeltaLin(logX, gridX.data(), 2, idxX, fracX);

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
      axes,
      alpha.data(),
      table.data(),
      0.0,
      out.data());
  REQUIRE(rc == 0);

  int idxT = 0;
  double fracT = 0.0;
  IndexAndDeltaLin(logT[0], gridT.data(), 2, idxT, fracT);

  for (std::size_t j = 0; j < sizeE; ++j) {
    for (std::size_t i = 0; i <= j; ++i) {
      double expected = 0.0;
      for (std::size_t l = 0; l < nAlpha; ++l) {
        int idxD = 0;
        double fracD = 0.0;
        IndexAndDeltaLin(logD[l], gridD.data(), 2, idxD, fracD);
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

  Axis axes4[4] = {
      MakeAxis(gridE.data(), static_cast<int>(sizeE), AxisScale::Linear),
      MakeAxis(gridE.data(), static_cast<int>(sizeE), AxisScale::Linear),
      MakeAxis(gridT.data(), 2, AxisScale::Linear),
      MakeAxis(gridX.data(), 2, AxisScale::Linear)};

  const int rc = LogInterpolateSingleVariable2D2DCustomPoint(
      gridE.data(), sizeE, logT, logX,
      axes4,
      table.data(), 0.0, out.data());
  REQUIRE(rc == 0);

  // Verify output against direct 4D interpolation
  // The function uses symmetric storage: out[i,j] = out[j,i] = interp(E[i], E[j], T, X)

  for (std::size_t j = 0; j < sizeE; ++j) {
    for (std::size_t i = 0; i <= j; ++i) {
      // For i <= j, coords are (E[i], E[j], T, X)
      double coords[4] = {gridE[i], gridE[j], logT, logX};
      const double expected = detail::LogInterpolatedValueDirect<4>(
          table.data(), layout, axes4, coords, 0.0);
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

  Axis axes4[4] = {
      MakeAxis(gridE.data(), static_cast<int>(sizeE), AxisScale::Linear),
      MakeAxis(gridE.data(), static_cast<int>(sizeE), AxisScale::Linear),
      MakeAxis(gridT.data(), 2, AxisScale::Linear),
      MakeAxis(gridX.data(), 2, AxisScale::Linear)};

  const int rc = LogInterpolateSingleVariable2D2DCustom(
      gridE.data(), sizeE,
      logT.data(), logX.data(), count,
      axes4,
      table.data(), 0.0, out.data());
  REQUIRE(rc == 0);

  // Verify against single point version
  for (std::size_t l = 0; l < count; ++l) {
    std::array<double, sizeE * sizeE> plane{};
    const int rcPoint = LogInterpolateSingleVariable2D2DCustomPoint(
        gridE.data(), sizeE, logT[l], logX[l],
        axes4,
        table.data(), 0.0, plane.data());
    REQUIRE(rcPoint == 0);

    for (std::size_t k = 0; k < sizeE * sizeE; ++k) {
      CHECK(out[l * sizeE * sizeE + k] == Catch::Approx(plane[k]).margin(kTol));
    }
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

  Axis axes[2] = {
      MakeAxis(gridT.data(), 2, AxisScale::Linear),
      MakeAxis(gridX.data(), 2, AxisScale::Linear)};

  const std::array<double, count> logT{1.3, 1.7};
  const std::array<double, count> logX{1.5, 2.5};
  const std::size_t planeSize = sizeE * sizeE;
  std::array<double, planeSize * count> outBatch{};

  const int rcBatch = LogInterpolateSingleVariable2D2DCustomAligned(
      sizeE,
      logT.data(), logX.data(), count,
      axes,
      table.data(),
      0.0,
      outBatch.data());
  REQUIRE(rcBatch == 0);

  for (std::size_t k = 0; k < count; ++k) {
    std::array<double, planeSize> outPoint{};
    const int rcPoint = LogInterpolateSingleVariable2D2DCustomAlignedPoint(
        sizeE, logT[k], logX[k],
        axes,
        table.data(),
        0.0,
        outPoint.data());
    REQUIRE(rcPoint == 0);

    for (std::size_t i = 0; i < planeSize; ++i) {
      CHECK(outBatch[k * planeSize + i] == Catch::Approx(outPoint[i]).margin(kTol));
    }
  }
}
