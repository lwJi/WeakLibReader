#define SIMPLE_CATCH_NO_MAIN
#include <catch2/catch_test_macros.hpp>

#include "interp/WeakLibReader_LogInterpolate.hpp"
#include "interp/WeakLibReader_InterpLogTable.hpp"
#include "base/WeakLibReader_Layout.hpp"
#include "base/WeakLibReader_AxisTypes.hpp"
#include "base/WeakLibReader_Math.hpp"

#include <array>
#include <cmath>
#include <limits>

namespace {

constexpr double Tol = 1.0e-12;

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
      CHECK(plane[lower] == Catch::Approx(expected).margin(Tol));
      CHECK(plane[upper] == Catch::Approx(expected).margin(Tol));
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
      CHECK(out[lower] == Catch::Approx(expected).margin(Tol));
      CHECK(out[upper] == Catch::Approx(expected).margin(Tol));
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
      CHECK(out[lower] == Catch::Approx(expected).margin(Tol));
      CHECK(out[upper] == Catch::Approx(expected).margin(Tol));
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
      CHECK(out[l * sizeE * sizeE + k] == Catch::Approx(plane[k]).margin(Tol));
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
      CHECK(outBatch[k * planeSize + i] == Catch::Approx(outPoint[i]).margin(Tol));
    }
  }
}

TEST_CASE("PreAlignScatteringKernelMoment interpolates correctly", "[loginterp][prealign]")
{
  using namespace WeakLibReader;

  // --- Setup: small 5D kernel [nE, nE, nMom, nDim3, nDim4] ---
  constexpr int nE_raw = 3;
  constexpr int nMom = 2;
  constexpr int nDim3 = 2;
  constexpr int nDim4 = 2;
  constexpr double offset = 1.0;
  constexpr std::size_t rawTotal = nE_raw * nE_raw * nMom * nDim3 * nDim4; // 72

  // Raw energy grid (Log10 scale)
  const std::array<double, nE_raw> gridE{1.0, 10.0, 100.0};

  // 5D kernel layout
  const int rawExtents[5] = {nE_raw, nE_raw, nMom, nDim3, nDim4};
  const Layout rawLayout = MakeLayout(rawExtents, 5);

  // Fill with known analytic function:
  //   value(E_in, E_out, iMom, iD3, iD4) =
  //       1.0 + 0.1*E_in + 0.2*E_out + 0.5*(iMom+1) + 0.3*(iD3+1) + 0.4*(iD4+1)
  // Store as log10(value + offset)
  std::array<double, rawTotal> rawKernel{};
  for (int iD4 = 0; iD4 < nDim4; ++iD4) {
    for (int iD3 = 0; iD3 < nDim3; ++iD3) {
      for (int iMom = 0; iMom < nMom; ++iMom) {
        for (int iE2 = 0; iE2 < nE_raw; ++iE2) {
          for (int iE1 = 0; iE1 < nE_raw; ++iE1) {
            const double value = 1.0 + 0.1 * gridE[iE1] + 0.2 * gridE[iE2] +
                                 0.5 * static_cast<double>(iMom + 1) +
                                 0.3 * static_cast<double>(iD3 + 1) +
                                 0.4 * static_cast<double>(iD4 + 1);
            rawKernel[rawLayout.Offset(iE1, iE2, iMom, iD3, iD4)] =
                std::log10(value + offset);
          }
        }
      }
    }
  }

  // Aligned energy grid (within raw range [1, 100])
  constexpr int nAligned = 4;
  const std::array<double, nAligned> alignedE{2.0, 5.0, 20.0, 50.0};

  // Energy axis
  const Axis energyAxis = MakeAxis(gridE.data(), nE_raw, AxisScale::Log10);
  const Axis energyAxes[2] = {energyAxis, energyAxis};

  // Output layout: [nAligned, nAligned, nDim3, nDim4]
  const int outExtents[4] = {nAligned, nAligned, nDim3, nDim4};
  const Layout outLayout = MakeLayout(outExtents, 4);
  constexpr std::size_t outSize = nAligned * nAligned * nDim3 * nDim4; // 64

  // --- Test each moment ---
  for (int iMom = 0; iMom < nMom; ++iMom) {
    std::array<double, outSize> output{};

    PreAlignScatteringKernelMoment(
        rawKernel.data(), rawLayout, energyAxis,
        iMom, nDim3, nDim4,
        alignedE.data(), nAligned,
        offset, output.data());

    // Verify against manual 2D interpolation calls
    for (int iD4 = 0; iD4 < nDim4; ++iD4) {
      for (int iD3 = 0; iD3 < nDim3; ++iD3) {
        // Get pointer to contiguous 2D energy slice [nE, nE] at (iMom, iD3, iD4)
        const int sliceIdx[5] = {0, 0, iMom, iD3, iD4};
        const double* slice2d = rawKernel.data() + rawLayout.Offset(sliceIdx);

        for (int iA2 = 0; iA2 < nAligned; ++iA2) {
          for (int iA1 = 0; iA1 < nAligned; ++iA1) {
            const double manual = LogInterpolateSingleVariable2DCustomPoint(
                alignedE[iA1], alignedE[iA2],
                energyAxes, slice2d, offset);
            const double expected = math::Log10(manual + offset);

            CHECK(output[outLayout.Offset(iA1, iA2, iD3, iD4)] ==
                  Catch::Approx(expected).margin(Tol));
          }
        }
      }
    }
  }
}
