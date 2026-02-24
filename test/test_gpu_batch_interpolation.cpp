#define SIMPLE_CATCH_NO_MAIN
#include <catch2/catch_test_macros.hpp>

#include <AMReX_Gpu.H>

#include "interp/WeakLibReader_LogInterpolate.hpp"
#include "interp/WeakLibReader_InterpLogTable.hpp"
#include "base/WeakLibReader_Layout.hpp"
#include "base/WeakLibReader_AxisTypes.hpp"
#include "test_amrex_guard.hpp"
#include "test_constants.hpp"

#include <array>
#include <cmath>
#include <vector>

using test_constants::Tol;

namespace {

template <typename T, std::size_t N>
amrex::Gpu::DeviceVector<T> ToDevice(const std::array<T, N>& host)
{
  amrex::Gpu::DeviceVector<T> dev(N);
  amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                   host.data(), host.data() + N,
                   dev.data());
  return dev;
}

template <typename T>
std::vector<T> ToHost(const amrex::Gpu::DeviceVector<T>& dev)
{
  std::vector<T> host(dev.size());
  amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                   dev.data(), dev.data() + dev.size(),
                   host.data());
  return host;
}

} // anonymous namespace

// =============================================================================
// GPU Batch Interpolation Tests
//
// Each test launches a single batch function from a device kernel via
// amrex::ParallelFor(1, ...) and verifies the output matches host-computed
// reference values.
// =============================================================================

TEST_CASE("GPU: Batch 2D log interpolation", "[gpu][batch][2d]")
{
  using namespace WeakLibReader;
  AmrexGuard amrex{};

  constexpr std::size_t count = 3;
  const std::array<double, 2> gridX{1.0, 2.0};
  const std::array<double, 2> gridY{1.0, 3.0};
  const std::array<double, 4> table{
      std::log10(2.0), std::log10(3.0),
      std::log10(4.0), std::log10(5.0)};

  std::array<double, count> x0{1.0, 1.5, 2.0};
  std::array<double, count> x1{1.0, 2.0, 3.0};

  // --- Host reference ---
  std::array<double, count> hostOut{};
  Axis hostAxes[2] = {
      MakeAxis(gridX.data(), 2, AxisScale::Linear),
      MakeAxis(gridY.data(), 2, AxisScale::Linear)};
  const int hostRc = LogInterpolateSingleVariable2DCustom(
      x0.data(), x1.data(), count,
      hostAxes, table.data(), 0.0, hostOut.data());
  REQUIRE(hostRc == 0);

  // --- Device copy ---
  auto d_gridX = ToDevice(gridX);
  auto d_gridY = ToDevice(gridY);
  auto d_table = ToDevice(table);
  auto d_x0 = ToDevice(x0);
  auto d_x1 = ToDevice(x1);
  amrex::Gpu::DeviceVector<double> d_out(count, 0.0);

  Axis ax0 = MakeAxis(d_gridX.data(), 2, AxisScale::Linear);
  Axis ax1 = MakeAxis(d_gridY.data(), 2, AxisScale::Linear);

  const double* p_x0 = d_x0.data();
  const double* p_x1 = d_x1.data();
  const double* p_table = d_table.data();
  double* p_out = d_out.data();

  amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE (int) {
    Axis axes[2] = {ax0, ax1};
    LogInterpolateSingleVariable2DCustom(
        p_x0, p_x1, count,
        axes, p_table, 0.0, p_out);
  });
  amrex::Gpu::streamSynchronize();

  // --- Verify ---
  const auto result = ToHost(d_out);
  for (std::size_t i = 0; i < count; ++i) {
    CHECK(result[i] == Catch::Approx(hostOut[i]).margin(Tol));
  }
}

TEST_CASE("GPU: Batch 3D log interpolation", "[gpu][batch][3d]")
{
  using namespace WeakLibReader;
  AmrexGuard amrex{};

  constexpr std::size_t count = 3;
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

  std::array<double, count> x0{1.0, 1.5, 2.0};
  std::array<double, count> x1{1.0, 2.0, 3.0};
  std::array<double, count> x2{0.0, 0.5, 1.0};

  // --- Host reference ---
  std::array<double, count> hostOut{};
  Axis hostAxes[3] = {
      MakeAxis(gridX.data(), 2, AxisScale::Log10),
      MakeAxis(gridY.data(), 2, AxisScale::Log10),
      MakeAxis(gridZ.data(), 2, AxisScale::Linear)};
  const int hostRc = LogInterpolateSingleVariable3DCustom(
      x0.data(), x1.data(), x2.data(), count,
      hostAxes, table.data(), 0.0, hostOut.data());
  REQUIRE(hostRc == 0);

  // --- Device copy ---
  auto d_gridX = ToDevice(gridX);
  auto d_gridY = ToDevice(gridY);
  auto d_gridZ = ToDevice(gridZ);
  auto d_table = ToDevice(table);
  auto d_x0 = ToDevice(x0);
  auto d_x1 = ToDevice(x1);
  auto d_x2 = ToDevice(x2);
  amrex::Gpu::DeviceVector<double> d_out(count, 0.0);

  Axis ax0 = MakeAxis(d_gridX.data(), 2, AxisScale::Log10);
  Axis ax1 = MakeAxis(d_gridY.data(), 2, AxisScale::Log10);
  Axis ax2 = MakeAxis(d_gridZ.data(), 2, AxisScale::Linear);

  const double* p_x0 = d_x0.data();
  const double* p_x1 = d_x1.data();
  const double* p_x2 = d_x2.data();
  const double* p_table = d_table.data();
  double* p_out = d_out.data();

  amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE (int) {
    Axis axes[3] = {ax0, ax1, ax2};
    LogInterpolateSingleVariable3DCustom(
        p_x0, p_x1, p_x2, count,
        axes, p_table, 0.0, p_out);
  });
  amrex::Gpu::streamSynchronize();

  // --- Verify ---
  const auto result = ToHost(d_out);
  for (std::size_t i = 0; i < count; ++i) {
    CHECK(result[i] == Catch::Approx(hostOut[i]).margin(Tol));
  }
}

TEST_CASE("GPU: Batch 4D log interpolation", "[gpu][batch][4d]")
{
  using namespace WeakLibReader;
  AmrexGuard amrex{};

  constexpr std::size_t count = 3;
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

  std::array<double, count> x0{1.0, 1.5, 2.0};
  std::array<double, count> x1{1.0, 2.0, 3.0};
  std::array<double, count> x2{0.0, 0.5, 1.0};
  std::array<double, count> x3{5.0, 7.0, 9.0};

  // --- Host reference ---
  std::array<double, count> hostOut{};
  Axis hostAxes[4] = {
      MakeAxis(gridA.data(), 2, AxisScale::Linear),
      MakeAxis(gridB.data(), 2, AxisScale::Linear),
      MakeAxis(gridC.data(), 2, AxisScale::Linear),
      MakeAxis(gridD.data(), 2, AxisScale::Linear)};
  const int hostRc = LogInterpolateSingleVariable4DCustom(
      x0.data(), x1.data(), x2.data(), x3.data(), count,
      hostAxes, table.data(), 0.0, hostOut.data());
  REQUIRE(hostRc == 0);

  // --- Device copy ---
  auto d_gridA = ToDevice(gridA);
  auto d_gridB = ToDevice(gridB);
  auto d_gridC = ToDevice(gridC);
  auto d_gridD = ToDevice(gridD);
  auto d_table = ToDevice(table);
  auto d_x0 = ToDevice(x0);
  auto d_x1 = ToDevice(x1);
  auto d_x2 = ToDevice(x2);
  auto d_x3 = ToDevice(x3);
  amrex::Gpu::DeviceVector<double> d_out(count, 0.0);

  Axis ax0 = MakeAxis(d_gridA.data(), 2, AxisScale::Linear);
  Axis ax1 = MakeAxis(d_gridB.data(), 2, AxisScale::Linear);
  Axis ax2 = MakeAxis(d_gridC.data(), 2, AxisScale::Linear);
  Axis ax3 = MakeAxis(d_gridD.data(), 2, AxisScale::Linear);

  const double* p_x0 = d_x0.data();
  const double* p_x1 = d_x1.data();
  const double* p_x2 = d_x2.data();
  const double* p_x3 = d_x3.data();
  const double* p_table = d_table.data();
  double* p_out = d_out.data();

  amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE (int) {
    Axis axes[4] = {ax0, ax1, ax2, ax3};
    LogInterpolateSingleVariable4DCustom(
        p_x0, p_x1, p_x2, p_x3, count,
        axes, p_table, 0.0, p_out);
  });
  amrex::Gpu::streamSynchronize();

  // --- Verify ---
  const auto result = ToHost(d_out);
  for (std::size_t i = 0; i < count; ++i) {
    CHECK(result[i] == Catch::Approx(hostOut[i]).margin(Tol));
  }
}

TEST_CASE("GPU: Batch 1D3D sweep", "[gpu][batch][sweep]")
{
  using namespace WeakLibReader;
  AmrexGuard amrex{};

  constexpr std::size_t sizeE = 2;
  constexpr std::size_t count = 2;
  constexpr std::size_t totalOut = sizeE * count;

  const std::array<double, 2> gridE{1.0, 2.0};
  const std::array<double, 2> gridD{1.0, 3.0};
  const std::array<double, 2> gridT{10.0, 20.0};
  const std::array<double, 2> gridY{0.0, 1.0};

  const int extents[4] = {2, 2, 2, 2};
  const Layout layout = MakeLayout(extents, 4);
  std::array<double, 16> table{};
  for (int ie = 0; ie < 2; ++ie) {
    for (int id = 0; id < 2; ++id) {
      for (int it = 0; it < 2; ++it) {
        for (int iy = 0; iy < 2; ++iy) {
          table[layout.Offset(ie, id, it, iy)] =
              std::log10(1.0 + 0.2 * gridE[ie] + 0.3 * gridD[id] +
                         0.4 * gridT[it] + 0.5 * gridY[iy]);
        }
      }
    }
  }

  std::array<double, sizeE> logE{1.2, 1.8};
  std::array<double, count> logD{1.1, 2.5};
  std::array<double, count> logT{12.0, 18.0};
  std::array<double, count> y{0.25, 0.75};

  // --- Host reference ---
  std::array<double, totalOut> hostOut{};
  Axis hostAxes[4] = {
      MakeAxis(gridE.data(), 2, AxisScale::Linear),
      MakeAxis(gridD.data(), 2, AxisScale::Linear),
      MakeAxis(gridT.data(), 2, AxisScale::Linear),
      MakeAxis(gridY.data(), 2, AxisScale::Linear)};
  const int hostRc = LogInterpolateSingleVariable1D3DCustom(
      logE.data(), sizeE,
      logD.data(), logT.data(), y.data(), count,
      hostAxes, table.data(), 0.0, hostOut.data());
  REQUIRE(hostRc == 0);

  // --- Device copy ---
  auto d_gridE = ToDevice(gridE);
  auto d_gridD = ToDevice(gridD);
  auto d_gridT = ToDevice(gridT);
  auto d_gridY = ToDevice(gridY);
  auto d_table = ToDevice(table);
  auto d_logE = ToDevice(logE);
  auto d_logD = ToDevice(logD);
  auto d_logT = ToDevice(logT);
  auto d_y = ToDevice(y);
  amrex::Gpu::DeviceVector<double> d_out(totalOut, 0.0);

  Axis ax0 = MakeAxis(d_gridE.data(), 2, AxisScale::Linear);
  Axis ax1 = MakeAxis(d_gridD.data(), 2, AxisScale::Linear);
  Axis ax2 = MakeAxis(d_gridT.data(), 2, AxisScale::Linear);
  Axis ax3 = MakeAxis(d_gridY.data(), 2, AxisScale::Linear);

  const double* p_logE = d_logE.data();
  const double* p_logD = d_logD.data();
  const double* p_logT = d_logT.data();
  const double* p_y = d_y.data();
  const double* p_table = d_table.data();
  double* p_out = d_out.data();

  amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE (int) {
    Axis axes[4] = {ax0, ax1, ax2, ax3};
    LogInterpolateSingleVariable1D3DCustom(
        p_logE, sizeE,
        p_logD, p_logT, p_y, count,
        axes, p_table, 0.0, p_out);
  });
  amrex::Gpu::streamSynchronize();

  // --- Verify ---
  const auto result = ToHost(d_out);
  for (std::size_t i = 0; i < totalOut; ++i) {
    CHECK(result[i] == Catch::Approx(hostOut[i]).margin(Tol));
  }
}

TEST_CASE("GPU: Batch 2D2D sweep", "[gpu][batch][sweep]")
{
  using namespace WeakLibReader;
  AmrexGuard amrex{};

  constexpr std::size_t sizeE = 2;
  constexpr std::size_t count = 2;
  constexpr std::size_t planeSize = sizeE * sizeE;
  constexpr std::size_t totalOut = planeSize * count;

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

  // --- Host reference ---
  std::array<double, totalOut> hostOut{};
  Axis hostAxes[4] = {
      MakeAxis(gridE.data(), static_cast<int>(sizeE), AxisScale::Linear),
      MakeAxis(gridE.data(), static_cast<int>(sizeE), AxisScale::Linear),
      MakeAxis(gridT.data(), 2, AxisScale::Linear),
      MakeAxis(gridX.data(), 2, AxisScale::Linear)};
  const int hostRc = LogInterpolateSingleVariable2D2DCustom(
      gridE.data(), sizeE,
      logT.data(), logX.data(), count,
      hostAxes, table.data(), 0.0, hostOut.data());
  REQUIRE(hostRc == 0);

  // --- Device copy ---
  auto d_gridE = ToDevice(gridE);
  auto d_gridT = ToDevice(gridT);
  auto d_gridX = ToDevice(gridX);
  auto d_table = ToDevice(table);
  auto d_logT = ToDevice(logT);
  auto d_logX = ToDevice(logX);
  amrex::Gpu::DeviceVector<double> d_out(totalOut, 0.0);

  Axis ax0 = MakeAxis(d_gridE.data(), static_cast<int>(sizeE), AxisScale::Linear);
  Axis ax1 = MakeAxis(d_gridE.data(), static_cast<int>(sizeE), AxisScale::Linear);
  Axis ax2 = MakeAxis(d_gridT.data(), 2, AxisScale::Linear);
  Axis ax3 = MakeAxis(d_gridX.data(), 2, AxisScale::Linear);

  const double* p_logE = d_gridE.data();
  const double* p_logT = d_logT.data();
  const double* p_logX = d_logX.data();
  const double* p_table = d_table.data();
  double* p_out = d_out.data();

  amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE (int) {
    Axis axes[4] = {ax0, ax1, ax2, ax3};
    LogInterpolateSingleVariable2D2DCustom(
        p_logE, sizeE,
        p_logT, p_logX, count,
        axes, p_table, 0.0, p_out);
  });
  amrex::Gpu::streamSynchronize();

  // --- Verify ---
  const auto result = ToHost(d_out);
  for (std::size_t i = 0; i < totalOut; ++i) {
    CHECK(result[i] == Catch::Approx(hostOut[i]).margin(Tol));
  }
}

TEST_CASE("GPU: Batch aligned 2D2D sweep", "[gpu][batch][sweep]")
{
  using namespace WeakLibReader;
  AmrexGuard amrex{};

  constexpr std::size_t sizeE = 2;
  constexpr std::size_t count = 2;
  constexpr std::size_t planeSize = sizeE * sizeE;
  constexpr std::size_t totalOut = planeSize * count;

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

  // --- Host reference ---
  std::array<double, totalOut> hostOut{};
  Axis hostAxes[2] = {
      MakeAxis(gridT.data(), 2, AxisScale::Linear),
      MakeAxis(gridX.data(), 2, AxisScale::Linear)};
  const int hostRc = LogInterpolateSingleVariable2D2DCustomAligned(
      sizeE,
      logT.data(), logX.data(), count,
      hostAxes, table.data(), 0.0, hostOut.data());
  REQUIRE(hostRc == 0);

  // --- Device copy ---
  auto d_gridT = ToDevice(gridT);
  auto d_gridX = ToDevice(gridX);
  auto d_table = ToDevice(table);
  auto d_logT = ToDevice(logT);
  auto d_logX = ToDevice(logX);
  amrex::Gpu::DeviceVector<double> d_out(totalOut, 0.0);

  Axis ax0 = MakeAxis(d_gridT.data(), 2, AxisScale::Linear);
  Axis ax1 = MakeAxis(d_gridX.data(), 2, AxisScale::Linear);

  const double* p_logT = d_logT.data();
  const double* p_logX = d_logX.data();
  const double* p_table = d_table.data();
  double* p_out = d_out.data();

  amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE (int) {
    Axis axes[2] = {ax0, ax1};
    LogInterpolateSingleVariable2D2DCustomAligned(
        sizeE,
        p_logT, p_logX, count,
        axes, p_table, 0.0, p_out);
  });
  amrex::Gpu::streamSynchronize();

  // --- Verify ---
  const auto result = ToHost(d_out);
  for (std::size_t i = 0; i < totalOut; ++i) {
    CHECK(result[i] == Catch::Approx(hostOut[i]).margin(Tol));
  }
}

TEST_CASE("GPU: Batch 3D derivative", "[gpu][batch][deriv]")
{
  using namespace WeakLibReader;
  AmrexGuard amrex{};

  constexpr std::size_t count = 3;

  const std::array<double, 2> gridD{1.0, 10.0};
  const std::array<double, 2> gridT{1.0, 100.0};
  const std::array<double, 2> gridY{0.0, 1.0};

  const int extents[3] = {2, 2, 2};
  const Layout layout = MakeLayout(extents, 3);
  std::array<double, 8> table{};
  for (int id = 0; id < 2; ++id) {
    for (int it = 0; it < 2; ++it) {
      for (int iy = 0; iy < 2; ++iy) {
        table[layout.Offset(id, it, iy)] =
            std::log10(1.0 + 0.5 * gridD[id] + 0.25 * gridT[it] + 0.1 * gridY[iy]);
      }
    }
  }

  std::array<double, count> dCoord{2.0, 5.0, 8.0};
  std::array<double, count> tCoord{10.0, 50.0, 80.0};
  std::array<double, count> yCoord{0.2, 0.5, 0.8};

  // --- Host reference ---
  std::array<double, count> hostInterp{};
  std::array<double, count * 3> hostDeriv{};
  Axis hostAxes[3] = {
      MakeAxis(gridD.data(), 2, AxisScale::Log10),
      MakeAxis(gridT.data(), 2, AxisScale::Log10),
      MakeAxis(gridY.data(), 2, AxisScale::Linear)};
  const int hostRc = LogInterpolateDifferentiateSingleVariable3DCustom(
      dCoord.data(), tCoord.data(), yCoord.data(), count,
      hostAxes, table.data(), 0.0,
      hostInterp.data(), hostDeriv.data());
  REQUIRE(hostRc == 0);

  // --- Device copy ---
  auto d_gridD = ToDevice(gridD);
  auto d_gridT = ToDevice(gridT);
  auto d_gridY = ToDevice(gridY);
  auto d_table = ToDevice(table);
  auto d_dCoord = ToDevice(dCoord);
  auto d_tCoord = ToDevice(tCoord);
  auto d_yCoord = ToDevice(yCoord);
  amrex::Gpu::DeviceVector<double> d_interp(count, 0.0);
  amrex::Gpu::DeviceVector<double> d_deriv(count * 3, 0.0);

  Axis ax0 = MakeAxis(d_gridD.data(), 2, AxisScale::Log10);
  Axis ax1 = MakeAxis(d_gridT.data(), 2, AxisScale::Log10);
  Axis ax2 = MakeAxis(d_gridY.data(), 2, AxisScale::Linear);

  const double* p_dCoord = d_dCoord.data();
  const double* p_tCoord = d_tCoord.data();
  const double* p_yCoord = d_yCoord.data();
  const double* p_table = d_table.data();
  double* p_interp = d_interp.data();
  double* p_deriv = d_deriv.data();

  amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE (int) {
    Axis axes[3] = {ax0, ax1, ax2};
    LogInterpolateDifferentiateSingleVariable3DCustom(
        p_dCoord, p_tCoord, p_yCoord, count,
        axes, p_table, 0.0,
        p_interp, p_deriv);
  });
  amrex::Gpu::streamSynchronize();

  // --- Verify ---
  const auto resultInterp = ToHost(d_interp);
  const auto resultDeriv = ToHost(d_deriv);
  for (std::size_t i = 0; i < count; ++i) {
    CHECK(resultInterp[i] == Catch::Approx(hostInterp[i]).margin(Tol));
    CHECK(resultDeriv[i * 3 + 0] == Catch::Approx(hostDeriv[i * 3 + 0]).margin(Tol));
    CHECK(resultDeriv[i * 3 + 1] == Catch::Approx(hostDeriv[i * 3 + 1]).margin(Tol));
    CHECK(resultDeriv[i * 3 + 2] == Catch::Approx(hostDeriv[i * 3 + 2]).margin(Tol));
  }
}

TEST_CASE("GPU: Batch 2D2D derivative", "[gpu][batch][deriv]")
{
  using namespace WeakLibReader;
  AmrexGuard amrex{};

  constexpr std::size_t sizeE = 2;
  constexpr std::size_t count = 2;
  constexpr std::size_t planeSize = sizeE * sizeE;
  constexpr std::size_t totalOut = planeSize * count;

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

  // --- Host reference ---
  std::array<double, totalOut> hostInterp{};
  std::array<double, totalOut> hostDerivT{};
  std::array<double, totalOut> hostDerivX{};
  Axis hostAxes[4] = {
      MakeAxis(gridE.data(), static_cast<int>(sizeE), AxisScale::Linear),
      MakeAxis(gridE.data(), static_cast<int>(sizeE), AxisScale::Linear),
      MakeAxis(gridT.data(), 2, AxisScale::Linear),
      MakeAxis(gridX.data(), 2, AxisScale::Linear)};
  const int hostRc = LogInterpolateDifferentiateSingleVariable2D2DCustom(
      gridE.data(), sizeE,
      logT.data(), logX.data(), count,
      hostAxes, table.data(), 0.0,
      hostInterp.data(), hostDerivT.data(), hostDerivX.data());
  REQUIRE(hostRc == 0);

  // --- Device copy ---
  auto d_gridE = ToDevice(gridE);
  auto d_gridT = ToDevice(gridT);
  auto d_gridX = ToDevice(gridX);
  auto d_table = ToDevice(table);
  auto d_logT = ToDevice(logT);
  auto d_logX = ToDevice(logX);
  amrex::Gpu::DeviceVector<double> d_interp(totalOut, 0.0);
  amrex::Gpu::DeviceVector<double> d_derivT(totalOut, 0.0);
  amrex::Gpu::DeviceVector<double> d_derivX(totalOut, 0.0);

  Axis ax0 = MakeAxis(d_gridE.data(), static_cast<int>(sizeE), AxisScale::Linear);
  Axis ax1 = MakeAxis(d_gridE.data(), static_cast<int>(sizeE), AxisScale::Linear);
  Axis ax2 = MakeAxis(d_gridT.data(), 2, AxisScale::Linear);
  Axis ax3 = MakeAxis(d_gridX.data(), 2, AxisScale::Linear);

  const double* p_logE = d_gridE.data();
  const double* p_logT = d_logT.data();
  const double* p_logX = d_logX.data();
  const double* p_table = d_table.data();
  double* p_interp = d_interp.data();
  double* p_derivT = d_derivT.data();
  double* p_derivX = d_derivX.data();

  amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE (int) {
    Axis axes[4] = {ax0, ax1, ax2, ax3};
    LogInterpolateDifferentiateSingleVariable2D2DCustom(
        p_logE, sizeE,
        p_logT, p_logX, count,
        axes, p_table, 0.0,
        p_interp, p_derivT, p_derivX);
  });
  amrex::Gpu::streamSynchronize();

  // --- Verify ---
  const auto resultInterp = ToHost(d_interp);
  const auto resultDerivT = ToHost(d_derivT);
  const auto resultDerivX = ToHost(d_derivX);
  for (std::size_t i = 0; i < totalOut; ++i) {
    CHECK(resultInterp[i] == Catch::Approx(hostInterp[i]).margin(Tol));
    CHECK(resultDerivT[i] == Catch::Approx(hostDerivT[i]).margin(Tol));
    CHECK(resultDerivX[i] == Catch::Approx(hostDerivX[i]).margin(Tol));
  }
}

TEST_CASE("GPU: Batch aligned 2D2D derivative", "[gpu][batch][deriv]")
{
  using namespace WeakLibReader;
  AmrexGuard amrex{};

  constexpr std::size_t sizeE = 2;
  constexpr std::size_t count = 2;
  constexpr std::size_t planeSize = sizeE * sizeE;
  constexpr std::size_t totalOut = planeSize * count;

  const std::array<double, 2> gridT{1.0, 2.0};
  const std::array<double, 2> gridX{1.0, 3.0};

  const int extents[4] = {static_cast<int>(sizeE), static_cast<int>(sizeE), 2, 2};
  const Layout layout = MakeLayout(extents, 4);
  std::array<double, sizeE * sizeE * 2 * 2> table{};
  for (std::size_t j = 0; j < sizeE; ++j) {
    for (std::size_t i = 0; i < sizeE; ++i) {
      for (int it = 0; it < 2; ++it) {
        for (int ix = 0; ix < 2; ++ix) {
          table[layout.Offset(static_cast<int>(i), static_cast<int>(j), it, ix)] =
              std::log10(2.0 + 0.1 * static_cast<double>(i) +
                         0.2 * static_cast<double>(j) +
                         0.3 * gridT[it] + 0.4 * gridX[ix]);
        }
      }
    }
  }

  const std::array<double, count> logT{1.3, 1.7};
  const std::array<double, count> logX{1.5, 2.5};

  // --- Host reference ---
  std::array<double, totalOut> hostInterp{};
  std::array<double, totalOut> hostDerivT{};
  std::array<double, totalOut> hostDerivX{};
  Axis hostAxes[2] = {
      MakeAxis(gridT.data(), 2, AxisScale::Linear),
      MakeAxis(gridX.data(), 2, AxisScale::Linear)};
  const int hostRc = LogInterpolateDifferentiateSingleVariable2D2DCustomAligned(
      sizeE,
      logT.data(), logX.data(), count,
      hostAxes, table.data(), 0.0,
      hostInterp.data(), hostDerivT.data(), hostDerivX.data());
  REQUIRE(hostRc == 0);

  // --- Device copy ---
  auto d_gridT = ToDevice(gridT);
  auto d_gridX = ToDevice(gridX);
  auto d_table = ToDevice(table);
  auto d_logT = ToDevice(logT);
  auto d_logX = ToDevice(logX);
  amrex::Gpu::DeviceVector<double> d_interp(totalOut, 0.0);
  amrex::Gpu::DeviceVector<double> d_derivT(totalOut, 0.0);
  amrex::Gpu::DeviceVector<double> d_derivX(totalOut, 0.0);

  Axis ax0 = MakeAxis(d_gridT.data(), 2, AxisScale::Linear);
  Axis ax1 = MakeAxis(d_gridX.data(), 2, AxisScale::Linear);

  const double* p_logT = d_logT.data();
  const double* p_logX = d_logX.data();
  const double* p_table = d_table.data();
  double* p_interp = d_interp.data();
  double* p_derivT = d_derivT.data();
  double* p_derivX = d_derivX.data();

  amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE (int) {
    Axis axes[2] = {ax0, ax1};
    LogInterpolateDifferentiateSingleVariable2D2DCustomAligned(
        sizeE,
        p_logT, p_logX, count,
        axes, p_table, 0.0,
        p_interp, p_derivT, p_derivX);
  });
  amrex::Gpu::streamSynchronize();

  // --- Verify ---
  const auto resultInterp = ToHost(d_interp);
  const auto resultDerivT = ToHost(d_derivT);
  const auto resultDerivX = ToHost(d_derivX);
  for (std::size_t i = 0; i < totalOut; ++i) {
    CHECK(resultInterp[i] == Catch::Approx(hostInterp[i]).margin(Tol));
    CHECK(resultDerivT[i] == Catch::Approx(hostDerivT[i]).margin(Tol));
    CHECK(resultDerivX[i] == Catch::Approx(hostDerivX[i]).margin(Tol));
  }
}

TEST_CASE("GPU: Batch weighted sum aligned", "[gpu][batch][weighted]")
{
  using namespace WeakLibReader;
  AmrexGuard amrex{};

  constexpr std::size_t sizeE = 2;
  constexpr std::size_t nAlpha = 2;
  constexpr std::size_t count = 1;
  constexpr std::size_t planeSize = sizeE * sizeE;
  constexpr std::size_t totalOut = planeSize * count;

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

  const std::array<double, nAlpha> alpha{0.6, 1.1};
  const std::array<double, nAlpha * count> logD{1.5, 2.4};
  const std::array<double, count> logT{1.5};

  // --- Host reference ---
  std::array<double, totalOut> hostOut{};
  Axis hostAxes[2] = {
      MakeAxis(gridD.data(), 2, AxisScale::Linear),
      MakeAxis(gridT.data(), 2, AxisScale::Linear)};
  const int hostRc = SumLogInterpolateSingleVariable2D2DCustomAligned(
      sizeE,
      logD.data(), nAlpha,
      logT.data(), count,
      hostAxes, alpha.data(), table.data(), 0.0, hostOut.data());
  REQUIRE(hostRc == 0);

  // --- Device copy ---
  auto d_gridD = ToDevice(gridD);
  auto d_gridT = ToDevice(gridT);
  auto d_table = ToDevice(table);
  auto d_alpha = ToDevice(alpha);
  auto d_logD = ToDevice(logD);
  auto d_logT = ToDevice(logT);
  amrex::Gpu::DeviceVector<double> d_out(totalOut, 0.0);

  Axis ax0 = MakeAxis(d_gridD.data(), 2, AxisScale::Linear);
  Axis ax1 = MakeAxis(d_gridT.data(), 2, AxisScale::Linear);

  const double* p_logD = d_logD.data();
  const double* p_logT = d_logT.data();
  const double* p_alpha = d_alpha.data();
  const double* p_table = d_table.data();
  double* p_out = d_out.data();

  amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE (int) {
    Axis axes[2] = {ax0, ax1};
    SumLogInterpolateSingleVariable2D2DCustomAligned(
        sizeE,
        p_logD, nAlpha,
        p_logT, count,
        axes, p_alpha, p_table, 0.0, p_out);
  });
  amrex::Gpu::streamSynchronize();

  // --- Verify ---
  const auto result = ToHost(d_out);
  for (std::size_t i = 0; i < totalOut; ++i) {
    CHECK(result[i] == Catch::Approx(hostOut[i]).margin(Tol));
  }
}
