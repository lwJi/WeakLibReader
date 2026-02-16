#define SIMPLE_CATCH_NO_MAIN
#include <catch2/catch_test_macros.hpp>

#include "interp/WeakLibReader_LogInterpolate.hpp"
#include "base/WeakLibReader_Layout.hpp"
#include "base/WeakLibReader_AxisTypes.hpp"

#include <array>
#include <cmath>
#include <cstddef>

namespace {
constexpr double Tol = 1.0e-12;
}

TEST_CASE("Point wrappers reject null axis/data pointers", "[loginterp][guards]")
{
  using namespace WeakLibReader;

  const std::array<double, 2> grid{1.0, 2.0};
  const std::array<double, 4> table{0.0, 0.0, 0.0, 0.0};

  Axis axes2[2] = {
      MakeAxis(grid.data(), 2, AxisScale::Linear),
      MakeAxis(grid.data(), 2, AxisScale::Linear)};

  const double nanValue = LogInterpolateSingleVariable2DCustomPoint(
      1.2, 1.4,
      axes2,
      nullptr,
      0.0);
  CHECK(std::isnan(nanValue));

  axes2[1].grid = nullptr;
  const double nanAxis = LogInterpolateSingleVariable2DCustomPoint(
      1.2, 1.4,
      axes2,
      table.data(),
      0.0);
  CHECK(std::isnan(nanAxis));
}

TEST_CASE("Batch wrappers reject null pointers", "[loginterp][guards][batch]")
{
  using namespace WeakLibReader;

  const std::array<double, 2> grid{1.0, 2.0};
  const std::array<double, 16> table{};
  const std::array<double, 2> coord{1.1, 1.8};
  std::array<double, 2> out{};

  Axis axes4[4] = {
      MakeAxis(grid.data(), 2, AxisScale::Linear),
      MakeAxis(grid.data(), 2, AxisScale::Linear),
      MakeAxis(grid.data(), 2, AxisScale::Linear),
      MakeAxis(grid.data(), 2, AxisScale::Linear)};

  CHECK(LogInterpolateSingleVariable1D3DCustom(
            coord.data(), coord.size(),
            coord.data(), coord.data(), coord.data(), coord.size(),
            axes4,
            nullptr,
            0.0,
            out.data()) == 1);

  axes4[2].grid = nullptr;
  CHECK(LogInterpolateSingleVariable2D2DCustom(
            coord.data(), coord.size(),
            coord.data(), coord.data(), coord.size(),
            axes4,
            table.data(),
            0.0,
            out.data()) == 1);
}

TEST_CASE("Derivative 2D2D point output remains symmetric", "[loginterp][derivative][symmetry]")
{
  using namespace WeakLibReader;

  constexpr std::size_t sizeE = 3;
  const std::array<double, sizeE> gridE{1.0, 2.0, 4.0};
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

  Axis axes4[4] = {
      MakeAxis(gridE.data(), static_cast<int>(sizeE), AxisScale::Linear),
      MakeAxis(gridE.data(), static_cast<int>(sizeE), AxisScale::Linear),
      MakeAxis(gridT.data(), 2, AxisScale::Linear),
      MakeAxis(gridX.data(), 2, AxisScale::Linear)};

  std::array<double, sizeE * sizeE> interp{};
  std::array<double, sizeE * sizeE> dT{};
  std::array<double, sizeE * sizeE> dX{};

  REQUIRE(LogInterpolateDifferentiateSingleVariable2D2DCustomPoint(
              gridE.data(), sizeE,
              1.3, 2.1,
              axes4,
              table.data(),
              0.0,
              interp.data(), dT.data(), dX.data()) == 0);

  for (std::size_t j = 0; j < sizeE; ++j) {
    for (std::size_t i = 0; i < sizeE; ++i) {
      CHECK(interp[j * sizeE + i] == Catch::Approx(interp[i * sizeE + j]).margin(Tol));
      CHECK(dT[j * sizeE + i] == Catch::Approx(dT[i * sizeE + j]).margin(Tol));
      CHECK(dX[j * sizeE + i] == Catch::Approx(dX[i * sizeE + j]).margin(Tol));
    }
  }
}

TEST_CASE("Layout strides are column-major from extents", "[layout][column-major]")
{
  using namespace WeakLibReader;

  const int extents[4] = {3, 4, 5, 6};
  const Layout layout = MakeLayout(extents, 4);

  CHECK(layout.stride[0] == 1);
  CHECK(layout.stride[1] == 3);
  CHECK(layout.stride[2] == 12);
  CHECK(layout.stride[3] == 60);
}
