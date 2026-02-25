#define SIMPLE_CATCH_NO_MAIN
#include <catch2/catch_test_macros.hpp>

#include "base/WeakLibReader_AxisTypes.hpp"
#include "base/WeakLibReader_InterpBasis.hpp"
#include "base/WeakLibReader_Layout.hpp"
#include "interp/WeakLibReader_InterpLogTable.hpp"
#include "interp/WeakLibReader_LogInterpolate.hpp"
#include "test_constants.hpp"

#include <array>
#include <cmath>

using test_constants::Tol;

TEST_CASE("4D log interpolation matches quadrilinear expectation",
          "[loginterp][4d]") {
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

  Axis axes[4] = {MakeAxis(gridA.data(), 2, AxisScale::Linear),
                  MakeAxis(gridB.data(), 2, AxisScale::Linear),
                  MakeAxis(gridC.data(), 2, AxisScale::Linear),
                  MakeAxis(gridD.data(), 2, AxisScale::Linear)};

  const double a = 1.5;
  const double b = 2.0;
  const double c = 0.4;
  const double d = 7.0;
  const double result = LogInterpolateSingleVariable4DCustomPoint(
      a, b, c, d, axes, table.data(), 0.0);

  // Verify by computing via 4D wrapper
  double coords[4] = {a, b, c, d};
  const double expected = detail::LogInterpolatedValueDirect<4>(
      table.data(), layout, axes, coords, 0.0);

  CHECK(result == Catch::Approx(expected).margin(Tol));
}

TEST_CASE("Batch 4D log interpolation matches point wrapper",
          "[loginterp][4d][batch]") {
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

  Axis axes[4] = {MakeAxis(gridA.data(), 2, AxisScale::Linear),
                  MakeAxis(gridB.data(), 2, AxisScale::Linear),
                  MakeAxis(gridC.data(), 2, AxisScale::Linear),
                  MakeAxis(gridD.data(), 2, AxisScale::Linear)};

  std::array<double, 3> x0{1.0, 1.5, 2.0};
  std::array<double, 3> x1{1.0, 2.0, 3.0};
  std::array<double, 3> x2{0.0, 0.5, 1.0};
  std::array<double, 3> x3{5.0, 7.0, 9.0};
  std::array<double, 3> out{};

  const int rc = LogInterpolateSingleVariable4DCustom(
      x0.data(), x1.data(), x2.data(), x3.data(), x0.size(), axes, table.data(),
      0.0, out.data());
  REQUIRE(rc == 0);

  for (std::size_t i = 0; i < x0.size(); ++i) {
    const double point = LogInterpolateSingleVariable4DCustomPoint(
        x0[i], x1[i], x2[i], x3[i], axes, table.data(), 0.0);
    CHECK(out[i] == Catch::Approx(point).margin(Tol));
  }
}

TEST_CASE("4D mixed axes log interpolation respects offset",
          "[loginterp][4d][offset]") {
  using namespace WeakLibReader;

  const std::array<double, 2> gridA{1.0, 10.0}; // log10 axis
  const std::array<double, 2> gridB{2.0, 20.0}; // log10 axis
  const std::array<double, 2> gridC{0.0, 1.0};  // linear axis
  const std::array<double, 2> gridD{5.0, 9.0};  // linear axis

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

  Axis axes[4] = {MakeAxis(gridA.data(), 2, AxisScale::Log10),
                  MakeAxis(gridB.data(), 2, AxisScale::Log10),
                  MakeAxis(gridC.data(), 2, AxisScale::Linear),
                  MakeAxis(gridD.data(), 2, AxisScale::Linear)};

  double coords[4] = {3.0, 5.0, 0.25, 6.0};

  int idxA = 0, idxB = 0, idxC = 0, idxD = 0;
  double fracA = 0.0, fracB = 0.0, fracC = 0.0, fracD = 0.0;
  IndexAndDeltaLog10(coords[0], gridA.data(), 2, idxA, fracA);
  IndexAndDeltaLog10(coords[1], gridB.data(), 2, idxB, fracB);
  IndexAndDeltaLin(coords[2], gridC.data(), 2, idxC, fracC);
  IndexAndDeltaLin(coords[3], gridD.data(), 2, idxD, fracD);

  const double expected =
      LinearInterp4DPoint(idxA, idxB, idxC, idxD, fracA, fracB, fracC, fracD,
                          offset, table.data(), layout);

  const double result = detail::LogInterpolatedValueDirect<4>(
      table.data(), layout, axes, coords, offset);

  CHECK(result == Catch::Approx(expected).margin(Tol));
}

TEST_CASE("4D mixed axes derivative matches kernel",
          "[loginterp][4d][derivative]") {
  using namespace WeakLibReader;

  const std::array<double, 2> gridA{1.0, 10.0}; // log10 axis
  const std::array<double, 2> gridB{2.0, 20.0}; // log10 axis
  const std::array<double, 2> gridC{0.0, 1.0};  // linear axis
  const std::array<double, 2> gridD{5.0, 9.0};  // linear axis

  const int extents[4] = {2, 2, 2, 2};
  const Layout layout = MakeLayout(extents, 4);

  const double offset = 0.75;
  std::array<double, 16> table{};
  for (int ia = 0; ia < 2; ++ia) {
    for (int ib = 0; ib < 2; ++ib) {
      for (int ic = 0; ic < 2; ++ic) {
        for (int id = 0; id < 2; ++id) {
          const double actual =
              1.5 + 0.15 * ia + 0.25 * ib + 0.35 * ic + 0.45 * id;
          table[layout.Offset(ia, ib, ic, id)] = std::log10(actual + offset);
        }
      }
    }
  }

  Axis axes[4] = {MakeAxis(gridA.data(), 2, AxisScale::Log10),
                  MakeAxis(gridB.data(), 2, AxisScale::Log10),
                  MakeAxis(gridC.data(), 2, AxisScale::Linear),
                  MakeAxis(gridD.data(), 2, AxisScale::Linear)};

  double coords[4] = {4.0, 6.0, 0.4, 7.0};

  int idxA = 0, idxB = 0, idxC = 0, idxD = 0;
  double fracA = 0.0, fracB = 0.0, fracC = 0.0, fracD = 0.0;
  IndexAndDeltaLog10(coords[0], gridA.data(), 2, idxA, fracA);
  IndexAndDeltaLog10(coords[1], gridB.data(), 2, idxB, fracB);
  IndexAndDeltaLin(coords[2], gridC.data(), 2, idxC, fracC);
  IndexAndDeltaLin(coords[3], gridD.data(), 2, idxD, fracD);

  const double aA =
      1.0 / (coords[0] * WeakLibReader::math::Log10(gridA[1] / gridA[0]));
  const double aB =
      1.0 / (coords[1] * WeakLibReader::math::Log10(gridB[1] / gridB[0]));
  const double aC = WeakLibReader::math::Ln10 / (gridC[1] - gridC[0]);
  const double aD = WeakLibReader::math::Ln10 / (gridD[1] - gridD[0]);

  double expectedInterp = 0.0;
  double expectedDA = 0.0;
  double expectedDB = 0.0;
  double expectedDC = 0.0;
  double expectedDD = 0.0;
  LinearInterpDeriv4DPoint(idxA, idxB, idxC, idxD, fracA, fracB, fracC, fracD,
                           aA, aB, aC, aD, offset, table.data(), layout,
                           expectedInterp, expectedDA, expectedDB, expectedDC,
                           expectedDD);

  double interpolant = 0.0;
  double deriv[4] = {0.0, 0.0, 0.0, 0.0};
  detail::LogInterpolatedDerivativeDirect<4>(table.data(), layout, axes, coords,
                                             offset, interpolant, deriv);

  CHECK(interpolant == Catch::Approx(expectedInterp).margin(Tol));
  CHECK(deriv[0] == Catch::Approx(expectedDA).margin(Tol));
  CHECK(deriv[1] == Catch::Approx(expectedDB).margin(Tol));
  CHECK(deriv[2] == Catch::Approx(expectedDC).margin(Tol));
  CHECK(deriv[3] == Catch::Approx(expectedDD).margin(Tol));
}

TEST_CASE("5D log interpolation via LinearInterp5DPoint", "[loginterp][5d]") {
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
            table[layout.Offset(i0, i1, i2, i3, i4)] = std::log10(
                actual(grid0[i0], grid1[i1], grid2[i2], grid3[i3], grid4[i4]));
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

  const double result = LinearInterp5DPoint(0, 0, 0, 0, 0, d0, d1, d2, d3, d4,
                                            0.0, table.data(), layout);

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
      p00000, p10000, p01000, p11000, p00100, p10100, p01100, p11100, p00010,
      p10010, p01010, p11010, p00110, p10110, p01110, p11110, p00001, p10001,
      p01001, p11001, p00101, p10101, p01101, p11101, p00011, p10011, p01011,
      p11011, p00111, p10111, p01111, p11111, d0, d1, d2, d3, d4);
  const double expected = std::pow(10.0, logExpected);

  CHECK(result == Catch::Approx(expected).margin(Tol));
}

TEST_CASE("PentaLinear reduces to TetraLinear at boundaries",
          "[loginterp][5d][pentalinear]") {
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
      p00000, p10000, p01000, p11000, p00100, p10100, p01100, p11100, p00010,
      p10010, p01010, p11010, p00110, p10110, p01110, p11110, p00001, p10001,
      p01001, p11001, p00101, p10101, p01101, p11101, p00011, p10011, p01011,
      p11011, p00111, p10111, p01111, p11111, dX1, dX2, dX3, dX4, 0.0);

  const double tetraLo =
      TetraLinear(p00000, p10000, p01000, p11000, p00100, p10100, p01100,
                  p11100, p00010, p10010, p01010, p11010, p00110, p10110,
                  p01110, p11110, dX1, dX2, dX3, dX4);

  CHECK(penta0 == Catch::Approx(tetraLo).margin(Tol));

  // dX5 = 1 should give the "hi" tetralinear result
  const double penta1 = PentaLinear(
      p00000, p10000, p01000, p11000, p00100, p10100, p01100, p11100, p00010,
      p10010, p01010, p11010, p00110, p10110, p01110, p11110, p00001, p10001,
      p01001, p11001, p00101, p10101, p01101, p11101, p00011, p10011, p01011,
      p11011, p00111, p10111, p01111, p11111, dX1, dX2, dX3, dX4, 1.0);

  const double tetraHi =
      TetraLinear(p00001, p10001, p01001, p11001, p00101, p10101, p01101,
                  p11101, p00011, p10011, p01011, p11011, p00111, p10111,
                  p01111, p11111, dX1, dX2, dX3, dX4);

  CHECK(penta1 == Catch::Approx(tetraHi).margin(Tol));
}

TEST_CASE("1D3D sweep batch matches direct interpolation",
          "[loginterp][4d][batch]") {
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

  Axis axes[4] = {MakeAxis(gridE.data(), 2, AxisScale::Linear),
                  MakeAxis(gridD.data(), 2, AxisScale::Linear),
                  MakeAxis(gridT.data(), 2, AxisScale::Linear),
                  MakeAxis(gridY.data(), 2, AxisScale::Linear)};

  const int rc = LogInterpolateSingleVariable1D3DCustom(
      logE.data(), sizeE, logD.data(), logT.data(), y.data(), count, axes,
      table.data(), 0.0, out.data());
  REQUIRE(rc == 0);

  for (std::size_t j = 0; j < count; ++j) {
    for (std::size_t i = 0; i < sizeE; ++i) {
      double coords[4] = {logE[i], logD[j], logT[j], y[j]};
      const double expected = detail::LogInterpolatedValueDirect<4>(
          table.data(), layout, axes, coords, 0.0);
      const std::size_t idx = j * sizeE + i;
      CHECK(out[idx] == Catch::Approx(expected).margin(Tol));
    }
  }
}

TEST_CASE("1D3D single point matches batch with count=1",
          "[loginterp][4d][1d3d]") {
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

  Axis axes[4] = {MakeAxis(gridE.data(), 2, AxisScale::Linear),
                  MakeAxis(gridD.data(), 2, AxisScale::Linear),
                  MakeAxis(gridT.data(), 2, AxisScale::Linear),
                  MakeAxis(gridY.data(), 2, AxisScale::Linear)};

  std::array<double, sizeE> logE{1.0, 1.5, 2.0};
  const double logD = 2.0;
  const double logT = 15.0;
  const double y = 0.5;

  std::array<double, sizeE> outPoint{};
  const int rcPoint = LogInterpolateSingleVariable1D3DCustomPoint(
      logE.data(), sizeE, logD, logT, y, axes, table.data(), 0.0,
      outPoint.data());
  REQUIRE(rcPoint == 0);

  // Compare against batch version with count=1
  std::array<double, 1> logDArr{logD};
  std::array<double, 1> logTArr{logT};
  std::array<double, 1> yArr{y};
  std::array<double, sizeE> outBatch{};

  const int rcBatch = LogInterpolateSingleVariable1D3DCustom(
      logE.data(), sizeE, logDArr.data(), logTArr.data(), yArr.data(), 1, axes,
      table.data(), 0.0, outBatch.data());
  REQUIRE(rcBatch == 0);

  for (std::size_t i = 0; i < sizeE; ++i) {
    CHECK(outPoint[i] == Catch::Approx(outBatch[i]).margin(Tol));
  }
}
