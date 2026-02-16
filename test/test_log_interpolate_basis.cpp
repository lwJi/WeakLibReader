#define SIMPLE_CATCH_NO_MAIN
#include <catch2/catch_test_macros.hpp>

#include "base/WeakLibReader_InterpBasis.hpp"

#include <cmath>

namespace {

constexpr double Tol = 1.0e-12;

} // namespace

// =============================================================================
// Value Basis Function Tests (InterpBasis.hpp)
// =============================================================================

TEST_CASE("Linear interpolates between two values", "[basis][value][1d]")
{
  using namespace WeakLibReader;

  const double p0 = 2.0, p1 = 8.0;

  // At endpoints
  CHECK(Linear(p0, p1, 0.0) == Catch::Approx(p0).margin(Tol));
  CHECK(Linear(p0, p1, 1.0) == Catch::Approx(p1).margin(Tol));

  // At midpoint
  CHECK(Linear(p0, p1, 0.5) == Catch::Approx((p0 + p1) / 2.0).margin(Tol));

  // At known point: Linear(2.0, 8.0, 0.25) = 0.75*2 + 0.25*8 = 3.5
  CHECK(Linear(2.0, 8.0, 0.25) == Catch::Approx(3.5).margin(Tol));
}

TEST_CASE("BiLinear interpolates on 2D cell", "[basis][value][2d]")
{
  using namespace WeakLibReader;

  const double p00 = 1.0, p10 = 3.0, p01 = 5.0, p11 = 7.0;

  // At all 4 corners
  CHECK(BiLinear(p00, p10, p01, p11, 0.0, 0.0) == Catch::Approx(p00).margin(Tol));
  CHECK(BiLinear(p00, p10, p01, p11, 1.0, 0.0) == Catch::Approx(p10).margin(Tol));
  CHECK(BiLinear(p00, p10, p01, p11, 0.0, 1.0) == Catch::Approx(p01).margin(Tol));
  CHECK(BiLinear(p00, p10, p01, p11, 1.0, 1.0) == Catch::Approx(p11).margin(Tol));

  // At center (0.5, 0.5): average of 4 corners
  const double avg = (p00 + p10 + p01 + p11) / 4.0;
  CHECK(BiLinear(p00, p10, p01, p11, 0.5, 0.5) == Catch::Approx(avg).margin(Tol));

  // Hand-computed: (1-dX1)(1-dX2)*p00 + dX1*(1-dX2)*p10 + (1-dX1)*dX2*p01 + dX1*dX2*p11
  const double dX1 = 0.3, dX2 = 0.7;
  const double expected = (1.0 - dX1) * (1.0 - dX2) * p00 +
                          dX1 * (1.0 - dX2) * p10 +
                          (1.0 - dX1) * dX2 * p01 +
                          dX1 * dX2 * p11;
  CHECK(BiLinear(p00, p10, p01, p11, dX1, dX2) == Catch::Approx(expected).margin(Tol));
}

TEST_CASE("TriLinear interpolates on 3D cell", "[basis][value][3d]")
{
  using namespace WeakLibReader;

  const double p000 = 1.0, p100 = 2.0, p010 = 1.5, p110 = 3.0;
  const double p001 = 2.5, p101 = 3.5, p011 = 2.0, p111 = 4.0;

  // At all 8 corners
  CHECK(TriLinear(p000, p100, p010, p110, p001, p101, p011, p111,
                  0.0, 0.0, 0.0) == Catch::Approx(p000).margin(Tol));
  CHECK(TriLinear(p000, p100, p010, p110, p001, p101, p011, p111,
                  1.0, 0.0, 0.0) == Catch::Approx(p100).margin(Tol));
  CHECK(TriLinear(p000, p100, p010, p110, p001, p101, p011, p111,
                  0.0, 1.0, 0.0) == Catch::Approx(p010).margin(Tol));
  CHECK(TriLinear(p000, p100, p010, p110, p001, p101, p011, p111,
                  1.0, 1.0, 0.0) == Catch::Approx(p110).margin(Tol));
  CHECK(TriLinear(p000, p100, p010, p110, p001, p101, p011, p111,
                  0.0, 0.0, 1.0) == Catch::Approx(p001).margin(Tol));
  CHECK(TriLinear(p000, p100, p010, p110, p001, p101, p011, p111,
                  1.0, 0.0, 1.0) == Catch::Approx(p101).margin(Tol));
  CHECK(TriLinear(p000, p100, p010, p110, p001, p101, p011, p111,
                  0.0, 1.0, 1.0) == Catch::Approx(p011).margin(Tol));
  CHECK(TriLinear(p000, p100, p010, p110, p001, p101, p011, p111,
                  1.0, 1.0, 1.0) == Catch::Approx(p111).margin(Tol));

  // At center (0.5, 0.5, 0.5): average of all 8 corners
  const double avg = (p000 + p100 + p010 + p110 + p001 + p101 + p011 + p111) / 8.0;
  CHECK(TriLinear(p000, p100, p010, p110, p001, p101, p011, p111,
                  0.5, 0.5, 0.5) == Catch::Approx(avg).margin(Tol));

  // Dimensional reduction: TriLinear(..., dX1, dX2, 0.0) == BiLinear(p000, p100, p010, p110, dX1, dX2)
  for (double dX1 : {0.2, 0.5, 0.8}) {
    for (double dX2 : {0.2, 0.5, 0.8}) {
      const double tri = TriLinear(p000, p100, p010, p110, p001, p101, p011, p111,
                                   dX1, dX2, 0.0);
      const double bi = BiLinear(p000, p100, p010, p110, dX1, dX2);
      CHECK(tri == Catch::Approx(bi).margin(Tol));
    }
  }
}

TEST_CASE("TetraLinear interpolates on 4D cell", "[basis][value][4d]")
{
  using namespace WeakLibReader;

  // 16 corner values for 4D hypercube
  const double p0000 = 1.0, p1000 = 2.0, p0100 = 1.5, p1100 = 3.0;
  const double p0010 = 2.5, p1010 = 3.5, p0110 = 2.0, p1110 = 4.0;
  const double p0001 = 1.2, p1001 = 2.2, p0101 = 1.7, p1101 = 3.2;
  const double p0011 = 2.7, p1011 = 3.7, p0111 = 2.2, p1111 = 4.2;

  // Store corners in array indexed by binary encoding: bit0=X1, bit1=X2, bit2=X3, bit3=X4
  const double corners[16] = {
      p0000, p1000, p0100, p1100, p0010, p1010, p0110, p1110,
      p0001, p1001, p0101, p1101, p0011, p1011, p0111, p1111};

  // At all 16 corners: dX values are 0 or 1 from binary encoding
  for (int c = 0; c < 16; ++c) {
    const double dX1 = (c & 1) ? 1.0 : 0.0;
    const double dX2 = (c & 2) ? 1.0 : 0.0;
    const double dX3 = (c & 4) ? 1.0 : 0.0;
    const double dX4 = (c & 8) ? 1.0 : 0.0;
    CHECK(TetraLinear(p0000, p1000, p0100, p1100,
                      p0010, p1010, p0110, p1110,
                      p0001, p1001, p0101, p1101,
                      p0011, p1011, p0111, p1111,
                      dX1, dX2, dX3, dX4) == Catch::Approx(corners[c]).margin(Tol));
  }

  // Dimensional reduction: TetraLinear(..., dX1, dX2, dX3, 0.0) == TriLinear(lo-face, dX1, dX2, dX3)
  for (double dX1 : {0.2, 0.5, 0.8}) {
    for (double dX2 : {0.3, 0.7}) {
      for (double dX3 : {0.4, 0.6}) {
        const double tetra = TetraLinear(p0000, p1000, p0100, p1100,
                                         p0010, p1010, p0110, p1110,
                                         p0001, p1001, p0101, p1101,
                                         p0011, p1011, p0111, p1111,
                                         dX1, dX2, dX3, 0.0);
        const double tri = TriLinear(p0000, p1000, p0100, p1100,
                                     p0010, p1010, p0110, p1110,
                                     dX1, dX2, dX3);
        CHECK(tetra == Catch::Approx(tri).margin(Tol));
      }
    }
  }
}

// =============================================================================
// Derivative Basis Function Tests (InterpBasis.hpp)
// =============================================================================

TEST_CASE("BiLinearDerivativeX1 matches analytical formula", "[basis][deriv][2d]")
{
  using namespace WeakLibReader;

  // Test data: 2x2 grid values
  const double p00 = 1.0, p10 = 2.0, p01 = 3.0, p11 = 5.0;

  // Test at several dX2 values
  for (double dX2 : {0.0, 0.25, 0.5, 0.75, 1.0}) {
    const double result = BiLinearDerivativeX1(p00, p10, p01, p11, dX2);
    // Analytical: d/dX1[BiLinear] = Linear(p10, p11, dX2) - Linear(p00, p01, dX2)
    const double expected = Linear(p10, p11, dX2) - Linear(p00, p01, dX2);
    CHECK(result == Catch::Approx(expected).margin(Tol));
  }
}

TEST_CASE("BiLinearDerivativeX1 matches numerical differentiation", "[basis][deriv][2d][numerical]")
{
  using namespace WeakLibReader;

  const double p00 = 1.0, p10 = 2.0, p01 = 3.0, p11 = 5.0;
  const double dX2 = 0.3;
  const double h = 1.0e-7;

  // Numerical derivative: [BiLinear(dX1+h) - BiLinear(dX1-h)] / (2h)
  for (double dX1 : {0.2, 0.5, 0.8}) {
    const double fPlus = BiLinear(p00, p10, p01, p11, dX1 + h, dX2);
    const double fMinus = BiLinear(p00, p10, p01, p11, dX1 - h, dX2);
    const double numerical = (fPlus - fMinus) / (2.0 * h);
    const double analytical = BiLinearDerivativeX1(p00, p10, p01, p11, dX2);
    CHECK(analytical == Catch::Approx(numerical).epsilon(1.0e-6));
  }
}

TEST_CASE("BiLinearDerivativeX2 matches analytical formula", "[basis][deriv][2d]")
{
  using namespace WeakLibReader;

  const double p00 = 1.0, p10 = 2.0, p01 = 3.0, p11 = 5.0;

  for (double dX1 : {0.0, 0.25, 0.5, 0.75, 1.0}) {
    const double result = BiLinearDerivativeX2(p00, p10, p01, p11, dX1);
    // Analytical: d/dX2[BiLinear] = Linear(p01, p11, dX1) - Linear(p00, p10, dX1)
    const double expected = Linear(p01, p11, dX1) - Linear(p00, p10, dX1);
    CHECK(result == Catch::Approx(expected).margin(Tol));
  }
}

TEST_CASE("BiLinearDerivativeX2 matches numerical differentiation", "[basis][deriv][2d][numerical]")
{
  using namespace WeakLibReader;

  const double p00 = 1.0, p10 = 2.0, p01 = 3.0, p11 = 5.0;
  const double dX1 = 0.4;
  const double h = 1.0e-7;

  for (double dX2 : {0.2, 0.5, 0.8}) {
    const double fPlus = BiLinear(p00, p10, p01, p11, dX1, dX2 + h);
    const double fMinus = BiLinear(p00, p10, p01, p11, dX1, dX2 - h);
    const double numerical = (fPlus - fMinus) / (2.0 * h);
    const double analytical = BiLinearDerivativeX2(p00, p10, p01, p11, dX1);
    CHECK(analytical == Catch::Approx(numerical).epsilon(1.0e-6));
  }
}

TEST_CASE("TriLinearDerivativeX1 matches analytical formula", "[basis][deriv][3d]")
{
  using namespace WeakLibReader;

  const double p000 = 1.0, p100 = 2.0, p010 = 1.5, p110 = 3.0;
  const double p001 = 2.5, p101 = 3.5, p011 = 2.0, p111 = 4.0;

  for (double dX2 : {0.0, 0.5, 1.0}) {
    for (double dX3 : {0.0, 0.5, 1.0}) {
      const double result = TriLinearDerivativeX1(p000, p100, p010, p110,
                                                   p001, p101, p011, p111,
                                                   dX2, dX3);
      // Analytical: BiLinear(p100,p110,p101,p111) - BiLinear(p000,p010,p001,p011)
      const double expected = BiLinear(p100, p110, p101, p111, dX2, dX3) -
                              BiLinear(p000, p010, p001, p011, dX2, dX3);
      CHECK(result == Catch::Approx(expected).margin(Tol));
    }
  }
}

TEST_CASE("TriLinearDerivativeX1 matches numerical differentiation", "[basis][deriv][3d][numerical]")
{
  using namespace WeakLibReader;

  const double p000 = 1.0, p100 = 2.0, p010 = 1.5, p110 = 3.0;
  const double p001 = 2.5, p101 = 3.5, p011 = 2.0, p111 = 4.0;
  const double dX2 = 0.3, dX3 = 0.6;
  const double h = 1.0e-7;

  for (double dX1 : {0.2, 0.5, 0.8}) {
    const double fPlus = TriLinear(p000, p100, p010, p110, p001, p101, p011, p111,
                                   dX1 + h, dX2, dX3);
    const double fMinus = TriLinear(p000, p100, p010, p110, p001, p101, p011, p111,
                                    dX1 - h, dX2, dX3);
    const double numerical = (fPlus - fMinus) / (2.0 * h);
    const double analytical = TriLinearDerivativeX1(p000, p100, p010, p110,
                                                    p001, p101, p011, p111,
                                                    dX2, dX3);
    CHECK(analytical == Catch::Approx(numerical).epsilon(1.0e-6));
  }
}

TEST_CASE("TriLinearDerivativeX2 matches analytical formula", "[basis][deriv][3d]")
{
  using namespace WeakLibReader;

  const double p000 = 1.0, p100 = 2.0, p010 = 1.5, p110 = 3.0;
  const double p001 = 2.5, p101 = 3.5, p011 = 2.0, p111 = 4.0;

  for (double dX1 : {0.0, 0.5, 1.0}) {
    for (double dX3 : {0.0, 0.5, 1.0}) {
      const double result = TriLinearDerivativeX2(p000, p100, p010, p110,
                                                   p001, p101, p011, p111,
                                                   dX1, dX3);
      const double expected = BiLinear(p010, p110, p011, p111, dX1, dX3) -
                              BiLinear(p000, p100, p001, p101, dX1, dX3);
      CHECK(result == Catch::Approx(expected).margin(Tol));
    }
  }
}

TEST_CASE("TriLinearDerivativeX2 matches numerical differentiation", "[basis][deriv][3d][numerical]")
{
  using namespace WeakLibReader;

  const double p000 = 1.0, p100 = 2.0, p010 = 1.5, p110 = 3.0;
  const double p001 = 2.5, p101 = 3.5, p011 = 2.0, p111 = 4.0;
  const double dX1 = 0.4, dX3 = 0.7;
  const double h = 1.0e-7;

  for (double dX2 : {0.2, 0.5, 0.8}) {
    const double fPlus = TriLinear(p000, p100, p010, p110, p001, p101, p011, p111,
                                   dX1, dX2 + h, dX3);
    const double fMinus = TriLinear(p000, p100, p010, p110, p001, p101, p011, p111,
                                    dX1, dX2 - h, dX3);
    const double numerical = (fPlus - fMinus) / (2.0 * h);
    const double analytical = TriLinearDerivativeX2(p000, p100, p010, p110,
                                                    p001, p101, p011, p111,
                                                    dX1, dX3);
    CHECK(analytical == Catch::Approx(numerical).epsilon(1.0e-6));
  }
}

TEST_CASE("TriLinearDerivativeX3 matches analytical formula", "[basis][deriv][3d]")
{
  using namespace WeakLibReader;

  const double p000 = 1.0, p100 = 2.0, p010 = 1.5, p110 = 3.0;
  const double p001 = 2.5, p101 = 3.5, p011 = 2.0, p111 = 4.0;

  for (double dX1 : {0.0, 0.5, 1.0}) {
    for (double dX2 : {0.0, 0.5, 1.0}) {
      const double result = TriLinearDerivativeX3(p000, p100, p010, p110,
                                                   p001, p101, p011, p111,
                                                   dX1, dX2);
      const double expected = BiLinear(p001, p101, p011, p111, dX1, dX2) -
                              BiLinear(p000, p100, p010, p110, dX1, dX2);
      CHECK(result == Catch::Approx(expected).margin(Tol));
    }
  }
}

TEST_CASE("TriLinearDerivativeX3 matches numerical differentiation", "[basis][deriv][3d][numerical]")
{
  using namespace WeakLibReader;

  const double p000 = 1.0, p100 = 2.0, p010 = 1.5, p110 = 3.0;
  const double p001 = 2.5, p101 = 3.5, p011 = 2.0, p111 = 4.0;
  const double dX1 = 0.35, dX2 = 0.65;
  const double h = 1.0e-7;

  for (double dX3 : {0.2, 0.5, 0.8}) {
    const double fPlus = TriLinear(p000, p100, p010, p110, p001, p101, p011, p111,
                                   dX1, dX2, dX3 + h);
    const double fMinus = TriLinear(p000, p100, p010, p110, p001, p101, p011, p111,
                                    dX1, dX2, dX3 - h);
    const double numerical = (fPlus - fMinus) / (2.0 * h);
    const double analytical = TriLinearDerivativeX3(p000, p100, p010, p110,
                                                    p001, p101, p011, p111,
                                                    dX1, dX2);
    CHECK(analytical == Catch::Approx(numerical).epsilon(1.0e-6));
  }
}

TEST_CASE("TetraLinearDerivativeX1 matches numerical differentiation", "[basis][deriv][4d][numerical]")
{
  using namespace WeakLibReader;

  // 16 corner values for 4D hypercube
  const double p0000 = 1.0, p1000 = 2.0, p0100 = 1.5, p1100 = 3.0;
  const double p0010 = 2.5, p1010 = 3.5, p0110 = 2.0, p1110 = 4.0;
  const double p0001 = 1.2, p1001 = 2.2, p0101 = 1.7, p1101 = 3.2;
  const double p0011 = 2.7, p1011 = 3.7, p0111 = 2.2, p1111 = 4.2;

  const double dX2 = 0.3, dX3 = 0.5, dX4 = 0.7;
  const double h = 1.0e-7;

  for (double dX1 : {0.2, 0.5, 0.8}) {
    const double fPlus = TetraLinear(p0000, p1000, p0100, p1100,
                                     p0010, p1010, p0110, p1110,
                                     p0001, p1001, p0101, p1101,
                                     p0011, p1011, p0111, p1111,
                                     dX1 + h, dX2, dX3, dX4);
    const double fMinus = TetraLinear(p0000, p1000, p0100, p1100,
                                      p0010, p1010, p0110, p1110,
                                      p0001, p1001, p0101, p1101,
                                      p0011, p1011, p0111, p1111,
                                      dX1 - h, dX2, dX3, dX4);
    const double numerical = (fPlus - fMinus) / (2.0 * h);
    const double analytical = TetraLinearDerivativeX1(p0000, p1000, p0100, p1100,
                                                      p0010, p1010, p0110, p1110,
                                                      p0001, p1001, p0101, p1101,
                                                      p0011, p1011, p0111, p1111,
                                                      dX2, dX3, dX4);
    CHECK(analytical == Catch::Approx(numerical).epsilon(1.0e-6));
  }
}

TEST_CASE("TetraLinearDerivativeX2 matches numerical differentiation", "[basis][deriv][4d][numerical]")
{
  using namespace WeakLibReader;

  const double p0000 = 1.0, p1000 = 2.0, p0100 = 1.5, p1100 = 3.0;
  const double p0010 = 2.5, p1010 = 3.5, p0110 = 2.0, p1110 = 4.0;
  const double p0001 = 1.2, p1001 = 2.2, p0101 = 1.7, p1101 = 3.2;
  const double p0011 = 2.7, p1011 = 3.7, p0111 = 2.2, p1111 = 4.2;

  const double dX1 = 0.4, dX3 = 0.6, dX4 = 0.3;
  const double h = 1.0e-7;

  for (double dX2 : {0.2, 0.5, 0.8}) {
    const double fPlus = TetraLinear(p0000, p1000, p0100, p1100,
                                     p0010, p1010, p0110, p1110,
                                     p0001, p1001, p0101, p1101,
                                     p0011, p1011, p0111, p1111,
                                     dX1, dX2 + h, dX3, dX4);
    const double fMinus = TetraLinear(p0000, p1000, p0100, p1100,
                                      p0010, p1010, p0110, p1110,
                                      p0001, p1001, p0101, p1101,
                                      p0011, p1011, p0111, p1111,
                                      dX1, dX2 - h, dX3, dX4);
    const double numerical = (fPlus - fMinus) / (2.0 * h);
    const double analytical = TetraLinearDerivativeX2(p0000, p1000, p0100, p1100,
                                                      p0010, p1010, p0110, p1110,
                                                      p0001, p1001, p0101, p1101,
                                                      p0011, p1011, p0111, p1111,
                                                      dX1, dX3, dX4);
    CHECK(analytical == Catch::Approx(numerical).epsilon(1.0e-6));
  }
}

TEST_CASE("TetraLinearDerivativeX3 matches numerical differentiation", "[basis][deriv][4d][numerical]")
{
  using namespace WeakLibReader;

  const double p0000 = 1.0, p1000 = 2.0, p0100 = 1.5, p1100 = 3.0;
  const double p0010 = 2.5, p1010 = 3.5, p0110 = 2.0, p1110 = 4.0;
  const double p0001 = 1.2, p1001 = 2.2, p0101 = 1.7, p1101 = 3.2;
  const double p0011 = 2.7, p1011 = 3.7, p0111 = 2.2, p1111 = 4.2;

  const double dX1 = 0.35, dX2 = 0.55, dX4 = 0.45;
  const double h = 1.0e-7;

  for (double dX3 : {0.2, 0.5, 0.8}) {
    const double fPlus = TetraLinear(p0000, p1000, p0100, p1100,
                                     p0010, p1010, p0110, p1110,
                                     p0001, p1001, p0101, p1101,
                                     p0011, p1011, p0111, p1111,
                                     dX1, dX2, dX3 + h, dX4);
    const double fMinus = TetraLinear(p0000, p1000, p0100, p1100,
                                      p0010, p1010, p0110, p1110,
                                      p0001, p1001, p0101, p1101,
                                      p0011, p1011, p0111, p1111,
                                      dX1, dX2, dX3 - h, dX4);
    const double numerical = (fPlus - fMinus) / (2.0 * h);
    const double analytical = TetraLinearDerivativeX3(p0000, p1000, p0100, p1100,
                                                      p0010, p1010, p0110, p1110,
                                                      p0001, p1001, p0101, p1101,
                                                      p0011, p1011, p0111, p1111,
                                                      dX1, dX2, dX4);
    CHECK(analytical == Catch::Approx(numerical).epsilon(1.0e-6));
  }
}

TEST_CASE("TetraLinearDerivativeX4 matches numerical differentiation", "[basis][deriv][4d][numerical]")
{
  using namespace WeakLibReader;

  const double p0000 = 1.0, p1000 = 2.0, p0100 = 1.5, p1100 = 3.0;
  const double p0010 = 2.5, p1010 = 3.5, p0110 = 2.0, p1110 = 4.0;
  const double p0001 = 1.2, p1001 = 2.2, p0101 = 1.7, p1101 = 3.2;
  const double p0011 = 2.7, p1011 = 3.7, p0111 = 2.2, p1111 = 4.2;

  const double dX1 = 0.25, dX2 = 0.45, dX3 = 0.65;
  const double h = 1.0e-7;

  for (double dX4 : {0.2, 0.5, 0.8}) {
    const double fPlus = TetraLinear(p0000, p1000, p0100, p1100,
                                     p0010, p1010, p0110, p1110,
                                     p0001, p1001, p0101, p1101,
                                     p0011, p1011, p0111, p1111,
                                     dX1, dX2, dX3, dX4 + h);
    const double fMinus = TetraLinear(p0000, p1000, p0100, p1100,
                                      p0010, p1010, p0110, p1110,
                                      p0001, p1001, p0101, p1101,
                                      p0011, p1011, p0111, p1111,
                                      dX1, dX2, dX3, dX4 - h);
    const double numerical = (fPlus - fMinus) / (2.0 * h);
    const double analytical = TetraLinearDerivativeX4(p0000, p1000, p0100, p1100,
                                                      p0010, p1010, p0110, p1110,
                                                      p0001, p1001, p0101, p1101,
                                                      p0011, p1011, p0111, p1111,
                                                      dX1, dX2, dX3);
    CHECK(analytical == Catch::Approx(numerical).epsilon(1.0e-6));
  }
}
