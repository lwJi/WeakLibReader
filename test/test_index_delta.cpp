#define SIMPLE_CATCH_NO_MAIN
#include <catch2/catch_test_macros.hpp>

#include "base/WeakLibReader_IndexDelta.hpp"
#include "test_constants.hpp"

#include <cmath>

using test_constants::Tol;

// =============================================================================
// ClampIndex Tests
// =============================================================================

TEST_CASE("ClampIndex clamps below and above", "[indexdelta]")
{
  using WeakLibReader::detail::ClampIndex;

  // Below range
  CHECK(ClampIndex(-1, 5) == 0);
  CHECK(ClampIndex(-100, 5) == 0);

  // Above range
  CHECK(ClampIndex(6, 5) == 5);
  CHECK(ClampIndex(100, 5) == 5);

  // In range (no-op)
  CHECK(ClampIndex(3, 5) == 3);

  // Boundaries
  CHECK(ClampIndex(0, 5) == 0);
  CHECK(ClampIndex(5, 5) == 5);
}

// =============================================================================
// IndexAndDeltaLin Tests
// =============================================================================

TEST_CASE("IndexAndDeltaLin at exact grid points", "[indexdelta][linear]")
{
  using WeakLibReader::IndexAndDeltaLin;

  const double grid[] = {1.0, 2.0, 3.0, 4.0, 5.0};
  const int n = 5;
  int i;
  double t;

  // At grid[0]: i=0, delta=0
  IndexAndDeltaLin(1.0, grid, n, i, t);
  CHECK(i == 0);
  CHECK(t == Catch::Approx(0.0).margin(Tol));

  // At grid[1]: i=1, delta=0
  IndexAndDeltaLin(2.0, grid, n, i, t);
  CHECK(i == 1);
  CHECK(t == Catch::Approx(0.0).margin(Tol));

  // At grid[2]: i=2, delta=0
  IndexAndDeltaLin(3.0, grid, n, i, t);
  CHECK(i == 2);
  CHECK(t == Catch::Approx(0.0).margin(Tol));

  // At grid[3]: i=3, delta=0
  IndexAndDeltaLin(4.0, grid, n, i, t);
  CHECK(i == 3);
  CHECK(t == Catch::Approx(0.0).margin(Tol));

  // At grid[4] (last point): i = n-2 = 3, delta = 1.0
  IndexAndDeltaLin(5.0, grid, n, i, t);
  CHECK(i == 3);
  CHECK(t == Catch::Approx(1.0).margin(Tol));
}

TEST_CASE("IndexAndDeltaLin at midpoints", "[indexdelta][linear]")
{
  using WeakLibReader::IndexAndDeltaLin;

  const double grid[] = {1.0, 2.0, 3.0, 4.0, 5.0};
  const int n = 5;
  int i;
  double t;

  // At x=1.5: midpoint of [1, 2]
  IndexAndDeltaLin(1.5, grid, n, i, t);
  CHECK(i == 0);
  CHECK(t == Catch::Approx(0.5).margin(Tol));

  // At x=3.5: midpoint of [3, 4]
  IndexAndDeltaLin(3.5, grid, n, i, t);
  CHECK(i == 2);
  CHECK(t == Catch::Approx(0.5).margin(Tol));
}

TEST_CASE("IndexAndDeltaLin below range (extrapolation)", "[indexdelta][linear]")
{
  using WeakLibReader::IndexAndDeltaLin;

  const double grid[] = {1.0, 2.0, 3.0, 4.0, 5.0};
  const int n = 5;
  int i;
  double t;

  // At x=0.5 (below grid[0]=1.0)
  IndexAndDeltaLin(0.5, grid, n, i, t);
  CHECK(i == 0);
  CHECK(t < 0.0);
  // delta = (0.5 - 1.0) / (2.0 - 1.0) = -0.5
  CHECK(t == Catch::Approx(-0.5).margin(Tol));
}

TEST_CASE("IndexAndDeltaLin above range (extrapolation)", "[indexdelta][linear]")
{
  using WeakLibReader::IndexAndDeltaLin;

  const double grid[] = {1.0, 2.0, 3.0, 4.0, 5.0};
  const int n = 5;
  int i;
  double t;

  // At x=5.5 (above grid[4]=5.0): i clamped to n-2=3
  IndexAndDeltaLin(5.5, grid, n, i, t);
  CHECK(i == 3);
  CHECK(t > 1.0);
  // delta = (5.5 - 4.0) / (5.0 - 4.0) = 1.5
  CHECK(t == Catch::Approx(1.5).margin(Tol));
}

// =============================================================================
// IndexAndDeltaLog10 Tests
// =============================================================================

TEST_CASE("IndexAndDeltaLog10 at exact grid points", "[indexdelta][log10]")
{
  using WeakLibReader::IndexAndDeltaLog10;

  const double grid[] = {1.0, 10.0, 100.0, 1000.0};
  const int n = 4;
  int i;
  double t;

  // At grid[0]=1.0: i=0, delta=0
  IndexAndDeltaLog10(1.0, grid, n, i, t);
  CHECK(i == 0);
  CHECK(t == Catch::Approx(0.0).margin(Tol));

  // At grid[1]=10.0: i=1, delta=0
  IndexAndDeltaLog10(10.0, grid, n, i, t);
  CHECK(i == 1);
  CHECK(t == Catch::Approx(0.0).margin(Tol));

  // At grid[2]=100.0: i=2, delta=0
  IndexAndDeltaLog10(100.0, grid, n, i, t);
  CHECK(i == 2);
  CHECK(t == Catch::Approx(0.0).margin(Tol));

  // At grid[3]=1000.0 (last): i = n-2 = 2, delta = 1.0
  IndexAndDeltaLog10(1000.0, grid, n, i, t);
  CHECK(i == 2);
  CHECK(t == Catch::Approx(1.0).margin(Tol));
}

TEST_CASE("IndexAndDeltaLog10 at geometric midpoints", "[indexdelta][log10]")
{
  using WeakLibReader::IndexAndDeltaLog10;

  const double grid[] = {1.0, 10.0, 100.0, 1000.0};
  const int n = 4;
  int i;
  double t;

  // sqrt(1*10) = sqrt(10) ~ 3.162: geometric midpoint of [1, 10]
  IndexAndDeltaLog10(std::sqrt(10.0), grid, n, i, t);
  CHECK(i == 0);
  CHECK(t == Catch::Approx(0.5).margin(Tol));

  // sqrt(100*1000) = sqrt(100000) ~ 316.2: geometric midpoint of [100, 1000]
  IndexAndDeltaLog10(std::sqrt(100.0 * 1000.0), grid, n, i, t);
  CHECK(i == 2);
  CHECK(t == Catch::Approx(0.5).margin(Tol));
}

TEST_CASE("IndexAndDeltaLog10 below range (extrapolation)", "[indexdelta][log10]")
{
  using WeakLibReader::IndexAndDeltaLog10;

  const double grid[] = {1.0, 10.0, 100.0, 1000.0};
  const int n = 4;
  int i;
  double t;

  // At x=0.5 (below grid[0]=1.0): i=0, delta < 0
  IndexAndDeltaLog10(0.5, grid, n, i, t);
  CHECK(i == 0);
  CHECK(t < 0.0);
  // delta = log10(0.5 / 1.0) / log10(10.0 / 1.0) = log10(0.5) / 1.0
  CHECK(t == Catch::Approx(std::log10(0.5)).margin(Tol));
}

TEST_CASE("IndexAndDeltaLog10 above range (extrapolation)", "[indexdelta][log10]")
{
  using WeakLibReader::IndexAndDeltaLog10;

  const double grid[] = {1.0, 10.0, 100.0, 1000.0};
  const int n = 4;
  int i;
  double t;

  // At x=2000 (above grid[3]=1000): i = n-2 = 2
  IndexAndDeltaLog10(2000.0, grid, n, i, t);
  CHECK(i == 2);
  CHECK(t > 1.0);
  // delta = log10(2000 / 100) / log10(1000 / 100) = log10(20) / log10(10) = log10(20)
  CHECK(t == Catch::Approx(std::log10(20.0)).margin(Tol));
}

// =============================================================================
// Non-uniform Grid Test
// =============================================================================

TEST_CASE("IndexAndDeltaLin with non-uniform grid", "[indexdelta][linear]")
{
  using WeakLibReader::IndexAndDeltaLin;

  // Non-uniform grid: the initial index estimate assumes uniform spacing,
  // so for non-uniform grids the index may differ from the "geometrically
  // correct" cell.  The delta compensates so that overall interpolation
  // still converges.  We verify the actual (index, delta) pairs produced.
  const double grid[] = {0.0, 1.0, 4.0, 9.0};
  const int n = 4;
  int i;
  double t;

  // At x=0.5: scaled=0.5/9=0.0556, scaledIndex=0.167 -> rawIndex=0
  //   i=0, t=(0.5-0.0)/(1.0-0.0)=0.5
  IndexAndDeltaLin(0.5, grid, n, i, t);
  CHECK(i == 0);
  CHECK(t == Catch::Approx(0.5).margin(Tol));

  // At x=2.5: scaled=2.5/9, scaledIndex=0.833 -> rawIndex=0
  //   i=0, t=(2.5-0.0)/(1.0-0.0)=2.5
  IndexAndDeltaLin(2.5, grid, n, i, t);
  CHECK(i == 0);
  CHECK(t == Catch::Approx(2.5).margin(Tol));

  // At x=6.5: scaled=6.5/9, scaledIndex=2.167 -> rawIndex=2
  //   i=2, t=(6.5-4.0)/(9.0-4.0)=0.5
  IndexAndDeltaLin(6.5, grid, n, i, t);
  CHECK(i == 2);
  CHECK(t == Catch::Approx(0.5).margin(Tol));

  // At x=1.0: scaled=1/9, scaledIndex=0.333 -> rawIndex=0
  //   i=0, t=(1.0-0.0)/(1.0-0.0)=1.0
  IndexAndDeltaLin(1.0, grid, n, i, t);
  CHECK(i == 0);
  CHECK(t == Catch::Approx(1.0).margin(Tol));

  // At x=4.0: scaled=4/9, scaledIndex=1.333 -> rawIndex=1
  //   i=1, t=(4.0-1.0)/(4.0-1.0)=1.0
  IndexAndDeltaLin(4.0, grid, n, i, t);
  CHECK(i == 1);
  CHECK(t == Catch::Approx(1.0).margin(Tol));

  // At x=9.0: scaled=1.0, scaledIndex=3.0 -> rawIndex=3, clamped to 2
  //   i=2, t=(9.0-4.0)/(9.0-4.0)=1.0
  IndexAndDeltaLin(9.0, grid, n, i, t);
  CHECK(i == 2);
  CHECK(t == Catch::Approx(1.0).margin(Tol));
}
