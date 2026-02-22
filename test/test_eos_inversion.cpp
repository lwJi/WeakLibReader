#define SIMPLE_CATCH_NO_MAIN
#include <catch2/catch_test_macros.hpp>

#include "interp/WeakLibReader_EosInversion.hpp"
#include "interp/WeakLibReader_LogInterpolate.hpp"
#include "base/WeakLibReader_Layout.hpp"
#include "base/WeakLibReader_AxisTypes.hpp"

#include <cmath>
#include <limits>
#include <vector>

namespace {

// Synthetic energy function: monotonically increasing in T for all positive
// rho, Ye. Used to build a 3D EOS table for round-trip inversion testing.
double EnergyFunc(double rho, double T, double Ye)
{
  return std::pow(rho, 0.3) * std::pow(T, 1.5) * (1.0 + 0.2 * Ye);
}

// Synthetic pressure: monotonically increasing in T
double PressureFunc(double rho, double T, double Ye)
{
  return std::pow(rho, 0.4) * std::pow(T, 1.8) * (1.0 + 0.15 * Ye);
}

// Synthetic entropy: monotonically increasing in T
double EntropyFunc(double rho, double T, double Ye)
{
  return std::pow(rho, -0.1) * std::pow(T, 1.2) * (1.0 + 0.3 * Ye);
}

// Grid definitions
constexpr int nD = 5;
constexpr int nT = 5;
constexpr int nY = 5;
constexpr int totalSize = nD * nT * nY;

const double gridD[nD] = {1e6, 1e7, 1e8, 1e9, 1e10};
const double gridT[nT] = {1e9, 3e9, 1e10, 3e10, 1e11};
const double gridY[nY] = {0.1, 0.2, 0.3, 0.4, 0.5};

constexpr double Offset = 1.0;

// Build the synthetic table and axes. Returns the data vector and populates
// axes and layout.
std::vector<double> BuildSyntheticTable(
    WeakLibReader::Axis axes[3],
    WeakLibReader::Layout& layout)
{
  using namespace WeakLibReader;

  axes[0] = MakeAxis(gridD, nD, AxisScale::Log10);
  axes[1] = MakeAxis(gridT, nT, AxisScale::Log10);
  axes[2] = MakeAxis(gridY, nY, AxisScale::Linear);

  const int extents[3] = {nD, nT, nY};
  layout = MakeLayout(extents, 3);

  std::vector<double> data(totalSize);
  for (int iRho = 0; iRho < nD; ++iRho) {
    for (int iTemp = 0; iTemp < nT; ++iTemp) {
      for (int iYe = 0; iYe < nY; ++iYe) {
        const double E = EnergyFunc(gridD[iRho], gridT[iTemp], gridY[iYe]);
        data[layout.Offset(iRho, iTemp, iYe)] =
            std::log10(E + Offset);
      }
    }
  }
  return data;
}

// Extended synthetic data for pressure/entropy inversion tests.
struct SyntheticEosData {
  std::vector<double> energy;
  std::vector<double> pressure;
  std::vector<double> entropy;
  WeakLibReader::Axis axes[3];
  WeakLibReader::Layout layout;
  WeakLibReader::EosInversionBounds bounds;
};

SyntheticEosData BuildFullSyntheticEos()
{
  using namespace WeakLibReader;

  SyntheticEosData d;
  d.axes[0] = MakeAxis(gridD, nD, AxisScale::Log10);
  d.axes[1] = MakeAxis(gridT, nT, AxisScale::Log10);
  d.axes[2] = MakeAxis(gridY, nY, AxisScale::Linear);

  const int extents[3] = {nD, nT, nY};
  d.layout = MakeLayout(extents, 3);

  d.energy.resize(totalSize);
  d.pressure.resize(totalSize);
  d.entropy.resize(totalSize);

  for (int iRho = 0; iRho < nD; ++iRho) {
    for (int iTemp = 0; iTemp < nT; ++iTemp) {
      for (int iYe = 0; iYe < nY; ++iYe) {
        const auto off = d.layout.Offset(iRho, iTemp, iYe);
        d.energy[off]   = std::log10(EnergyFunc(gridD[iRho], gridT[iTemp], gridY[iYe]) + Offset);
        d.pressure[off] = std::log10(PressureFunc(gridD[iRho], gridT[iTemp], gridY[iYe]) + Offset);
        d.entropy[off]  = std::log10(EntropyFunc(gridD[iRho], gridT[iTemp], gridY[iYe]) + Offset);
      }
    }
  }

  d.bounds = InitializeEosInversionBounds(
      d.axes, totalSize,
      d.energy.data(), Offset,
      d.pressure.data(), Offset,
      d.entropy.data(), Offset);

  return d;
}

constexpr double Tol = 1.0e-10;

} // namespace

TEST_CASE("InverseLogInterp recovers T from known bracket", "[eosinversion]")
{
  using namespace WeakLibReader;

  const double T_a = 1e9;
  const double T_b = 3e9;
  const double X_a = 100.0;
  const double X_b = 500.0;
  const double offset = 1.0;

  // Pick T_target at the geometric midpoint of [T_a, T_b]
  const double T_target = std::sqrt(T_a * T_b);

  // Compute X at T_target using the InverseLogInterp forward formula:
  // log10(T_target) = log10(T_a) + log10(T_b/T_a) * log10((X+OS)/(X_a+OS))
  //                                                 / log10((X_b+OS)/(X_a+OS))
  // => frac = (log10(T_target) - log10(T_a)) / (log10(T_b) - log10(T_a))
  // => X = (X_a+OS) * ((X_b+OS)/(X_a+OS))^frac - OS
  const double frac =
      (std::log10(T_target) - std::log10(T_a))
      / (std::log10(T_b) - std::log10(T_a));
  const double X_target =
      (X_a + offset)
      * std::pow((X_b + offset) / (X_a + offset), frac)
      - offset;

  const double T_recovered = detail::InverseLogInterp(
      T_a, T_b, X_a, X_b, X_target, offset);

  CHECK(std::abs(T_recovered - T_target) / T_target < 1e-12);

  // Also test at a non-midpoint fraction
  const double T_target2 = std::pow(10.0,
      0.25 * std::log10(T_a) + 0.75 * std::log10(T_b));
  const double frac2 = 0.75;
  const double X_target2 =
      (X_a + offset)
      * std::pow((X_b + offset) / (X_a + offset), frac2)
      - offset;

  const double T_recovered2 = detail::InverseLogInterp(
      T_a, T_b, X_a, X_b, X_target2, offset);

  CHECK(std::abs(T_recovered2 - T_target2) / T_target2 < 1e-12);
}

TEST_CASE("CheckInversionInputError returns correct codes", "[eosinversion]")
{
  using namespace WeakLibReader;

  EosInversionBounds bounds;
  bounds.minD = 1e6;  bounds.maxD = 1e10;
  bounds.minT = 1e9;  bounds.maxT = 1e11;
  bounds.minY = 0.1;  bounds.maxY = 0.5;
  bounds.minE = 10.0; bounds.maxE = 1e20;
  bounds.minP = 1.0;  bounds.maxP = 1e18;
  bounds.minS = 0.5;  bounds.maxS = 100.0;

  // Valid input
  CHECK(CheckInversionInputError(1e8, 1e10, 0.3, bounds,
                                  bounds.minE, bounds.maxE) == EosInversionError::Success);

  // D out of range (below)
  CHECK(CheckInversionInputError(1e5, 1e10, 0.3, bounds,
                                  bounds.minE, bounds.maxE) == EosInversionError::DensityOutOfRange);
  // D out of range (above)
  CHECK(CheckInversionInputError(1e11, 1e10, 0.3, bounds,
                                  bounds.minE, bounds.maxE) == EosInversionError::DensityOutOfRange);

  // X out of range (below)
  CHECK(CheckInversionInputError(1e8, 1.0, 0.3, bounds,
                                  bounds.minE, bounds.maxE) == EosInversionError::VariableOutOfRange);
  // X out of range (above)
  CHECK(CheckInversionInputError(1e8, 1e21, 0.3, bounds,
                                  bounds.minE, bounds.maxE) == EosInversionError::VariableOutOfRange);

  // Y out of range (below)
  CHECK(CheckInversionInputError(1e8, 1e10, 0.05, bounds,
                                  bounds.minE, bounds.maxE) == EosInversionError::ElectronFractionOutOfRange);
  // Y out of range (above)
  CHECK(CheckInversionInputError(1e8, 1e10, 0.6, bounds,
                                  bounds.minE, bounds.maxE) == EosInversionError::ElectronFractionOutOfRange);

  // NaN inputs
  const double nan = std::numeric_limits<double>::quiet_NaN();
  CHECK(CheckInversionInputError(nan, 1e10, 0.3, bounds,
                                  bounds.minE, bounds.maxE) == EosInversionError::NaNInput);
  CHECK(CheckInversionInputError(1e8, nan, 0.3, bounds,
                                  bounds.minE, bounds.maxE) == EosInversionError::NaNInput);
  CHECK(CheckInversionInputError(1e8, 1e10, nan, bounds,
                                  bounds.minE, bounds.maxE) == EosInversionError::NaNInput);
}

TEST_CASE("EvalAtFixedTIndex matches full 3D interp at grid T", "[eosinversion]")
{
  using namespace WeakLibReader;

  Axis axes[3];
  Layout layout;
  const auto data = BuildSyntheticTable(axes, layout);

  // Test at several D/Y points between grid nodes, at exact T grid points
  const double testD[] = {3e6, 5e8, 2e9};
  const double testY[] = {0.15, 0.35, 0.45};

  for (double D : testD) {
    for (double Y : testY) {
      int iD, iY;
      double dD, dY;
      detail::IndexAndDelta(axes[0], D, iD, dD);
      detail::IndexAndDelta(axes[2], Y, iY, dY);

      for (int iTemp = 0; iTemp < nT; ++iTemp) {
        const double T = gridT[iTemp];

        // EvalAtFixedTIndex
        const double valFixed = detail::EvalAtFixedTIndex(
            iD, iY, dD, dY, iTemp, Offset, data.data(), layout);

        // Full 3D interpolation at the same point
        const double valFull = LogInterpolateSingleVariable3DCustomPoint(
            D, T, Y, axes, data.data(), Offset);

        CHECK(std::abs(valFixed - valFull)
              / (std::abs(valFull) + 1e-30) < 1e-12);
      }
    }
  }
}

TEST_CASE("Round-trip inversion recovers T without guess", "[eosinversion]")
{
  using namespace WeakLibReader;

  Axis axes[3];
  Layout layout;
  const auto data = BuildSyntheticTable(axes, layout);

  const auto bounds = InitializeEosInversionBounds(
      axes, totalSize,
      data.data(), Offset,
      nullptr, 0.0,
      nullptr, 0.0);

  // Test at multiple interior points
  const double testD[] = {3e6, 5e7, 2e8, 7e9};
  const double testT[] = {2e9, 5e9, 2e10, 5e10};
  const double testY[] = {0.15, 0.25, 0.35, 0.45};

  for (double D : testD) {
    for (double T_known : testT) {
      for (double Y : testY) {
        // Forward interpolate
        const double E = LogInterpolateSingleVariable3DCustomPoint(
            D, T_known, Y, axes, data.data(), Offset);

        // Invert
        double T_recovered = 0.0;
        const auto error = ComputeTemperatureFromEnergy(
            D, E, Y, axes, data.data(), layout, Offset, bounds,
            T_recovered);

        CHECK(error == EosInversionError::Success);
        const double relErr = std::abs(T_recovered - T_known) / T_known;
        CHECK(relErr < Tol);
      }
    }
  }
}

TEST_CASE("Round-trip inversion recovers T with good guess", "[eosinversion]")
{
  using namespace WeakLibReader;

  Axis axes[3];
  Layout layout;
  const auto data = BuildSyntheticTable(axes, layout);

  const auto bounds = InitializeEosInversionBounds(
      axes, totalSize,
      data.data(), Offset,
      nullptr, 0.0,
      nullptr, 0.0);

  const double D = 5e8;
  const double T_known = 1.5e10;
  const double Y = 0.3;

  const double E = LogInterpolateSingleVariable3DCustomPoint(
      D, T_known, Y, axes, data.data(), Offset);

  const double tGuess = 1.2e10; // close to T_known
  double T_recovered = 0.0;
  const auto error = ComputeTemperatureFromEnergy(
      D, E, Y, axes, data.data(), layout, Offset, bounds,
      tGuess, T_recovered);

  CHECK(error == EosInversionError::Success);
  const double relErr = std::abs(T_recovered - T_known) / T_known;
  CHECK(relErr < Tol);
}

TEST_CASE("Round-trip inversion recovers T with bad guess", "[eosinversion]")
{
  using namespace WeakLibReader;

  Axis axes[3];
  Layout layout;
  const auto data = BuildSyntheticTable(axes, layout);

  const auto bounds = InitializeEosInversionBounds(
      axes, totalSize,
      data.data(), Offset,
      nullptr, 0.0,
      nullptr, 0.0);

  const double D = 5e8;
  const double T_known = 1.5e10;
  const double Y = 0.3;

  const double E = LogInterpolateSingleVariable3DCustomPoint(
      D, T_known, Y, axes, data.data(), Offset);

  const double tGuess = 1e9; // far from T_known=1.5e10
  double T_recovered = 0.0;
  const auto error = ComputeTemperatureFromEnergy(
      D, E, Y, axes, data.data(), layout, Offset, bounds,
      tGuess, T_recovered);

  CHECK(error == EosInversionError::Success);
  const double relErr = std::abs(T_recovered - T_known) / T_known;
  CHECK(relErr < Tol);
}

TEST_CASE("Round-trip inversion recovers T with upper boundary guess", "[eosinversion]")
{
  using namespace WeakLibReader;

  Axis axes[3];
  Layout layout;
  const auto data = BuildSyntheticTable(axes, layout);

  const auto bounds = InitializeEosInversionBounds(
      axes, totalSize,
      data.data(), Offset,
      nullptr, 0.0,
      nullptr, 0.0);

  const double D = 5e8;
  const double T_known = 1.5e10;
  const double Y = 0.3;

  const double E = LogInterpolateSingleVariable3DCustomPoint(
      D, T_known, Y, axes, data.data(), Offset);

  const double tGuess = 9e10; // near upper T boundary
  double T_recovered = 0.0;
  const auto error = ComputeTemperatureFromEnergy(
      D, E, Y, axes, data.data(), layout, Offset, bounds,
      tGuess, T_recovered);

  CHECK(error == EosInversionError::Success);
  const double relErr = std::abs(T_recovered - T_known) / T_known;
  CHECK(relErr < Tol);
}

TEST_CASE("Inversion at grid boundary points", "[eosinversion]")
{
  using namespace WeakLibReader;

  Axis axes[3];
  Layout layout;
  const auto data = BuildSyntheticTable(axes, layout);

  const auto bounds = InitializeEosInversionBounds(
      axes, totalSize,
      data.data(), Offset,
      nullptr, 0.0,
      nullptr, 0.0);

  // Test at exact grid corners and edges
  struct TestPoint { double D, T, Y; };
  const TestPoint points[] = {
    // Corner: low D, low T, low Y
    {gridD[0], gridT[0], gridY[0]},
    // Corner: high D, high T, high Y
    {gridD[nD-1], gridT[nT-1], gridY[nY-1]},
    // Edge: low D, mid T, high Y
    {gridD[0], gridT[2], gridY[nY-1]},
    // Edge: mid D, low T, mid Y
    {gridD[2], gridT[0], gridY[2]},
    // Edge: high D, high T, low Y
    {gridD[nD-1], gridT[nT-1], gridY[0]},
    // Exact grid point in interior
    {gridD[2], gridT[2], gridY[2]},
  };

  for (const auto& pt : points) {
    // Forward interpolate
    const double E = LogInterpolateSingleVariable3DCustomPoint(
        pt.D, pt.T, pt.Y, axes, data.data(), Offset);

    // Invert (no guess)
    double T_recovered = 0.0;
    const auto error = ComputeTemperatureFromEnergy(
        pt.D, E, pt.Y, axes, data.data(), layout, Offset, bounds,
        T_recovered);

    CHECK(error == EosInversionError::Success);
    const double relErr = std::abs(T_recovered - pt.T) / pt.T;
    CHECK(relErr < Tol);
  }
}

TEST_CASE("InitializeEosInversionBounds computes correct bounds", "[eosinversion]")
{
  using namespace WeakLibReader;

  Axis axes[3];
  Layout layout;
  const auto data = BuildSyntheticTable(axes, layout);

  const auto bounds = InitializeEosInversionBounds(
      axes, totalSize,
      data.data(), Offset,
      nullptr, 0.0,
      nullptr, 0.0);

  // Axis bounds
  CHECK(bounds.minD == gridD[0]);
  CHECK(bounds.maxD == gridD[nD - 1]);
  CHECK(bounds.minT == gridT[0]);
  CHECK(bounds.maxT == gridT[nT - 1]);
  CHECK(bounds.minY == gridY[0]);
  CHECK(bounds.maxY == gridY[nY - 1]);

  // Energy bounds: compute expected min/max from the synthetic function
  double expectedMinE = std::numeric_limits<double>::max();
  double expectedMaxE = std::numeric_limits<double>::lowest();
  for (int iRho = 0; iRho < nD; ++iRho) {
    for (int iTemp = 0; iTemp < nT; ++iTemp) {
      for (int iYe = 0; iYe < nY; ++iYe) {
        const double E = EnergyFunc(gridD[iRho], gridT[iTemp], gridY[iYe]);
        if (E < expectedMinE) expectedMinE = E;
        if (E > expectedMaxE) expectedMaxE = E;
      }
    }
  }

  CHECK(std::abs(bounds.minE - expectedMinE)
        / (std::abs(expectedMinE) + 1e-30) < 1e-12);
  CHECK(std::abs(bounds.maxE - expectedMaxE)
        / (std::abs(expectedMaxE) + 1e-30) < 1e-12);

  // Pressure and entropy should be 0 since nullptr was passed
  CHECK(bounds.minP == 0.0);
  CHECK(bounds.maxP == 0.0);
  CHECK(bounds.minS == 0.0);
  CHECK(bounds.maxS == 0.0);
}

TEST_CASE("InitializeEosInversionBounds populates pressure and entropy bounds",
          "[eosinversion]")
{
  using namespace WeakLibReader;

  const auto d = BuildFullSyntheticEos();

  // Pressure bounds: compute expected min/max from the synthetic function
  double expectedMinP = std::numeric_limits<double>::max();
  double expectedMaxP = std::numeric_limits<double>::lowest();
  for (int iRho = 0; iRho < nD; ++iRho) {
    for (int iTemp = 0; iTemp < nT; ++iTemp) {
      for (int iYe = 0; iYe < nY; ++iYe) {
        const double P = PressureFunc(gridD[iRho], gridT[iTemp], gridY[iYe]);
        if (P < expectedMinP) expectedMinP = P;
        if (P > expectedMaxP) expectedMaxP = P;
      }
    }
  }

  CHECK(std::abs(d.bounds.minP - expectedMinP)
        / (std::abs(expectedMinP) + 1e-30) < 1e-12);
  CHECK(std::abs(d.bounds.maxP - expectedMaxP)
        / (std::abs(expectedMaxP) + 1e-30) < 1e-12);

  // Entropy bounds: compute expected min/max from the synthetic function
  double expectedMinS = std::numeric_limits<double>::max();
  double expectedMaxS = std::numeric_limits<double>::lowest();
  for (int iRho = 0; iRho < nD; ++iRho) {
    for (int iTemp = 0; iTemp < nT; ++iTemp) {
      for (int iYe = 0; iYe < nY; ++iYe) {
        const double S = EntropyFunc(gridD[iRho], gridT[iTemp], gridY[iYe]);
        if (S < expectedMinS) expectedMinS = S;
        if (S > expectedMaxS) expectedMaxS = S;
      }
    }
  }

  CHECK(std::abs(d.bounds.minS - expectedMinS)
        / (std::abs(expectedMinS) + 1e-30) < 1e-12);
  CHECK(std::abs(d.bounds.maxS - expectedMaxS)
        / (std::abs(expectedMaxS) + 1e-30) < 1e-12);
}

TEST_CASE("Round-trip pressure inversion without guess",
          "[eosinversion][pressure]")
{
  using namespace WeakLibReader;

  const auto d = BuildFullSyntheticEos();

  const double testD[] = {3e6, 5e7, 2e8, 7e9};
  const double testT[] = {2e9, 5e9, 2e10, 5e10};
  const double testY[] = {0.15, 0.25, 0.35, 0.45};

  for (double D : testD) {
    for (double T_known : testT) {
      for (double Y : testY) {
        const double P = LogInterpolateSingleVariable3DCustomPoint(
            D, T_known, Y, d.axes, d.pressure.data(), Offset);

        double T_recovered = 0.0;
        const auto error = ComputeTemperatureFromPressure(
            D, P, Y, d.axes, d.pressure.data(), d.layout, Offset,
            d.bounds, T_recovered);

        CHECK(error == EosInversionError::Success);
        const double relErr = std::abs(T_recovered - T_known) / T_known;
        CHECK(relErr < Tol);
      }
    }
  }
}

TEST_CASE("Round-trip pressure inversion with good guess",
          "[eosinversion][pressure]")
{
  using namespace WeakLibReader;

  const auto d = BuildFullSyntheticEos();

  const double D = 5e8;
  const double T_known = 1.5e10;
  const double Y = 0.3;

  const double P = LogInterpolateSingleVariable3DCustomPoint(
      D, T_known, Y, d.axes, d.pressure.data(), Offset);

  const double tGuess = 1.2e10; // close to T_known
  double T_recovered = 0.0;
  const auto error = ComputeTemperatureFromPressure(
      D, P, Y, d.axes, d.pressure.data(), d.layout, Offset,
      d.bounds, tGuess, T_recovered);

  CHECK(error == EosInversionError::Success);
  const double relErr = std::abs(T_recovered - T_known) / T_known;
  CHECK(relErr < Tol);
}

TEST_CASE("Round-trip pressure inversion with bad guess",
          "[eosinversion][pressure]")
{
  using namespace WeakLibReader;

  const auto d = BuildFullSyntheticEos();

  const double D = 5e8;
  const double T_known = 1.5e10;
  const double Y = 0.3;

  const double P = LogInterpolateSingleVariable3DCustomPoint(
      D, T_known, Y, d.axes, d.pressure.data(), Offset);

  const double tGuess = 1e9; // far from T_known=1.5e10
  double T_recovered = 0.0;
  const auto error = ComputeTemperatureFromPressure(
      D, P, Y, d.axes, d.pressure.data(), d.layout, Offset,
      d.bounds, tGuess, T_recovered);

  CHECK(error == EosInversionError::Success);
  const double relErr = std::abs(T_recovered - T_known) / T_known;
  CHECK(relErr < Tol);
}

TEST_CASE("Round-trip entropy inversion without guess",
          "[eosinversion][entropy]")
{
  using namespace WeakLibReader;

  const auto d = BuildFullSyntheticEos();

  const double testD[] = {3e6, 5e7, 2e8, 7e9};
  const double testT[] = {2e9, 5e9, 2e10, 5e10};
  const double testY[] = {0.15, 0.25, 0.35, 0.45};

  for (double D : testD) {
    for (double T_known : testT) {
      for (double Y : testY) {
        const double S = LogInterpolateSingleVariable3DCustomPoint(
            D, T_known, Y, d.axes, d.entropy.data(), Offset);

        double T_recovered = 0.0;
        const auto error = ComputeTemperatureFromEntropy(
            D, S, Y, d.axes, d.entropy.data(), d.layout, Offset,
            d.bounds, T_recovered);

        CHECK(error == EosInversionError::Success);
        const double relErr = std::abs(T_recovered - T_known) / T_known;
        CHECK(relErr < Tol);
      }
    }
  }
}

TEST_CASE("Round-trip entropy inversion with good guess",
          "[eosinversion][entropy]")
{
  using namespace WeakLibReader;

  const auto d = BuildFullSyntheticEos();

  const double D = 5e8;
  const double T_known = 1.5e10;
  const double Y = 0.3;

  const double S = LogInterpolateSingleVariable3DCustomPoint(
      D, T_known, Y, d.axes, d.entropy.data(), Offset);

  const double tGuess = 1.2e10; // close to T_known
  double T_recovered = 0.0;
  const auto error = ComputeTemperatureFromEntropy(
      D, S, Y, d.axes, d.entropy.data(), d.layout, Offset,
      d.bounds, tGuess, T_recovered);

  CHECK(error == EosInversionError::Success);
  const double relErr = std::abs(T_recovered - T_known) / T_known;
  CHECK(relErr < Tol);
}

TEST_CASE("Round-trip entropy inversion with bad guess",
          "[eosinversion][entropy]")
{
  using namespace WeakLibReader;

  const auto d = BuildFullSyntheticEos();

  const double D = 5e8;
  const double T_known = 1.5e10;
  const double Y = 0.3;

  const double S = LogInterpolateSingleVariable3DCustomPoint(
      D, T_known, Y, d.axes, d.entropy.data(), Offset);

  const double tGuess = 1e9; // far from T_known=1.5e10
  double T_recovered = 0.0;
  const auto error = ComputeTemperatureFromEntropy(
      D, S, Y, d.axes, d.entropy.data(), d.layout, Offset,
      d.bounds, tGuess, T_recovered);

  CHECK(error == EosInversionError::Success);
  const double relErr = std::abs(T_recovered - T_known) / T_known;
  CHECK(relErr < Tol);
}
