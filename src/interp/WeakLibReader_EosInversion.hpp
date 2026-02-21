#pragma once

#include <AMReX_GpuQualifiers.H>
#include <AMReX_Extension.H>

#include "base/WeakLibReader_AxisTypes.hpp"
#include "base/WeakLibReader_InterpBasis.hpp"
#include "base/WeakLibReader_Layout.hpp"
#include "base/WeakLibReader_Math.hpp"

namespace WeakLibReader {

// Lightweight struct for inversion bounds (12 doubles = 96 bytes).
// Trivially copyable, captured by value in device lambdas.
struct EosInversionBounds {
  double minD = 0.0, maxD = 0.0;
  double minT = 0.0, maxT = 0.0;
  double minY = 0.0, maxY = 0.0;
  double minE = 0.0, maxE = 0.0;
  double minP = 0.0, maxP = 0.0;
  double minS = 0.0, maxS = 0.0;
};

// Host-only: scan axis grids and variable data to compute inversion bounds.
// Takes raw Axis + data pointers (no dependency on hdf5/ types).
// Variable data is in log-space: log10(value + offset).
// Accepts nullptr for variables not present (sets min/max to 0).
// Matches Fortran InitializeEOSInversion (wlEOSInversionModule.F90:139-185).
inline EosInversionBounds InitializeEosInversionBounds(
    const Axis axes[3],
    int totalSize,
    const double* energyData,   double energyOffset,
    const double* pressureData, double pressureOffset,
    const double* entropyData,  double entropyOffset) noexcept
{
  EosInversionBounds bounds;

  // Axis bounds from grid endpoints (monotonically increasing)
  bounds.minD = axes[0].grid[0];
  bounds.maxD = axes[0].grid[axes[0].n - 1];
  bounds.minT = axes[1].grid[0];
  bounds.maxT = axes[1].grid[axes[1].n - 1];
  bounds.minY = axes[2].grid[0];
  bounds.maxY = axes[2].grid[axes[2].n - 1];

  // Scan variable data for min/max in physical space
  auto ScanMinMax = [totalSize](const double* data, double offset,
                                double& minVal, double& maxVal) {
    if (data == nullptr) {
      minVal = 0.0;
      maxVal = 0.0;
      return;
    }
    double lo = math::Pow10(data[0]) - offset;
    double hi = lo;
    for (int i = 1; i < totalSize; ++i) {
      const double val = math::Pow10(data[i]) - offset;
      if (val < lo) lo = val;
      if (val > hi) hi = val;
    }
    minVal = lo;
    maxVal = hi;
  };

  ScanMinMax(energyData,   energyOffset,   bounds.minE, bounds.maxE);
  ScanMinMax(pressureData, pressureOffset, bounds.minP, bounds.maxP);
  ScanMinMax(entropyData,  entropyOffset,  bounds.minS, bounds.maxS);

  return bounds;
}

// Device input validation.
// Error codes: 0=OK, 1=D outside bounds, 2=X outside bounds,
//              3=Y outside bounds, 11=NaN in argument(s).
// Matches Fortran CheckInputError (wlEOSInversionModule.F90:188-227).
// Note: Fortran error 10 (not initialized) omitted — struct is always valid.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int CheckInversionInputError(
    double D, double X, double Y,
    const EosInversionBounds& bounds,
    double minX, double maxX) noexcept
{
  if (D != D || X != X || Y != Y) return 11;
  if (D < bounds.minD || D > bounds.maxD) return 1;
  if (X < minX || X > maxX) return 2;
  if (Y < bounds.minY || Y > bounds.maxY) return 3;
  return 0;
}

namespace detail {

// Log-space bracket interpolation for final T refinement.
// T = 10^(log10(T_a) + log10(T_b/T_a) * log10((X+OS)/(X_a+OS))
//                                       / log10((X_b+OS)/(X_a+OS)))
// Matches Fortran InverseLogInterp (wlEOSInversionModule.F90:253-267).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double InverseLogInterp(
    double T_a, double T_b,
    double X_a, double X_b,
    double X, double offset) noexcept
{
  return math::Pow10(
      math::Log10(T_a)
      + math::Log10(T_b / T_a)
        * math::Log10((X + offset) / (X_a + offset))
        / math::Log10((X_b + offset) / (X_a + offset)));
}

// 2D bilinear interpolation in (D, Y) at fixed T-index from 3D data.
// Pre-computed iD, iY, dD, dY are reused across all T evaluations
// during bisection (more efficient than Fortran which recomputes each call).
// Data layout is column-major [nRho, nT, nYe].
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double EvalAtFixedTIndex(
    int iD, int iY, double dD, double dY,
    int iT,
    double offset,
    const double* data,
    const Layout& layout) noexcept
{
  const auto base = layout.Offset(iD, iT, iY);
  const auto s0 = layout.stride[0];
  const auto s2 = layout.stride[2];

  const double p00 = data[base];
  const double p10 = data[base + s0];
  const double p01 = data[base + s2];
  const double p11 = data[base + s0 + s2];

  const double logValue = BiLinear(p00, p10, p01, p11, dD, dY);
  return math::Pow10(logValue) - offset;
}

} // namespace detail

// Core bisection with initial guess.
// Matches Fortran ComputeTemperatureWithDxyGuess
// (wlEOSInversionModule.F90:270-440).
// Returns error code: 0=success, 13=no root found.
//
// Algorithm:
//   Phase 1: Check guess vicinity [iT, iT+1]
//   Phase 2: Evaluate at full T range endpoints [0, nT-1]
//   Phase 3: Bisection biased toward guess
//   Phase 4: Multi-root scan, pick bracket nearest to guess
//
// Index mapping (Fortran 1-based -> C++ 0-based):
//   Fortran i_a=1, i_b=SizeTs -> C++ i_a=0, i_b=nT-1
//   Fortran DO i=2,SizeTs-1   -> C++ for(i=1; i<=nT-2; ++i)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ComputeTemperatureWithDxyGuess(
    double D, double X, double Y,
    const Axis axes[3],
    const double* xData,
    const Layout& layout,
    double xOffset,
    double tGuess,
    double& T) noexcept
{
  T = 0.0;
  int error = 0;

  const int nT = axes[1].n;

  // Pre-compute D and Y indices/fractions (fixed during bisection)
  int iD, iY, iT;
  double dD, dY, dT;
  detail::IndexAndDelta(axes[0], D, iD, dD);
  detail::IndexAndDelta(axes[2], Y, iY, dY);
  detail::IndexAndDelta(axes[1], tGuess, iT, dT);
  (void)dT; // T fraction not needed; only index used for bisection

  // --- Phase 1: Check initial guess vicinity [iT, iT+1] ---
  // (Fortran lines 321-344)

  int i_a = iT;
  double T_a = axes[1].grid[i_a];
  double X_a = detail::EvalAtFixedTIndex(
      iD, iY, dD, dY, i_a, xOffset, xData, layout);
  double f_a = X - X_a;

  int i_b = i_a + 1;
  double T_b = axes[1].grid[i_b];
  double X_b = detail::EvalAtFixedTIndex(
      iD, iY, dD, dY, i_b, xOffset, xData, layout);
  double f_b = X - X_b;

  if (f_a * f_b <= 0.0) {
    T = detail::InverseLogInterp(T_a, T_b, X_a, X_b, X, xOffset);
    return 0;
  }

  // --- Phase 2: Evaluate at full T range endpoints ---
  // (Fortran lines 348-365)

  i_a = 0;
  T_a = axes[1].grid[0];
  X_a = detail::EvalAtFixedTIndex(
      iD, iY, dD, dY, 0, xOffset, xData, layout);
  f_a = X - X_a;

  i_b = nT - 1;
  T_b = axes[1].grid[nT - 1];
  X_b = detail::EvalAtFixedTIndex(
      iD, iY, dD, dY, nT - 1, xOffset, xData, layout);
  f_b = X - X_b;

  if (f_a * f_b <= 0.0) {

    // --- Phase 3: Bisection biased toward guess ---
    // (Fortran lines 366-393)

    // Initial midpoint biased toward guess: min(max(i_a+1, iT), i_b-1)
    int i_c = (iT > i_a + 1) ? iT : (i_a + 1);
    if (i_c > i_b - 1) i_c = i_b - 1;

    while (i_b > i_a + 1) {
      const double T_c = axes[1].grid[i_c];
      const double X_c = detail::EvalAtFixedTIndex(
          iD, iY, dD, dY, i_c, xOffset, xData, layout);
      const double f_c = X - X_c;

      if (f_a * f_c <= 0.0) {
        i_b = i_c;
        T_b = T_c;
        X_b = X_c;
        f_b = f_c;
      } else {
        i_a = i_c;
        T_a = T_c;
        X_a = X_c;
        f_a = f_c;
      }

      // Standard bisection midpoint: max(i_a+1, (i_a+i_b)/2)
      i_c = (i_a + i_b) / 2;
      if (i_c < i_a + 1) i_c = i_a + 1;
    }

  } else {

    // --- Phase 4: Pick root nearest to initial guess ---
    // (Fortran lines 395-432)

    int d_i = nT;
    int i_c = iT;
    // Clamp guess index to valid range [0, nT-2]
    // (IndexAndDelta already clamps, but be explicit for faithfulness)
    if (i_c < 0) i_c = 0;
    if (i_c > nT - 2) i_c = nT - 2;

    double T_c = T_a; // From Phase 2: axes[1].grid[0]
    double X_c = X_a; // From Phase 2: value at T-index 0

    // Scan all interior T indices (Fortran: DO i = 2, SizeTs-1)
    for (int i = 1; i <= nT - 2; ++i) {
      const double T_i = axes[1].grid[i];
      const double X_i = detail::EvalAtFixedTIndex(
          iD, iY, dD, dY, i, xOffset, xData, layout);

      const double f_c_val = X - X_c;
      const double f_b_val = X - X_i;

      const int d_c = (i >= i_c) ? (i - i_c) : (i_c - i);

      if (f_c_val * f_b_val <= 0.0 && d_c < d_i) {
        d_i = d_c;
        i_a = i - 1;
        T_a = T_c;
        X_a = X_c;
        i_b = i;
        T_b = T_i;
        X_b = X_i;
      } else {
        T_c = T_i;
        X_c = X_i;
      }
    }

    if (d_i >= nT) error = 13;
  }

  if (error != 0) {
    T = 0.0;
  } else {
    T = detail::InverseLogInterp(T_a, T_b, X_a, X_b, X, xOffset);
  }

  return error;
}

// Core bisection without initial guess.
// Matches Fortran ComputeTemperatureWithDxyNoGuess
// (wlEOSInversionModule.F90:443-573).
// Returns error code: 0=success, 13=no root found.
//
// Algorithm:
//   Phase 1: Evaluate at full T range endpoints [0, nT-1]
//   Phase 2: Standard bisection on T-index
//   Phase 3: Multi-root scan backwards (picks highest-T root)
//
// Index mapping (Fortran 1-based -> C++ 0-based):
//   Fortran i_a=1, i_b=SizeTs     -> C++ i_a=0, i_b=nT-1
//   Fortran DO i=SizeTs-1,2,-1    -> C++ for(i=nT-2; i>=1; --i)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ComputeTemperatureWithDxyNoGuess(
    double D, double X, double Y,
    const Axis axes[3],
    const double* xData,
    const Layout& layout,
    double xOffset,
    double& T) noexcept
{
  int error = 0;

  const int nT = axes[1].n;

  // Pre-compute D and Y indices/fractions
  int iD, iY;
  double dD, dY;
  detail::IndexAndDelta(axes[0], D, iD, dD);
  detail::IndexAndDelta(axes[2], Y, iY, dY);

  // --- Phase 1: Evaluate at full T range endpoints ---
  // (Fortran lines 488-504)

  int i_a = 0;
  double T_a = axes[1].grid[0];
  double X_a = detail::EvalAtFixedTIndex(
      iD, iY, dD, dY, 0, xOffset, xData, layout);
  double f_a = X - X_a;

  int i_b = nT - 1;
  double T_b = axes[1].grid[nT - 1];
  double X_b = detail::EvalAtFixedTIndex(
      iD, iY, dD, dY, nT - 1, xOffset, xData, layout);
  double f_b = X - X_b;

  if (f_a * f_b <= 0.0) {

    // --- Phase 2: Standard bisection ---
    // (Fortran lines 508-531)

    while (i_b > i_a + 1) {
      // max(i_a + 1, (i_a + i_b) / 2)
      int i_c = (i_a + i_b) / 2;
      if (i_c < i_a + 1) i_c = i_a + 1;

      const double T_c = axes[1].grid[i_c];
      const double X_c = detail::EvalAtFixedTIndex(
          iD, iY, dD, dY, i_c, xOffset, xData, layout);
      const double f_c = X - X_c;

      if (f_a * f_c <= 0.0) {
        i_b = i_c;
        T_b = T_c;
        X_b = X_c;
        f_b = f_c;
      } else {
        i_a = i_c;
        T_a = T_c;
        X_a = X_c;
        f_a = f_c;
      }
    }

  } else {

    // --- Phase 3: Pick highest temperature root ---
    // Scan backwards from second-to-last to second T-index.
    // (Fortran lines 536-558)

    for (int i = nT - 2; i >= 1; --i) {
      const double T_i = axes[1].grid[i];
      const double X_i = detail::EvalAtFixedTIndex(
          iD, iY, dD, dY, i, xOffset, xData, layout);

      const double f_a_val = X - X_i;
      const double f_b_val = X - X_b;

      if (f_a_val * f_b_val <= 0.0) {
        i_a = i;
        T_a = T_i;
        X_a = X_i;
        break;
      } else {
        i_b = i;
        T_b = T_i;
        X_b = X_i;
      }
    }

    // Check if root was found (Fortran lines 560-563)
    f_a = X - X_a;
    f_b = X - X_b;
    if (f_a * f_b > 0.0) error = 13;
  }

  if (error != 0) {
    T = 0.0;
  } else {
    T = detail::InverseLogInterp(T_a, T_b, X_a, X_b, X, xOffset);
  }

  return error;
}

namespace detail {

// Shared check-and-dispatch for all ComputeTemperatureFrom* wrappers (no guess).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ComputeTemperatureFromVariable(
    double D, double X, double Y,
    const Axis axes[3],
    const double* xData,
    const Layout& layout,
    double xOffset,
    const EosInversionBounds& bounds,
    double minX, double maxX,
    double& T) noexcept
{
  T = 0.0;
  const int error = CheckInversionInputError(D, X, Y, bounds, minX, maxX);
  if (error != 0) return error;
  return ComputeTemperatureWithDxyNoGuess(
      D, X, Y, axes, xData, layout, xOffset, T);
}

// Shared check-and-dispatch for all ComputeTemperatureFrom* wrappers (with guess).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ComputeTemperatureFromVariable(
    double D, double X, double Y,
    const Axis axes[3],
    const double* xData,
    const Layout& layout,
    double xOffset,
    const EosInversionBounds& bounds,
    double minX, double maxX,
    double tGuess,
    double& T) noexcept
{
  T = 0.0;
  const int error = CheckInversionInputError(D, X, Y, bounds, minX, maxX);
  if (error != 0) return error;
  return ComputeTemperatureWithDxyGuess(
      D, X, Y, axes, xData, layout, xOffset, tGuess, T);
}

} // namespace detail

// Convenience wrappers for temperature inversion from different EOS variables.

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ComputeTemperatureFromEnergy(
    double D, double E, double Y,
    const Axis axes[3],
    const double* energyData,
    const Layout& layout,
    double energyOffset,
    const EosInversionBounds& bounds,
    double& T) noexcept
{
  return detail::ComputeTemperatureFromVariable(
      D, E, Y, axes, energyData, layout, energyOffset,
      bounds, bounds.minE, bounds.maxE, T);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ComputeTemperatureFromEnergy(
    double D, double E, double Y,
    const Axis axes[3],
    const double* energyData,
    const Layout& layout,
    double energyOffset,
    const EosInversionBounds& bounds,
    double tGuess,
    double& T) noexcept
{
  return detail::ComputeTemperatureFromVariable(
      D, E, Y, axes, energyData, layout, energyOffset,
      bounds, bounds.minE, bounds.maxE, tGuess, T);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ComputeTemperatureFromPressure(
    double D, double P, double Y,
    const Axis axes[3],
    const double* pressureData,
    const Layout& layout,
    double pressureOffset,
    const EosInversionBounds& bounds,
    double& T) noexcept
{
  return detail::ComputeTemperatureFromVariable(
      D, P, Y, axes, pressureData, layout, pressureOffset,
      bounds, bounds.minP, bounds.maxP, T);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ComputeTemperatureFromPressure(
    double D, double P, double Y,
    const Axis axes[3],
    const double* pressureData,
    const Layout& layout,
    double pressureOffset,
    const EosInversionBounds& bounds,
    double tGuess,
    double& T) noexcept
{
  return detail::ComputeTemperatureFromVariable(
      D, P, Y, axes, pressureData, layout, pressureOffset,
      bounds, bounds.minP, bounds.maxP, tGuess, T);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ComputeTemperatureFromEntropy(
    double D, double S, double Y,
    const Axis axes[3],
    const double* entropyData,
    const Layout& layout,
    double entropyOffset,
    const EosInversionBounds& bounds,
    double& T) noexcept
{
  return detail::ComputeTemperatureFromVariable(
      D, S, Y, axes, entropyData, layout, entropyOffset,
      bounds, bounds.minS, bounds.maxS, T);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ComputeTemperatureFromEntropy(
    double D, double S, double Y,
    const Axis axes[3],
    const double* entropyData,
    const Layout& layout,
    double entropyOffset,
    const EosInversionBounds& bounds,
    double tGuess,
    double& T) noexcept
{
  return detail::ComputeTemperatureFromVariable(
      D, S, Y, axes, entropyData, layout, entropyOffset,
      bounds, bounds.minS, bounds.maxS, tGuess, T);
}

} // namespace WeakLibReader
