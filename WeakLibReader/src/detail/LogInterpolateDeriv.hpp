#pragma once

#include <AMReX_GpuQualifiers.H>
#include <cstddef>
#include <limits>

#include "LogInterpolateCore.hpp"

namespace WeakLibReader {

/// GPU-optimized 3D log-interpolation with derivatives (single point)
/// Uses compile-time dimensionality for zero runtime branching
inline int LogInterpolateDifferentiateSingleVariable3DCustomPoint(
    double d, double t, double y,
    const double* gridD, int nD,
    const double* gridT, int nT,
    const double* gridY, int nY,
    const double* data,
    double offset,
    double& interpolant,
    double derivatives[3],
    const InterpConfig& cfg = InterpConfig{}) noexcept
{
  Axis axesLocal[3] = {
      MakeAxis(gridD, nD, AxisScale::Log10),
      MakeAxis(gridT, nT, AxisScale::Log10),
      MakeAxis(gridY, nY, AxisScale::Linear)};
  int extents[3] = {nD, nT, nY};
  const Layout layout = MakeLayout(extents, 3);
  return detail::LogInterpolateDifferentiateSingleVariable3DCustomPointImpl(
      d, t, y, data, layout, axesLocal, offset, interpolant, derivatives, cfg);
}

/// GPU-optimized 3D log-interpolation with derivatives (batch)
/// Uses compile-time dimensionality for zero runtime branching
inline int LogInterpolateDifferentiateSingleVariable3DCustom(
    const double* d, const double* t, const double* y, std::size_t count,
    const double* gridD, int nD,
    const double* gridT, int nT,
    const double* gridY, int nY,
    const double* data,
    double offset,
    double* interpolants,
    double* derivatives,
    const InterpConfig& cfg = InterpConfig{}) noexcept
{
  if (d == nullptr || t == nullptr || y == nullptr ||
      data == nullptr || interpolants == nullptr || derivatives == nullptr ||
      gridD == nullptr || gridT == nullptr || gridY == nullptr) {
    return 1;
  }
  Axis axesLocal[3] = {
      MakeAxis(gridD, nD, AxisScale::Log10),
      MakeAxis(gridT, nT, AxisScale::Log10),
      MakeAxis(gridY, nY, AxisScale::Linear)};
  int extents[3] = {nD, nT, nY};
  const Layout layout = MakeLayout(extents, 3);

  for (std::size_t i = 0; i < count; ++i) {
    double deriv[3] = {0.0, 0.0, 0.0};
    double interp = 0.0;
    const int rc = detail::LogInterpolateDifferentiateSingleVariable3DCustomPointImpl(
        d[i], t[i], y[i],
        data, layout, axesLocal,
        offset, interp, deriv, cfg);
    if (rc != 0) {
      return rc;
    }
    interpolants[i] = interp;
    derivatives[i * 3 + 0] = deriv[0];
    derivatives[i * 3 + 1] = deriv[1];
    derivatives[i * 3 + 2] = deriv[2];
  }
  return 0;
}

inline int LogInterpolateDifferentiateSingleVariable2D2DCustomPoint(
    const double* logE, std::size_t sizeE,
    double logT, double logX,
    const double* gridE, int nE,
    const double* gridT, int nT,
    const double* gridX, int nX,
    const double* data,
    double offset,
    double* interpolantPlane,
    double* derivativeTPlane,
    double* derivativeXPlane,
    const InterpConfig& cfg = InterpConfig{}) noexcept
{
  if (logE == nullptr || data == nullptr ||
      interpolantPlane == nullptr || derivativeTPlane == nullptr || derivativeXPlane == nullptr ||
      gridE == nullptr || gridT == nullptr || gridX == nullptr) {
    return 1;
  }
  if (sizeE == 0) {
    return 0;
  }

  Axis axes[4] = {
      MakeAxis(gridE, nE, AxisScale::Linear),
      MakeAxis(gridE, nE, AxisScale::Linear),
      MakeAxis(gridT, nT, AxisScale::Linear),
      MakeAxis(gridX, nX, AxisScale::Linear)};
  int extents[4] = {nE, nE, nT, nX};
  const Layout layout = MakeLayout(extents, 4);

  int idxT = 0;
  int idxX = 0;
  double fracT = 0.0;
  double fracX = 0.0;
  const bool outT = detail::IndexAndDelta(axes[2], logT, idxT, fracT);
  const bool outX = detail::IndexAndDelta(axes[3], logX, idxX, fracX);

  if (outT || outX) {
    if (cfg.outOfRange == OutOfRangePolicy::Error) {
      return 4;
    }
    if (cfg.outOfRange == OutOfRangePolicy::FillNaN) {
      detail::FillNaNPlane(interpolantPlane, sizeE);
      detail::FillNaNPlane(derivativeTPlane, sizeE);
      detail::FillNaNPlane(derivativeXPlane, sizeE);
      return 0;
    }
    fracT = detail::Clamp01(fracT);
    fracX = detail::Clamp01(fracX);
  }

  const double spanT = axes[2].grid[idxT + 1] - axes[2].grid[idxT];
  const double spanX = axes[3].grid[idxX + 1] - axes[3].grid[idxX];

  if (!(spanT > 0.0) || !(spanX > 0.0)) {
    detail::FillNaNPlane(interpolantPlane, sizeE);
    detail::FillNaNPlane(derivativeTPlane, sizeE);
    detail::FillNaNPlane(derivativeXPlane, sizeE);
    return 5;
  }

  const double aT = 1.0 / (spanT * math::Pow10(logT));
  const double aX = 1.0 / (spanX * math::Pow10(logX));

  for (std::size_t j = 0; j < sizeE; ++j) {
    int idxE2 = 0;
    double fracE2 = 0.0;
    bool outE2 = detail::IndexAndDelta(axes[1], logE[j], idxE2, fracE2);

    if (outE2 && cfg.outOfRange == OutOfRangePolicy::Error) {
      return 6;
    }
    if (outE2 && cfg.outOfRange == OutOfRangePolicy::FillNaN) {
      const double nanValue = std::numeric_limits<double>::quiet_NaN();
      for (std::size_t i = 0; i <= j; ++i) {
        detail::StoreSymmetric(interpolantPlane, sizeE, i, j, nanValue);
        detail::StoreSymmetric(derivativeTPlane, sizeE, i, j, nanValue);
        detail::StoreSymmetric(derivativeXPlane, sizeE, i, j, nanValue);
      }
      continue;
    }
    if (outE2) {
      fracE2 = detail::Clamp01(fracE2);
    }

    for (std::size_t i = 0; i <= j; ++i) {
      int idxE1 = 0;
      double fracE1 = 0.0;
      bool outE1 = detail::IndexAndDelta(axes[0], logE[i], idxE1, fracE1);
      if (outE1 && cfg.outOfRange == OutOfRangePolicy::Error) {
        return 7;
      }
      if (outE1 && cfg.outOfRange == OutOfRangePolicy::FillNaN) {
        detail::StoreSymmetric(interpolantPlane, sizeE, i, j, std::numeric_limits<double>::quiet_NaN());
        detail::StoreSymmetric(derivativeTPlane, sizeE, i, j, std::numeric_limits<double>::quiet_NaN());
        detail::StoreSymmetric(derivativeXPlane, sizeE, i, j, std::numeric_limits<double>::quiet_NaN());
        continue;
      }
      if (outE1) {
        fracE1 = detail::Clamp01(fracE1);
      }

      double interpValue = 0.0;
      double derivE1 = 0.0;
      double derivE2 = 0.0;
      double derivTVal = 0.0;
      double derivXVal = 0.0;

      LinearInterpDeriv4DPoint(
          idxE1, idxE2, idxT, idxX,
          fracE1, fracE2, fracT, fracX,
          1.0, 1.0, aT, aX,
          offset, data, layout,
          interpValue, derivE1, derivE2, derivTVal, derivXVal);

      detail::StoreSymmetric(interpolantPlane, sizeE, i, j, interpValue);
      detail::StoreSymmetric(derivativeTPlane, sizeE, i, j, derivTVal);
      detail::StoreSymmetric(derivativeXPlane, sizeE, i, j, derivXVal);
    }
  }

  return 0;
}

inline int LogInterpolateDifferentiateSingleVariable2D2DCustom(
    const double* logE, std::size_t sizeE,
    const double* logT, const double* logX, std::size_t count,
    const double* gridE, int nE,
    const double* gridT, int nT,
    const double* gridX, int nX,
    const double* data,
    double offset,
    double* interpolant,
    double* derivativeT,
    double* derivativeX,
    const InterpConfig& cfg = InterpConfig{}) noexcept
{
  if (logE == nullptr || logT == nullptr || logX == nullptr ||
      data == nullptr || interpolant == nullptr ||
      derivativeT == nullptr || derivativeX == nullptr ||
      gridE == nullptr || gridT == nullptr || gridX == nullptr) {
    return 1;
  }
  if (sizeE == 0 || count == 0) {
    return 0;
  }

  const std::size_t planeSize = sizeE * sizeE;
  for (std::size_t l = 0; l < count; ++l) {
    double* planeInterp = interpolant + l * planeSize;
    double* planeDerivT = derivativeT + l * planeSize;
    double* planeDerivX = derivativeX + l * planeSize;
    const int rc = LogInterpolateDifferentiateSingleVariable2D2DCustomPoint(
        logE, sizeE,
        logT[l], logX[l],
        gridE, nE,
        gridT, nT,
        gridX, nX,
        data, offset,
        planeInterp, planeDerivT, planeDerivX,
        cfg);
    if (rc != 0) {
      return rc;
    }
  }

  return 0;
}

inline int LogInterpolateDifferentiateSingleVariable2D2DCustomAlignedPoint(
    std::size_t sizeE,
    double logT, double logX,
    const double* gridT, int nT,
    const double* gridX, int nX,
    const double* data,
    double offset,
    double* interpolantPlane,
    double* derivativeTPlane,
    double* derivativeXPlane,
    const InterpConfig& cfg = InterpConfig{}) noexcept
{
  if (data == nullptr || interpolantPlane == nullptr ||
      derivativeTPlane == nullptr || derivativeXPlane == nullptr ||
      gridT == nullptr || gridX == nullptr) {
    return 1;
  }
  if (sizeE == 0) {
    return 0;
  }

  Axis axes[2] = {
      MakeAxis(gridT, nT, AxisScale::Linear),
      MakeAxis(gridX, nX, AxisScale::Linear)};
  int extents[4] = {
      static_cast<int>(sizeE),
      static_cast<int>(sizeE),
      nT,
      nX};
  const Layout layout = MakeLayout(extents, 4);

  int idxT = 0;
  int idxX = 0;
  double fracT = 0.0;
  double fracX = 0.0;
  const bool outT = detail::IndexAndDelta(axes[0], logT, idxT, fracT);
  const bool outX = detail::IndexAndDelta(axes[1], logX, idxX, fracX);

  if (outT || outX) {
    if (cfg.outOfRange == OutOfRangePolicy::Error) {
      return 4;
    }
    if (cfg.outOfRange == OutOfRangePolicy::FillNaN) {
      detail::FillNaNPlane(interpolantPlane, sizeE);
      detail::FillNaNPlane(derivativeTPlane, sizeE);
      detail::FillNaNPlane(derivativeXPlane, sizeE);
      return 0;
    }
    fracT = detail::Clamp01(fracT);
    fracX = detail::Clamp01(fracX);
  }

  const double spanT = axes[0].grid[idxT + 1] - axes[0].grid[idxT];
  const double spanX = axes[1].grid[idxX + 1] - axes[1].grid[idxX];

  if (!(spanT > 0.0) || !(spanX > 0.0)) {
    detail::FillNaNPlane(interpolantPlane, sizeE);
    detail::FillNaNPlane(derivativeTPlane, sizeE);
    detail::FillNaNPlane(derivativeXPlane, sizeE);
    return 5;
  }

  const double aT = 1.0 / (spanT * math::Pow10(logT));
  const double aX = 1.0 / (spanX * math::Pow10(logX));

  for (std::size_t j = 0; j < sizeE; ++j) {
    for (std::size_t i = 0; i <= j; ++i) {
      double interpValue = 0.0;
      double derivTVal = 0.0;
      double derivXVal = 0.0;

      LinearInterpDeriv2D4DArray2DAlignedPoint(
          static_cast<int>(i), static_cast<int>(j),
          idxT, idxX,
          fracT, fracX,
          aT, aX,
          offset,
          data, layout,
          interpValue, derivTVal, derivXVal);

      detail::StoreSymmetric(interpolantPlane, sizeE, i, j, interpValue);
      detail::StoreSymmetric(derivativeTPlane, sizeE, i, j, derivTVal);
      detail::StoreSymmetric(derivativeXPlane, sizeE, i, j, derivXVal);
    }
  }

  return 0;
}

inline int LogInterpolateDifferentiateSingleVariable2D2DCustomAligned(
    std::size_t sizeE,
    const double* logT, const double* logX, std::size_t count,
    const double* gridT, int nT,
    const double* gridX, int nX,
    const double* data,
    double offset,
    double* interpolant,
    double* derivativeT,
    double* derivativeX,
    const InterpConfig& cfg = InterpConfig{}) noexcept
{
  if (logT == nullptr || logX == nullptr ||
      data == nullptr || interpolant == nullptr ||
      derivativeT == nullptr || derivativeX == nullptr ||
      gridT == nullptr || gridX == nullptr) {
    return 1;
  }
  if (sizeE == 0 || count == 0) {
    return 0;
  }

  const std::size_t planeSize = sizeE * sizeE;
  for (std::size_t k = 0; k < count; ++k) {
    double* planeInterp = interpolant + k * planeSize;
    double* planeDerivT = derivativeT + k * planeSize;
    double* planeDerivX = derivativeX + k * planeSize;
    const int rc = LogInterpolateDifferentiateSingleVariable2D2DCustomAlignedPoint(
        sizeE, logT[k], logX[k],
        gridT, nT, gridX, nX,
        data, offset,
        planeInterp, planeDerivT, planeDerivX,
        cfg);
    if (rc != 0) {
      return rc;
    }
  }

  return 0;
}

} // namespace WeakLibReader
