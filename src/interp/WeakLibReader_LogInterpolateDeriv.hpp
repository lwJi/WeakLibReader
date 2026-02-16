#pragma once

#include <AMReX_GpuQualifiers.H>
#include <cstddef>

#include "interp/WeakLibReader_LogInterpolateCore.hpp"

namespace WeakLibReader {

/// GPU-optimized 3D log-interpolation with derivatives (single point)
inline int LogInterpolateDifferentiateSingleVariable3DCustomPoint(
    double d, double t, double y,
    const Axis axes[3],
    const double* data,
    double offset,
    double& interpolant,
    double derivatives[3]) noexcept
{
  constexpr int ND = 3;
  if (!detail::AxesAndDataValid<ND>(axes, data)) {
    return 1;
  }

  const Layout layout = detail::MakeAxisLayout<ND>(axes);

  detail::LogInterpolateDifferentiateSingleVariable3DCustomPointImpl(
      d, t, y, data, layout, axes, offset, interpolant, derivatives);
  return 0;
}

/// GPU-optimized 3D log-interpolation with derivatives (batch)
inline int LogInterpolateDifferentiateSingleVariable3DCustom(
    const double* d, const double* t, const double* y, std::size_t count,
    const Axis axes[3],
    const double* data,
    double offset,
    double* interpolants,
    double* derivatives) noexcept
{
  constexpr int ND = 3;
  if (d == nullptr || t == nullptr || y == nullptr ||
      interpolants == nullptr || derivatives == nullptr ||
      !detail::AxesAndDataValid<ND>(axes, data)) {
    return 1;
  }

  const Layout layout = detail::MakeAxisLayout<ND>(axes);

  for (std::size_t i = 0; i < count; ++i) {
    double deriv[3] = {0.0, 0.0, 0.0};
    double interp = 0.0;
    detail::LogInterpolateDifferentiateSingleVariable3DCustomPointImpl(
        d[i], t[i], y[i],
        data, layout, axes,
        offset, interp, deriv);
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
    const Axis axes[4],
    const double* data,
    double offset,
    double* interpolantPlane,
    double* derivativeTPlane,
    double* derivativeXPlane) noexcept
{
  constexpr int ND = 4;
  if (logE == nullptr || interpolantPlane == nullptr ||
      derivativeTPlane == nullptr || derivativeXPlane == nullptr ||
      !detail::AxesAndDataValid<ND>(axes, data)) {
    return 1;
  }
  if (sizeE == 0) {
    return 0;
  }

  const Layout layout = detail::MakeAxisLayout<ND>(axes);

  int idxT = 0;
  int idxX = 0;
  double fracT = 0.0;
  double fracX = 0.0;
  detail::IndexAndDelta(axes[2], logT, idxT, fracT);
  detail::IndexAndDelta(axes[3], logX, idxX, fracX);

  const double spanT = axes[2].grid[idxT + 1] - axes[2].grid[idxT];
  const double spanX = axes[3].grid[idxX + 1] - axes[3].grid[idxX];

  const double aT = 1.0 / (spanT * math::Pow10(logT));
  const double aX = 1.0 / (spanX * math::Pow10(logX));

  detail::ForEachSymmetricUpper(sizeE, [&](std::size_t i, std::size_t j) noexcept {
    int idxE1 = 0;
    int idxE2 = 0;
    double fracE1 = 0.0;
    double fracE2 = 0.0;
    detail::IndexAndDelta(axes[0], logE[i], idxE1, fracE1);
    detail::IndexAndDelta(axes[1], logE[j], idxE2, fracE2);

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
  });

  return 0;
}

inline int LogInterpolateDifferentiateSingleVariable2D2DCustom(
    const double* logE, std::size_t sizeE,
    const double* logT, const double* logX, std::size_t count,
    const Axis axes[4],
    const double* data,
    double offset,
    double* interpolant,
    double* derivativeT,
    double* derivativeX) noexcept
{
  constexpr int ND = 4;
  if (logE == nullptr || logT == nullptr || logX == nullptr ||
      interpolant == nullptr || derivativeT == nullptr || derivativeX == nullptr ||
      !detail::AxesAndDataValid<ND>(axes, data)) {
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
        axes,
        data, offset,
        planeInterp, planeDerivT, planeDerivX);
    if (rc != 0) {
      return rc;
    }
  }

  return 0;
}

inline int LogInterpolateDifferentiateSingleVariable2D2DCustomAlignedPoint(
    std::size_t sizeE,
    double logT, double logX,
    const Axis axes[2],
    const double* data,
    double offset,
    double* interpolantPlane,
    double* derivativeTPlane,
    double* derivativeXPlane) noexcept
{
  if (data == nullptr || interpolantPlane == nullptr ||
      derivativeTPlane == nullptr || derivativeXPlane == nullptr ||
      axes[0].grid == nullptr || axes[1].grid == nullptr) {
    return 1;
  }
  if (sizeE == 0) {
    return 0;
  }

  int extents[4] = {
      static_cast<int>(sizeE),
      static_cast<int>(sizeE),
      axes[0].n,
      axes[1].n};
  const Layout layout = MakeLayout(extents, 4);

  int idxT = 0;
  int idxX = 0;
  double fracT = 0.0;
  double fracX = 0.0;
  detail::IndexAndDelta(axes[0], logT, idxT, fracT);
  detail::IndexAndDelta(axes[1], logX, idxX, fracX);

  const double spanT = axes[0].grid[idxT + 1] - axes[0].grid[idxT];
  const double spanX = axes[1].grid[idxX + 1] - axes[1].grid[idxX];

  const double aT = 1.0 / (spanT * math::Pow10(logT));
  const double aX = 1.0 / (spanX * math::Pow10(logX));

  detail::ForEachSymmetricUpper(sizeE, [&](std::size_t i, std::size_t j) noexcept {
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
  });

  return 0;
}

inline int LogInterpolateDifferentiateSingleVariable2D2DCustomAligned(
    std::size_t sizeE,
    const double* logT, const double* logX, std::size_t count,
    const Axis axes[2],
    const double* data,
    double offset,
    double* interpolant,
    double* derivativeT,
    double* derivativeX) noexcept
{
  if (logT == nullptr || logX == nullptr ||
      data == nullptr || interpolant == nullptr ||
      derivativeT == nullptr || derivativeX == nullptr ||
      axes[0].grid == nullptr || axes[1].grid == nullptr) {
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
        axes,
        data, offset,
        planeInterp, planeDerivT, planeDerivX);
    if (rc != 0) {
      return rc;
    }
  }

  return 0;
}

} // namespace WeakLibReader
