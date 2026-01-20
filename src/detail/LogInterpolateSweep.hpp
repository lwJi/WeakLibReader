#pragma once

#include <AMReX_GpuQualifiers.H>
#include <cstddef>

#include "LogInterpolateCore.hpp"

namespace WeakLibReader {

/// GPU-optimized 4D log-interpolation with 1D sweep (single point, multiple E values)
inline int LogInterpolateSingleVariable1D3DCustomPoint(
    const double* logE, std::size_t sizeE,
    double logD, double logT, double y,
    const double* gridE, int nE,
    const double* gridD, int nD,
    const double* gridT, int nT,
    const double* gridY, int nY,
    const double* data,
    double offset,
    double* out) noexcept
{
  if (logE == nullptr || out == nullptr || data == nullptr ||
      gridE == nullptr || gridD == nullptr || gridT == nullptr || gridY == nullptr) {
    return 1;
  }
  if (sizeE == 0) {
    return 0;
  }

  constexpr int ND = 4;
  Axis axes[ND] = {
      MakeAxis(gridE, nE, AxisScale::Linear),
      MakeAxis(gridD, nD, AxisScale::Linear),
      MakeAxis(gridT, nT, AxisScale::Linear),
      MakeAxis(gridY, nY, AxisScale::Linear)};
  int extents[ND] = {nE, nD, nT, nY};
  const Layout layout = MakeLayout(extents, ND);

  for (std::size_t i = 0; i < sizeE; ++i) {
    double coords[ND] = {logE[i], logD, logT, y};
    out[i] = detail::LogInterpolatedValueDirect<ND>(data, layout, axes, coords, offset);
  }

  return 0;
}

/// GPU-optimized 4D log-interpolation with 1D sweep (batch)
inline int LogInterpolateSingleVariable1D3DCustom(
    const double* logE, std::size_t sizeE,
    const double* logD, const double* logT, const double* y, std::size_t count,
    const double* gridE, int nE,
    const double* gridD, int nD,
    const double* gridT, int nT,
    const double* gridY, int nY,
    const double* data,
    double offset,
    double* out) noexcept
{
  if (logE == nullptr || logD == nullptr || logT == nullptr || y == nullptr ||
      out == nullptr || data == nullptr ||
      gridE == nullptr || gridD == nullptr || gridT == nullptr || gridY == nullptr) {
    return 1;
  }
  if (sizeE == 0 || count == 0) {
    return 0;
  }

  constexpr int ND = 4;
  Axis axes[ND] = {
      MakeAxis(gridE, nE, AxisScale::Linear),
      MakeAxis(gridD, nD, AxisScale::Linear),
      MakeAxis(gridT, nT, AxisScale::Linear),
      MakeAxis(gridY, nY, AxisScale::Linear)};
  int extents[ND] = {nE, nD, nT, nY};
  const Layout layout = MakeLayout(extents, ND);

  for (std::size_t j = 0; j < count; ++j) {
    double* row = out + j * sizeE;
    for (std::size_t i = 0; i < sizeE; ++i) {
      double coords[ND] = {logE[i], logD[j], logT[j], y[j]};
      row[i] = detail::LogInterpolatedValueDirect<ND>(data, layout, axes, coords, offset);
    }
  }

  return 0;
}

/// GPU-optimized 4D log-interpolation with 2D symmetric sweep (single point)
inline int LogInterpolateSingleVariable2D2DCustomPoint(
    const double* logE, std::size_t sizeE,
    double logT, double logX,
    const double* gridE, int nE,
    const double* gridT, int nT,
    const double* gridX, int nX,
    const double* data,
    double offset,
    double* out) noexcept
{
  if (logE == nullptr || data == nullptr || out == nullptr ||
      gridE == nullptr || gridT == nullptr || gridX == nullptr) {
    return 1;
  }
  if (sizeE == 0) {
    return 0;
  }

  constexpr int ND = 4;
  Axis axes[ND] = {
      MakeAxis(gridE, nE, AxisScale::Linear),
      MakeAxis(gridE, nE, AxisScale::Linear),
      MakeAxis(gridT, nT, AxisScale::Linear),
      MakeAxis(gridX, nX, AxisScale::Linear)};
  int extents[ND] = {nE, nE, nT, nX};
  const Layout layout = MakeLayout(extents, ND);

  for (std::size_t j = 0; j < sizeE; ++j) {
    for (std::size_t i = 0; i <= j; ++i) {
      double coords[ND] = {logE[i], logE[j], logT, logX};
      const double value = detail::LogInterpolatedValueDirect<ND>(data, layout, axes, coords, offset);
      detail::StoreSymmetric(out, sizeE, i, j, value);
    }
  }

  return 0;
}

/// GPU-optimized 4D log-interpolation with 2D symmetric sweep (batch)
inline int LogInterpolateSingleVariable2D2DCustom(
    const double* logE, std::size_t sizeE,
    const double* logT, const double* logX, std::size_t count,
    const double* gridE, int nE,
    const double* gridT, int nT,
    const double* gridX, int nX,
    const double* data,
    double offset,
    double* out) noexcept
{
  if (logE == nullptr || logT == nullptr || logX == nullptr ||
      out == nullptr || data == nullptr ||
      gridE == nullptr || gridT == nullptr || gridX == nullptr) {
    return 1;
  }
  if (sizeE == 0 || count == 0) {
    return 0;
  }

  constexpr int ND = 4;
  Axis axes[ND] = {
      MakeAxis(gridE, nE, AxisScale::Linear),
      MakeAxis(gridE, nE, AxisScale::Linear),
      MakeAxis(gridT, nT, AxisScale::Linear),
      MakeAxis(gridX, nX, AxisScale::Linear)};
  int extents[ND] = {nE, nE, nT, nX};
  const Layout layout = MakeLayout(extents, ND);

  const std::size_t planeSize = sizeE * sizeE;
  for (std::size_t l = 0; l < count; ++l) {
    double* plane = out + l * planeSize;
    for (std::size_t j = 0; j < sizeE; ++j) {
      for (std::size_t i = 0; i <= j; ++i) {
        double coords[ND] = {logE[i], logE[j], logT[l], logX[l]};
        const double value = detail::LogInterpolatedValueDirect<ND>(data, layout, axes, coords, offset);
        detail::StoreSymmetric(plane, sizeE, i, j, value);
      }
    }
  }

  return 0;
}

inline int LogInterpolateSingleVariable2D2DCustomAlignedPoint(
    std::size_t sizeE,
    double logT, double logX,
    const double* gridT, int nT,
    const double* gridX, int nX,
    const double* data,
    double offset,
    double* out) noexcept
{
  if (data == nullptr || out == nullptr) {
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
  detail::IndexAndDelta(axes[0], logT, idxT, fracT);
  detail::IndexAndDelta(axes[1], logX, idxX, fracX);

  for (std::size_t j = 0; j < sizeE; ++j) {
    for (std::size_t i = 0; i <= j; ++i) {
      const double value = LinearInterp2D4DArray2DAlignedPoint(
          static_cast<int>(i), static_cast<int>(j),
          idxT, idxX, fracT, fracX, offset,
          data, layout);
      detail::StoreSymmetric(out, sizeE, i, j, value);
    }
  }

  return 0;
}

inline int LogInterpolateSingleVariable2D2DCustomAligned(
    std::size_t sizeE,
    const double* logT, const double* logX, std::size_t count,
    const double* gridT, int nT,
    const double* gridX, int nX,
    const double* data,
    double offset,
    double* out) noexcept
{
  if (logT == nullptr || logX == nullptr || data == nullptr || out == nullptr ||
      gridT == nullptr || gridX == nullptr) {
    return 1;
  }
  if (sizeE == 0 || count == 0) {
    return 0;
  }

  const std::size_t planeSize = sizeE * sizeE;
  for (std::size_t k = 0; k < count; ++k) {
    double* plane = out + k * planeSize;
    const int rc = LogInterpolateSingleVariable2D2DCustomAlignedPoint(
        sizeE, logT[k], logX[k],
        gridT, nT, gridX, nX,
        data, offset, plane);
    if (rc != 0) {
      return rc;
    }
  }

  return 0;
}

} // namespace WeakLibReader
