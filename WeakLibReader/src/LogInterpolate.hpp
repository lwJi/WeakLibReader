#pragma once

#include <AMReX_GpuQualifiers.H>
#include <cstddef>
#include <limits>
#include <vector>

#include "InterpLogTable.hpp"
#include "Math.hpp"
#include "WeakLibReader.hpp"

namespace WeakLibReader {
namespace detail {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double LogInterpolatedValue(const double* data,
                            const Layout& layout,
                            const Axis axes[5],
                            const double coords[5],
                            double offset,
                            const InterpConfig& cfg,
                            int nd) noexcept
{
  const double logValue = InterpLinearND(data, layout, axes, coords, cfg, nd);
  return math::Pow10(logValue) - offset;
}

inline void StoreSymmetric(double* plane, std::size_t size,
                           std::size_t i, std::size_t j,
                           double value) noexcept
{
  const std::size_t idxLower = j * size + i;
  plane[idxLower] = value;
  if (i != j) {
    plane[i * size + j] = value;
  }
}

inline void FillNaNPlane(double* plane, std::size_t size) noexcept
{
  const std::size_t planeSize = size * size;
  const double nanValue = std::numeric_limits<double>::quiet_NaN();
  for (std::size_t idx = 0; idx < planeSize; ++idx) {
    plane[idx] = nanValue;
  }
}

inline void FillNaNVector(double* values, std::size_t count, std::size_t stride) noexcept
{
  const double nanValue = std::numeric_limits<double>::quiet_NaN();
  for (std::size_t i = 0; i < count; ++i) {
    for (std::size_t j = 0; j < stride; ++j) {
      values[i * stride + j] = nanValue;
    }
  }
}

inline bool ComputeLinearAxisScale(const Axis& axis, int idx, double& scale) noexcept
{
  const double span = axis.grid[idx + 1] - axis.grid[idx];
  if (!(span > 0.0)) {
    scale = std::numeric_limits<double>::quiet_NaN();
    return false;
  }
  scale = math::Ln10 / span;
  return true;
}

inline bool ComputeLogAxisScale(const Axis& axis, int idx, double coord,
                                double& scale) noexcept
{
  if (!(coord > 0.0)) {
    scale = std::numeric_limits<double>::quiet_NaN();
    return false;
  }
  const double ratio = axis.grid[idx + 1] / axis.grid[idx];
  if (!(ratio > 0.0)) {
    scale = std::numeric_limits<double>::quiet_NaN();
    return false;
  }
  const double denom = math::Log10(ratio);
  if (denom == 0.0) {
    scale = std::numeric_limits<double>::quiet_NaN();
    return false;
  }
  scale = 1.0 / (coord * denom);
  return true;
}

inline bool ComputeAxisScale(const Axis& axis, int idx, double coord,
                             double& scale) noexcept
{
  if (axis.grid == nullptr) {
    scale = std::numeric_limits<double>::quiet_NaN();
    return false;
  }
  if (axis.scale == AxisScale::Linear) {
    return ComputeLinearAxisScale(axis, idx, scale);
  }
  return ComputeLogAxisScale(axis, idx, coord, scale);
}

inline void SetNaN(double& value0, double& value1, double& value2) noexcept
{
  const double nanValue = std::numeric_limits<double>::quiet_NaN();
  value0 = nanValue;
  value1 = nanValue;
  value2 = nanValue;
}

inline void SetNaN(double& value0, double& value1,
                   double& value2, double& value3) noexcept
{
  const double nanValue = std::numeric_limits<double>::quiet_NaN();
  value0 = nanValue;
  value1 = nanValue;
  value2 = nanValue;
  value3 = nanValue;
}

inline int LogInterpolateDifferentiateSingleVariable3DCustomPointImpl(
    double d, double t, double y,
    const double* data, const Layout& layout,
    const Axis axes[3],
    double offset,
    double& interpolant,
    double derivatives[3],
    const InterpConfig& cfg) noexcept
{
  if (data == nullptr || axes == nullptr) {
    return 1;
  }
  if (layout.nd < 3) {
    return 2;
  }
  if (axes[0].grid == nullptr || axes[1].grid == nullptr || axes[2].grid == nullptr) {
    return 3;
  }

  const double coords[3] = {d, t, y};
  int indices[3] = {0, 0, 0};
  double fractions[3] = {0.0, 0.0, 0.0};
  bool outOfRange = false;

  for (int dim = 0; dim < 3; ++dim) {
    bool axisOut = detail::IndexAndDelta(axes[dim], coords[dim], indices[dim], fractions[dim]);
    if (axisOut) {
      outOfRange = true;
      fractions[dim] = detail::Clamp01(fractions[dim]);
    }
  }

  if (outOfRange) {
    if (cfg.outOfRange == OutOfRangePolicy::Error) {
      return 4;
    }
    if (cfg.outOfRange == OutOfRangePolicy::FillNaN) {
      detail::SetNaN(interpolant, derivatives[0], derivatives[1], derivatives[2]);
      return 0;
    }
  }

  double scales[3] = {0.0, 0.0, 0.0};
  bool ok =
      detail::ComputeAxisScale(axes[0], indices[0], coords[0], scales[0]) &&
      detail::ComputeAxisScale(axes[1], indices[1], coords[1], scales[1]) &&
      detail::ComputeAxisScale(axes[2], indices[2], coords[2], scales[2]);

  if (!ok) {
    detail::SetNaN(interpolant, derivatives[0], derivatives[1], derivatives[2]);
    return 5;
  }

  double dIdX0 = 0.0;
  double dIdX1 = 0.0;
  double dIdX2 = 0.0;

  LinearInterpDeriv3DPoint(indices[0], indices[1], indices[2],
                           fractions[0], fractions[1], fractions[2],
                           scales[0], scales[1], scales[2],
                           offset, data, layout,
                           interpolant, dIdX0, dIdX1, dIdX2);

  derivatives[0] = dIdX0;
  derivatives[1] = dIdX1;
  derivatives[2] = dIdX2;

  return 0;
}

} // namespace detail

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double LogInterpolateSingleVariable2DCustomPoint(
    double x0, double x1,
    const double* grid0, int n0,
    const double* grid1, int n1,
    const double* data,
    double offset,
    const InterpConfig& cfg = InterpConfig{}) noexcept
{
  if (data == nullptr || grid0 == nullptr || grid1 == nullptr) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  Axis axes[5] = {
      MakeAxis(grid0, n0, AxisScale::Linear),
      MakeAxis(grid1, n1, AxisScale::Linear)};
  int extents[2] = {n0, n1};
  const Layout layout = MakeLayout(extents, 2);
  double coords[5] = {x0, x1, 0.0, 0.0, 0.0};
  return detail::LogInterpolatedValue(data, layout, axes, coords, offset, cfg, 2);
}

inline int LogInterpolateSingleVariable2DCustom(
    const double* x0, const double* x1, std::size_t count,
    const double* grid0, int n0,
    const double* grid1, int n1,
    const double* data,
    double offset,
    double* out,
    const InterpConfig& cfg = InterpConfig{}) noexcept
{
  if (x0 == nullptr || x1 == nullptr || out == nullptr ||
      data == nullptr || grid0 == nullptr || grid1 == nullptr) {
    return 1;
  }
  Axis axes[5] = {
      MakeAxis(grid0, n0, AxisScale::Linear),
      MakeAxis(grid1, n1, AxisScale::Linear)};
  int extents[2] = {n0, n1};
  const Layout layout = MakeLayout(extents, 2);
  for (std::size_t i = 0; i < count; ++i) {
    double coords[5] = {x0[i], x1[i], 0.0, 0.0, 0.0};
    out[i] = detail::LogInterpolatedValue(data, layout, axes, coords, offset, cfg, 2);
  }
  return 0;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double LogInterpolateSingleVariable3DCustomPoint(
    double x0, double x1, double x2,
    const double* grid0, int n0,
    const double* grid1, int n1,
    const double* grid2, int n2,
    const double* data,
    double offset,
    const InterpConfig& cfg = InterpConfig{}) noexcept
{
  if (data == nullptr || grid0 == nullptr || grid1 == nullptr || grid2 == nullptr) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  Axis axes[5] = {
      MakeAxis(grid0, n0, AxisScale::Linear),
      MakeAxis(grid1, n1, AxisScale::Linear),
      MakeAxis(grid2, n2, AxisScale::Linear)};
  int extents[3] = {n0, n1, n2};
  const Layout layout = MakeLayout(extents, 3);
  double coords[5] = {x0, x1, x2, 0.0, 0.0};
  return detail::LogInterpolatedValue(data, layout, axes, coords, offset, cfg, 3);
}

inline int LogInterpolateSingleVariable3DCustom(
    const double* x0, const double* x1, const double* x2, std::size_t count,
    const double* grid0, int n0,
    const double* grid1, int n1,
    const double* grid2, int n2,
    const double* data,
    double offset,
    double* out,
    const InterpConfig& cfg = InterpConfig{}) noexcept
{
  if (x0 == nullptr || x1 == nullptr || x2 == nullptr ||
      out == nullptr || data == nullptr ||
      grid0 == nullptr || grid1 == nullptr || grid2 == nullptr) {
    return 1;
  }
  Axis axes[5] = {
      MakeAxis(grid0, n0, AxisScale::Linear),
      MakeAxis(grid1, n1, AxisScale::Linear),
      MakeAxis(grid2, n2, AxisScale::Linear)};
  int extents[3] = {n0, n1, n2};
  const Layout layout = MakeLayout(extents, 3);
  for (std::size_t i = 0; i < count; ++i) {
    double coords[5] = {x0[i], x1[i], x2[i], 0.0, 0.0};
    out[i] = detail::LogInterpolatedValue(data, layout, axes, coords, offset, cfg, 3);
  }
  return 0;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double LogInterpolateSingleVariable4DCustomPoint(
    double x0, double x1, double x2, double x3,
    const double* grid0, int n0,
    const double* grid1, int n1,
    const double* grid2, int n2,
    const double* grid3, int n3,
    const double* data,
    double offset,
    const InterpConfig& cfg = InterpConfig{}) noexcept
{
  if (data == nullptr || grid0 == nullptr || grid1 == nullptr ||
      grid2 == nullptr || grid3 == nullptr) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  Axis axes[5] = {
      MakeAxis(grid0, n0, AxisScale::Linear),
      MakeAxis(grid1, n1, AxisScale::Linear),
      MakeAxis(grid2, n2, AxisScale::Linear),
      MakeAxis(grid3, n3, AxisScale::Linear)};
  int extents[4] = {n0, n1, n2, n3};
  const Layout layout = MakeLayout(extents, 4);
  double coords[5] = {x0, x1, x2, x3, 0.0};
  return detail::LogInterpolatedValue(data, layout, axes, coords, offset, cfg, 4);
}

inline int LogInterpolateSingleVariable4DCustom(
    const double* x0, const double* x1, const double* x2, const double* x3,
    std::size_t count,
    const double* grid0, int n0,
    const double* grid1, int n1,
    const double* grid2, int n2,
    const double* grid3, int n3,
    const double* data,
    double offset,
    double* out,
    const InterpConfig& cfg = InterpConfig{}) noexcept
{
  if (x0 == nullptr || x1 == nullptr || x2 == nullptr || x3 == nullptr ||
      out == nullptr || data == nullptr ||
      grid0 == nullptr || grid1 == nullptr || grid2 == nullptr || grid3 == nullptr) {
    return 1;
  }
  Axis axes[5] = {
      MakeAxis(grid0, n0, AxisScale::Linear),
      MakeAxis(grid1, n1, AxisScale::Linear),
      MakeAxis(grid2, n2, AxisScale::Linear),
      MakeAxis(grid3, n3, AxisScale::Linear)};
  int extents[4] = {n0, n1, n2, n3};
  const Layout layout = MakeLayout(extents, 4);
  for (std::size_t i = 0; i < count; ++i) {
    double coords[5] = {x0[i], x1[i], x2[i], x3[i], 0.0};
    out[i] = detail::LogInterpolatedValue(data, layout, axes, coords, offset, cfg, 4);
  }
  return 0;
}

inline int LogInterpolateSingleVariable1D3DCustomPoint(
    const double* logE, std::size_t sizeE,
    double logD, double logT, double y,
    const double* gridE, int nE,
    const double* gridD, int nD,
    const double* gridT, int nT,
    const double* gridY, int nY,
    const double* data,
    double offset,
    double* out,
    const InterpConfig& cfg = InterpConfig{}) noexcept
{
  if (logE == nullptr || out == nullptr || data == nullptr ||
      gridE == nullptr || gridD == nullptr || gridT == nullptr || gridY == nullptr) {
    return 1;
  }
  if (sizeE == 0) {
    return 0;
  }

  Axis axes[5] = {
      MakeAxis(gridE, nE, AxisScale::Linear),
      MakeAxis(gridD, nD, AxisScale::Linear),
      MakeAxis(gridT, nT, AxisScale::Linear),
      MakeAxis(gridY, nY, AxisScale::Linear)};
  int extents[4] = {nE, nD, nT, nY};
  const Layout layout = MakeLayout(extents, 4);

  for (std::size_t i = 0; i < sizeE; ++i) {
    double coords[5] = {logE[i], logD, logT, y, 0.0};
    out[i] = detail::LogInterpolatedValue(data, layout, axes, coords, offset, cfg, 4);
  }

  return 0;
}

inline int LogInterpolateSingleVariable1D3DCustom(
    const double* logE, std::size_t sizeE,
    const double* logD, const double* logT, const double* y, std::size_t count,
    const double* gridE, int nE,
    const double* gridD, int nD,
    const double* gridT, int nT,
    const double* gridY, int nY,
    const double* data,
    double offset,
    double* out,
    const InterpConfig& cfg = InterpConfig{}) noexcept
{
  if (logE == nullptr || logD == nullptr || logT == nullptr || y == nullptr ||
      out == nullptr || data == nullptr ||
      gridE == nullptr || gridD == nullptr || gridT == nullptr || gridY == nullptr) {
    return 1;
  }
  if (sizeE == 0 || count == 0) {
    return 0;
  }

  Axis axes[5] = {
      MakeAxis(gridE, nE, AxisScale::Linear),
      MakeAxis(gridD, nD, AxisScale::Linear),
      MakeAxis(gridT, nT, AxisScale::Linear),
      MakeAxis(gridY, nY, AxisScale::Linear)};
  int extents[4] = {nE, nD, nT, nY};
  const Layout layout = MakeLayout(extents, 4);

  for (std::size_t j = 0; j < count; ++j) {
    double* row = out + j * sizeE;
    for (std::size_t i = 0; i < sizeE; ++i) {
      double coords[5] = {logE[i], logD[j], logT[j], y[j], 0.0};
      row[i] = detail::LogInterpolatedValue(data, layout, axes, coords, offset, cfg, 4);
    }
  }

  return 0;
}

inline int LogInterpolateSingleVariable2D2DCustomPoint(
    const double* logE, std::size_t sizeE,
    double logT, double logX,
    const double* gridE, int nE,
    const double* gridT, int nT,
    const double* gridX, int nX,
    const double* data,
    double offset,
    double* out,
    const InterpConfig& cfg = InterpConfig{}) noexcept
{
  if (logE == nullptr || data == nullptr || out == nullptr ||
      gridE == nullptr || gridT == nullptr || gridX == nullptr) {
    return 1;
  }
  if (sizeE == 0) {
    return 0;
  }

  Axis axes[5] = {
      MakeAxis(gridE, nE, AxisScale::Linear),
      MakeAxis(gridE, nE, AxisScale::Linear),
      MakeAxis(gridT, nT, AxisScale::Linear),
      MakeAxis(gridX, nX, AxisScale::Linear)};
  int extents[4] = {nE, nE, nT, nX};
  const Layout layout = MakeLayout(extents, 4);

  for (std::size_t j = 0; j < sizeE; ++j) {
    for (std::size_t i = 0; i <= j; ++i) {
      double coords[5] = {logE[i], logE[j], logT, logX, 0.0};
      const double value = detail::LogInterpolatedValue(data, layout, axes, coords, offset, cfg, 4);
      detail::StoreSymmetric(out, sizeE, i, j, value);
    }
  }

  return 0;
}

inline int LogInterpolateSingleVariable2D2DCustom(
    const double* logE, std::size_t sizeE,
    const double* logT, const double* logX, std::size_t count,
    const double* gridE, int nE,
    const double* gridT, int nT,
    const double* gridX, int nX,
    const double* data,
    double offset,
    double* out,
    const InterpConfig& cfg = InterpConfig{}) noexcept
{
  if (logE == nullptr || logT == nullptr || logX == nullptr ||
      out == nullptr || data == nullptr ||
      gridE == nullptr || gridT == nullptr || gridX == nullptr) {
    return 1;
  }
  if (sizeE == 0 || count == 0) {
    return 0;
  }

  Axis axes[5] = {
      MakeAxis(gridE, nE, AxisScale::Linear),
      MakeAxis(gridE, nE, AxisScale::Linear),
      MakeAxis(gridT, nT, AxisScale::Linear),
      MakeAxis(gridX, nX, AxisScale::Linear)};
  int extents[4] = {nE, nE, nT, nX};
  const Layout layout = MakeLayout(extents, 4);

  const std::size_t planeSize = sizeE * sizeE;
  for (std::size_t l = 0; l < count; ++l) {
    double* plane = out + l * planeSize;
    for (std::size_t j = 0; j < sizeE; ++j) {
      for (std::size_t i = 0; i <= j; ++i) {
        double coords[5] = {logE[i], logE[j], logT[l], logX[l], 0.0};
        const double value = detail::LogInterpolatedValue(data, layout, axes, coords, offset, cfg, 4);
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
    double* out,
    const InterpConfig& cfg = InterpConfig{}) noexcept
{
  Axis axes[2] = {
      MakeAxis(gridT, nT, AxisScale::Linear),
      MakeAxis(gridX, nX, AxisScale::Linear)};
  int extents[4] = {
      static_cast<int>(sizeE),
      static_cast<int>(sizeE),
      nT,
      nX};
  const Layout layout = MakeLayout(extents, 4);
  if (data == nullptr || out == nullptr) {
    return 1;
  }
  if (sizeE == 0) {
    return 0;
  }

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
      detail::FillNaNPlane(out, sizeE);
      return 0;
    }
    fracT = detail::Clamp01(fracT);
    fracX = detail::Clamp01(fracX);
  }

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
    double* out,
    const InterpConfig& cfg = InterpConfig{}) noexcept
{
  if (logT == nullptr || logX == nullptr || data == nullptr || out == nullptr ||
      gridT == nullptr || gridX == nullptr) {
    return 1;
  }
  if (sizeE == 0 || count == 0) {
    return 0;
  }

  Axis axesLocal[2] = {
      MakeAxis(gridT, nT, AxisScale::Linear),
      MakeAxis(gridX, nX, AxisScale::Linear)};
  int extents[4] = {
      static_cast<int>(sizeE),
      static_cast<int>(sizeE),
      nT,
      nX};
  const Layout layout = MakeLayout(extents, 4);

  const std::size_t planeSize = sizeE * sizeE;
  for (std::size_t k = 0; k < count; ++k) {
    double* plane = out + k * planeSize;
    const int rc = LogInterpolateSingleVariable2D2DCustomAlignedPoint(
        sizeE, logT[k], logX[k],
        gridT, nT, gridX, nX,
        data, offset, plane, cfg);
    if (rc != 0) {
      return rc;
    }
  }

  return 0;
}

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

inline int SumLogInterpolateSingleVariable2D2DCustomAligned(
    std::size_t sizeE,
    const double* logD, std::size_t nAlpha,
    const double* logT, std::size_t count,
    const double* gridD, int nD,
    const double* gridT, int nT,
    const double* alpha,
    const double* data,
    double offset,
    double* out,
    const InterpConfig& cfg = InterpConfig{}) noexcept
{
  if (logD == nullptr || logT == nullptr || data == nullptr ||
      alpha == nullptr || out == nullptr ||
      gridD == nullptr || gridT == nullptr) {
    return 1;
  }
  if (sizeE == 0 || nAlpha == 0 || count == 0) {
    return 0;
  }

  Axis axes[2] = {
      MakeAxis(gridD, nD, AxisScale::Linear),
      MakeAxis(gridT, nT, AxisScale::Linear)};
  int extents[4] = {
      static_cast<int>(sizeE),
      static_cast<int>(sizeE),
      nD,
      nT};
  const Layout layout = MakeLayout(extents, 4);

  const std::size_t planeSize = sizeE * sizeE;
  std::vector<int> idxD(nAlpha);
  std::vector<double> fracD(nAlpha);

  for (std::size_t k = 0; k < count; ++k) {
    double* plane = out + k * planeSize;

    int idxT = 0;
    double fracT = 0.0;
    bool outT = detail::IndexAndDelta(axes[1], logT[k], idxT, fracT);

    if (outT) {
      if (cfg.outOfRange == OutOfRangePolicy::Error) {
        return 4;
      }
      if (cfg.outOfRange == OutOfRangePolicy::FillNaN) {
        detail::FillNaNPlane(plane, sizeE);
        continue;
      }
      fracT = detail::Clamp01(fracT);
    }

    bool skipPlane = false;

    for (std::size_t l = 0; l < nAlpha; ++l) {
      int idx = 0;
      double frac = 0.0;
      const double value = logD[k * nAlpha + l];
      bool outD = detail::IndexAndDelta(axes[0], value, idx, frac);
      if (outD) {
        if (cfg.outOfRange == OutOfRangePolicy::Error) {
          return 5;
        }
        if (cfg.outOfRange == OutOfRangePolicy::FillNaN) {
          detail::FillNaNPlane(plane, sizeE);
          skipPlane = true;
          break;
        }
        frac = detail::Clamp01(frac);
      }
      idxD[l] = idx;
      fracD[l] = frac;
    }

    if (skipPlane) {
      continue;
    }

    for (std::size_t j = 0; j < sizeE; ++j) {
      for (std::size_t i = 0; i <= j; ++i) {
        double sum = 0.0;
        for (std::size_t l = 0; l < nAlpha; ++l) {
          const double interp = LinearInterp2D4DArray2DAlignedPoint(
              static_cast<int>(i), static_cast<int>(j),
              idxD[l], idxT,
              fracD[l], fracT,
              offset,
              data, layout);
          sum += alpha[l] * interp;
        }
        detail::StoreSymmetric(plane, sizeE, i, j, sum);
      }
    }
  }

  return 0;
}

} // namespace WeakLibReader
