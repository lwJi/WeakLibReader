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

template <int ND>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void PackAxes(const Axis* src, Axis* dst) noexcept
{
  for (int dim = 0; dim < ND; ++dim) {
    dst[dim] = src[dim];
  }
  for (int dim = ND; dim < 5; ++dim) {
    dst[dim] = Axis{};
  }
}

template <int ND>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void PackCoords(const double* src, double* dst) noexcept
{
  for (int dim = 0; dim < ND; ++dim) {
    dst[dim] = src[dim];
  }
  for (int dim = ND; dim < 5; ++dim) {
    dst[dim] = 0.0;
  }
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

inline void StoreSymmetricTriple(double* plane0, double* plane1, double* plane2,
                                 std::size_t size,
                                 std::size_t i, std::size_t j,
                                 double v0, double v1, double v2) noexcept
{
  detail::StoreSymmetric(plane0, size, i, j, v0);
  detail::StoreSymmetric(plane1, size, i, j, v1);
  detail::StoreSymmetric(plane2, size, i, j, v2);
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

inline void SetNaN(double& value) noexcept
{
  value = std::numeric_limits<double>::quiet_NaN();
}

inline void SetNaN(double& value0, double& value1) noexcept
{
  const double nanValue = std::numeric_limits<double>::quiet_NaN();
  value0 = nanValue;
  value1 = nanValue;
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

} // namespace detail

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double LogInterpolateLinearND(const double* data, const Layout& layout,
                              const Axis axes[5], const double coords[5],
                              double offset,
                              const InterpConfig& cfg,
                              int nd) noexcept
{
  return detail::LogInterpolatedValue(data, layout, axes, coords, offset, cfg, nd);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double LogInterpolateSingleVariable1DCustomPoint(
    const double* data, const Layout& layout,
    const Axis axes[1],
    double x0,
    double offset,
    const InterpConfig& cfg = InterpConfig{}) noexcept
{
  Axis packedAxes[5];
  double coords[5];
  const double localCoords[1] = {x0};
  detail::PackAxes<1>(axes, packedAxes);
  detail::PackCoords<1>(localCoords, coords);
  return detail::LogInterpolatedValue(data, layout, packedAxes, coords, offset, cfg, 1);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double LogInterpolateSingleVariable2DCustomPoint(
    const double* data, const Layout& layout,
    const Axis axes[2],
    double x0, double x1,
    double offset,
    const InterpConfig& cfg = InterpConfig{}) noexcept
{
  Axis packedAxes[5];
  double coords[5];
  const double localCoords[2] = {x0, x1};
  detail::PackAxes<2>(axes, packedAxes);
  detail::PackCoords<2>(localCoords, coords);
  return detail::LogInterpolatedValue(data, layout, packedAxes, coords, offset, cfg, 2);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double LogInterpolateSingleVariable3DCustomPoint(
    const double* data, const Layout& layout,
    const Axis axes[3],
    double x0, double x1, double x2,
    double offset,
    const InterpConfig& cfg = InterpConfig{}) noexcept
{
  Axis packedAxes[5];
  double coords[5];
  const double localCoords[3] = {x0, x1, x2};
  detail::PackAxes<3>(axes, packedAxes);
  detail::PackCoords<3>(localCoords, coords);
  return detail::LogInterpolatedValue(data, layout, packedAxes, coords, offset, cfg, 3);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double LogInterpolateSingleVariable4DCustomPoint(
    const double* data, const Layout& layout,
    const Axis axes[4],
    double x0, double x1, double x2, double x3,
    double offset,
    const InterpConfig& cfg = InterpConfig{}) noexcept
{
  Axis packedAxes[5];
  double coords[5];
  const double localCoords[4] = {x0, x1, x2, x3};
  detail::PackAxes<4>(axes, packedAxes);
  detail::PackCoords<4>(localCoords, coords);
  return detail::LogInterpolatedValue(data, layout, packedAxes, coords, offset, cfg, 4);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double LogInterpolateSingleVariable5DCustomPoint(
    const double* data, const Layout& layout,
    const Axis axes[5],
    double x0, double x1, double x2, double x3, double x4,
    double offset,
    const InterpConfig& cfg = InterpConfig{}) noexcept
{
  Axis packedAxes[5];
  double coords[5];
  const double localCoords[5] = {x0, x1, x2, x3, x4};
  detail::PackAxes<5>(axes, packedAxes);
  detail::PackCoords<5>(localCoords, coords);
  return detail::LogInterpolatedValue(data, layout, packedAxes, coords, offset, cfg, 5);
}

inline int LogInterpolateDifferentiateSingleVariable3DCustomPoint(
    double d, double t, double y,
    const double* data, const Layout& layout,
    const Axis axes[3],
    double offset,
    double& interpolant,
    double derivatives[3],
    const InterpConfig& cfg = InterpConfig{}) noexcept
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

inline int LogInterpolateDifferentiateSingleVariable3DCustom(
    const double* d, const double* t, const double* y, std::size_t count,
    const double* data, const Layout& layout,
    const Axis axes[3],
    double offset,
    double* interpolants,
    double* derivatives,
    const InterpConfig& cfg = InterpConfig{}) noexcept
{
  if (d == nullptr || t == nullptr || y == nullptr ||
      data == nullptr || interpolants == nullptr || derivatives == nullptr ||
      axes == nullptr) {
    return 1;
  }
  if (layout.nd < 3) {
    return 2;
  }

  for (std::size_t i = 0; i < count; ++i) {
    double deriv[3] = {0.0, 0.0, 0.0};
    double interp = 0.0;
    const int rc = LogInterpolateDifferentiateSingleVariable3DCustomPoint(
        d[i], t[i], y[i],
        data, layout, axes,
        offset, interp, deriv, cfg);
    if (rc != 0) {
      if (rc == 4 && cfg.outOfRange == OutOfRangePolicy::FillNaN) {
        // Should not reach here due to internal handling, but keep for completeness.
        interpolants[i] = std::numeric_limits<double>::quiet_NaN();
        derivatives[i * 3 + 0] = std::numeric_limits<double>::quiet_NaN();
        derivatives[i * 3 + 1] = std::numeric_limits<double>::quiet_NaN();
        derivatives[i * 3 + 2] = std::numeric_limits<double>::quiet_NaN();
        continue;
      }
      return rc;
    }
    interpolants[i] = interp;
    derivatives[i * 3 + 0] = deriv[0];
    derivatives[i * 3 + 1] = deriv[1];
    derivatives[i * 3 + 2] = deriv[2];
  }

  return 0;
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
  Axis axesLocal[3] = {
      MakeAxis(gridD, nD, AxisScale::Log10),
      MakeAxis(gridT, nT, AxisScale::Log10),
      MakeAxis(gridY, nY, AxisScale::Linear)};
  int extents[3] = {nD, nT, nY};
  const Layout layout = MakeLayout(extents, 3);
  return LogInterpolateDifferentiateSingleVariable3DCustom(
      d, t, y, count,
      data, layout, axesLocal,
      offset, interpolants, derivatives, cfg);
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
  return LogInterpolateDifferentiateSingleVariable3DCustomPoint(
      d, t, y, data, layout, axesLocal,
      offset, interpolant, derivatives, cfg);
}

inline int LogInterpolateSingleVariable2DCustom(
    const double* x0, const double* x1, std::size_t count,
    const double* data, const Layout& layout,
    const Axis axes[2],
    double offset,
    double* out,
    const InterpConfig& cfg = InterpConfig{}) noexcept
{
  if (x0 == nullptr || x1 == nullptr || data == nullptr || out == nullptr ||
      axes == nullptr) {
    return 1;
  }
  if (layout.nd < 2) {
    return 2;
  }
  if (axes[0].grid == nullptr || axes[1].grid == nullptr) {
    return 3;
  }
  for (std::size_t i = 0; i < count; ++i) {
    out[i] = LogInterpolateSingleVariable2DCustomPoint(
        data, layout, axes, x0[i], x1[i], offset, cfg);
  }
  return 0;
}

inline int LogInterpolateSingleVariable2DCustom(
    const double* x0, const double* x1, std::size_t count,
    const double* grid0, int n0, AxisScale scale0,
    const double* grid1, int n1, AxisScale scale1,
    const double* data,
    double offset,
    double* out,
    const InterpConfig& cfg) noexcept
{
  Axis axesLocal[2] = {
      MakeAxis(grid0, n0, scale0),
      MakeAxis(grid1, n1, scale1)};
  int extents[2] = {n0, n1};
  const Layout layout = MakeLayout(extents, 2);
  return LogInterpolateSingleVariable2DCustom(
      x0, x1, count, data, layout, axesLocal, offset, out, cfg);
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
  return LogInterpolateSingleVariable2DCustom(
      x0, x1, count,
      grid0, n0, AxisScale::Linear,
      grid1, n1, AxisScale::Linear,
      data, offset, out, cfg);
}

inline int LogInterpolateSingleVariable2D2DCustomAlignedPoint(
    std::size_t sizeE,
    double logT, double logX,
    const double* data, const Layout& layout,
    const Axis axes[2],
    double offset,
    double* out,
    const InterpConfig& cfg = InterpConfig{}) noexcept
{
  if (data == nullptr || out == nullptr || axes == nullptr) {
    return 1;
  }
  if (sizeE == 0) {
    return 0;
  }
  if (layout.nd < 4) {
    return 2;
  }
  if (axes[0].grid == nullptr || axes[1].grid == nullptr) {
    return 3;
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

inline int LogInterpolateSingleVariable2D2DCustomAlignedPoint(
    std::size_t sizeE,
    double logT, double logX,
    const double* gridT, int nT, AxisScale scaleT,
    const double* gridX, int nX, AxisScale scaleX,
    const double* data,
    double offset,
    double* out,
    const InterpConfig& cfg) noexcept
{
  Axis axesLocal[2] = {
      MakeAxis(gridT, nT, scaleT),
      MakeAxis(gridX, nX, scaleX)};
  int extents[4] = {
      static_cast<int>(sizeE),
      static_cast<int>(sizeE),
      nT,
      nX};
  const Layout layout = MakeLayout(extents, 4);
  return LogInterpolateSingleVariable2D2DCustomAlignedPoint(
      sizeE, logT, logX, data, layout, axesLocal, offset, out, cfg);
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
  return LogInterpolateSingleVariable2D2DCustomAlignedPoint(
      sizeE, logT, logX,
      gridT, nT, AxisScale::Linear,
      gridX, nX, AxisScale::Linear,
      data, offset, out, cfg);
}

inline int LogInterpolateSingleVariable2D2DCustomPoint(
    const double* logE, std::size_t sizeE,
    double logT, double logX,
    const double* data, const Layout& layout,
    const Axis axes[4],
    double offset,
    double* out,
    const InterpConfig& cfg = InterpConfig{}) noexcept
{
  if (logE == nullptr || data == nullptr || out == nullptr || axes == nullptr) {
    return 1;
  }
  if (sizeE == 0) {
    return 0;
  }
  if (layout.nd < 4) {
    return 2;
  }
  if (axes[0].grid == nullptr || axes[1].grid == nullptr ||
      axes[2].grid == nullptr || axes[3].grid == nullptr) {
    return 3;
  }

  for (std::size_t j = 0; j < sizeE; ++j) {
    for (std::size_t i = 0; i <= j; ++i) {
      const double value = LogInterpolateSingleVariable4DCustomPoint(
          data, layout, axes, logE[i], logE[j], logT, logX, offset, cfg);
      detail::StoreSymmetric(out, sizeE, i, j, value);
    }
  }

  return 0;
}

inline int LogInterpolateSingleVariable2D2DCustomAligned(
    std::size_t sizeE,
    const double* logT, const double* logX, std::size_t count,
    const double* data, const Layout& layout,
    const Axis axes[2],
    double offset,
    double* out,
    const InterpConfig& cfg = InterpConfig{}) noexcept
{
  if (logT == nullptr || logX == nullptr ||
      data == nullptr || out == nullptr || axes == nullptr) {
    return 1;
  }
  if (sizeE == 0 || count == 0) {
    return 0;
  }
  if (layout.nd < 4) {
    return 2;
  }
  if (axes[0].grid == nullptr || axes[1].grid == nullptr) {
    return 3;
  }

  const std::size_t planeSize = sizeE * sizeE;
  for (std::size_t k = 0; k < count; ++k) {
    double* plane = out + k * planeSize;
    const int rc = LogInterpolateSingleVariable2D2DCustomAlignedPoint(
        sizeE, logT[k], logX[k], data, layout, axes, offset, plane, cfg);
    if (rc != 0) {
      return rc;
    }
  }

  return 0;
}

inline int LogInterpolateSingleVariable2D2DCustomAligned(
    std::size_t sizeE,
    const double* logT, const double* logX, std::size_t count,
    const double* gridT, int nT, AxisScale scaleT,
    const double* gridX, int nX, AxisScale scaleX,
    const double* data,
    double offset,
    double* out,
    const InterpConfig& cfg) noexcept
{
  Axis axesLocal[2] = {
      MakeAxis(gridT, nT, scaleT),
      MakeAxis(gridX, nX, scaleX)};
  int extents[4] = {
      static_cast<int>(sizeE),
      static_cast<int>(sizeE),
      nT,
      nX};
  const Layout layout = MakeLayout(extents, 4);
  return LogInterpolateSingleVariable2D2DCustomAligned(
      sizeE, logT, logX, count, data, layout, axesLocal, offset, out, cfg);
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
  return LogInterpolateSingleVariable2D2DCustomAligned(
      sizeE, logT, logX, count,
      gridT, nT, AxisScale::Linear,
      gridX, nX, AxisScale::Linear,
      data, offset, out, cfg);
}

inline int LogInterpolateSingleVariable2D2DCustom(
    const double* logE, std::size_t sizeE,
    const double* logT, const double* logX, std::size_t count,
    const double* data, const Layout& layout,
    const Axis axes[4],
    double offset,
    double* out,
    const InterpConfig& cfg = InterpConfig{}) noexcept
{
  if (logE == nullptr || logT == nullptr || logX == nullptr ||
      data == nullptr || out == nullptr || axes == nullptr) {
    return 1;
  }
  if (sizeE == 0 || count == 0) {
    return 0;
  }
  if (layout.nd < 4) {
    return 2;
  }
  if (axes[0].grid == nullptr || axes[1].grid == nullptr ||
      axes[2].grid == nullptr || axes[3].grid == nullptr) {
    return 3;
  }

  const std::size_t planeSize = sizeE * sizeE;
  for (std::size_t l = 0; l < count; ++l) {
    double* plane = out + l * planeSize;
    const double tCoord = logT[l];
    const double xCoord = logX[l];
    for (std::size_t j = 0; j < sizeE; ++j) {
      for (std::size_t i = 0; i <= j; ++i) {
        const double value = LogInterpolateSingleVariable4DCustomPoint(
            data, layout, axes, logE[i], logE[j], tCoord, xCoord, offset, cfg);
        detail::StoreSymmetric(plane, sizeE, i, j, value);
      }
    }
  }

  return 0;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double LogInterpolateSingleVariable1DCustomPoint(
    double x0,
    const double* grid0, int n0, AxisScale scale0 = AxisScale::Linear,
    double offset = 0.0,
    const double* data = nullptr,
    const InterpConfig& cfg = InterpConfig{}) noexcept
{
  Axis axesLocal[1] = {MakeAxis(grid0, n0, scale0)};
  int extents[1] = {n0};
  const Layout layout = MakeLayout(extents, 1);
  return LogInterpolateSingleVariable1DCustomPoint(data, layout, axesLocal, x0, offset, cfg);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double LogInterpolateSingleVariable2DCustomPoint(
    double x0, double x1,
    const double* grid0, int n0, AxisScale scale0 = AxisScale::Linear,
    const double* grid1 = nullptr, int n1 = 0, AxisScale scale1 = AxisScale::Linear,
    double offset = 0.0,
    const double* data = nullptr,
    const InterpConfig& cfg = InterpConfig{}) noexcept
{
  Axis axesLocal[2] = {
      MakeAxis(grid0, n0, scale0),
      MakeAxis(grid1, n1, scale1)};
  int extents[2] = {n0, n1};
  const Layout layout = MakeLayout(extents, 2);
  return LogInterpolateSingleVariable2DCustomPoint(data, layout, axesLocal, x0, x1, offset, cfg);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double LogInterpolateSingleVariable3DCustomPoint(
    double x0, double x1, double x2,
    const double* grid0, int n0, AxisScale scale0 = AxisScale::Linear,
    const double* grid1 = nullptr, int n1 = 0, AxisScale scale1 = AxisScale::Linear,
    const double* grid2 = nullptr, int n2 = 0, AxisScale scale2 = AxisScale::Linear,
    double offset = 0.0,
    const double* data = nullptr,
    const InterpConfig& cfg = InterpConfig{}) noexcept
{
  Axis axesLocal[3] = {
      MakeAxis(grid0, n0, scale0),
      MakeAxis(grid1, n1, scale1),
      MakeAxis(grid2, n2, scale2)};
  int extents[3] = {n0, n1, n2};
  const Layout layout = MakeLayout(extents, 3);
  return LogInterpolateSingleVariable3DCustomPoint(data, layout, axesLocal, x0, x1, x2, offset, cfg);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double LogInterpolateSingleVariable4DCustomPoint(
    double x0, double x1, double x2, double x3,
    const double* grid0, int n0, AxisScale scale0 = AxisScale::Linear,
    const double* grid1 = nullptr, int n1 = 0, AxisScale scale1 = AxisScale::Linear,
    const double* grid2 = nullptr, int n2 = 0, AxisScale scale2 = AxisScale::Linear,
    const double* grid3 = nullptr, int n3 = 0, AxisScale scale3 = AxisScale::Linear,
    double offset = 0.0,
    const double* data = nullptr,
    const InterpConfig& cfg = InterpConfig{}) noexcept
{
  Axis axesLocal[4] = {
      MakeAxis(grid0, n0, scale0),
      MakeAxis(grid1, n1, scale1),
      MakeAxis(grid2, n2, scale2),
      MakeAxis(grid3, n3, scale3)};
  int extents[4] = {n0, n1, n2, n3};
  const Layout layout = MakeLayout(extents, 4);
  return LogInterpolateSingleVariable4DCustomPoint(data, layout, axesLocal,
                                                   x0, x1, x2, x3, offset, cfg);
}

inline int LogInterpolateSingleVariable3DCustom(
    const double* d, const double* t, const double* y, std::size_t count,
    const double* data, const Layout& layout,
    const Axis axes[3],
    double offset,
    double* out,
    const InterpConfig& cfg = InterpConfig{}) noexcept
{
  if (d == nullptr || t == nullptr || y == nullptr ||
      data == nullptr || out == nullptr || axes == nullptr) {
    return 1;
  }
  if (layout.nd < 3) {
    return 2;
  }
  if (axes[0].grid == nullptr || axes[1].grid == nullptr || axes[2].grid == nullptr) {
    return 3;
  }
  for (std::size_t i = 0; i < count; ++i) {
    out[i] = LogInterpolateSingleVariable3DCustomPoint(
        data, layout, axes, d[i], t[i], y[i], offset, cfg);
  }
  return 0;
}

inline int LogInterpolateSingleVariable3DCustom(
    const double* d, const double* t, const double* y, std::size_t count,
    const double* grid0, int n0, AxisScale scale0,
    const double* grid1, int n1, AxisScale scale1,
    const double* grid2, int n2, AxisScale scale2,
    const double* data,
    double offset,
    double* out,
    const InterpConfig& cfg = InterpConfig{}) noexcept
{
  Axis axesLocal[3] = {
      MakeAxis(grid0, n0, scale0),
      MakeAxis(grid1, n1, scale1),
      MakeAxis(grid2, n2, scale2)};
  int extents[3] = {n0, n1, n2};
  const Layout layout = MakeLayout(extents, 3);
  return LogInterpolateSingleVariable3DCustom(
      d, t, y, count, data, layout, axesLocal, offset, out, cfg);
}

inline int LogInterpolateSingleVariable1D3DCustomPoint(
    const double* logE, std::size_t sizeE,
    double logD, double logT, double y,
    const double* data, const Layout& layout,
    const Axis axes[4],
    double offset,
    double* out,
    const InterpConfig& cfg = InterpConfig{}) noexcept
{
  if (logE == nullptr || data == nullptr || out == nullptr || axes == nullptr) {
    return 1;
  }
  if (sizeE == 0) {
    return 0;
  }
  if (layout.nd < 4) {
    return 2;
  }
  if (axes[0].grid == nullptr || axes[1].grid == nullptr ||
      axes[2].grid == nullptr || axes[3].grid == nullptr) {
    return 3;
  }

  for (std::size_t i = 0; i < sizeE; ++i) {
    out[i] = LogInterpolateSingleVariable4DCustomPoint(
        data, layout, axes,
        logE[i], logD, logT, y,
        offset, cfg);
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
  Axis axesLocal[4] = {
      MakeAxis(gridE, nE, AxisScale::Linear),
      MakeAxis(gridD, nD, AxisScale::Linear),
      MakeAxis(gridT, nT, AxisScale::Linear),
      MakeAxis(gridY, nY, AxisScale::Linear)};
  int extents[4] = {nE, nD, nT, nY};
  const Layout layout = MakeLayout(extents, 4);
  return LogInterpolateSingleVariable1D3DCustomPoint(
      logE, sizeE, logD, logT, y,
      data, layout, axesLocal,
      offset, out, cfg);
}

inline int LogInterpolateSingleVariable1D3DCustom(
    const double* logE, std::size_t sizeE,
    const double* logD, const double* logT, const double* y, std::size_t count,
    const double* data, const Layout& layout,
    const Axis axes[4],
    double offset,
    double* out,
    const InterpConfig& cfg = InterpConfig{}) noexcept
{
  if (logE == nullptr || logD == nullptr || logT == nullptr || y == nullptr ||
      data == nullptr || out == nullptr || axes == nullptr) {
    return 1;
  }
  if (sizeE == 0 || count == 0) {
    return 0;
  }
  if (layout.nd < 4) {
    return 2;
  }
  if (axes[0].grid == nullptr || axes[1].grid == nullptr ||
      axes[2].grid == nullptr || axes[3].grid == nullptr) {
    return 3;
  }

  for (std::size_t j = 0; j < count; ++j) {
    double* row = out + j * sizeE;
    const int rc = LogInterpolateSingleVariable1D3DCustomPoint(
        logE, sizeE,
        logD[j], logT[j], y[j],
        data, layout, axes,
        offset, row, cfg);
    if (rc != 0) {
      return rc;
    }
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
  Axis axesLocal[4] = {
      MakeAxis(gridE, nE, AxisScale::Linear),
      MakeAxis(gridD, nD, AxisScale::Linear),
      MakeAxis(gridT, nT, AxisScale::Linear),
      MakeAxis(gridY, nY, AxisScale::Linear)};
  int extents[4] = {nE, nD, nT, nY};
  const Layout layout = MakeLayout(extents, 4);
  return LogInterpolateSingleVariable1D3DCustom(
      logE, sizeE, logD, logT, y, count,
      data, layout, axesLocal,
      offset, out, cfg);
}

inline int LogInterpolateSingleVariable4DCustom(
    const double* logE, const double* logD, const double* logT, const double* y,
    std::size_t count,
    const double* data, const Layout& layout,
    const Axis axes[4],
    double offset,
    double* out,
    const InterpConfig& cfg = InterpConfig{}) noexcept
{
  if (logE == nullptr || logD == nullptr || logT == nullptr || y == nullptr ||
      data == nullptr || out == nullptr || axes == nullptr) {
    return 1;
  }
  if (count == 0) {
    return 0;
  }
  if (layout.nd < 4) {
    return 2;
  }
  if (axes[0].grid == nullptr || axes[1].grid == nullptr ||
      axes[2].grid == nullptr || axes[3].grid == nullptr) {
    return 3;
  }

  for (std::size_t i = 0; i < count; ++i) {
    out[i] = LogInterpolateSingleVariable4DCustomPoint(
        data, layout, axes,
        logE[i], logD[i], logT[i], y[i],
        offset, cfg);
  }

  return 0;
}

inline int LogInterpolateSingleVariable4DCustom(
    const double* logE, const double* logD, const double* logT, const double* y,
    std::size_t count,
    const double* gridE, int nE,
    const double* gridD, int nD,
    const double* gridT, int nT,
    const double* gridY, int nY,
    const double* data,
    double offset,
    double* out,
    const InterpConfig& cfg = InterpConfig{}) noexcept
{
  Axis axesLocal[4] = {
      MakeAxis(gridE, nE, AxisScale::Linear),
      MakeAxis(gridD, nD, AxisScale::Linear),
      MakeAxis(gridT, nT, AxisScale::Linear),
      MakeAxis(gridY, nY, AxisScale::Linear)};
  int extents[4] = {nE, nD, nT, nY};
  const Layout layout = MakeLayout(extents, 4);
  return LogInterpolateSingleVariable4DCustom(
      logE, logD, logT, y, count,
      data, layout, axesLocal,
      offset, out, cfg);
}

inline int LogInterpolateDifferentiateSingleVariable2D2DCustomPoint(
    const double* logE, std::size_t sizeE,
    double logT, double logX,
    const double* data, const Layout& layout,
    const Axis axes[4],
    double offset,
    double* interpolantPlane,
    double* derivativeTPlane,
    double* derivativeXPlane,
    const InterpConfig& cfg) noexcept
{
  if (logE == nullptr || data == nullptr ||
      interpolantPlane == nullptr || derivativeTPlane == nullptr || derivativeXPlane == nullptr ||
      axes == nullptr) {
    return 1;
  }
  if (sizeE == 0) {
    return 0;
  }
  if (layout.nd < 4) {
    return 2;
  }
  if (axes[0].grid == nullptr || axes[1].grid == nullptr ||
      axes[2].grid == nullptr || axes[3].grid == nullptr) {
    return 3;
  }

  int idxT = 0;
  int idxX = 0;
  double fracT = 0.0;
  double fracX = 0.0;
  const bool outT = detail::IndexAndDelta(axes[2], logT, idxT, fracT);
  const bool outX = detail::IndexAndDelta(axes[3], logX, idxX, fracX);

  const double nanValue = std::numeric_limits<double>::quiet_NaN();

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
        detail::StoreSymmetric(interpolantPlane, sizeE, i, j, nanValue);
        detail::StoreSymmetric(derivativeTPlane, sizeE, i, j, nanValue);
        detail::StoreSymmetric(derivativeXPlane, sizeE, i, j, nanValue);
        continue;
      }
      if (outE1) {
        fracE1 = detail::Clamp01(fracE1);
      }

      double interpValue = 0.0;
      double dDummy1 = 0.0;
      double dDummy2 = 0.0;
      double derivTVal = 0.0;
      double derivXVal = 0.0;

      LinearInterpDeriv4DPoint(
          idxE1, idxE2, idxT, idxX,
          fracE1, fracE2, fracT, fracX,
          1.0, 1.0, aT, aX,
          offset, data, layout,
          interpValue, dDummy1, dDummy2, derivTVal, derivXVal);

      detail::StoreSymmetric(interpolantPlane, sizeE, i, j, interpValue);
      detail::StoreSymmetric(derivativeTPlane, sizeE, i, j, derivTVal);
      detail::StoreSymmetric(derivativeXPlane, sizeE, i, j, derivXVal);
    }
  }

  return 0;
}

inline int LogInterpolateDifferentiateSingleVariable2D2DCustomPoint(
    const double* logE, std::size_t sizeE,
    double logT, double logX,
    const double* gridE, int nE,
    const double* gridT, int nT, AxisScale scaleT = AxisScale::Linear,
    const double* gridX = nullptr, int nX = 0, AxisScale scaleX = AxisScale::Linear,
    const double* data = nullptr,
    double offset = 0.0,
    double* interpolantPlane = nullptr,
    double* derivativeTPlane = nullptr,
    double* derivativeXPlane = nullptr,
    const InterpConfig& cfg = InterpConfig{}) noexcept
{
  Axis axesLocal[4] = {
      MakeAxis(gridE, nE, AxisScale::Linear),
      MakeAxis(gridE, nE, AxisScale::Linear),
      MakeAxis(gridT, nT, scaleT),
      MakeAxis(gridX, nX, scaleX)};
  int extents[4] = {nE, nE, nT, nX};
  const Layout layout = MakeLayout(extents, 4);
  return LogInterpolateDifferentiateSingleVariable2D2DCustomPoint(
      logE, sizeE, logT, logX,
      data, layout, axesLocal,
      offset, interpolantPlane, derivativeTPlane, derivativeXPlane, cfg);
}

inline int LogInterpolateDifferentiateSingleVariable2D2DCustom(
    const double* logE, std::size_t sizeE,
    const double* logT, const double* logX, std::size_t count,
    const double* data, const Layout& layout,
    const Axis axes[4],
    double offset,
    double* interpolant,
    double* derivativeT,
    double* derivativeX,
    const InterpConfig& cfg) noexcept
{
  if (logE == nullptr || logT == nullptr || logX == nullptr ||
      data == nullptr || interpolant == nullptr ||
      derivativeT == nullptr || derivativeX == nullptr ||
      axes == nullptr) {
    return 1;
  }
  if (sizeE == 0 || count == 0) {
    return 0;
  }
  if (layout.nd < 4) {
    return 2;
  }
  if (axes[0].grid == nullptr || axes[1].grid == nullptr ||
      axes[2].grid == nullptr || axes[3].grid == nullptr) {
    return 3;
  }

  const std::size_t planeSize = sizeE * sizeE;
  for (std::size_t l = 0; l < count; ++l) {
    double* planeInterp = interpolant + l * planeSize;
    double* planeDerivT = derivativeT + l * planeSize;
    double* planeDerivX = derivativeX + l * planeSize;
    const int rc = LogInterpolateDifferentiateSingleVariable2D2DCustomPoint(
        logE, sizeE, logT[l], logX[l],
        data, layout, axes, offset,
        planeInterp, planeDerivT, planeDerivX, cfg);
    if (rc != 0) {
      return rc;
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
  Axis axesLocal[4] = {
      MakeAxis(gridE, nE, AxisScale::Linear),
      MakeAxis(gridE, nE, AxisScale::Linear),
      MakeAxis(gridT, nT, AxisScale::Linear),
      MakeAxis(gridX, nX, AxisScale::Linear)};
  int extents[4] = {nE, nE, nT, nX};
  const Layout layout = MakeLayout(extents, 4);
  return LogInterpolateDifferentiateSingleVariable2D2DCustom(
      logE, sizeE, logT, logX, count,
      data, layout, axesLocal, offset,
      interpolant, derivativeT, derivativeX, cfg);
}

inline int LogInterpolateDifferentiateSingleVariable2D2DCustomAlignedPoint(
    std::size_t sizeE,
    double logT, double logX,
    const double* data, const Layout& layout,
    const Axis axes[2],
    double offset,
    double* interpolantPlane,
    double* derivativeTPlane,
    double* derivativeXPlane,
    const InterpConfig& cfg) noexcept
{
  if (data == nullptr || interpolantPlane == nullptr ||
      derivativeTPlane == nullptr || derivativeXPlane == nullptr ||
      axes == nullptr) {
    return 1;
  }
  if (sizeE == 0) {
    return 0;
  }
  if (layout.nd < 4) {
    return 2;
  }
  if (axes[0].grid == nullptr || axes[1].grid == nullptr) {
    return 3;
  }

  int idxT = 0;
  int idxX = 0;
  double fracT = 0.0;
  double fracX = 0.0;
  const bool outT = detail::IndexAndDelta(axes[0], logT, idxT, fracT);
  const bool outX = detail::IndexAndDelta(axes[1], logX, idxX, fracX);

  const double nanValue = std::numeric_limits<double>::quiet_NaN();

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
          idxT, idxX, fracT, fracX,
          aT, aX, offset,
          data, layout,
          interpValue, derivTVal, derivXVal);

      detail::StoreSymmetric(interpolantPlane, sizeE, i, j, interpValue);
      detail::StoreSymmetric(derivativeTPlane, sizeE, i, j, derivTVal);
      detail::StoreSymmetric(derivativeXPlane, sizeE, i, j, derivXVal);
    }
  }

  return 0;
}

inline int LogInterpolateDifferentiateSingleVariable2D2DCustomAlignedPoint(
    std::size_t sizeE,
    double logT, double logX,
    const double* gridT, int nT, AxisScale scaleT = AxisScale::Linear,
    const double* gridX = nullptr, int nX = 0, AxisScale scaleX = AxisScale::Linear,
    const double* data = nullptr,
    double offset = 0.0,
    double* interpolantPlane = nullptr,
    double* derivativeTPlane = nullptr,
    double* derivativeXPlane = nullptr,
    const InterpConfig& cfg = InterpConfig{}) noexcept
{
  Axis axesLocal[2] = {
      MakeAxis(gridT, nT, scaleT),
      MakeAxis(gridX, nX, scaleX)};
  int extents[4] = {
      static_cast<int>(sizeE),
      static_cast<int>(sizeE),
      nT,
      nX};
  const Layout layout = MakeLayout(extents, 4);
  return LogInterpolateDifferentiateSingleVariable2D2DCustomAlignedPoint(
      sizeE, logT, logX,
      data, layout, axesLocal, offset,
      interpolantPlane, derivativeTPlane, derivativeXPlane, cfg);
}

inline int LogInterpolateDifferentiateSingleVariable2D2DCustomAligned(
    std::size_t sizeE,
    const double* logT, const double* logX, std::size_t count,
    const double* data, const Layout& layout,
    const Axis axes[2],
    double offset,
    double* interpolant,
    double* derivativeT,
    double* derivativeX,
    const InterpConfig& cfg) noexcept
{
  if (logT == nullptr || logX == nullptr ||
      data == nullptr || interpolant == nullptr ||
      derivativeT == nullptr || derivativeX == nullptr ||
      axes == nullptr) {
    return 1;
  }
  if (sizeE == 0 || count == 0) {
    return 0;
  }
  if (layout.nd < 4) {
    return 2;
  }
  if (axes[0].grid == nullptr || axes[1].grid == nullptr) {
    return 3;
  }

  const std::size_t planeSize = sizeE * sizeE;
  for (std::size_t k = 0; k < count; ++k) {
    double* planeInterp = interpolant + k * planeSize;
    double* planeDerivT = derivativeT + k * planeSize;
    double* planeDerivX = derivativeX + k * planeSize;
    const int rc = LogInterpolateDifferentiateSingleVariable2D2DCustomAlignedPoint(
        sizeE, logT[k], logX[k],
        data, layout, axes, offset,
        planeInterp, planeDerivT, planeDerivX, cfg);
    if (rc != 0) {
      return rc;
    }
  }

  return 0;
}

inline int LogInterpolateDifferentiateSingleVariable2D2DCustomAligned(
    std::size_t sizeE,
    const double* logT, const double* logX, std::size_t count,
    const double* gridT, int nT, AxisScale scaleT = AxisScale::Linear,
    const double* gridX = nullptr, int nX = 0, AxisScale scaleX = AxisScale::Linear,
    const double* data = nullptr,
    double offset = 0.0,
    double* interpolant = nullptr,
    double* derivativeT = nullptr,
    double* derivativeX = nullptr,
    const InterpConfig& cfg = InterpConfig{}) noexcept
{
  Axis axesLocal[2] = {
      MakeAxis(gridT, nT, scaleT),
      MakeAxis(gridX, nX, scaleX)};
  int extents[4] = {
      static_cast<int>(sizeE),
      static_cast<int>(sizeE),
      nT,
      nX};
  const Layout layout = MakeLayout(extents, 4);
  return LogInterpolateDifferentiateSingleVariable2D2DCustomAligned(
      sizeE, logT, logX, count,
      data, layout, axesLocal,
      offset, interpolant, derivativeT, derivativeX, cfg);
}

inline int SumLogInterpolateSingleVariable2D2DCustomAligned(
    std::size_t sizeE,
    const double* logD, std::size_t nAlpha,
    const double* logT, std::size_t count,
    const double* data, const Layout& layout,
    const Axis axes[2],
    const double* alpha,
    double offset,
    double* out,
    const InterpConfig& cfg = InterpConfig{}) noexcept
{
  if (logD == nullptr || logT == nullptr || data == nullptr ||
      alpha == nullptr || out == nullptr || axes == nullptr) {
    return 1;
  }
  if (sizeE == 0 || nAlpha == 0 || count == 0) {
    return 0;
  }
  if (layout.nd < 4) {
    return 2;
  }
  if (axes[0].grid == nullptr || axes[1].grid == nullptr) {
    return 3;
  }

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
  Axis axesLocal[2] = {
      MakeAxis(gridD, nD, AxisScale::Linear),
      MakeAxis(gridT, nT, AxisScale::Linear)};
  int extents[4] = {
      static_cast<int>(sizeE),
      static_cast<int>(sizeE),
      nD,
      nT};
  const Layout layout = MakeLayout(extents, 4);
  return SumLogInterpolateSingleVariable2D2DCustomAligned(
      sizeE,
      logD, nAlpha,
      logT, count,
      data, layout, axesLocal,
      alpha,
      offset, out, cfg);
}

} // namespace WeakLibReader
