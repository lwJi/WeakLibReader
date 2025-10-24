#pragma once

#include <AMReX_GpuQualifiers.H>
#include <cstddef>

#include "WeakLibReader/Math.hpp"
#include "WeakLibReader/WeakLibReader.hpp"

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
    const double* grid0, int n0, AxisScale scale0 = AxisScale::Linear,
    const double* grid1, int n1, AxisScale scale1 = AxisScale::Linear,
    const double* data,
    double offset,
    double* out,
    const InterpConfig& cfg = InterpConfig{}) noexcept
{
  Axis axesLocal[2] = {
      MakeAxis(grid0, n0, scale0),
      MakeAxis(grid1, n1, scale1)};
  int extents[2] = {n0, n1};
  const Layout layout = MakeLayout(extents, 2);
  return LogInterpolateSingleVariable2DCustom(
      x0, x1, count, data, layout, axesLocal, offset, out, cfg);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double LogInterpolateSingleVariable1DCustomPoint(
    double x0,
    const double* grid0, int n0, AxisScale scale0 = AxisScale::Linear,
    double offset,
    const double* data,
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
    const double* grid1, int n1, AxisScale scale1 = AxisScale::Linear,
    double offset,
    const double* data,
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
    const double* grid1, int n1, AxisScale scale1 = AxisScale::Linear,
    const double* grid2, int n2, AxisScale scale2 = AxisScale::Linear,
    double offset,
    const double* data,
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
    const double* grid1, int n1, AxisScale scale1 = AxisScale::Linear,
    const double* grid2, int n2, AxisScale scale2 = AxisScale::Linear,
    const double* grid3, int n3, AxisScale scale3 = AxisScale::Linear,
    double offset,
    const double* data,
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

} // namespace WeakLibReader
