#pragma once

#include <AMReX_GpuQualifiers.H>
#include <cstddef>
#include <limits>

#include "LogInterpolateCore.hpp"

namespace WeakLibReader {

/// GPU-optimized 2D log-interpolation (single point)
/// Uses compile-time dimensionality for zero runtime branching
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double LogInterpolateSingleVariable2DCustomPoint(
    double x0, double x1,
    const double* grid0, int n0,
    const double* grid1, int n1,
    const double* data,
    double offset) noexcept
{
  if (data == nullptr || grid0 == nullptr || grid1 == nullptr) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  // Compile-time known dimensionality (eliminates runtime branching!)
  constexpr int ND = 2;
  Axis axes[ND] = {
      MakeAxis(grid0, n0, AxisScale::Linear),
      MakeAxis(grid1, n1, AxisScale::Linear)};
  int extents[ND] = {n0, n1};
  const Layout layout = MakeLayout(extents, ND);
  double coords[ND] = {x0, x1};
  // Template parameter known at compile time - zero branching!
  return detail::LogInterpolatedValueDirect<ND>(data, layout, axes, coords, offset);
}

/// GPU-optimized 2D log-interpolation (batch)
/// Uses compile-time dimensionality for zero runtime branching
inline int LogInterpolateSingleVariable2DCustom(
    const double* x0, const double* x1, std::size_t count,
    const double* grid0, int n0,
    const double* grid1, int n1,
    const double* data,
    double offset,
    double* out) noexcept
{
  if (x0 == nullptr || x1 == nullptr || out == nullptr ||
      data == nullptr || grid0 == nullptr || grid1 == nullptr) {
    return 1;
  }
  constexpr int ND = 2;
  Axis axes[ND] = {
      MakeAxis(grid0, n0, AxisScale::Linear),
      MakeAxis(grid1, n1, AxisScale::Linear)};
  int extents[ND] = {n0, n1};
  const Layout layout = MakeLayout(extents, ND);
  for (std::size_t i = 0; i < count; ++i) {
    double coords[ND] = {x0[i], x1[i]};
    out[i] = detail::LogInterpolatedValueDirect<ND>(data, layout, axes, coords, offset);
  }
  return 0;
}

/// GPU-optimized 3D log-interpolation (single point)
/// Uses compile-time dimensionality for zero runtime branching
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double LogInterpolateSingleVariable3DCustomPoint(
    double x0, double x1, double x2,
    const double* grid0, int n0,
    const double* grid1, int n1,
    const double* grid2, int n2,
    const double* data,
    double offset) noexcept
{
  if (data == nullptr || grid0 == nullptr || grid1 == nullptr || grid2 == nullptr) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  constexpr int ND = 3;
  Axis axes[ND] = {
      MakeAxis(grid0, n0, AxisScale::Log10),
      MakeAxis(grid1, n1, AxisScale::Log10),
      MakeAxis(grid2, n2, AxisScale::Linear)};
  int extents[ND] = {n0, n1, n2};
  const Layout layout = MakeLayout(extents, ND);
  double coords[ND] = {x0, x1, x2};
  return detail::LogInterpolatedValueDirect<ND>(data, layout, axes, coords, offset);
}

/// GPU-optimized 3D log-interpolation (batch)
/// Uses compile-time dimensionality for zero runtime branching
inline int LogInterpolateSingleVariable3DCustom(
    const double* x0, const double* x1, const double* x2, std::size_t count,
    const double* grid0, int n0,
    const double* grid1, int n1,
    const double* grid2, int n2,
    const double* data,
    double offset,
    double* out) noexcept
{
  if (x0 == nullptr || x1 == nullptr || x2 == nullptr ||
      out == nullptr || data == nullptr ||
      grid0 == nullptr || grid1 == nullptr || grid2 == nullptr) {
    return 1;
  }
  constexpr int ND = 3;
  Axis axes[ND] = {
      MakeAxis(grid0, n0, AxisScale::Log10),
      MakeAxis(grid1, n1, AxisScale::Log10),
      MakeAxis(grid2, n2, AxisScale::Linear)};
  int extents[ND] = {n0, n1, n2};
  const Layout layout = MakeLayout(extents, ND);
  for (std::size_t i = 0; i < count; ++i) {
    double coords[ND] = {x0[i], x1[i], x2[i]};
    out[i] = detail::LogInterpolatedValueDirect<ND>(data, layout, axes, coords, offset);
  }
  return 0;
}

/// GPU-optimized 4D log-interpolation (single point)
/// Uses compile-time dimensionality for zero runtime branching
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double LogInterpolateSingleVariable4DCustomPoint(
    double x0, double x1, double x2, double x3,
    const double* grid0, int n0,
    const double* grid1, int n1,
    const double* grid2, int n2,
    const double* grid3, int n3,
    const double* data,
    double offset) noexcept
{
  if (data == nullptr || grid0 == nullptr || grid1 == nullptr ||
      grid2 == nullptr || grid3 == nullptr) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  constexpr int ND = 4;
  Axis axes[ND] = {
      MakeAxis(grid0, n0, AxisScale::Linear),
      MakeAxis(grid1, n1, AxisScale::Linear),
      MakeAxis(grid2, n2, AxisScale::Linear),
      MakeAxis(grid3, n3, AxisScale::Linear)};
  int extents[ND] = {n0, n1, n2, n3};
  const Layout layout = MakeLayout(extents, ND);
  double coords[ND] = {x0, x1, x2, x3};
  return detail::LogInterpolatedValueDirect<ND>(data, layout, axes, coords, offset);
}

/// GPU-optimized 4D log-interpolation (batch)
/// Uses compile-time dimensionality for zero runtime branching
inline int LogInterpolateSingleVariable4DCustom(
    const double* x0, const double* x1, const double* x2, const double* x3,
    std::size_t count,
    const double* grid0, int n0,
    const double* grid1, int n1,
    const double* grid2, int n2,
    const double* grid3, int n3,
    const double* data,
    double offset,
    double* out) noexcept
{
  if (x0 == nullptr || x1 == nullptr || x2 == nullptr || x3 == nullptr ||
      out == nullptr || data == nullptr ||
      grid0 == nullptr || grid1 == nullptr || grid2 == nullptr || grid3 == nullptr) {
    return 1;
  }
  constexpr int ND = 4;
  Axis axes[ND] = {
      MakeAxis(grid0, n0, AxisScale::Linear),
      MakeAxis(grid1, n1, AxisScale::Linear),
      MakeAxis(grid2, n2, AxisScale::Linear),
      MakeAxis(grid3, n3, AxisScale::Linear)};
  int extents[ND] = {n0, n1, n2, n3};
  const Layout layout = MakeLayout(extents, ND);
  for (std::size_t i = 0; i < count; ++i) {
    double coords[ND] = {x0[i], x1[i], x2[i], x3[i]};
    out[i] = detail::LogInterpolatedValueDirect<ND>(data, layout, axes, coords, offset);
  }
  return 0;
}

} // namespace WeakLibReader
