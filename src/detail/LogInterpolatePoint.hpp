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
    const Axis axes[2],
    const double* data,
    double offset) noexcept
{
  if (data == nullptr || axes[0].grid == nullptr || axes[1].grid == nullptr) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  constexpr int ND = 2;
  const Layout layout = MakeLayoutFromAxes(axes, ND);
  double coords[ND] = {x0, x1};
  return detail::LogInterpolatedValueDirect<ND>(data, layout, axes, coords, offset);
}

/// GPU-optimized 2D log-interpolation (batch)
/// Uses compile-time dimensionality for zero runtime branching
inline int LogInterpolateSingleVariable2DCustom(
    const double* x0, const double* x1, std::size_t count,
    const Axis axes[2],
    const double* data,
    double offset,
    double* out) noexcept
{
  if (x0 == nullptr || x1 == nullptr || out == nullptr ||
      data == nullptr || axes[0].grid == nullptr || axes[1].grid == nullptr) {
    return 1;
  }
  constexpr int ND = 2;
  const Layout layout = MakeLayoutFromAxes(axes, ND);
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
    const Axis axes[3],
    const double* data,
    double offset) noexcept
{
  if (data == nullptr ||
      axes[0].grid == nullptr || axes[1].grid == nullptr || axes[2].grid == nullptr) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  constexpr int ND = 3;
  const Layout layout = MakeLayoutFromAxes(axes, ND);
  double coords[ND] = {x0, x1, x2};
  return detail::LogInterpolatedValueDirect<ND>(data, layout, axes, coords, offset);
}

/// GPU-optimized 3D log-interpolation (batch)
/// Uses compile-time dimensionality for zero runtime branching
inline int LogInterpolateSingleVariable3DCustom(
    const double* x0, const double* x1, const double* x2, std::size_t count,
    const Axis axes[3],
    const double* data,
    double offset,
    double* out) noexcept
{
  if (x0 == nullptr || x1 == nullptr || x2 == nullptr ||
      out == nullptr || data == nullptr ||
      axes[0].grid == nullptr || axes[1].grid == nullptr || axes[2].grid == nullptr) {
    return 1;
  }
  constexpr int ND = 3;
  const Layout layout = MakeLayoutFromAxes(axes, ND);
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
    const Axis axes[4],
    const double* data,
    double offset) noexcept
{
  if (data == nullptr ||
      axes[0].grid == nullptr || axes[1].grid == nullptr ||
      axes[2].grid == nullptr || axes[3].grid == nullptr) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  constexpr int ND = 4;
  const Layout layout = MakeLayoutFromAxes(axes, ND);
  double coords[ND] = {x0, x1, x2, x3};
  return detail::LogInterpolatedValueDirect<ND>(data, layout, axes, coords, offset);
}

/// GPU-optimized 4D log-interpolation (batch)
/// Uses compile-time dimensionality for zero runtime branching
inline int LogInterpolateSingleVariable4DCustom(
    const double* x0, const double* x1, const double* x2, const double* x3,
    std::size_t count,
    const Axis axes[4],
    const double* data,
    double offset,
    double* out) noexcept
{
  if (x0 == nullptr || x1 == nullptr || x2 == nullptr || x3 == nullptr ||
      out == nullptr || data == nullptr ||
      axes[0].grid == nullptr || axes[1].grid == nullptr ||
      axes[2].grid == nullptr || axes[3].grid == nullptr) {
    return 1;
  }
  constexpr int ND = 4;
  const Layout layout = MakeLayoutFromAxes(axes, ND);
  for (std::size_t i = 0; i < count; ++i) {
    double coords[ND] = {x0[i], x1[i], x2[i], x3[i]};
    out[i] = detail::LogInterpolatedValueDirect<ND>(data, layout, axes, coords, offset);
  }
  return 0;
}

} // namespace WeakLibReader
