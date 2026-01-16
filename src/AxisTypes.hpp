#pragma once

#include <AMReX_GpuQualifiers.H>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "IndexDelta.hpp"
#include "InterpBasis.hpp"
#include "Layout.hpp"

namespace WeakLibReader {

enum class AxisScale : std::uint8_t { Linear, Log10 };

struct Axis {
  const double* grid = nullptr;
  int n = 0;
  AxisScale scale = AxisScale::Linear;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Axis MakeAxis(const double* grid, int n, AxisScale scale) noexcept
{
  Axis axis{};
  axis.grid = grid;
  axis.n = n;
  axis.scale = scale;
  return axis;
}

enum class OutOfRangePolicy : std::uint8_t { Clamp, Error, FillNaN };

struct InterpConfig {
  OutOfRangePolicy outOfRange = OutOfRangePolicy::Clamp;
};

namespace detail {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double Clamp01(double value) noexcept
{
  if (value < 0.0) {
    return 0.0;
  }
  if (value > 1.0) {
    return 1.0;
  }
  return value;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double NaN() noexcept
{
  return std::numeric_limits<double>::quiet_NaN();
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool IndexAndDelta(const Axis& axis, double x,
                   int& idx, double& frac) noexcept
{
  const bool out =
      (axis.scale == AxisScale::Linear)
          ? IndexAndDeltaLin(x, axis.grid, axis.n, idx, frac)
          : IndexAndDeltaLog10(x, axis.grid, axis.n, idx, frac);
  return out;
}

} // namespace detail

} // namespace WeakLibReader
