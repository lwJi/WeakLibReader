#pragma once

#include <AMReX_GpuQualifiers.H>
#include <cstddef>
#include <cstdint>

#include "WeakLibReader_IndexDelta.hpp"

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

namespace detail {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void IndexAndDelta(const Axis& axis, double x,
                   int& idx, double& frac) noexcept
{
  if (axis.scale == AxisScale::Linear) {
    IndexAndDeltaLin(x, axis.grid, axis.n, idx, frac);
  } else {
    IndexAndDeltaLog10(x, axis.grid, axis.n, idx, frac);
  }
}

} // namespace detail

} // namespace WeakLibReader
