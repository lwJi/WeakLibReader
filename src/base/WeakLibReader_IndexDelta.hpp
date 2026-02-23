#pragma once

#include <AMReX_BLassert.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>

#include "base/WeakLibReader_Math.hpp"

namespace WeakLibReader {
namespace detail {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ClampIndex(int raw, int maxIndex) noexcept
{
  if (raw < 0) {
    return 0;
  }
  if (raw > maxIndex) {
    return maxIndex;
  }
  return raw;
}

} // namespace detail

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void IndexAndDeltaLin(double x, const double* grid, int n,
                      int& idx, double& frac) noexcept
{
  AMREX_ASSERT(grid != nullptr);
  AMREX_ASSERT(n >= 2);

  const double first = grid[0];
  const double last = grid[n - 1];

  AMREX_ASSERT(last > first);

  const double span = last - first;

  const double scaled = (x - first) / span;
  const double scaledIndex = (n - 1) * scaled;
  const int rawIndex = static_cast<int>(math::Floor(scaledIndex));

  const int maxIndex = n - 2;
  idx = detail::ClampIndex(rawIndex, maxIndex);

  const double cellSpan = grid[idx + 1] - grid[idx];

  AMREX_ASSERT(cellSpan > 0.0);

  frac = (x - grid[idx]) / cellSpan;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void IndexAndDeltaLog10(double x, const double* grid, int n,
                        int& idx, double& frac) noexcept
{
  AMREX_ASSERT(grid != nullptr);
  AMREX_ASSERT(n >= 2);

  const double first = grid[0];
  const double last = grid[n - 1];

  AMREX_ASSERT(first > 0.0 && last > first);
  AMREX_ASSERT(x > 0.0);

  const double logSpan = math::Log10(last / first);

  const double scaled = math::Log10(x / first) / logSpan;
  const double scaledIndex = (n - 1) * scaled;
  const int rawIndex = static_cast<int>(math::Floor(scaledIndex));

  const int maxIndex = n - 2;
  idx = detail::ClampIndex(rawIndex, maxIndex);

  const double logCellRatio = math::Log10(grid[idx + 1] / grid[idx]);

  AMREX_ASSERT(logCellRatio > 0.0);

  frac = math::Log10(x / grid[idx]) / logCellRatio;
}

} // namespace WeakLibReader
