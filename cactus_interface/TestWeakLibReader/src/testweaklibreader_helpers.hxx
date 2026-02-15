#pragma once

#include <base/WeakLibReader_AxisTypes.hpp>
#include <base/WeakLibReader_Math.hpp>

namespace TestWeakLibReader {

// Map a coordinate t in [0,1) to the physical range of the given axis.
// For Log10 axes the mapping is uniform in log-space.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double RescaleToAxis(double t, const WeakLibReader::Axis& axis) noexcept
{
  const double lo = axis.grid[0];
  const double hi = axis.grid[axis.n - 1];
  if (axis.scale == WeakLibReader::AxisScale::Log10) {
    return lo * WeakLibReader::math::Pow10(
        WeakLibReader::math::Log10(hi / lo) * t);
  }
  return lo + (hi - lo) * t;
}

} // namespace TestWeakLibReader
