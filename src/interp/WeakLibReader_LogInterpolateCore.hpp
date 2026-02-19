#pragma once

#include <AMReX_BLassert.H>
#include <AMReX_GpuQualifiers.H>

#include <cstddef>

#include "base/WeakLibReader_AxisTypes.hpp"
#include "base/WeakLibReader_Math.hpp"
#include "interp/WeakLibReader_InterpLogTable.hpp"

namespace WeakLibReader {
namespace detail {

template<int ND>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double LogInterpolatedValueDirect(const double* data,
                                  const Layout& layout,
                                  const Axis axes[ND],
                                  const double coords[ND],
                                  double offset) noexcept
{
  int indices[ND];
  double fractions[ND];

  for (int d = 0; d < ND; ++d) {
    IndexAndDelta(axes[d], coords[d], indices[d], fractions[d]);
  }

  return LinearInterpPointDirect<ND>(indices, fractions, offset, data, layout);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void StoreSymmetric(double* plane, std::size_t size,
                    std::size_t i, std::size_t j,
                    double value) noexcept
{
  const std::size_t idxLower = j * size + i;
  plane[idxLower] = value;
  if (i != j) {
    plane[i * size + j] = value;
  }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void ComputeAxisScale(const Axis& axis, int idx, double coord,
                      double& scale) noexcept
{
  AMREX_ASSERT(axis.grid != nullptr);
  if (axis.scale == AxisScale::Linear) {
    const double span = axis.grid[idx + 1] - axis.grid[idx];
    AMREX_ASSERT(span > 0.0);
    scale = math::Ln10 / span;
  } else {
    AMREX_ASSERT(coord > 0.0);
    const double ratio = axis.grid[idx + 1] / axis.grid[idx];
    AMREX_ASSERT(ratio > 0.0);
    const double denom = math::Log10(ratio);
    AMREX_ASSERT(denom != 0.0);
    scale = 1.0 / (coord * denom);
  }
}

template<int ND>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void LogInterpolatedDerivativeDirect(const double* data,
                                     const Layout& layout,
                                     const Axis axes[ND],
                                     const double coords[ND],
                                     double offset,
                                     double& interpolant,
                                     double derivatives[ND]) noexcept
{
  int indices[ND];
  double fractions[ND];
  double scales[ND];

  for (int d = 0; d < ND; ++d) {
    IndexAndDelta(axes[d], coords[d], indices[d], fractions[d]);
    ComputeAxisScale(axes[d], indices[d], coords[d], scales[d]);
  }

  LinearInterpDerivPointDirect<ND>(indices, fractions, scales, offset,
                                   data, layout, interpolant, derivatives);
}

} // namespace detail
} // namespace WeakLibReader
