#pragma once

#include <AMReX_BLassert.H>
#include <AMReX_GpuQualifiers.H>
#include <cstddef>

#include "interp/WeakLibReader_InterpLogTable.hpp"
#include "base/WeakLibReader_Math.hpp"
#include "base/WeakLibReader_AxisTypes.hpp"

namespace WeakLibReader {
namespace detail {

/// GPU-optimized log-interpolation using compile-time dimension dispatch
/// @tparam ND Number of dimensions (1-5), known at compile time
/// @param data Raw data array in column-major order (log10-stored values)
/// @param layout Layout describing array dimensions and strides
/// @param axes Array of Axis descriptors (length ND)
/// @param coords Query coordinates (length ND)
/// @param offset Offset to subtract after converting from log space
/// @return Interpolated value: 10^(interp_log_value) - offset
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
void ComputeLinearAxisScale(const Axis& axis, int idx, double& scale) noexcept
{
  AMREX_ASSERT(axis.grid != nullptr);
  const double span = axis.grid[idx + 1] - axis.grid[idx];
  AMREX_ASSERT(span > 0.0);
  scale = math::Ln10 / span;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void ComputeLogAxisScale(const Axis& axis, int idx, double coord,
                         double& scale) noexcept
{
  AMREX_ASSERT(axis.grid != nullptr);
  AMREX_ASSERT(coord > 0.0);
  const double ratio = axis.grid[idx + 1] / axis.grid[idx];
  AMREX_ASSERT(ratio > 0.0);
  const double denom = math::Log10(ratio);
  AMREX_ASSERT(denom != 0.0);
  scale = 1.0 / (coord * denom);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void ComputeAxisScale(const Axis& axis, int idx, double coord,
                      double& scale) noexcept
{
  AMREX_ASSERT(axis.grid != nullptr);
  if (axis.scale == AxisScale::Linear) {
    ComputeLinearAxisScale(axis, idx, scale);
  } else {
    ComputeLogAxisScale(axis, idx, coord, scale);
  }
}

/// GPU-optimized log-interpolation with derivatives using compile-time dimension dispatch
/// @tparam ND Number of dimensions (2-4), known at compile time
/// @param data Raw data array in column-major order (log10-stored values)
/// @param layout Layout describing array dimensions and strides
/// @param axes Array of Axis descriptors (length ND)
/// @param coords Query coordinates (length ND)
/// @param offset Offset to subtract after converting from log space
/// @param[out] interpolant Interpolated value: 10^(interp_log_value) - offset
/// @param[out] derivatives Partial derivatives w.r.t. each dimension (array of size ND)
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

/// GPU-optimized 3D log-interpolation with derivatives (implementation)
/// Uses compile-time dimensionality for zero runtime branching
inline void LogInterpolateDifferentiateSingleVariable3DCustomPointImpl(
    double d, double t, double y,
    const double* data, const Layout& layout,
    const Axis axes[3],
    double offset,
    double& interpolant,
    double derivatives[3]) noexcept
{
  constexpr int ND = 3;
  const double coords[ND] = {d, t, y};

  detail::LogInterpolatedDerivativeDirect<ND>(
      data, layout, axes, coords, offset, interpolant, derivatives);
}

} // namespace detail
} // namespace WeakLibReader
