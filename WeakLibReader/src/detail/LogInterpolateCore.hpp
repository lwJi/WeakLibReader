#pragma once

#include <AMReX_GpuQualifiers.H>
#include <cstddef>
#include <limits>

#include "../InterpLogTable.hpp"
#include "../Math.hpp"
#include "../AxisTypes.hpp"

namespace WeakLibReader {
namespace detail {

/// GPU-optimized log-interpolation using compile-time dimension dispatch
/// @tparam ND Number of dimensions (1-5), known at compile time
/// @param data Raw data array in row-major order (log10-stored values)
/// @param layout Layout describing array dimensions and strides
/// @param axes Array of Axis descriptors (length ND)
/// @param coords Query coordinates (length ND)
/// @param offset Offset to subtract after converting from log space
/// @param cfg Interpolation configuration (out-of-range policy)
/// @return Interpolated value: 10^(interp_log_value) - offset
template<int ND>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double LogInterpolatedValueDirect(const double* data,
                                  const Layout& layout,
                                  const Axis axes[ND],
                                  const double coords[ND],
                                  double offset,
                                  const InterpConfig& cfg) noexcept
{
  int indices[ND];
  double fractions[ND];

  for (int d = 0; d < ND; ++d) {
    bool out = IndexAndDelta(axes[d], coords[d], indices[d], fractions[d]);
    if (out) {
      if (cfg.outOfRange == OutOfRangePolicy::Error) {
        return std::numeric_limits<double>::quiet_NaN();
      }
      // Early exit for FillNaN policy
      if (cfg.outOfRange == OutOfRangePolicy::FillNaN) {
        return std::numeric_limits<double>::quiet_NaN();
      }
      // Clamp fraction for Clamp policy
      fractions[d] = Clamp01(fractions[d]);
    }
  }

  // Compile-time dispatch to dimension-specific kernel (zero runtime branching!)
  return LinearInterpPointDirect<ND>(indices, fractions, offset, data, layout);
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

inline bool ComputeLinearAxisScale(const Axis& axis, int idx, double& scale) noexcept
{
  if (axis.grid == nullptr) {
    scale = std::numeric_limits<double>::quiet_NaN();
    return false;
  }
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
  if (axis.grid == nullptr) {
    scale = std::numeric_limits<double>::quiet_NaN();
    return false;
  }
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

/// Helper: Set interpolant and derivative array to NaN
template<int ND>
inline void SetNaN(double& interpolant, double derivatives[ND]) noexcept
{
  const double nanValue = std::numeric_limits<double>::quiet_NaN();
  interpolant = nanValue;
  for (int d = 0; d < ND; ++d) {
    derivatives[d] = nanValue;
  }
}

/// GPU-optimized log-interpolation with derivatives using compile-time dimension dispatch
/// @tparam ND Number of dimensions (2-4), known at compile time
/// @param data Raw data array in row-major order (log10-stored values)
/// @param layout Layout describing array dimensions and strides
/// @param axes Array of Axis descriptors (length ND)
/// @param coords Query coordinates (length ND)
/// @param offset Offset to subtract after converting from log space
/// @param cfg Interpolation configuration (out-of-range policy)
/// @param[out] interpolant Interpolated value: 10^(interp_log_value) - offset
/// @param[out] derivatives Partial derivatives w.r.t. each dimension (array of size ND)
/// @return true if successful, false if error occurred
template<int ND>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool LogInterpolatedDerivativeDirect(const double* data,
                                     const Layout& layout,
                                     const Axis axes[ND],
                                     const double coords[ND],
                                     double offset,
                                     const InterpConfig& cfg,
                                     double& interpolant,
                                     double derivatives[ND]) noexcept
{
  // Fixed-size arrays for indices, fractions, and scales
  int indices[ND];
  double fractions[ND];
  double scales[ND];
  bool outOfRange = false;

  // Index lookup and scale computation for each dimension
  // Note: Loop fully unrolls at compile time (ND known), ensuring optimal GPU performance
  for (int d = 0; d < ND; ++d) {
    bool out = IndexAndDelta(axes[d], coords[d], indices[d], fractions[d]);
    if (out) {
      outOfRange = true;
      if (cfg.outOfRange == OutOfRangePolicy::Error) {
        return false;
      }
      if (cfg.outOfRange == OutOfRangePolicy::FillNaN) {
        SetNaN<ND>(interpolant, derivatives);
        return true;
      }
      fractions[d] = Clamp01(fractions[d]);
    }

    // Compute axis scale for derivative calculation
    if (!ComputeAxisScale(axes[d], indices[d], coords[d], scales[d])) {
      SetNaN<ND>(interpolant, derivatives);
      return false;
    }
  }

  // Compile-time dispatch to dimension-specific derivative kernel
  LinearInterpDerivPointDirect<ND>(indices, fractions, scales, offset,
                                   data, layout, interpolant, derivatives);
  return true;
}

/// GPU-optimized 3D log-interpolation with derivatives (implementation)
/// Uses compile-time dimensionality for zero runtime branching
inline int LogInterpolateDifferentiateSingleVariable3DCustomPointImpl(
    double d, double t, double y,
    const double* data, const Layout& layout,
    const Axis axes[3],
    double offset,
    double& interpolant,
    double derivatives[3],
    const InterpConfig& cfg) noexcept
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

  constexpr int ND = 3;
  const double coords[ND] = {d, t, y};

  // Use the templated version for GPU optimization
  bool success = detail::LogInterpolatedDerivativeDirect<ND>(
      data, layout, axes, coords, offset, cfg, interpolant, derivatives);

  if (!success) {
    return 4;  // Error occurred
  }

  return 0;
}

} // namespace detail
} // namespace WeakLibReader
