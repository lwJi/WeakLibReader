#pragma once

#include <AMReX_GpuQualifiers.H>
#include <cmath>

namespace WeakLibReader {
namespace math {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double Floor(double value) noexcept
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return ::floor(value);
#else
  return std::floor(value);
#endif
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double Log10(double value) noexcept
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return ::log10(value);
#else
  return std::log10(value);
#endif
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double Pow10(double value) noexcept
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return ::pow(10.0, value);
#else
  return std::pow(10.0, value);
#endif
}

} // namespace math
} // namespace WeakLibReader

