#pragma once

#include <AMReX_GpuQualifiers.H>
#include <cmath>

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

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool InvalidSpan(double span) noexcept
{
  return !(span > 0.0);
}

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

} // namespace detail

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool IndexAndDeltaLin(double x, const double* grid, int n,
                      int& i, double& t) noexcept
{
  if (grid == nullptr || n < 2) {
    i = 0;
    t = 0.0;
    return true;
  }

  const double first = grid[0];
  const double last = grid[n - 1];
  const double span = last - first;

  if (detail::InvalidSpan(span)) {
    i = 0;
    t = 0.0;
    return true;
  }

  const double scaled = (x - first) / span;
  const double scaledIndex = (n - 1) * scaled;
  const int rawIndex = static_cast<int>(detail::Floor(scaledIndex));

  const int maxIndex = n - 2;
  const int clampedIndex = detail::ClampIndex(rawIndex, maxIndex);

  i = clampedIndex;

  const double cellSpan = grid[i + 1] - grid[i];
  t = detail::InvalidSpan(cellSpan) ? 0.0 : (x - grid[i]) / cellSpan;

  const bool outside = (x < first) || (x > last);
  return outside || (rawIndex != clampedIndex);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool IndexAndDeltaLog10(double x, const double* grid, int n,
                        int& i, double& t) noexcept
{
  if (grid == nullptr || n < 2) {
    i = 0;
    t = 0.0;
    return true;
  }

  const double first = grid[0];
  const double last = grid[n - 1];

  if (!(first > 0.0) || !(last > first) || !(x > 0.0)) {
    i = 0;
    t = 0.0;
    return true;
  }

  const double logSpan = detail::Log10(last / first);

  if (detail::InvalidSpan(logSpan)) {
    i = 0;
    t = 0.0;
    return true;
  }

  const double scaled = detail::Log10(x / first) / logSpan;
  const double scaledIndex = (n - 1) * scaled;
  const int rawIndex = static_cast<int>(detail::Floor(scaledIndex));

  const int maxIndex = n - 2;
  const int clampedIndex = detail::ClampIndex(rawIndex, maxIndex);

  i = clampedIndex;

  const double cellRatio = grid[i + 1] / grid[i];
  if (!(cellRatio > 0.0)) {
    t = 0.0;
    return true;
  }

  const double logCellRatio = detail::Log10(cellRatio);
  if (detail::InvalidSpan(logCellRatio)) {
    t = 0.0;
    return true;
  }

  t = detail::Log10(x / grid[i]) / logCellRatio;

  const bool outside = (x < first) || (x > last);
  return outside || (rawIndex != clampedIndex);
}

} // namespace WeakLibReader
