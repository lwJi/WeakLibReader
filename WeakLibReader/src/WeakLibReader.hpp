#pragma once

#include <AMReX_GpuQualifiers.H>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "WeakLibReader/IndexDelta.hpp"
#include "WeakLibReader/InterpBasis.hpp"
#include "WeakLibReader/Layout.hpp"

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

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double InterpCorner(const double* data,
                    const Layout& layout,
                    const int idx[5],
                    const double frac[5],
                    int nd) noexcept
{
  switch (nd) {
    case 1: {
      const std::size_t base = layout.Offset(idx[0]);
      const std::size_t s0 = layout.stride[0];
      const double p0 = data[base];
      const double p1 = data[base + s0];
      return Linear(p0, p1, frac[0]);
    }
    case 2: {
      const std::size_t base = layout.Offset(idx[0], idx[1]);
      const std::size_t s0 = layout.stride[0];
      const std::size_t s1 = layout.stride[1];
      const double p00 = data[base];
      const double p10 = data[base + s0];
      const double p01 = data[base + s1];
      const double p11 = data[base + s0 + s1];
      return BiLinear(p00, p10, p01, p11, frac[0], frac[1]);
    }
    case 3: {
      const std::size_t base = layout.Offset(idx[0], idx[1], idx[2]);
      const std::size_t s0 = layout.stride[0];
      const std::size_t s1 = layout.stride[1];
      const std::size_t s2 = layout.stride[2];
      const double p000 = data[base];
      const double p100 = data[base + s0];
      const double p010 = data[base + s1];
      const double p110 = data[base + s0 + s1];
      const double p001 = data[base + s2];
      const double p101 = data[base + s0 + s2];
      const double p011 = data[base + s1 + s2];
      const double p111 = data[base + s0 + s1 + s2];
      return TriLinear(p000, p100, p010, p110,
                       p001, p101, p011, p111,
                       frac[0], frac[1], frac[2]);
    }
    case 4: {
      const std::size_t base = layout.Offset(idx[0], idx[1], idx[2], idx[3]);
      const std::size_t s0 = layout.stride[0];
      const std::size_t s1 = layout.stride[1];
      const std::size_t s2 = layout.stride[2];
      const std::size_t s3 = layout.stride[3];
      const double p0000 = data[base];
      const double p1000 = data[base + s0];
      const double p0100 = data[base + s1];
      const double p1100 = data[base + s0 + s1];
      const double p0010 = data[base + s2];
      const double p1010 = data[base + s0 + s2];
      const double p0110 = data[base + s1 + s2];
      const double p1110 = data[base + s0 + s1 + s2];
      const double p0001 = data[base + s3];
      const double p1001 = data[base + s0 + s3];
      const double p0101 = data[base + s1 + s3];
      const double p1101 = data[base + s0 + s1 + s3];
      const double p0011 = data[base + s2 + s3];
      const double p1011 = data[base + s0 + s2 + s3];
      const double p0111 = data[base + s1 + s2 + s3];
      const double p1111 = data[base + s0 + s1 + s2 + s3];
      return TetraLinear(p0000, p1000, p0100, p1100,
                         p0010, p1010, p0110, p1110,
                         p0001, p1001, p0101, p1101,
                         p0011, p1011, p0111, p1111,
                         frac[0], frac[1], frac[2], frac[3]);
    }
    case 5: {
      const std::size_t base = layout.Offset(idx[0], idx[1], idx[2], idx[3], idx[4]);
      const std::size_t s0 = layout.stride[0];
      const std::size_t s1 = layout.stride[1];
      const std::size_t s2 = layout.stride[2];
      const std::size_t s3 = layout.stride[3];
      const std::size_t s4 = layout.stride[4];
      const double p00000 = data[base];
      const double p10000 = data[base + s0];
      const double p01000 = data[base + s1];
      const double p11000 = data[base + s0 + s1];
      const double p00100 = data[base + s2];
      const double p10100 = data[base + s0 + s2];
      const double p01100 = data[base + s1 + s2];
      const double p11100 = data[base + s0 + s1 + s2];
      const double p00010 = data[base + s3];
      const double p10010 = data[base + s0 + s3];
      const double p01010 = data[base + s1 + s3];
      const double p11010 = data[base + s0 + s1 + s3];
      const double p00110 = data[base + s2 + s3];
      const double p10110 = data[base + s0 + s2 + s3];
      const double p01110 = data[base + s1 + s2 + s3];
      const double p11110 = data[base + s0 + s1 + s2 + s3];
      const double p00001 = data[base + s4];
      const double p10001 = data[base + s0 + s4];
      const double p01001 = data[base + s1 + s4];
      const double p11001 = data[base + s0 + s1 + s4];
      const double p00101 = data[base + s2 + s4];
      const double p10101 = data[base + s0 + s2 + s4];
      const double p01101 = data[base + s1 + s2 + s4];
      const double p11101 = data[base + s0 + s1 + s2 + s4];
      const double p00011 = data[base + s3 + s4];
      const double p10011 = data[base + s0 + s3 + s4];
      const double p01011 = data[base + s1 + s3 + s4];
      const double p11011 = data[base + s0 + s1 + s3 + s4];
      const double p00111 = data[base + s2 + s3 + s4];
      const double p10111 = data[base + s0 + s2 + s3 + s4];
      const double p01111 = data[base + s1 + s2 + s3 + s4];
      const double p11111 = data[base + s0 + s1 + s2 + s3 + s4];
      return PentaLinear(p00000, p10000, p01000, p11000,
                         p00100, p10100, p01100, p11100,
                         p00010, p10010, p01010, p11010,
                         p00110, p10110, p01110, p11110,
                         p00001, p10001, p01001, p11001,
                         p00101, p10101, p01101, p11101,
                         p00011, p10011, p01011, p11011,
                         p00111, p10111, p01111, p11111,
                         frac[0], frac[1], frac[2], frac[3], frac[4]);
    }
    default:
      return data[0];
  }
}

} // namespace detail

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double InterpLinearND(const double* data, const Layout& layout,
                      const Axis axes[5], const double x[5],
                      const InterpConfig& cfg, int nd) noexcept
{
  const int dims = (nd > 0) ? nd : layout.nd;
  if (dims <= 0 || dims > 5) {
    return detail::NaN();
  }

  int indices[5] = {0, 0, 0, 0, 0};
  double frac[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  bool outOfRange = false;

  for (int dim = 0; dim < dims; ++dim) {
    const bool status = detail::IndexAndDelta(axes[dim], x[dim], indices[dim], frac[dim]);
    if (status) {
      outOfRange = true;
      if (cfg.outOfRange == OutOfRangePolicy::Error) {
        return detail::NaN();
      }
      if (cfg.outOfRange == OutOfRangePolicy::FillNaN) {
        return detail::NaN();
      }
      frac[dim] = detail::Clamp01(frac[dim]);
    }
  }

  if (outOfRange && cfg.outOfRange == OutOfRangePolicy::FillNaN) {
    return detail::NaN();
  }

  return detail::InterpCorner(data, layout, indices, frac, dims);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double InterpLinearND(const double* data, const Layout& layout,
                      const Axis axes[5], const double x[5]) noexcept
{
  const InterpConfig cfg{};
  return InterpLinearND(data, layout, axes, x, cfg, layout.nd);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double InterpLinear1D(const double* data, const Layout& layout,
                      const Axis axes[5], double x0,
                      const InterpConfig& cfg) noexcept
{
  double coords[5] = {x0, 0.0, 0.0, 0.0, 0.0};
  return InterpLinearND(data, layout, axes, coords, cfg, 1);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double InterpLinear2D(const double* data, const Layout& layout,
                      const Axis axes[5], double x0, double x1,
                      const InterpConfig& cfg) noexcept
{
  double coords[5] = {x0, x1, 0.0, 0.0, 0.0};
  return InterpLinearND(data, layout, axes, coords, cfg, 2);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double InterpLinear3D(const double* data, const Layout& layout,
                      const Axis axes[5],
                      double x0, double x1, double x2,
                      const InterpConfig& cfg) noexcept
{
  double coords[5] = {x0, x1, x2, 0.0, 0.0};
  return InterpLinearND(data, layout, axes, coords, cfg, 3);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double InterpLinear4D(const double* data, const Layout& layout,
                      const Axis axes[5],
                      double x0, double x1, double x2, double x3,
                      const InterpConfig& cfg) noexcept
{
  double coords[5] = {x0, x1, x2, x3, 0.0};
  return InterpLinearND(data, layout, axes, coords, cfg, 4);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double InterpLinear5D(const double* data, const Layout& layout,
                      const Axis axes[5],
                      double x0, double x1, double x2, double x3, double x4,
                      const InterpConfig& cfg) noexcept
{
  double coords[5] = {x0, x1, x2, x3, x4};
  return InterpLinearND(data, layout, axes, coords, cfg, 5);
}

} // namespace WeakLibReader
