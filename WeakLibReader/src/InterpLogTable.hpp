#pragma once

#include <AMReX_GpuQualifiers.H>

#include "InterpBasis.hpp"
#include "Layout.hpp"
#include "Math.hpp"

namespace WeakLibReader {
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double LinearInterp1DPoint(int i0, double d0, double os,
                           const double* data,
                           const Layout& layout) noexcept
{
  const std::size_t base = layout.Offset(i0);
  const double p0 = data[base];
  const double p1 = data[base + layout.stride[0]];
  return math::Pow10(Linear(p0, p1, d0)) - os;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double LinearInterp2DPoint(int i0, int i1,
                           double d0, double d1, double os,
                           const double* data,
                           const Layout& layout) noexcept
{
  const std::size_t base = layout.Offset(i0, i1);
  const std::size_t s0 = layout.stride[0];
  const std::size_t s1 = layout.stride[1];

  const double p00 = data[base];
  const double p10 = data[base + s0];
  const double p01 = data[base + s1];
  const double p11 = data[base + s0 + s1];

  return math::Pow10(BiLinear(p00, p10, p01, p11, d0, d1)) - os;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double LinearInterp3DPoint(int i0, int i1, int i2,
                           double d0, double d1, double d2, double os,
                           const double* data,
                           const Layout& layout) noexcept
{
  const std::size_t base = layout.Offset(i0, i1, i2);
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

  return math::Pow10(TriLinear(p000, p100, p010, p110,
                               p001, p101, p011, p111,
                               d0, d1, d2)) - os;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double LinearInterp4DPoint(int i0, int i1, int i2, int i3,
                           double d0, double d1, double d2, double d3,
                           double os,
                           const double* data,
                           const Layout& layout) noexcept
{
  const std::size_t base = layout.Offset(i0, i1, i2, i3);
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

  return math::Pow10(TetraLinear(p0000, p1000, p0100, p1100,
                                 p0010, p1010, p0110, p1110,
                                 p0001, p1001, p0101, p1101,
                                 p0011, p1011, p0111, p1111,
                                 d0, d1, d2, d3)) - os;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double LinearInterp5DPoint(int i0, int i1, int i2, int i3, int i4,
                           double d0, double d1, double d2, double d3, double d4,
                           double os,
                           const double* data,
                           const Layout& layout) noexcept
{
  const std::size_t base = layout.Offset(i0, i1, i2, i3, i4);
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

  return math::Pow10(PentaLinear(p00000, p10000, p01000, p11000,
                                 p00100, p10100, p01100, p11100,
                                 p00010, p10010, p01010, p11010,
                                 p00110, p10110, p01110, p11110,
                                 p00001, p10001, p01001, p11001,
                                 p00101, p10101, p01101, p11101,
                                 p00011, p10011, p01011, p11011,
                                 p00111, p10111, p01111, p11111,
                                 d0, d1, d2, d3, d4)) - os;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void LinearInterpDeriv2DPoint(int i0, int i1,
                              double d0, double d1,
                              double a0, double a1,
                              double os,
                              const double* data,
                              const Layout& layout,
                              double& interpolant,
                              double& dIdX0,
                              double& dIdX1) noexcept
{
  const std::size_t base = layout.Offset(i0, i1);
  const std::size_t s0 = layout.stride[0];
  const std::size_t s1 = layout.stride[1];

  const double p00 = data[base];
  const double p10 = data[base + s0];
  const double p01 = data[base + s1];
  const double p11 = data[base + s0 + s1];

  const double logValue = BiLinear(p00, p10, p01, p11, d0, d1);
  interpolant = math::Pow10(logValue) - os;

  const double scale = interpolant + os;
  dIdX0 = scale * a0 * BiLinearDerivativeX1(p00, p10, p01, p11, d1);
  dIdX1 = scale * a1 * BiLinearDerivativeX2(p00, p10, p01, p11, d0);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void LinearInterpDeriv3DPoint(int i0, int i1, int i2,
                              double d0, double d1, double d2,
                              double a0, double a1, double a2,
                              double os,
                              const double* data,
                              const Layout& layout,
                              double& interpolant,
                              double& dIdX0,
                              double& dIdX1,
                              double& dIdX2) noexcept
{
  const std::size_t base = layout.Offset(i0, i1, i2);
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

  const double logValue = TriLinear(p000, p100, p010, p110,
                                    p001, p101, p011, p111,
                                    d0, d1, d2);
  interpolant = math::Pow10(logValue) - os;

  const double scale = interpolant + os;
  dIdX0 = scale * a0 * TriLinearDerivativeX1(p000, p100, p010, p110,
                                             p001, p101, p011, p111,
                                             d1, d2);
  dIdX1 = scale * a1 * TriLinearDerivativeX2(p000, p100, p010, p110,
                                             p001, p101, p011, p111,
                                             d0, d2);
  dIdX2 = scale * a2 * TriLinearDerivativeX3(p000, p100, p010, p110,
                                             p001, p101, p011, p111,
                                             d0, d1);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void LinearInterpDeriv4DPoint(int i0, int i1, int i2, int i3,
                              double d0, double d1, double d2, double d3,
                              double a0, double a1, double a2, double a3,
                              double os,
                              const double* data,
                              const Layout& layout,
                              double& interpolant,
                              double& dIdX0,
                              double& dIdX1,
                              double& dIdX2,
                              double& dIdX3) noexcept
{
  const std::size_t base = layout.Offset(i0, i1, i2, i3);
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

  const double logValue = TetraLinear(p0000, p1000, p0100, p1100,
                                      p0010, p1010, p0110, p1110,
                                      p0001, p1001, p0101, p1101,
                                      p0011, p1011, p0111, p1111,
                                      d0, d1, d2, d3);
  interpolant = math::Pow10(logValue) - os;

  const double scale = interpolant + os;
  dIdX0 = scale * a0 * TetraLinearDerivativeX1(p0000, p1000, p0100, p1100,
                                               p0010, p1010, p0110, p1110,
                                               p0001, p1001, p0101, p1101,
                                               p0011, p1011, p0111, p1111,
                                               d1, d2, d3);
  dIdX1 = scale * a1 * TetraLinearDerivativeX2(p0000, p1000, p0100, p1100,
                                               p0010, p1010, p0110, p1110,
                                               p0001, p1001, p0101, p1101,
                                               p0011, p1011, p0111, p1111,
                                               d0, d2, d3);
  dIdX2 = scale * a2 * TetraLinearDerivativeX3(p0000, p1000, p0100, p1100,
                                               p0010, p1010, p0110, p1110,
                                               p0001, p1001, p0101, p1101,
                                               p0011, p1011, p0111, p1111,
                                               d0, d1, d3);
  dIdX3 = scale * a3 * TetraLinearDerivativeX4(p0000, p1000, p0100, p1100,
                                               p0010, p1010, p0110, p1110,
                                               p0001, p1001, p0101, p1101,
                                               p0011, p1011, p0111, p1111,
                                               d0, d1, d2);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double LinearInterp2D3DArray1DAlignedPoint(int iFixed,
                                           int i0, int i1,
                                           double d0, double d1,
                                           double os,
                                           const double* data,
                                           const Layout& layout) noexcept
{
  const double* slice = data + layout.Offset(iFixed, 0, 0);
  const Layout sliceLayout = SliceLeading(layout, 1);
  return LinearInterp2DPoint(i0, i1, d0, d1, os, slice, sliceLayout);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double LinearInterp2D4DArray2DAlignedPoint(int iFixed0, int iFixed1,
                                           int i0, int i1,
                                           double d0, double d1,
                                           double os,
                                           const double* data,
                                           const Layout& layout) noexcept
{
  const double* slice = data + layout.Offset(iFixed0, iFixed1, 0, 0);
  const Layout sliceLayout = SliceLeading(layout, 2);
  return LinearInterp2DPoint(i0, i1, d0, d1, os, slice, sliceLayout);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double LinearInterp3D4DArray1DAlignedPoint(int iFixed,
                                           int i0, int i1, int i2,
                                           double d0, double d1, double d2,
                                           double os,
                                           const double* data,
                                           const Layout& layout) noexcept
{
  const double* slice = data + layout.Offset(iFixed, 0, 0, 0);
  const Layout sliceLayout = SliceLeading(layout, 1);
  return LinearInterp3DPoint(i0, i1, i2, d0, d1, d2, os, slice, sliceLayout);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double LinearInterp3D5DArray2DAlignedPoint(int iFixed0, int iFixed1,
                                           int i0, int i1, int i2,
                                           double d0, double d1, double d2,
                                           double os,
                                           const double* data,
                                           const Layout& layout) noexcept
{
  const double* slice = data + layout.Offset(iFixed0, iFixed1, 0, 0, 0);
  const Layout sliceLayout = SliceLeading(layout, 2);
  return LinearInterp3DPoint(i0, i1, i2, d0, d1, d2, os, slice, sliceLayout);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double LinearInterp4D5DArray1DAlignedPoint(int iFixed,
                                           int i0, int i1, int i2, int i3,
                                           double d0, double d1, double d2, double d3,
                                           double os,
                                           const double* data,
                                           const Layout& layout) noexcept
{
  const double* slice = data + layout.Offset(iFixed, 0, 0, 0, 0);
  const Layout sliceLayout = SliceLeading(layout, 1);
  return LinearInterp4DPoint(i0, i1, i2, i3, d0, d1, d2, d3, os, slice, sliceLayout);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void LinearInterpDeriv2D4DArray2DAlignedPoint(int iFixed0, int iFixed1,
                                              int i0, int i1,
                                              double d0, double d1,
                                              double a0, double a1,
                                              double os,
                                              const double* data,
                                              const Layout& layout,
                                              double& interpolant,
                                              double& dIdX0,
                                              double& dIdX1) noexcept
{
  const double* slice = data + layout.Offset(iFixed0, iFixed1, 0, 0);
  const Layout sliceLayout = SliceLeading(layout, 2);
  LinearInterpDeriv2DPoint(i0, i1, d0, d1, a0, a1, os, slice, sliceLayout,
                           interpolant, dIdX0, dIdX1);
}

// ============================================================================
// Template-based compile-time dispatch wrappers for GPU optimization
// ============================================================================
// These templates eliminate runtime branching by dispatching to the correct
// dimension-specific function at compile time using 'if constexpr' (C++17).
// This improves GPU performance by avoiding warp divergence.
// ============================================================================

/// Template wrapper for dimension-specific log-interpolation (compile-time dispatch)
/// @tparam ND Number of dimensions (1-5), known at compile time
/// @param indices Cell indices for each dimension (output from IndexAndDelta*)
/// @param fractions Interpolation fractions [0,1] for each dimension
/// @param offset Offset to subtract after converting from log space
/// @param data Raw data array in row-major order
/// @param layout Layout describing array dimensions and strides
/// @return Interpolated value: 10^(interp_log_value) - offset
template<int ND>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double LinearInterpPointDirect(const int indices[ND],
                               const double fractions[ND],
                               double offset,
                               const double* data,
                               const Layout& layout) noexcept
{
  // Compile-time dispatch: branches eliminated at compile time, zero runtime overhead
  if constexpr (ND == 1) {
    return LinearInterp1DPoint(indices[0], fractions[0], offset, data, layout);
  } else if constexpr (ND == 2) {
    return LinearInterp2DPoint(indices[0], indices[1],
                               fractions[0], fractions[1],
                               offset, data, layout);
  } else if constexpr (ND == 3) {
    return LinearInterp3DPoint(indices[0], indices[1], indices[2],
                               fractions[0], fractions[1], fractions[2],
                               offset, data, layout);
  } else if constexpr (ND == 4) {
    return LinearInterp4DPoint(indices[0], indices[1], indices[2], indices[3],
                               fractions[0], fractions[1], fractions[2], fractions[3],
                               offset, data, layout);
  } else if constexpr (ND == 5) {
    return LinearInterp5DPoint(indices[0], indices[1], indices[2], indices[3], indices[4],
                               fractions[0], fractions[1], fractions[2], fractions[3], fractions[4],
                               offset, data, layout);
  }
}

/// Template wrapper for dimension-specific log-interpolation with derivatives (compile-time dispatch)
/// @tparam ND Number of dimensions (2-4), known at compile time
/// @param indices Cell indices for each dimension
/// @param fractions Interpolation fractions [0,1] for each dimension
/// @param scales Axis-dependent scaling factors for derivative computation
/// @param offset Offset to subtract after converting from log space
/// @param data Raw data array in row-major order
/// @param layout Layout describing array dimensions and strides
/// @param[out] interpolant Interpolated value: 10^(interp_log_value) - offset
/// @param[out] derivatives Partial derivatives w.r.t. each dimension (array of size ND)
template<int ND>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void LinearInterpDerivPointDirect(const int indices[ND],
                                  const double fractions[ND],
                                  const double scales[ND],
                                  double offset,
                                  const double* data,
                                  const Layout& layout,
                                  double& interpolant,
                                  double derivatives[ND]) noexcept
{
  // Compile-time dispatch for derivative functions (2D, 3D, 4D only)
  if constexpr (ND == 2) {
    LinearInterpDeriv2DPoint(indices[0], indices[1],
                             fractions[0], fractions[1],
                             scales[0], scales[1],
                             offset, data, layout,
                             interpolant, derivatives[0], derivatives[1]);
  } else if constexpr (ND == 3) {
    LinearInterpDeriv3DPoint(indices[0], indices[1], indices[2],
                             fractions[0], fractions[1], fractions[2],
                             scales[0], scales[1], scales[2],
                             offset, data, layout,
                             interpolant, derivatives[0], derivatives[1], derivatives[2]);
  } else if constexpr (ND == 4) {
    LinearInterpDeriv4DPoint(indices[0], indices[1], indices[2], indices[3],
                             fractions[0], fractions[1], fractions[2], fractions[3],
                             scales[0], scales[1], scales[2], scales[3],
                             offset, data, layout,
                             interpolant, derivatives[0], derivatives[1], derivatives[2], derivatives[3]);
  }
}

} // namespace WeakLibReader
