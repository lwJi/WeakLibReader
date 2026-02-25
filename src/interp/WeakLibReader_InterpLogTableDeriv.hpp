#pragma once

#include <AMReX_GpuQualifiers.H>

#include "base/WeakLibReader_InterpBasis.hpp"
#include "base/WeakLibReader_Layout.hpp"
#include "base/WeakLibReader_Math.hpp"

namespace WeakLibReader {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
LinearInterpDeriv2DPoint(int i0, int i1, double d0, double d1, double a0,
                         double a1, double offset, const double *data,
                         const Layout &layout, double &interpolant,
                         double &dIdX0, double &dIdX1) noexcept {
  const std::size_t base = layout.Offset(i0, i1);
  const std::size_t s0 = layout.stride[0];
  const std::size_t s1 = layout.stride[1];

  const double p00 = data[base];
  const double p10 = data[base + s0];
  const double p01 = data[base + s1];
  const double p11 = data[base + s0 + s1];

  const double logValue = BiLinear(p00, p10, p01, p11, d0, d1);
  interpolant = math::Pow10(logValue) - offset;

  const double scale = interpolant + offset;
  dIdX0 = scale * a0 * BiLinearDerivativeX1(p00, p10, p01, p11, d1);
  dIdX1 = scale * a1 * BiLinearDerivativeX2(p00, p10, p01, p11, d0);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
LinearInterpDeriv3DPoint(int i0, int i1, int i2, double d0, double d1,
                         double d2, double a0, double a1, double a2,
                         double offset, const double *data,
                         const Layout &layout, double &interpolant,
                         double &dIdX0, double &dIdX1, double &dIdX2) noexcept {
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

  const double logValue =
      TriLinear(p000, p100, p010, p110, p001, p101, p011, p111, d0, d1, d2);
  interpolant = math::Pow10(logValue) - offset;

  const double scale = interpolant + offset;
  dIdX0 = scale * a0 *
          TriLinearDerivativeX1(p000, p100, p010, p110, p001, p101, p011, p111,
                                d1, d2);
  dIdX1 = scale * a1 *
          TriLinearDerivativeX2(p000, p100, p010, p110, p001, p101, p011, p111,
                                d0, d2);
  dIdX2 = scale * a2 *
          TriLinearDerivativeX3(p000, p100, p010, p110, p001, p101, p011, p111,
                                d0, d1);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void LinearInterpDeriv4DPoint(
    int i0, int i1, int i2, int i3, double d0, double d1, double d2, double d3,
    double a0, double a1, double a2, double a3, double offset,
    const double *data, const Layout &layout, double &interpolant,
    double &dIdX0, double &dIdX1, double &dIdX2, double &dIdX3) noexcept {
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

  const double logValue = TetraLinear(
      p0000, p1000, p0100, p1100, p0010, p1010, p0110, p1110, p0001, p1001,
      p0101, p1101, p0011, p1011, p0111, p1111, d0, d1, d2, d3);
  interpolant = math::Pow10(logValue) - offset;

  const double scale = interpolant + offset;
  dIdX0 = scale * a0 *
          TetraLinearDerivativeX1(p0000, p1000, p0100, p1100, p0010, p1010,
                                  p0110, p1110, p0001, p1001, p0101, p1101,
                                  p0011, p1011, p0111, p1111, d1, d2, d3);
  dIdX1 = scale * a1 *
          TetraLinearDerivativeX2(p0000, p1000, p0100, p1100, p0010, p1010,
                                  p0110, p1110, p0001, p1001, p0101, p1101,
                                  p0011, p1011, p0111, p1111, d0, d2, d3);
  dIdX2 = scale * a2 *
          TetraLinearDerivativeX3(p0000, p1000, p0100, p1100, p0010, p1010,
                                  p0110, p1110, p0001, p1001, p0101, p1101,
                                  p0011, p1011, p0111, p1111, d0, d1, d3);
  dIdX3 = scale * a3 *
          TetraLinearDerivativeX4(p0000, p1000, p0100, p1100, p0010, p1010,
                                  p0110, p1110, p0001, p1001, p0101, p1101,
                                  p0011, p1011, p0111, p1111, d0, d1, d2);
}

template <int ND>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void LinearInterpDerivPointDirect(
    const int indices[ND], const double fractions[ND], const double scales[ND],
    double offset, const double *data, const Layout &layout,
    double &interpolant, double derivatives[ND]) noexcept {
  static_assert(ND >= 2 && ND <= 4, "Derivatives only supported for 2D-4D");

  if constexpr (ND == 2) {
    LinearInterpDeriv2DPoint(indices[0], indices[1], fractions[0], fractions[1],
                             scales[0], scales[1], offset, data, layout,
                             interpolant, derivatives[0], derivatives[1]);
  } else if constexpr (ND == 3) {
    LinearInterpDeriv3DPoint(indices[0], indices[1], indices[2], fractions[0],
                             fractions[1], fractions[2], scales[0], scales[1],
                             scales[2], offset, data, layout, interpolant,
                             derivatives[0], derivatives[1], derivatives[2]);
  } else if constexpr (ND == 4) {
    LinearInterpDeriv4DPoint(
        indices[0], indices[1], indices[2], indices[3], fractions[0],
        fractions[1], fractions[2], fractions[3], scales[0], scales[1],
        scales[2], scales[3], offset, data, layout, interpolant, derivatives[0],
        derivatives[1], derivatives[2], derivatives[3]);
  }
}

} // namespace WeakLibReader
