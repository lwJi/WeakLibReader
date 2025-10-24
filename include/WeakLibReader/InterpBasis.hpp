#pragma once

#include <AMReX_GpuQualifiers.H>

namespace WeakLibReader {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double Linear(double p0, double p1, double dX1) noexcept
{
  const double oneMinus = 1.0 - dX1;
  return oneMinus * p0 + dX1 * p1;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double BiLinear(double p00, double p10, double p01, double p11,
                double dX1, double dX2) noexcept
{
  const double lo = Linear(p00, p10, dX1);
  const double hi = Linear(p01, p11, dX1);
  return Linear(lo, hi, dX2);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double BiLinearDerivativeX1(double p00, double p10,
                            double p01, double p11,
                            double dX2) noexcept
{
  return Linear(p10, p11, dX2) - Linear(p00, p01, dX2);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double BiLinearDerivativeX2(double p00, double p10,
                            double p01, double p11,
                            double dX1) noexcept
{
  return Linear(p01, p11, dX1) - Linear(p00, p10, dX1);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double TriLinear(double p000, double p100, double p010, double p110,
                 double p001, double p101, double p011, double p111,
                 double dX1, double dX2, double dX3) noexcept
{
  const double lo = BiLinear(p000, p100, p010, p110, dX1, dX2);
  const double hi = BiLinear(p001, p101, p011, p111, dX1, dX2);
  return Linear(lo, hi, dX3);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double TriLinearDerivativeX1(double p000, double p100, double p010, double p110,
                             double p001, double p101, double p011, double p111,
                             double dX2, double dX3) noexcept
{
  return BiLinear(p100, p110, p101, p111, dX2, dX3) -
         BiLinear(p000, p010, p001, p011, dX2, dX3);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double TriLinearDerivativeX2(double p000, double p100, double p010, double p110,
                             double p001, double p101, double p011, double p111,
                             double dX1, double dX3) noexcept
{
  return BiLinear(p010, p110, p011, p111, dX1, dX3) -
         BiLinear(p000, p100, p001, p101, dX1, dX3);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double TriLinearDerivativeX3(double p000, double p100, double p010, double p110,
                             double p001, double p101, double p011, double p111,
                             double dX1, double dX2) noexcept
{
  return BiLinear(p001, p101, p011, p111, dX1, dX2) -
         BiLinear(p000, p100, p010, p110, dX1, dX2);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double TetraLinear(double p0000, double p1000, double p0100, double p1100,
                   double p0010, double p1010, double p0110, double p1110,
                   double p0001, double p1001, double p0101, double p1101,
                   double p0011, double p1011, double p0111, double p1111,
                   double dX1, double dX2, double dX3, double dX4) noexcept
{
  const double lo = TriLinear(p0000, p1000, p0100, p1100,
                              p0010, p1010, p0110, p1110,
                              dX1, dX2, dX3);
  const double hi = TriLinear(p0001, p1001, p0101, p1101,
                              p0011, p1011, p0111, p1111,
                              dX1, dX2, dX3);
  return Linear(lo, hi, dX4);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double TetraLinearDerivativeX1(double p0000, double p1000, double p0100, double p1100,
                               double p0010, double p1010, double p0110, double p1110,
                               double p0001, double p1001, double p0101, double p1101,
                               double p0011, double p1011, double p0111, double p1111,
                               double dX2, double dX3, double dX4) noexcept
{
  return TriLinear(p1000, p1100, p1010, p1110,
                   p1001, p1101, p1011, p1111,
                   dX2, dX3, dX4) -
         TriLinear(p0000, p0100, p0010, p0110,
                   p0001, p0101, p0011, p0111,
                   dX2, dX3, dX4);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double TetraLinearDerivativeX2(double p0000, double p1000, double p0100, double p1100,
                               double p0010, double p1010, double p0110, double p1110,
                               double p0001, double p1001, double p0101, double p1101,
                               double p0011, double p1011, double p0111, double p1111,
                               double dX1, double dX3, double dX4) noexcept
{
  return TriLinear(p0100, p1100, p0110, p1110,
                   p0101, p1101, p0111, p1111,
                   dX1, dX3, dX4) -
         TriLinear(p0000, p1000, p0010, p1010,
                   p0001, p1001, p0011, p1011,
                   dX1, dX3, dX4);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double TetraLinearDerivativeX3(double p0000, double p1000, double p0100, double p1100,
                               double p0010, double p1010, double p0110, double p1110,
                               double p0001, double p1001, double p0101, double p1101,
                               double p0011, double p1011, double p0111, double p1111,
                               double dX1, double dX2, double dX4) noexcept
{
  return TriLinear(p0010, p1010, p0110, p1110,
                   p0011, p1011, p0111, p1111,
                   dX1, dX2, dX4) -
         TriLinear(p0000, p1000, p0100, p1100,
                   p0001, p1001, p0101, p1101,
                   dX1, dX2, dX4);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double TetraLinearDerivativeX4(double p0000, double p1000, double p0100, double p1100,
                               double p0010, double p1010, double p0110, double p1110,
                               double p0001, double p1001, double p0101, double p1101,
                               double p0011, double p1011, double p0111, double p1111,
                               double dX1, double dX2, double dX3) noexcept
{
  return TriLinear(p0001, p1001, p0101, p1101,
                   p0011, p1011, p0111, p1111,
                   dX1, dX2, dX3) -
         TriLinear(p0000, p1000, p0100, p1100,
                   p0010, p1010, p0110, p1110,
                   dX1, dX2, dX3);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double PentaLinear(double p00000, double p10000, double p01000, double p11000,
                   double p00100, double p10100, double p01100, double p11100,
                   double p00010, double p10010, double p01010, double p11010,
                   double p00110, double p10110, double p01110, double p11110,
                   double p00001, double p10001, double p01001, double p11001,
                   double p00101, double p10101, double p01101, double p11101,
                   double p00011, double p10011, double p01011, double p11011,
                   double p00111, double p10111, double p01111, double p11111,
                   double dX1, double dX2, double dX3, double dX4, double dX5) noexcept
{
  const double lo = TetraLinear(p00000, p10000, p01000, p11000,
                                p00100, p10100, p01100, p11100,
                                p00010, p10010, p01010, p11010,
                                p00110, p10110, p01110, p11110,
                                dX1, dX2, dX3, dX4);
  const double hi = TetraLinear(p00001, p10001, p01001, p11001,
                                p00101, p10101, p01101, p11101,
                                p00011, p10011, p01011, p11011,
                                p00111, p10111, p01111, p11111,
                                dX1, dX2, dX3, dX4);
  return Linear(lo, hi, dX5);
}

} // namespace WeakLibReader

