#pragma once

#include <AMReX_GpuQualifiers.H>

#include "interp/WeakLibReader_InterpLogTablePoint.hpp"
#include "interp/WeakLibReader_InterpLogTableDeriv.hpp"

namespace WeakLibReader {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double LinearInterp2D3DArray1DAlignedPoint(int iFixed,
                                           int i0, int i1,
                                           double d0, double d1,
                                           double offset,
                                           const double* data,
                                           const Layout& layout) noexcept
{
  const double* slice = data + layout.Offset(iFixed, 0, 0);
  const Layout sliceLayout = SliceLeading(layout, 1);
  return LinearInterp2DPoint(i0, i1, d0, d1, offset, slice, sliceLayout);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double LinearInterp2D4DArray2DAlignedPoint(int iFixed0, int iFixed1,
                                           int i0, int i1,
                                           double d0, double d1,
                                           double offset,
                                           const double* data,
                                           const Layout& layout) noexcept
{
  const double* slice = data + layout.Offset(iFixed0, iFixed1, 0, 0);
  const Layout sliceLayout = SliceLeading(layout, 2);
  return LinearInterp2DPoint(i0, i1, d0, d1, offset, slice, sliceLayout);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double LinearInterp3D4DArray1DAlignedPoint(int iFixed,
                                           int i0, int i1, int i2,
                                           double d0, double d1, double d2,
                                           double offset,
                                           const double* data,
                                           const Layout& layout) noexcept
{
  const double* slice = data + layout.Offset(iFixed, 0, 0, 0);
  const Layout sliceLayout = SliceLeading(layout, 1);
  return LinearInterp3DPoint(i0, i1, i2, d0, d1, d2, offset, slice, sliceLayout);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double LinearInterp3D5DArray2DAlignedPoint(int iFixed0, int iFixed1,
                                           int i0, int i1, int i2,
                                           double d0, double d1, double d2,
                                           double offset,
                                           const double* data,
                                           const Layout& layout) noexcept
{
  const double* slice = data + layout.Offset(iFixed0, iFixed1, 0, 0, 0);
  const Layout sliceLayout = SliceLeading(layout, 2);
  return LinearInterp3DPoint(i0, i1, i2, d0, d1, d2, offset, slice, sliceLayout);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double LinearInterp4D5DArray1DAlignedPoint(int iFixed,
                                           int i0, int i1, int i2, int i3,
                                           double d0, double d1, double d2, double d3,
                                           double offset,
                                           const double* data,
                                           const Layout& layout) noexcept
{
  const double* slice = data + layout.Offset(iFixed, 0, 0, 0, 0);
  const Layout sliceLayout = SliceLeading(layout, 1);
  return LinearInterp4DPoint(i0, i1, i2, i3, d0, d1, d2, d3, offset, slice, sliceLayout);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void LinearInterpDeriv2D4DArray2DAlignedPoint(int iFixed0, int iFixed1,
                                              int i0, int i1,
                                              double d0, double d1,
                                              double a0, double a1,
                                              double offset,
                                              const double* data,
                                              const Layout& layout,
                                              double& interpolant,
                                              double& dIdX0,
                                              double& dIdX1) noexcept
{
  const double* slice = data + layout.Offset(iFixed0, iFixed1, 0, 0);
  const Layout sliceLayout = SliceLeading(layout, 2);
  LinearInterpDeriv2DPoint(i0, i1, d0, d1, a0, a1, offset, slice, sliceLayout,
                           interpolant, dIdX0, dIdX1);
}

} // namespace WeakLibReader
