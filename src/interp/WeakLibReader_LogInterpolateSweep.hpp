#pragma once

#include <AMReX_GpuQualifiers.H>
#include <cstddef>

#include "interp/WeakLibReader_LogInterpolateCore.hpp"

namespace WeakLibReader {

/// GPU-optimized 4D log-interpolation with 1D sweep (single point, multiple E values)
inline int LogInterpolateSingleVariable1D3DCustomPoint(
    const double* logE, std::size_t sizeE,
    double logD, double logT, double y,
    const Axis axes[4],
    const double* data,
    double offset,
    double* out) noexcept
{
  if (logE == nullptr || out == nullptr || data == nullptr ||
      axes[0].grid == nullptr || axes[1].grid == nullptr ||
      axes[2].grid == nullptr || axes[3].grid == nullptr) {
    return 1;
  }
  if (sizeE == 0) {
    return 0;
  }

  constexpr int ND = 4;
  int extents[ND] = {axes[0].n, axes[1].n, axes[2].n, axes[3].n};
  const Layout layout = MakeLayout(extents, ND);

  for (std::size_t i = 0; i < sizeE; ++i) {
    double coords[ND] = {logE[i], logD, logT, y};
    out[i] = detail::LogInterpolatedValueDirect<ND>(data, layout, axes, coords, offset);
  }

  return 0;
}

/// GPU-optimized 4D log-interpolation with 1D sweep (batch)
inline int LogInterpolateSingleVariable1D3DCustom(
    const double* logE, std::size_t sizeE,
    const double* logD, const double* logT, const double* y, std::size_t count,
    const Axis axes[4],
    const double* data,
    double offset,
    double* out) noexcept
{
  if (logE == nullptr || logD == nullptr || logT == nullptr || y == nullptr ||
      out == nullptr || data == nullptr ||
      axes[0].grid == nullptr || axes[1].grid == nullptr ||
      axes[2].grid == nullptr || axes[3].grid == nullptr) {
    return 1;
  }
  if (sizeE == 0 || count == 0) {
    return 0;
  }

  constexpr int ND = 4;
  int extents[ND] = {axes[0].n, axes[1].n, axes[2].n, axes[3].n};
  const Layout layout = MakeLayout(extents, ND);

  for (std::size_t j = 0; j < count; ++j) {
    double* row = out + j * sizeE;
    for (std::size_t i = 0; i < sizeE; ++i) {
      double coords[ND] = {logE[i], logD[j], logT[j], y[j]};
      row[i] = detail::LogInterpolatedValueDirect<ND>(data, layout, axes, coords, offset);
    }
  }

  return 0;
}

/// GPU-optimized 4D log-interpolation with 2D symmetric sweep (single point)
inline int LogInterpolateSingleVariable2D2DCustomPoint(
    const double* logE, std::size_t sizeE,
    double logT, double logX,
    const Axis axes[4],
    const double* data,
    double offset,
    double* out) noexcept
{
  if (logE == nullptr || data == nullptr || out == nullptr ||
      axes[0].grid == nullptr || axes[1].grid == nullptr ||
      axes[2].grid == nullptr || axes[3].grid == nullptr) {
    return 1;
  }
  if (sizeE == 0) {
    return 0;
  }

  constexpr int ND = 4;
  int extents[ND] = {axes[0].n, axes[1].n, axes[2].n, axes[3].n};
  const Layout layout = MakeLayout(extents, ND);

  for (std::size_t j = 0; j < sizeE; ++j) {
    for (std::size_t i = 0; i <= j; ++i) {
      double coords[ND] = {logE[i], logE[j], logT, logX};
      const double value = detail::LogInterpolatedValueDirect<ND>(data, layout, axes, coords, offset);
      detail::StoreSymmetric(out, sizeE, i, j, value);
    }
  }

  return 0;
}

/// GPU-optimized 4D log-interpolation with 2D symmetric sweep (batch)
inline int LogInterpolateSingleVariable2D2DCustom(
    const double* logE, std::size_t sizeE,
    const double* logT, const double* logX, std::size_t count,
    const Axis axes[4],
    const double* data,
    double offset,
    double* out) noexcept
{
  if (logE == nullptr || logT == nullptr || logX == nullptr ||
      out == nullptr || data == nullptr ||
      axes[0].grid == nullptr || axes[1].grid == nullptr ||
      axes[2].grid == nullptr || axes[3].grid == nullptr) {
    return 1;
  }
  if (sizeE == 0 || count == 0) {
    return 0;
  }

  constexpr int ND = 4;
  int extents[ND] = {axes[0].n, axes[1].n, axes[2].n, axes[3].n};
  const Layout layout = MakeLayout(extents, ND);

  const std::size_t planeSize = sizeE * sizeE;
  for (std::size_t l = 0; l < count; ++l) {
    double* plane = out + l * planeSize;
    for (std::size_t j = 0; j < sizeE; ++j) {
      for (std::size_t i = 0; i <= j; ++i) {
        double coords[ND] = {logE[i], logE[j], logT[l], logX[l]};
        const double value = detail::LogInterpolatedValueDirect<ND>(data, layout, axes, coords, offset);
        detail::StoreSymmetric(plane, sizeE, i, j, value);
      }
    }
  }

  return 0;
}

inline int LogInterpolateSingleVariable2D2DCustomAlignedPoint(
    std::size_t sizeE,
    double logT, double logX,
    const Axis axes[2],
    const double* data,
    double offset,
    double* out) noexcept
{
  if (data == nullptr || out == nullptr ||
      axes[0].grid == nullptr || axes[1].grid == nullptr) {
    return 1;
  }
  if (sizeE == 0) {
    return 0;
  }

  int extents[4] = {
      static_cast<int>(sizeE),
      static_cast<int>(sizeE),
      axes[0].n,
      axes[1].n};
  const Layout layout = MakeLayout(extents, 4);

  int idxT = 0;
  int idxX = 0;
  double fracT = 0.0;
  double fracX = 0.0;
  detail::IndexAndDelta(axes[0], logT, idxT, fracT);
  detail::IndexAndDelta(axes[1], logX, idxX, fracX);

  for (std::size_t j = 0; j < sizeE; ++j) {
    for (std::size_t i = 0; i <= j; ++i) {
      const double value = LinearInterp2D4DArray2DAlignedPoint(
          static_cast<int>(i), static_cast<int>(j),
          idxT, idxX, fracT, fracX, offset,
          data, layout);
      detail::StoreSymmetric(out, sizeE, i, j, value);
    }
  }

  return 0;
}

inline int LogInterpolateSingleVariable2D2DCustomAligned(
    std::size_t sizeE,
    const double* logT, const double* logX, std::size_t count,
    const Axis axes[2],
    const double* data,
    double offset,
    double* out) noexcept
{
  if (logT == nullptr || logX == nullptr || data == nullptr || out == nullptr ||
      axes[0].grid == nullptr || axes[1].grid == nullptr) {
    return 1;
  }
  if (sizeE == 0 || count == 0) {
    return 0;
  }

  const std::size_t planeSize = sizeE * sizeE;
  for (std::size_t k = 0; k < count; ++k) {
    double* plane = out + k * planeSize;
    const int rc = LogInterpolateSingleVariable2D2DCustomAlignedPoint(
        sizeE, logT[k], logX[k],
        axes,
        data, offset, plane);
    if (rc != 0) {
      return rc;
    }
  }

  return 0;
}

/// Pre-align one moment of a scattering kernel onto aligned energy nodes.
///
/// Transforms a 5D raw kernel [nE, nE, nMom, nDim3, nDim4] into a 4D aligned
/// table [nAlignedE, nAlignedE, nDim3, nDim4] stored as log10(value + offset).
///
/// Relies on column-major layout: the 2D energy slice [nE_in, nE_out] at any
/// fixed (iMom, iD3, iD4) is contiguous in memory, so we can pass a pointer
/// directly to LogInterpolateSingleVariable2DCustomPoint.
///
/// @param rawKernel    Flat 5D kernel data
/// @param rawLayout    Layout of the 5D kernel [nE, nE, nMom, nDim3, nDim4]
/// @param energyAxis   Energy axis (raw values, same for both E dims)
/// @param iMom         Which moment to extract (0-based index into dim 2)
/// @param nDim3        Size of dimension 3 (nT for NES/Pair, nRho for Brem)
/// @param nDim4        Size of dimension 4 (nEta for NES/Pair, nT for Brem)
/// @param alignedE     Raw energy values for aligned grid
/// @param nAlignedE    Number of aligned energy points
/// @param offset       Opacity offset
/// @param output       Output: [nAlignedE, nAlignedE, nDim3, nDim4] log10-stored
inline void PreAlignScatteringKernelMoment(
    const double* rawKernel,
    const Layout& rawLayout,
    const Axis& energyAxis,
    int iMom, int nDim3, int nDim4,
    const double* alignedE, int nAlignedE,
    double offset,
    double* output) noexcept
{
  // Output layout: [nAlignedE, nAlignedE, nDim3, nDim4]
  const int outExtents[4] = {nAlignedE, nAlignedE, nDim3, nDim4};
  const Layout outLayout = MakeLayout(outExtents, 4);

  // 2D energy axes for interpolation
  const Axis energyAxes[2] = {energyAxis, energyAxis};

  for (int iD4 = 0; iD4 < nDim4; ++iD4) {
    for (int iD3 = 0; iD3 < nDim3; ++iD3) {
      // Pointer to contiguous 2D energy slice [nE, nE] at (iMom, iD3, iD4)
      // Column-major: first two dims (E_in, E_out) are fastest-varying
      const int sliceIdx[5] = {0, 0, iMom, iD3, iD4};
      const double* slice2d = rawKernel + rawLayout.Offset(sliceIdx);

      for (int iA2 = 0; iA2 < nAlignedE; ++iA2) {
        for (int iA1 = 0; iA1 < nAlignedE; ++iA1) {
          const double value = LogInterpolateSingleVariable2DCustomPoint(
              alignedE[iA1], alignedE[iA2],
              energyAxes, slice2d, offset);

          const int outIdx[4] = {iA1, iA2, iD3, iD4};
          output[outLayout.Offset(outIdx)] = math::Log10(value + offset);
        }
      }
    }
  }
}

} // namespace WeakLibReader
