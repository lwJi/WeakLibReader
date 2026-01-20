#pragma once

#include <AMReX_GpuQualifiers.H>
#include <cstddef>
#include <vector>

#include "LogInterpolateCore.hpp"

namespace WeakLibReader {

inline int SumLogInterpolateSingleVariable2D2DCustomAligned(
    std::size_t sizeE,
    const double* logD, std::size_t nAlpha,
    const double* logT, std::size_t count,
    const double* gridD, int nD,
    const double* gridT, int nT,
    const double* alpha,
    const double* data,
    double offset,
    double* out) noexcept
{
  if (logD == nullptr || logT == nullptr || data == nullptr ||
      alpha == nullptr || out == nullptr ||
      gridD == nullptr || gridT == nullptr) {
    return 1;
  }
  if (sizeE == 0 || nAlpha == 0 || count == 0) {
    return 0;
  }

  Axis axes[2] = {
      MakeAxis(gridD, nD, AxisScale::Linear),
      MakeAxis(gridT, nT, AxisScale::Linear)};
  int extents[4] = {
      static_cast<int>(sizeE),
      static_cast<int>(sizeE),
      nD,
      nT};
  const Layout layout = MakeLayout(extents, 4);

  const std::size_t planeSize = sizeE * sizeE;
  std::vector<int> idxD(nAlpha);
  std::vector<double> fracD(nAlpha);

  for (std::size_t k = 0; k < count; ++k) {
    double* plane = out + k * planeSize;

    int idxT = 0;
    double fracT = 0.0;
    detail::IndexAndDelta(axes[1], logT[k], idxT, fracT);

    for (std::size_t l = 0; l < nAlpha; ++l) {
      int idx = 0;
      double frac = 0.0;
      const double value = logD[k * nAlpha + l];
      detail::IndexAndDelta(axes[0], value, idx, frac);
      idxD[l] = idx;
      fracD[l] = frac;
    }

    for (std::size_t j = 0; j < sizeE; ++j) {
      for (std::size_t i = 0; i <= j; ++i) {
        double sum = 0.0;
        for (std::size_t l = 0; l < nAlpha; ++l) {
          const double interp = LinearInterp2D4DArray2DAlignedPoint(
              static_cast<int>(i), static_cast<int>(j),
              idxD[l], idxT,
              fracD[l], fracT,
              offset,
              data, layout);
          sum += alpha[l] * interp;
        }
        detail::StoreSymmetric(plane, sizeE, i, j, sum);
      }
    }
  }

  return 0;
}

} // namespace WeakLibReader
