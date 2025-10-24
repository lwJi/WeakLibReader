#pragma once

#include <AMReX_GpuQualifiers.H>
#include <cstddef>

namespace WeakLibReader {

struct Layout {
  int nd = 0;                  // number of dimensions, 1..5
  int n[5] = {1, 1, 1, 1, 1};   // extent along each axis (row-major ordering)
  std::size_t stride[5] = {0, 0, 0, 0, 0}; // row-major stride for each axis

  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
  std::size_t Offset(const int idx[5]) const noexcept
  {
    std::size_t k = 0;
    for (int dim = 0; dim < nd; ++dim) {
      k += static_cast<std::size_t>(idx[dim]) * stride[dim];
    }
    return k;
  }

  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
  std::size_t Offset(int i0) const noexcept
  {
    const int idx[5] = {i0, 0, 0, 0, 0};
    return Offset(idx);
  }

  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
  std::size_t Offset(int i0, int i1) const noexcept
  {
    const int idx[5] = {i0, i1, 0, 0, 0};
    return Offset(idx);
  }

  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
  std::size_t Offset(int i0, int i1, int i2) const noexcept
  {
    const int idx[5] = {i0, i1, i2, 0, 0};
    return Offset(idx);
  }

  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
  std::size_t Offset(int i0, int i1, int i2, int i3) const noexcept
  {
    const int idx[5] = {i0, i1, i2, i3, 0};
    return Offset(idx);
  }

  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
  std::size_t Offset(int i0, int i1, int i2, int i3, int i4) const noexcept
  {
    const int idx[5] = {i0, i1, i2, i3, i4};
    return Offset(idx);
  }
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Layout MakeLayout(const int* extents, int nd) noexcept
{
  Layout layout{};
  layout.nd = nd;
  std::size_t stride = 1;
  for (int dim = 0; dim < nd; ++dim) {
    layout.n[dim] = extents[dim];
    layout.stride[dim] = stride;
    stride *= static_cast<std::size_t>(extents[dim]);
  }
  return layout;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Layout SliceLeading(const Layout& layout, int drop) noexcept
{
  Layout result{};
  result.nd = layout.nd - drop;
  for (int dim = drop; dim < layout.nd; ++dim) {
    const int out = dim - drop;
    result.n[out] = layout.n[dim];
  }
  std::size_t stride = 1;
  for (int dim = 0; dim < result.nd; ++dim) {
    result.stride[dim] = stride;
    stride *= static_cast<std::size_t>(result.n[dim]);
  }
  return result;
}

} // namespace WeakLibReader
