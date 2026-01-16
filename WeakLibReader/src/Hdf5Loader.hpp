#pragma once

// Hdf5Loader.hpp - Umbrella header for HDF5 table loading
//
// This header includes all HDF5 loading functionality. Users should
// continue to #include "Hdf5Loader.hpp" for the complete API.

#include <AMReX_Arena.H>
#include <AMReX_Array.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_ParallelDescriptor.H>

#include <limits>
#include <string>
#include <utility>

#include "Hdf5Types.hpp"
#include "detail/Hdf5LoaderDetail.hpp"

namespace WeakLibReader {

inline Hdf5LoadStatus LoadHdf5Table(const std::string& filePath,
                                    Hdf5Table& output,
                                    const Hdf5LoadConfig& cfg = Hdf5LoadConfig{})
{
  Hdf5Table result;

  detail::ScopedHandle file(H5Fopen(filePath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
  if (!file.Valid()) {
    return Hdf5LoadStatus::FileOpenFailed;
  }

  detail::ScopedHandle dataset(H5Dopen(file.Get(), cfg.valueDataset.c_str(), H5P_DEFAULT), H5Dclose);
  if (!dataset.Valid()) {
    return Hdf5LoadStatus::DatasetOpenFailed;
  }

  detail::ScopedHandle dataspace(H5Dget_space(dataset.Get()), H5Sclose);
  if (!dataspace.Valid()) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }

  const int rank = H5Sget_simple_extent_ndims(dataspace.Get());
  if (rank <= 0 || rank > 5) {
    return Hdf5LoadStatus::DatasetRankInvalid;
  }

  std::array<hsize_t, 5> nativeDims{{1, 1, 1, 1, 1}};
  if (H5Sget_simple_extent_dims(dataspace.Get(), nativeDims.data(), nullptr) < 0) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }

  result.nd = rank;
  std::array<int, 5> extents{{1, 1, 1, 1, 1}};
  for (int dim = 0; dim < rank; ++dim) {
    const hsize_t source = nativeDims[rank - 1 - dim];
    if (source == 0 || source > static_cast<hsize_t>(std::numeric_limits<int>::max())) {
      return Hdf5LoadStatus::IncompatibleDatasetExtent;
    }
    extents[dim] = static_cast<int>(source);
  }
  result.extents = extents;

  const std::size_t totalSize = detail::ComputeTotalSize(rank, extents);
  if (totalSize == 0) {
    return Hdf5LoadStatus::IncompatibleDatasetExtent;
  }

  const amrex::Array<int, 4> lo{{0, 0, 0, 0}};
  bool extentOverflow = false;
  amrex::Array<int, 4> hi = detail::MakeHiArray(rank, extents, extentOverflow);
  if (extentOverflow) {
    return Hdf5LoadStatus::IncompatibleDatasetExtent;
  }
  result.values.resize(lo, hi, amrex::The_Pinned_Arena());

  if (H5Dread(dataset.Get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
              result.values.table().p) < 0) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }

  const Hdf5LoadStatus axisStatus = detail::LoadAxes(file.Get(), rank, cfg, result);
  if (axisStatus != Hdf5LoadStatus::Success) {
    return axisStatus;
  }

  result.layout = MakeLayout(result.extents.data(), result.nd);
  output = std::move(result);
  return Hdf5LoadStatus::Success;
}

inline TableDevice MakeDeviceCopy(const Hdf5Table& host,
                                  amrex::Arena* arena = amrex::The_Device_Arena())
{
  TableDevice device{};
  device.nd = host.nd;
  device.layout = host.layout;

  const amrex::Array<int, 4> lo{{0, 0, 0, 0}};
  bool overflow = false;
  const amrex::Array<int, 4> hi = detail::MakeHiArray(host.nd, host.extents, overflow);
  if (overflow) {
    return device;
  }

  device.values.resize(lo, hi, arena);
  device.values.copy(host.values);

  for (int dim = 0; dim < host.nd; ++dim) {
    const auto& hostAxis = host.axisStorage[dim];
    amrex::Gpu::DeviceVector<double>& deviceAxis = device.axisStorage[dim];
    deviceAxis.resize(hostAxis.size());
    if (!hostAxis.empty()) {
      amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                       hostAxis.begin(), hostAxis.end(),
                       deviceAxis.begin());
    }
    Axis axis{};
    axis.grid = deviceAxis.data();
    axis.n = static_cast<int>(deviceAxis.size());
    axis.scale = host.axes[dim].scale;
    device.axes[dim] = axis;
  }

  for (int dim = host.nd; dim < 5; ++dim) {
    device.axisStorage[dim].clear();
    device.axes[dim] = Axis{};
  }

  return device;
}

inline Hdf5LoadStatus LoadHdf5TableParallel(const std::string& filePath,
                                            Hdf5Table& output,
                                            const Hdf5LoadConfig& cfg = Hdf5LoadConfig{},
                                            int readerRank = amrex::ParallelDescriptor::IOProcessorNumber())
{
  const int nProcs = amrex::ParallelDescriptor::NProcs();
  if (nProcs <= 1) {
    return LoadHdf5Table(filePath, output, cfg);
  }

  int root = readerRank;
  if (root < 0 || root >= nProcs) {
    root = amrex::ParallelDescriptor::IOProcessorNumber();
  }

  const int myRank = amrex::ParallelDescriptor::MyProc();

  Hdf5Table localTable;
  Hdf5LoadStatus status = Hdf5LoadStatus::Success;
  if (myRank == root) {
    status = LoadHdf5Table(filePath, localTable, cfg);
  }

  int statusInt = static_cast<int>(status);
  amrex::ParallelDescriptor::Bcast(&statusInt, 1, root);
  status = static_cast<Hdf5LoadStatus>(statusInt);
  if (status != Hdf5LoadStatus::Success) {
    return status;
  }

  int header[6] = {0, 1, 1, 1, 1, 1};
  if (myRank == root) {
    header[0] = localTable.nd;
    for (int dim = 0; dim < 5; ++dim) {
      header[1 + dim] = localTable.extents[dim];
    }
  }
  amrex::ParallelDescriptor::Bcast(header, 6, root);

  const int nd = header[0];
  std::array<int, 5> extents{{1, 1, 1, 1, 1}};
  for (int dim = 0; dim < 5; ++dim) {
    extents[dim] = header[1 + dim];
  }

  int axisCounts[5] = {0, 0, 0, 0, 0};
  int axisScales[5] = {static_cast<int>(AxisScale::Linear),
                       static_cast<int>(AxisScale::Linear),
                       static_cast<int>(AxisScale::Linear),
                       static_cast<int>(AxisScale::Linear),
                       static_cast<int>(AxisScale::Linear)};

  if (myRank == root) {
    for (int dim = 0; dim < 5; ++dim) {
      axisCounts[dim] = localTable.axes[dim].n;
      axisScales[dim] = static_cast<int>(localTable.axes[dim].scale);
    }
  }

  amrex::ParallelDescriptor::Bcast(axisCounts, 5, root);
  amrex::ParallelDescriptor::Bcast(axisScales, 5, root);

  const std::size_t totalSize = detail::ComputeTotalSize(nd, extents);
  if (totalSize > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return Hdf5LoadStatus::IncompatibleDatasetExtent;
  }

  if (myRank != root) {
    output = Hdf5Table{};
    output.nd = nd;
    output.extents = extents;
    bool overflow = false;
    const amrex::Array<int, 4> lo{{0, 0, 0, 0}};
    const amrex::Array<int, 4> hi = detail::MakeHiArray(nd, output.extents, overflow);
    if (overflow || totalSize == 0) {
      return Hdf5LoadStatus::IncompatibleDatasetExtent;
    }
    output.values.resize(lo, hi, amrex::The_Pinned_Arena());
    output.layout = MakeLayout(output.extents.data(), output.nd);
    for (int dim = 0; dim < 5; ++dim) {
      amrex::Vector<double>& storage = output.axisStorage[dim];
      storage.resize(static_cast<std::size_t>(axisCounts[dim]));
      if (!storage.empty()) {
        output.axes[dim].grid = storage.data();
      } else {
        output.axes[dim].grid = nullptr;
      }
      output.axes[dim].n = axisCounts[dim];
      output.axes[dim].scale = static_cast<AxisScale>(axisScales[dim]);
    }
  }

  double* dataPtr = (myRank == root)
                        ? localTable.values.table().p
                        : output.values.table().p;
  if (totalSize > 0) {
    amrex::ParallelDescriptor::Bcast(dataPtr,
                                     static_cast<int>(totalSize),
                                     root);
  }

  for (int dim = 0; dim < 5; ++dim) {
    const int count = axisCounts[dim];
    if (count <= 0) {
      continue;
    }
    double* axisPtr = (myRank == root)
                          ? localTable.axisStorage[dim].data()
                          : output.axisStorage[dim].data();
    amrex::ParallelDescriptor::Bcast(axisPtr, count, root);
    if (myRank != root) {
      output.axes[dim].grid = axisPtr;
    }
  }

  if (myRank == root) {
    output = std::move(localTable);
  }

  return status;
}

} // namespace WeakLibReader
