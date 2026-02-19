#pragma once

#include "hdf5/WeakLibReader_Hdf5Types.hpp"
#include "hdf5/WeakLibReader_Hdf5LoaderDetail.hpp"

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
  result.layout = MakeLayout(extents.data(), result.nd);

  const std::size_t totalSize = detail::ComputeTotalSize(rank, extents);
  if (totalSize == 0) {
    return Hdf5LoadStatus::IncompatibleDatasetExtent;
  }

  result.values.resize(totalSize);

  if (H5Dread(dataset.Get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
              result.values.data()) < 0) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }

  const Hdf5LoadStatus axisStatus = detail::LoadAxes(file.Get(), rank, cfg, result);
  if (axisStatus != Hdf5LoadStatus::Success) {
    return axisStatus;
  }
  output = std::move(result);
  return Hdf5LoadStatus::Success;
}

inline TableDevice MakeDeviceCopy(const Hdf5Table& host)
{
  TableDevice device{};
  device.nd = host.nd;
  device.layout = host.layout;

  detail::CopyVectorToDevice<double>(host.values, device.values);

  for (int dim = 0; dim < host.nd; ++dim) {
    detail::CopyVectorToDevice<double>(host.axisStorage[dim], device.axisStorage[dim]);
    device.axes[dim] = Axis{device.axisStorage[dim].data(),
                            static_cast<int>(device.axisStorage[dim].size()),
                            host.axes[dim].scale};
  }

  for (int dim = host.nd; dim < 5; ++dim) {
    device.axisStorage[dim].clear();
    device.axes[dim] = Axis{};
  }

  return device;
}

} // namespace WeakLibReader
