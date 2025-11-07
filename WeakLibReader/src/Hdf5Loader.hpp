#pragma once

#include <AMReX_Array.H>
#include <AMReX_Arena.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_TableData.H>
#include <AMReX_Vector.H>
#include <AMReX_ParallelDescriptor.H>

#include <array>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "Layout.hpp"
#include "WeakLibReader.hpp"

#include <hdf5.h>

namespace WeakLibReader {

enum class Hdf5LoadStatus : std::uint8_t {
  Success = 0,
  FileOpenFailed,
  DatasetOpenFailed,
  DatasetRankInvalid,
  DatasetReadFailed,
  AxisDatasetOpenFailed,
  AxisExtentMismatch,
  AxisNotMonotone,
  AxisInvalidScale,
  AxisReadFailed,
  IncompatibleDatasetExtent
};

struct Hdf5LoadConfig {
  std::string valueDataset = "values";
  std::string axisPrefix = "axis";
  std::string axisScaleAttribute = "scale";
};

struct TableView {
  int nd = 0;
  Layout layout{};
  Axis axes[5]{};
  const double* data = nullptr;
};

struct TableDevice {
  int nd = 0;
  Layout layout{};
  Axis axes[5]{};
  amrex::TableData<double, 4> values{};
  std::array<amrex::Gpu::DeviceVector<double>, 5> axisStorage{};

  [[nodiscard]] TableView View() const noexcept
  {
    TableView view{};
    view.nd = nd;
    view.layout = layout;
    view.data = values.const_table().p;
    for (int dim = 0; dim < 5; ++dim) {
      view.axes[dim] = axes[dim];
    }
    return view;
  }
};

struct Hdf5Table {
  int nd = 0;
  std::array<int, 5> extents{{1, 1, 1, 1, 1}};
  Layout layout{};
  Axis axes[5]{};
  amrex::TableData<double, 4> values{};
  std::array<amrex::Vector<double>, 5> axisStorage{};

  Hdf5Table() = default;
  Hdf5Table(Hdf5Table&&) = default;
  Hdf5Table& operator=(Hdf5Table&&) = default;
  Hdf5Table(const Hdf5Table&) = delete;
  Hdf5Table& operator=(const Hdf5Table&) = delete;

  [[nodiscard]] double* DataPtr() noexcept { return values.table().p; }
  [[nodiscard]] const double* DataPtr() const noexcept { return values.const_table().p; }

  [[nodiscard]] TableView View() const noexcept
  {
    TableView view{};
    view.nd = nd;
    view.layout = layout;
    view.data = values.const_table().p;
    for (int dim = 0; dim < 5; ++dim) {
      view.axes[dim] = axes[dim];
    }
    return view;
  }
};

namespace detail {

inline void CloseHandle(hid_t handle, herr_t (*closer)(hid_t)) noexcept
{
  if (handle >= 0 && closer != nullptr) {
    closer(handle);
  }
}

struct ScopedHandle {
  hid_t id = -1;
  herr_t (*closer)(hid_t) = nullptr;

  ScopedHandle() = default;
  ScopedHandle(hid_t handle, herr_t (*fn)(hid_t)) noexcept : id(handle), closer(fn) {}
  ScopedHandle(const ScopedHandle&) = delete;
  ScopedHandle& operator=(const ScopedHandle&) = delete;
  ScopedHandle(ScopedHandle&& other) noexcept : id(other.id), closer(other.closer)
  {
    other.id = -1;
    other.closer = nullptr;
  }
  ScopedHandle& operator=(ScopedHandle&& other) noexcept
  {
    if (this != &other) {
      Reset();
      id = other.id;
      closer = other.closer;
      other.id = -1;
      other.closer = nullptr;
    }
    return *this;
  }
  ~ScopedHandle() { Reset(); }

  void Reset() noexcept { CloseHandle(id, closer); id = -1; closer = nullptr; }
  [[nodiscard]] hid_t Get() const noexcept { return id; }
  [[nodiscard]] bool Valid() const noexcept { return id >= 0; }
};

inline AxisScale ParseAxisScale(const std::string& raw)
{
  std::string lower;
  lower.reserve(raw.size());
  for (char c : raw) {
    lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  }
  if (lower == "linear") {
    return AxisScale::Linear;
  }
  if (lower == "log10" || lower == "log") {
    return AxisScale::Log10;
  }
  return AxisScale::Linear;
}

inline bool ReadStringAttribute(hid_t parent, const std::string& name, std::string& out)
{
  if (parent < 0) {
    return false;
  }
  ScopedHandle attr(H5Aopen(parent, name.c_str(), H5P_DEFAULT), H5Aclose);
  if (!attr.Valid()) {
    return false;
  }

  ScopedHandle type(H5Aget_type(attr.Get()), H5Tclose);
  if (!type.Valid()) {
    return false;
  }

  const bool isVariable = H5Tis_variable_str(type.Get()) > 0;
  ScopedHandle native(H5Tget_native_type(type.Get(), H5T_DIR_ASCEND), H5Tclose);
  if (!native.Valid()) {
    return false;
  }

  if (isVariable) {
    char* buffer = nullptr;
    if (H5Aread(attr.Get(), native.Get(), &buffer) < 0 || buffer == nullptr) {
      return false;
    }
    out.assign(buffer);
    H5free_memory(buffer);
    return true;
  }

  const std::size_t size = static_cast<std::size_t>(H5Tget_size(native.Get()));
  std::vector<char> storage(size + 1, '\0');
  if (H5Aread(attr.Get(), native.Get(), storage.data()) < 0) {
    return false;
  }
  out.assign(storage.data());
  return true;
}

inline bool ValidateAxis(const amrex::Vector<double>& values, AxisScale scale)
{
  if (values.empty()) {
    return false;
  }
  if (values.size() >= 2) {
    for (std::size_t i = 1; i < values.size(); ++i) {
      if (!(values[i] > values[i - 1])) {
        return false;
      }
    }
  }
  if (scale == AxisScale::Log10) {
    for (double v : values) {
      if (!(v > 0.0)) {
        return false;
      }
    }
  }
  return true;
}

inline amrex::Array<int, 4> MakeHiArray(int nd, const std::array<int, 5>& extents,
                                        bool& overflow) noexcept
{
  amrex::Array<int, 4> hi{{0, 0, 0, 0}};
  if (nd >= 1) {
    hi[0] = extents[0] - 1;
  }
  if (nd >= 2) {
    hi[1] = extents[1] - 1;
  }
  if (nd >= 3) {
    hi[2] = extents[2] - 1;
  }
  if (nd == 4) {
    hi[3] = extents[3] - 1;
  } else if (nd == 5) {
    const long long product =
        static_cast<long long>(extents[3]) * static_cast<long long>(extents[4]);
    if (product > static_cast<long long>(std::numeric_limits<int>::max())) {
      overflow = true;
      return hi;
    }
    hi[3] = static_cast<int>(product) - 1;
  }
  return hi;
}

inline std::size_t ComputeTotalSize(int nd, const std::array<int, 5>& extents)
{
  std::size_t size = 1;
  const std::size_t maxSize = std::numeric_limits<std::size_t>::max();
  for (int dim = 0; dim < nd; ++dim) {
    const int extent = extents[dim];
    if (extent <= 0) {
      return 0;
    }
    if (size > maxSize / static_cast<std::size_t>(extent)) {
      return 0;
    }
    size *= static_cast<std::size_t>(extent);
  }
  return size;
}

inline Hdf5LoadStatus LoadAxes(hid_t file,
                               int nd,
                               const Hdf5LoadConfig& cfg,
                               Hdf5Table& table)
{
  for (int dim = 0; dim < nd; ++dim) {
    const std::string datasetName = cfg.axisPrefix + std::to_string(dim);
    ScopedHandle axisDataset(H5Dopen(file, datasetName.c_str(), H5P_DEFAULT), H5Dclose);
    if (!axisDataset.Valid()) {
      return Hdf5LoadStatus::AxisDatasetOpenFailed;
    }

    ScopedHandle axisSpace(H5Dget_space(axisDataset.Get()), H5Sclose);
    if (!axisSpace.Valid()) {
      return Hdf5LoadStatus::AxisReadFailed;
    }

    const int rank = H5Sget_simple_extent_ndims(axisSpace.Get());
    if (rank != 1) {
      return Hdf5LoadStatus::AxisReadFailed;
    }

    hsize_t length = 0;
    if (H5Sget_simple_extent_dims(axisSpace.Get(), &length, nullptr) < 0) {
      return Hdf5LoadStatus::AxisReadFailed;
    }

    if (length != static_cast<hsize_t>(table.extents[dim])) {
      return Hdf5LoadStatus::AxisExtentMismatch;
    }

    amrex::Vector<double>& storage = table.axisStorage[dim];
    storage.resize(static_cast<std::size_t>(length));
    if (H5Dread(axisDataset.Get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                storage.data()) < 0) {
      return Hdf5LoadStatus::AxisReadFailed;
    }

    std::string scaleAttr;
    AxisScale scale = AxisScale::Linear;
    if (ReadStringAttribute(axisDataset.Get(), cfg.axisScaleAttribute, scaleAttr)) {
      scale = ParseAxisScale(scaleAttr);
    }

    if (!ValidateAxis(storage, scale)) {
      return Hdf5LoadStatus::AxisNotMonotone;
    }

    Axis axis{};
    axis.grid = storage.data();
    axis.n = static_cast<int>(storage.size());
    axis.scale = scale;
    table.axes[dim] = axis;
  }

  for (int dim = nd; dim < 5; ++dim) {
    table.axisStorage[dim].clear();
    table.axes[dim] = Axis{};
  }

  return Hdf5LoadStatus::Success;
}

} // namespace detail

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
    return Hdf5LoadStatus::DatasetOpenFailed;
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

inline Hdf5LoadStatus LoadHdf5TableDistributed(const std::string& filePath,
                                               Hdf5Table& output,
                                               const Hdf5LoadConfig& cfg = Hdf5LoadConfig{},
                                               int readerRank = 0)
{
  int myRank = 0;
  int nProcs = 1;
#ifdef AMREX_PARALLEL
  myRank = amrex::ParallelDescriptor::MyProc();
  nProcs = amrex::ParallelDescriptor::NProcs();
#endif

  if (nProcs == 1) {
    return LoadHdf5Table(filePath, output, cfg);
  }

  Hdf5LoadStatus status = Hdf5LoadStatus::Success;

  if (myRank == readerRank) {
    status = LoadHdf5Table(filePath, output, cfg);
  }

#ifdef AMREX_PARALLEL
  int statusInt = static_cast<int>(status);
  amrex::ParallelDescriptor::Bcast(&statusInt, 1, readerRank);
  status = static_cast<Hdf5LoadStatus>(statusInt);
  if (status != Hdf5LoadStatus::Success) {
    return status;
  }

  if (myRank == readerRank) {
    const std::size_t metaSize = sizeof(int) * (1 + 5) + sizeof(Layout);
    const std::size_t axisCounts = static_cast<std::size_t>(output.nd);
    std::size_t axisTotal = 0;
    for (int dim = 0; dim < output.nd; ++dim) {
      axisTotal += static_cast<std::size_t>(output.axisStorage[dim].size());
    }
    const std::size_t payloadSize =
        (axisTotal + output.values.size()) * sizeof(double) + metaSize;

    amrex::Vector<char> buffer(payloadSize);
    char* cursor = buffer.data();

    std::memcpy(cursor, &output.nd, sizeof(int));
    cursor += sizeof(int);
    std::memcpy(cursor, output.extents.data(), sizeof(int) * 5);
    cursor += sizeof(int) * 5;
    std::memcpy(cursor, &output.layout, sizeof(Layout));
    cursor += sizeof(Layout);

    for (int dim = 0; dim < output.nd; ++dim) {
      const int count = output.axisStorage[dim].empty()
                            ? output.axes[dim].n
                            : static_cast<int>(output.axisStorage[dim].size());
      std::memcpy(cursor, &count, sizeof(int));
      cursor += sizeof(int);
      const double* dataPtr = output.axisStorage[dim].empty()
                                  ? output.axes[dim].grid
                                  : output.axisStorage[dim].data();
      std::memcpy(cursor, dataPtr, sizeof(double) * count);
      cursor += sizeof(double) * count;
      const auto scale = output.axes[dim].scale;
      std::memcpy(cursor, &scale, sizeof(scale));
      cursor += sizeof(scale);
    }

    const double* valuesPtr = output.DataPtr();
    std::memcpy(cursor, valuesPtr, sizeof(double) * output.values.size());
    cursor += sizeof(double) * output.values.size();

    const int byteCount = static_cast<int>(payloadSize);
    amrex::ParallelDescriptor::Bcast(&byteCount, 1, readerRank);
    amrex::ParallelDescriptor::Bcast(buffer.data(), byteCount, readerRank);
  } else {
    int byteCount = 0;
    amrex::ParallelDescriptor::Bcast(&byteCount, 1, readerRank);
    amrex::Vector<char> buffer(byteCount);
    amrex::ParallelDescriptor::Bcast(buffer.data(), byteCount, readerRank);

    const char* cursor = buffer.data();
    Hdf5Table incoming;
    std::memcpy(&incoming.nd, cursor, sizeof(int));
    cursor += sizeof(int);
    std::memcpy(incoming.extents.data(), cursor, sizeof(int) * 5);
    cursor += sizeof(int) * 5;
    std::memcpy(&incoming.layout, cursor, sizeof(Layout));
    cursor += sizeof(Layout);

    for (int dim = 0; dim < incoming.nd; ++dim) {
      int count = 0;
      std::memcpy(&count, cursor, sizeof(int));
      cursor += sizeof(int);
      incoming.axisStorage[dim].resize(count);
      std::memcpy(incoming.axisStorage[dim].data(), cursor, sizeof(double) * count);
      cursor += sizeof(double) * count;
      Axis axis{};
      axis.grid = incoming.axisStorage[dim].data();
      axis.n = count;
      std::memcpy(&axis.scale, cursor, sizeof(axis.scale));
      cursor += sizeof(axis.scale);
      incoming.axes[dim] = axis;
    }

    const amrex::Array<int, 4> lo{{0, 0, 0, 0}};
    bool overflow = false;
    const amrex::Array<int, 4> hi = detail::MakeHiArray(incoming.nd, incoming.extents, overflow);
    if (overflow) {
      return Hdf5LoadStatus::IncompatibleDatasetExtent;
    }
    incoming.values.resize(lo, hi, amrex::The_Pinned_Arena());
    std::memcpy(incoming.values.table().p, cursor,
                sizeof(double) * incoming.values.size());

    output = std::move(incoming);
  }
#endif

  return status;
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

} // namespace WeakLibReader
