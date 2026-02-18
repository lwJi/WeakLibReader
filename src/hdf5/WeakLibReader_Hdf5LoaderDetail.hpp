#pragma once

#include <AMReX_ParallelDescriptor.H>

#include <array>
#include <cctype>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "hdf5/WeakLibReader_Hdf5Types.hpp"

#include <hdf5.h>

namespace WeakLibReader {
namespace detail {

inline void CloseHandle(hid_t handle, herr_t (*closer)(hid_t)) noexcept
{
  if (handle >= 0 && closer != nullptr) {
    closer(handle);
  }
}

struct ScopedH5ErrorSuppressor {
  H5E_auto2_t oldFunc;
  void* oldClientData;
  ScopedH5ErrorSuppressor() {
    H5Eget_auto2(H5E_DEFAULT, &oldFunc, &oldClientData);
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
  }
  ~ScopedH5ErrorSuppressor() {
    H5Eset_auto2(H5E_DEFAULT, oldFunc, oldClientData);
  }
  ScopedH5ErrorSuppressor(const ScopedH5ErrorSuppressor&) = delete;
  ScopedH5ErrorSuppressor& operator=(const ScopedH5ErrorSuppressor&) = delete;
};

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

inline bool ParseAxisScale(const std::string& raw, AxisScale& scale)
{
  std::string lower;
  lower.reserve(raw.size());
  for (char c : raw) {
    lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  }
  if (lower == "linear") {
    scale = AxisScale::Linear;
    return true;
  }
  if (lower == "log10" || lower == "log") {
    scale = AxisScale::Log10;
    return true;
  }
  return false;
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

inline bool ReadIntArray(hid_t parent, const char* name, int* out, std::size_t count)
{
  if (parent < 0) {
    return false;
  }
  ScopedHandle dataset(H5Dopen(parent, name, H5P_DEFAULT), H5Dclose);
  if (!dataset.Valid()) {
    return false;
  }
  ScopedHandle space(H5Dget_space(dataset.Get()), H5Sclose);
  if (!space.Valid()) {
    return false;
  }
  hsize_t dims = 0;
  if (H5Sget_simple_extent_dims(space.Get(), &dims, nullptr) < 0) {
    return false;
  }
  if (dims != static_cast<hsize_t>(count)) {
    return false;
  }
  return H5Dread(dataset.Get(), H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, out) >= 0;
}

inline bool ValidateAxis(const std::vector<double>& values, AxisScale scale)
{
  if (values.size() < 2) {
    return false;
  }
  // Verify strict monotonicity
  for (std::size_t i = 1; i < values.size(); ++i) {
    if (!(values[i] > values[i - 1])) {
      return false;
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

// Read a single integer from a 1-element dataset.
// Matches pattern used in Fortran ReadDependentVariablesHDF for index datasets.
inline bool ReadScalarInt(hid_t parent, const char* name, int& out)
{
  if (parent < 0) {
    return false;
  }
  ScopedHandle dataset(H5Dopen(parent, name, H5P_DEFAULT), H5Dclose);
  if (!dataset.Valid()) {
    return false;
  }
  ScopedHandle space(H5Dget_space(dataset.Get()), H5Sclose);
  if (!space.Valid()) {
    return false;
  }

  const int rank = H5Sget_simple_extent_ndims(space.Get());
  if (rank != 1) {
    return false;
  }

  hsize_t dims = 0;
  if (H5Sget_simple_extent_dims(space.Get(), &dims, nullptr) < 0) {
    return false;
  }
  if (dims != 1) {
    return false;
  }

  return H5Dread(dataset.Get(), H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &out) >= 0;
}

// Read a 1D string array dataset.
// Matches Fortran Read1dHDF_string (wlIOModuleHDF.F90:601-618).
inline bool ReadStringArray(hid_t parent, const char* name, std::vector<std::string>& out)
{
  if (parent < 0) {
    return false;
  }
  ScopedHandle dataset(H5Dopen(parent, name, H5P_DEFAULT), H5Dclose);
  if (!dataset.Valid()) {
    return false;
  }
  ScopedHandle space(H5Dget_space(dataset.Get()), H5Sclose);
  if (!space.Valid()) {
    return false;
  }

  const int rank = H5Sget_simple_extent_ndims(space.Get());
  if (rank != 1) {
    return false;
  }

  hsize_t count = 0;
  if (H5Sget_simple_extent_dims(space.Get(), &count, nullptr) < 0) {
    return false;
  }

  ScopedHandle dtype(H5Dget_type(dataset.Get()), H5Tclose);
  if (!dtype.Valid()) {
    return false;
  }

  const bool isVariable = H5Tis_variable_str(dtype.Get()) > 0;

  if (isVariable) {
    // Variable-length strings
    std::vector<char*> buffer(count, nullptr);
    ScopedHandle memtype(H5Tcopy(H5T_C_S1), H5Tclose);
    H5Tset_size(memtype.Get(), H5T_VARIABLE);

    if (H5Dread(dataset.Get(), memtype.Get(), H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer.data()) < 0) {
      return false;
    }

    out.resize(count);
    for (hsize_t i = 0; i < count; ++i) {
      if (buffer[i] != nullptr) {
        out[i] = buffer[i];
      }
    }

    // Free HDF5-allocated memory
    ScopedHandle vlenSpace(H5Dget_space(dataset.Get()), H5Sclose);
    H5Dvlen_reclaim(memtype.Get(), vlenSpace.Get(), H5P_DEFAULT, buffer.data());
  } else {
    // Fixed-length strings
    const std::size_t strLen = H5Tget_size(dtype.Get());
    std::vector<char> buffer(count * strLen);

    if (H5Dread(dataset.Get(), dtype.Get(), H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer.data()) < 0) {
      return false;
    }

    out.resize(count);
    for (hsize_t i = 0; i < count; ++i) {
      // Extract string, trimming trailing spaces/nulls
      const char* start = buffer.data() + i * strLen;
      std::size_t len = strLen;
      while (len > 0 && (start[len - 1] == '\0' || start[len - 1] == ' ')) {
        --len;
      }
      out[i].assign(start, len);
    }
  }

  return true;
}

template <int ND>
inline bool OpenWeakLibDataset(hid_t parent, const char* name,
                               ScopedHandle& dataset,
                               std::array<hsize_t, ND>& fileDims)
{
  if (parent < 0) {
    return false;
  }

  dataset = ScopedHandle(H5Dopen(parent, name, H5P_DEFAULT), H5Dclose);
  if (!dataset.Valid()) {
    return false;
  }

  ScopedHandle dataspace(H5Dget_space(dataset.Get()), H5Sclose);
  if (!dataspace.Valid()) {
    return false;
  }

  const int rank = H5Sget_simple_extent_ndims(dataspace.Get());
  if (rank != ND) {
    return false;
  }

  fileDims.fill(0);
  return H5Sget_simple_extent_dims(dataspace.Get(), fileDims.data(), nullptr) >= 0;
}

template <int ND>
inline bool ValidateFortranDims(const std::array<hsize_t, ND>& fileDims,
                                const std::array<int, ND>& expectedDims)
{
  for (int i = 0; i < ND; ++i) {
    if (expectedDims[i] <= 0) {
      return false;
    }
    if (static_cast<int>(fileDims[i]) != expectedDims[ND - 1 - i]) {
      return false;
    }
  }
  return true;
}

template <typename Container, typename T, int ND>
bool ReadWeakLibArrayNdImpl(hid_t parent, const char* name,
                             Container& output,
                             const std::array<int, ND>& expectedDims)
{
  ScopedHandle dataset;
  std::array<hsize_t, ND> fileDims{};
  if (!OpenWeakLibDataset<ND>(parent, name, dataset, fileDims)) {
    return false;
  }

  if (!ValidateFortranDims<ND>(fileDims, expectedDims)) {
    return false;
  }

  std::size_t totalSize = 1;
  for (int i = 0; i < ND; ++i) {
    totalSize *= static_cast<std::size_t>(expectedDims[i]);
  }
  output.resize(totalSize);

  const hid_t memType = std::is_same_v<T, double> ? H5T_NATIVE_DOUBLE : H5T_NATIVE_INT;
  return H5Dread(dataset.Get(), memType, H5S_ALL, H5S_ALL, H5P_DEFAULT, output.data()) >= 0;
}

template <typename T, int ND>
bool ReadWeakLibArrayNd(hid_t parent, const char* name,
                        amrex::Gpu::PinnedVector<T>& output,
                        const std::array<int, ND>& expectedDims)
{
  return ReadWeakLibArrayNdImpl<amrex::Gpu::PinnedVector<T>, T, ND>(
      parent, name, output, expectedDims);
}

inline bool GroupExists(hid_t loc, const char* name)
{
  ScopedH5ErrorSuppressor suppress;
  htri_t exists = H5Lexists(loc, name, H5P_DEFAULT);
  return exists > 0;
}

// Load an opacity grid group (EnergyGrid or EtaGrid) from an HDF5 file.
inline Hdf5LoadStatus LoadWeakLibOpacityGrid(hid_t file, const char* groupName,
                                              WeakLibOpacityGrid& grid)
{
  ScopedHandle group(H5Gopen(file, groupName, H5P_DEFAULT), H5Gclose);
  if (!group.Valid()) {
    return Hdf5LoadStatus::DatasetOpenFailed;
  }

  std::vector<std::string> nameVec, unitVec;
  if (!ReadStringArray(group.Get(), "Name", nameVec) || nameVec.empty()) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }
  if (!ReadStringArray(group.Get(), "Unit", unitVec) || unitVec.empty()) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }
  grid.name = nameVec[0];
  grid.unit = unitVec[0];

  if (!ReadScalarInt(group.Get(), "nPoints", grid.nPoints)) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }

  int logInterp = 0;
  if (!ReadScalarInt(group.Get(), "LogInterp", logInterp)) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }
  grid.scale = (logInterp == 1) ? AxisScale::Log10 : AxisScale::Linear;

  grid.values.resize(grid.nPoints);
  {
    ScopedHandle dataset(H5Dopen(group.Get(), "Values", H5P_DEFAULT), H5Dclose);
    if (!dataset.Valid()) {
      return Hdf5LoadStatus::DatasetOpenFailed;
    }
    if (H5Dread(dataset.Get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                grid.values.data()) < 0) {
      return Hdf5LoadStatus::DatasetReadFailed;
    }
  }

  if (!ValidateAxis(grid.values, grid.scale)) {
    return Hdf5LoadStatus::AxisNotMonotone;
  }

  grid.minValue = grid.values[0];
  grid.maxValue = grid.values[grid.nPoints - 1];

  // Zoom is optional (geometric grid parameter)
  {
    ScopedH5ErrorSuppressor suppress;
    ScopedHandle zoomDs(H5Dopen(group.Get(), "Zoom", H5P_DEFAULT), H5Dclose);
    if (zoomDs.Valid()) {
      double zoom = 0.0;
      if (H5Dread(zoomDs.Get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                  H5P_DEFAULT, &zoom) >= 0) {
        grid.zoom = zoom;
      }
    }
  }

  return Hdf5LoadStatus::Success;
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

inline Hdf5LoadStatus ReadAxisDataset1D(hid_t datasetId,
                                        int expectedExtent,
                                        AxisScale scale,
                                        std::vector<double>& storage,
                                        Axis& outAxis)
{
  ScopedHandle space(H5Dget_space(datasetId), H5Sclose);
  if (!space.Valid()) {
    return Hdf5LoadStatus::AxisReadFailed;
  }

  const int rank = H5Sget_simple_extent_ndims(space.Get());
  if (rank != 1) {
    return Hdf5LoadStatus::AxisReadFailed;
  }

  hsize_t length = 0;
  if (H5Sget_simple_extent_dims(space.Get(), &length, nullptr) < 0) {
    return Hdf5LoadStatus::AxisReadFailed;
  }
  if (length != static_cast<hsize_t>(expectedExtent)) {
    return Hdf5LoadStatus::AxisExtentMismatch;
  }

  storage.resize(static_cast<std::size_t>(length));
  if (H5Dread(datasetId, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
              storage.data()) < 0) {
    return Hdf5LoadStatus::AxisReadFailed;
  }

  if (!ValidateAxis(storage, scale)) {
    return Hdf5LoadStatus::AxisNotMonotone;
  }

  outAxis = Axis{storage.data(), static_cast<int>(storage.size()), scale};
  return Hdf5LoadStatus::Success;
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

    std::string scaleAttr;
    AxisScale scale = AxisScale::Linear;
    if (ReadStringAttribute(axisDataset.Get(), cfg.axisScaleAttribute, scaleAttr)) {
      if (!ParseAxisScale(scaleAttr, scale)) {
        return Hdf5LoadStatus::AxisInvalidScale;
      }
    }

    const Hdf5LoadStatus axisStatus = ReadAxisDataset1D(
        axisDataset.Get(), table.layout.n[dim], scale,
        table.axisStorage[dim], table.axes[dim]);
    if (axisStatus != Hdf5LoadStatus::Success) {
      return axisStatus;
    }
  }

  for (int dim = nd; dim < 5; ++dim) {
    table.axisStorage[dim].clear();
    table.axes[dim] = Axis{};
  }

  return Hdf5LoadStatus::Success;
}

inline Hdf5LoadStatus LoadWeakLibAxis(hid_t thermoGroup,
                                      const char* datasetName,
                                      int expectedExtent,
                                      AxisScale scale,
                                      std::vector<double>& storage,
                                      Axis& outAxis)
{
  ScopedHandle dataset(H5Dopen(thermoGroup, datasetName, H5P_DEFAULT), H5Dclose);
  if (!dataset.Valid()) {
    return Hdf5LoadStatus::AxisDatasetOpenFailed;
  }

  return ReadAxisDataset1D(dataset.Get(), expectedExtent, scale, storage, outAxis);
}

/// Broadcast a vector of strings from root to all ranks.
/// Format: [count][len0][len1]...[lenN-1][chars...]
inline void BcastStringVector(std::vector<std::string>& strings, int root)
{
  const int myRank = amrex::ParallelDescriptor::MyProc();

  // Broadcast count
  int count = (myRank == root) ? static_cast<int>(strings.size()) : 0;
  amrex::ParallelDescriptor::Bcast(&count, 1, root);

  if (count == 0) {
    if (myRank != root) {
      strings.clear();
    }
    return;
  }

  // Broadcast lengths
  std::vector<int> lengths(count);
  if (myRank == root) {
    for (int i = 0; i < count; ++i) {
      lengths[i] = static_cast<int>(strings[i].size());
    }
  }
  amrex::ParallelDescriptor::Bcast(lengths.data(), count, root);

  // Compute total chars and broadcast concatenated buffer
  int totalChars = 0;
  for (int i = 0; i < count; ++i) {
    totalChars += lengths[i];
  }

  std::vector<char> buffer(totalChars);
  if (myRank == root) {
    int offset = 0;
    for (int i = 0; i < count; ++i) {
      std::memcpy(buffer.data() + offset, strings[i].data(), lengths[i]);
      offset += lengths[i];
    }
  }

  if (totalChars > 0) {
    amrex::ParallelDescriptor::Bcast(buffer.data(), totalChars, root);
  }

  // Non-root: reconstruct strings
  if (myRank != root) {
    strings.resize(count);
    int offset = 0;
    for (int i = 0; i < count; ++i) {
      strings[i].assign(buffer.data() + offset, lengths[i]);
      offset += lengths[i];
    }
  }
}

/// Broadcast a fixed-size array of strings from root to all ranks.
template <std::size_t N>
inline void BcastStringArray(std::array<std::string, N>& strings, int root)
{
  std::vector<std::string> vec(strings.begin(), strings.end());
  BcastStringVector(vec, root);
  for (std::size_t i = 0; i < N && i < vec.size(); ++i) {
    strings[i] = std::move(vec[i]);
  }
}

} // namespace detail
} // namespace WeakLibReader
