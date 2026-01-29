#pragma once

#include <AMReX_Vector.H>

#include <array>
#include <cctype>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "../WeakLibReader_Hdf5Types.hpp"

#include <hdf5.h>

namespace WeakLibReader {
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
  if (H5Dread(dataset.Get(), H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, out) < 0) {
    return false;
  }
  return true;
}

inline bool ValidateAxis(const amrex::Vector<double>& values, AxisScale scale)
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

// Read a single integer from a 1-element dataset
// Matches pattern used in Fortran ReadDependentVariablesHDF for index datasets
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

  int buffer[1] = {0};
  if (H5Dread(dataset.Get(), H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer) < 0) {
    return false;
  }
  out = buffer[0];
  return true;
}

// Read a 1D string array dataset
// Matches Fortran Read1dHDF_string (wlIOModuleHDF.F90:601-618)
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

// Read a 3D integer array dataset
// Matches Fortran Read3dHDF_integer (wlIOModuleHDF.F90:361-375)
inline bool ReadIntArray3d(hid_t parent, const char* name,
                           amrex::TableData<int, 3>& out,
                           const std::array<int, 3>& expectedDims)
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
  if (rank != 3) {
    return false;
  }

  hsize_t dims[3] = {0, 0, 0};
  if (H5Sget_simple_extent_dims(space.Get(), dims, nullptr) < 0) {
    return false;
  }

  // Verify dimensions match (in reversed order due to Fortran->C conversion)
  // File order: [Ye, T, rho], C order: [rho, T, Ye]
  if (static_cast<int>(dims[2]) != expectedDims[0] ||
      static_cast<int>(dims[1]) != expectedDims[1] ||
      static_cast<int>(dims[0]) != expectedDims[2]) {
    return false;
  }

  // Allocate output array
  const amrex::Array<int, 3> lo{{0, 0, 0}};
  const amrex::Array<int, 3> hi{{expectedDims[0] - 1, expectedDims[1] - 1, expectedDims[2] - 1}};
  out.resize(lo, hi, amrex::The_Pinned_Arena());

  if (H5Dread(dataset.Get(), H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.table().p) < 0) {
    return false;
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
      if (!ParseAxisScale(scaleAttr, scale)) {
        return Hdf5LoadStatus::AxisInvalidScale;
      }
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

inline Hdf5LoadStatus LoadWeakLibAxis(hid_t thermoGroup,
                                      const char* datasetName,
                                      int expectedExtent,
                                      AxisScale scale,
                                      int axisIndex,
                                      Hdf5Table& table)
{
  ScopedHandle dataset(H5Dopen(thermoGroup, datasetName, H5P_DEFAULT), H5Dclose);
  if (!dataset.Valid()) {
    return Hdf5LoadStatus::AxisDatasetOpenFailed;
  }

  ScopedHandle space(H5Dget_space(dataset.Get()), H5Sclose);
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

  if (static_cast<int>(length) != expectedExtent) {
    return Hdf5LoadStatus::AxisExtentMismatch;
  }

  amrex::Vector<double>& storage = table.axisStorage[axisIndex];
  storage.resize(static_cast<std::size_t>(length));
  if (H5Dread(dataset.Get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
              storage.data()) < 0) {
    return Hdf5LoadStatus::AxisReadFailed;
  }

  if (!ValidateAxis(storage, scale)) {
    return Hdf5LoadStatus::AxisNotMonotone;
  }

  Axis axis{};
  axis.grid = storage.data();
  axis.n = static_cast<int>(storage.size());
  axis.scale = scale;
  table.axes[axisIndex] = axis;

  return Hdf5LoadStatus::Success;
}

} // namespace detail
} // namespace WeakLibReader
