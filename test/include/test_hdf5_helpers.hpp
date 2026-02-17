#pragma once

#include <catch2/catch_test_macros.hpp>

#include <hdf5.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <string>
#include <vector>

namespace TestHelpers {

inline void WriteStringAttribute(hid_t parent, const std::string& name, const char* value)
{
  hid_t type = H5Tcopy(H5T_C_S1);
  H5Tset_size(type, std::strlen(value));
  H5Tset_strpad(type, H5T_STR_NULLTERM);

  hid_t space = H5Screate(H5S_SCALAR);
  hid_t attr = H5Acreate(parent, name.c_str(), type, space, H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(attr, type, value);
  H5Aclose(attr);
  H5Sclose(space);
  H5Tclose(type);
}

inline void WriteIntArrayDataset(hid_t parent,
                                  const std::string& name,
                                  const std::vector<int>& values)
{
  const hsize_t dims = static_cast<hsize_t>(values.size());
  hid_t space = H5Screate_simple(1, &dims, nullptr);
  hid_t dataset = H5Dcreate(parent, name.c_str(), H5T_NATIVE_INT, space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(dataset >= 0);
  H5Dwrite(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data());
  H5Dclose(dataset);
  H5Sclose(space);
}

inline void WriteDoubleArrayDataset(hid_t parent,
                                     const std::string& name,
                                     const std::vector<double>& values)
{
  const hsize_t dims = static_cast<hsize_t>(values.size());
  hid_t space = H5Screate_simple(1, &dims, nullptr);
  hid_t dataset = H5Dcreate(parent, name.c_str(), H5T_IEEE_F64LE, space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(dataset >= 0);
  H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data());
  H5Dclose(dataset);
  H5Sclose(space);
}

inline void WriteStringArrayDataset(hid_t parent,
                                     const std::string& name,
                                     const std::vector<std::string>& values)
{
  const hsize_t dims = static_cast<hsize_t>(values.size());
  hid_t space = H5Screate_simple(1, &dims, nullptr);

  std::size_t maxLen = 1;
  for (const auto& value : values) {
    maxLen = std::max(maxLen, value.size());
  }
  const std::size_t stride = maxLen + 1;

  hid_t type = H5Tcopy(H5T_C_S1);
  H5Tset_size(type, stride);
  H5Tset_strpad(type, H5T_STR_NULLTERM);

  hid_t dataset = H5Dcreate(parent, name.c_str(), type, space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(dataset >= 0);

  std::vector<char> buffer(values.size() * stride, '\0');
  for (std::size_t i = 0; i < values.size(); ++i) {
    std::memcpy(buffer.data() + i * stride, values[i].c_str(), values[i].size());
  }

  H5Dwrite(dataset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer.data());
  H5Dclose(dataset);
  H5Sclose(space);
  H5Tclose(type);
}

// Write an N-dimensional double dataset. Replaces WriteDoubleArray{2,3,4,5}dDataset.
template <std::size_t N>
void WriteDoubleNdDataset(hid_t parent,
                          const std::string& name,
                          const std::array<hsize_t, N>& dims,
                          const std::vector<double>& values)
{
  hid_t space = H5Screate_simple(static_cast<int>(N), dims.data(), nullptr);
  hid_t dataset = H5Dcreate(parent, name.c_str(), H5T_IEEE_F64LE, space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(dataset >= 0);
  H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data());
  H5Dclose(dataset);
  H5Sclose(space);
}

// Write an N-dimensional int dataset. Replaces WriteIntArray3dDataset.
template <std::size_t N>
void WriteIntNdDataset(hid_t parent,
                       const std::string& name,
                       const std::array<hsize_t, N>& dims,
                       const std::vector<int>& values)
{
  hid_t space = H5Screate_simple(static_cast<int>(N), dims.data(), nullptr);
  hid_t dataset = H5Dcreate(parent, name.c_str(), H5T_NATIVE_INT, space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(dataset >= 0);
  H5Dwrite(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data());
  H5Dclose(dataset);
  H5Sclose(space);
}

// Backward-compatible aliases for existing callers
inline void WriteDoubleArray2dDataset(hid_t parent, const std::string& name,
                                       const std::array<hsize_t, 2>& dims,
                                       const std::vector<double>& values)
{ WriteDoubleNdDataset<2>(parent, name, dims, values); }

inline void WriteDoubleArray3dDataset(hid_t parent, const std::string& name,
                                       const std::array<hsize_t, 3>& dims,
                                       const std::vector<double>& values)
{ WriteDoubleNdDataset<3>(parent, name, dims, values); }

inline void WriteIntArray3dDataset(hid_t parent, const std::string& name,
                                    const std::array<hsize_t, 3>& dims,
                                    const std::vector<int>& values)
{ WriteIntNdDataset<3>(parent, name, dims, values); }

inline void WriteDoubleArray4dDataset(hid_t parent, const std::string& name,
                                       const std::array<hsize_t, 4>& dims,
                                       const std::vector<double>& values)
{ WriteDoubleNdDataset<4>(parent, name, dims, values); }

inline void WriteDoubleArray5dDataset(hid_t parent, const std::string& name,
                                       const std::array<hsize_t, 5>& dims,
                                       const std::vector<double>& values)
{ WriteDoubleNdDataset<5>(parent, name, dims, values); }

// Create an axis dataset with a scale attribute (used by test_hdf5_loader.cpp)
inline void CreateAxisDataset(hid_t file,
                               const std::string& name,
                               const std::vector<double>& values,
                               const char* scale)
{
  const hsize_t dims = static_cast<hsize_t>(values.size());
  hid_t space = H5Screate_simple(1, &dims, nullptr);
  hid_t dataset = H5Dcreate(file, name.c_str(), H5T_IEEE_F64LE, space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data());
  WriteStringAttribute(dataset, "scale", scale);
  H5Dclose(dataset);
  H5Sclose(space);
}

// Create a raw axis dataset without a scale attribute (used by test_weaklib_eos_loader.cpp)
inline void CreateAxisDataset(hid_t parent, const char* name, const double* values, std::size_t count)
{
  const hsize_t dims = static_cast<hsize_t>(count);
  hid_t space = H5Screate_simple(1, &dims, nullptr);
  REQUIRE(space >= 0);

  hid_t dataset = H5Dcreate(parent, name, H5T_IEEE_F64LE, space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(dataset >= 0);
  REQUIRE(H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values) >= 0);

  H5Dclose(dataset);
  H5Sclose(space);
}

inline void CreateIntDataset(hid_t parent, const char* name, const int* values, std::size_t count)
{
  const hsize_t dims = static_cast<hsize_t>(count);
  hid_t space = H5Screate_simple(1, &dims, nullptr);
  REQUIRE(space >= 0);

  hid_t dataset = H5Dcreate(parent, name, H5T_STD_I32LE, space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(dataset >= 0);
  REQUIRE(H5Dwrite(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, values) >= 0);

  H5Dclose(dataset);
  H5Sclose(space);
}

} // namespace TestHelpers
