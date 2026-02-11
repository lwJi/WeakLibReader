#define SIMPLE_CATCH_NO_MAIN
#include <catch2/catch_test_macros.hpp>

#include "hdf5/WeakLibReader_Hdf5Loader.hpp"

#include <AMReX.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuDevice.H>

#include <hdf5.h>

#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {

constexpr double kTol = 1.0e-12;

/// Global AMReX guard that initializes once and finalizes at program exit.
/// This avoids the MPI re-initialization error that occurs when each test
/// creates its own AmrexGuard (MPI cannot be re-initialized after finalization).
struct GlobalAmrexGuard {
  GlobalAmrexGuard()
  {
    int argc = 0;
    char** argv = nullptr;
    amrex::Initialize(argc, argv);
  }
  ~GlobalAmrexGuard() { amrex::Finalize(); }

  static GlobalAmrexGuard& Instance()
  {
    static GlobalAmrexGuard guard;
    return guard;
  }
};

/// Per-test helper that ensures AMReX is initialized (via the global guard)
struct AmrexGuard {
  AmrexGuard() { (void)GlobalAmrexGuard::Instance(); }
};

/// RAII helper that temporarily silences HDF5 automatic error printing.
/// Used in tests that intentionally trigger HDF5 errors.
struct ScopedHdf5ErrorSilencer {
  H5E_auto2_t oldFunc = nullptr;
  void* oldClientData = nullptr;

  ScopedHdf5ErrorSilencer()
  {
    H5Eget_auto2(H5E_DEFAULT, &oldFunc, &oldClientData);
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
  }

  ~ScopedHdf5ErrorSilencer()
  {
    H5Eset_auto2(H5E_DEFAULT, oldFunc, oldClientData);
  }
};

void WriteStringAttribute(hid_t parent, const std::string& name, const char* value)
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

void CreateAxisDataset(hid_t file,
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

} // namespace

TEST_CASE("HDF5 loader reads table and axes", "[hdf5][loader]")
{
  AmrexGuard amrex{};

  const std::filesystem::path filePath =
      std::filesystem::temp_directory_path() / "weaklibreader_hdf5_test.h5";

  const hsize_t dims[3] = {4, 3, 2}; // reversed (axis2, axis1, axis0)
  std::vector<double> rawData;
  rawData.reserve(static_cast<std::size_t>(dims[0] * dims[1] * dims[2]));
  for (hsize_t k = 0; k < dims[0]; ++k) {
    for (hsize_t j = 0; j < dims[1]; ++j) {
      for (hsize_t i = 0; i < dims[2]; ++i) {
        rawData.push_back(100.0 * static_cast<double>(k) +
                          10.0 * static_cast<double>(j) +
                          static_cast<double>(i));
      }
    }
  }

  hid_t file = H5Fcreate(filePath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(file >= 0);

  hid_t space = H5Screate_simple(3, dims, nullptr);
  hid_t dataset = H5Dcreate(file, "values", H5T_IEEE_F64LE, space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(dataset >= 0);
  H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, rawData.data());
  H5Dclose(dataset);
  H5Sclose(space);

  CreateAxisDataset(file, "axis0", {0.0, 1.0}, "Linear");
  CreateAxisDataset(file, "axis1", {0.0, 2.0, 4.0}, "Linear");
  CreateAxisDataset(file, "axis2", {1.0, 2.0, 3.0, 4.0}, "Log10");

  H5Fclose(file);

  WeakLibReader::Hdf5Table table;
  const auto status = WeakLibReader::LoadHdf5Table(filePath.string(), table);

  REQUIRE(status == WeakLibReader::Hdf5LoadStatus::Success);
  REQUIRE(table.nd == 3);

  CHECK(table.extents[0] == 2);
  CHECK(table.extents[1] == 3);
  CHECK(table.extents[2] == 4);

  const WeakLibReader::Layout& layout = table.layout;
  CHECK(layout.nd == 3);
  CHECK(layout.n[0] == 2);
  CHECK(layout.n[1] == 3);
  CHECK(layout.n[2] == 4);

  const double* data = table.DataPtr();
  const std::size_t idx =
      1 * static_cast<std::size_t>(layout.stride[0]) +
      2 * static_cast<std::size_t>(layout.stride[1]) +
      3 * static_cast<std::size_t>(layout.stride[2]);
  CHECK(data[idx] == Catch::Approx(100.0 * 3.0 + 10.0 * 2.0 + 1.0));

  const auto& axis0 = table.axes[0];
  CHECK(axis0.scale == WeakLibReader::AxisScale::Linear);
  CHECK(axis0.n == 2);
  CHECK(axis0.grid[1] == Catch::Approx(1.0));

  const auto& axis2 = table.axes[2];
  CHECK(axis2.scale == WeakLibReader::AxisScale::Log10);
  CHECK(axis2.n == 4);
  CHECK(axis2.grid[3] == Catch::Approx(4.0));

  const auto deviceTable = WeakLibReader::MakeDeviceCopy(table);

  amrex::Gpu::PinnedVector<double> roundtrip(deviceTable.values.size());
  amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                   deviceTable.values.begin(), deviceTable.values.end(),
                   roundtrip.begin());

  const double* originalPtr = table.DataPtr();
  const double* roundtripPtr = roundtrip.data();
  std::size_t total = 1;
  for (int dim = 0; dim < table.nd; ++dim) {
    total *= static_cast<std::size_t>(table.extents[dim]);
  }
  for (std::size_t i = 0; i < total; ++i) {
    CHECK(roundtripPtr[i] == Catch::Approx(originalPtr[i]).margin(kTol));
  }

  for (int dim = 0; dim < table.nd; ++dim) {
    const auto& deviceAxis = deviceTable.axisStorage[dim];
    std::vector<double> hostAxis(deviceAxis.size());
    amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                     deviceAxis.begin(), deviceAxis.end(),
                     hostAxis.begin());
    REQUIRE(hostAxis.size() == table.axisStorage[dim].size());
    for (std::size_t i = 0; i < hostAxis.size(); ++i) {
      CHECK(hostAxis[i] == Catch::Approx(table.axisStorage[dim][i]).margin(kTol));
    }
  }

  WeakLibReader::Hdf5Table parallelTable;
  const auto parallelStatus =
      WeakLibReader::LoadHdf5TableParallel(filePath.string(), parallelTable);
  REQUIRE(parallelStatus == WeakLibReader::Hdf5LoadStatus::Success);
  CHECK(parallelTable.nd == table.nd);
  CHECK(parallelTable.layout.n[0] == table.layout.n[0]);
  CHECK(parallelTable.axes[2].scale == table.axes[2].scale);
  const double* parallelData = parallelTable.DataPtr();
  CHECK(parallelData[idx] == Catch::Approx(data[idx]).margin(kTol));

  std::filesystem::remove(filePath);
}

TEST_CASE("HDF5 loader rejects single-point axes", "[hdf5][loader][validation]")
{
  AmrexGuard amrex{};

  const std::filesystem::path filePath =
      std::filesystem::temp_directory_path() / "weaklibreader_hdf5_single_axis.h5";

  const hsize_t dims[1] = {1};
  const double payload = 3.14;

  hid_t file = H5Fcreate(filePath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(file >= 0);

  hid_t space = H5Screate_simple(1, dims, nullptr);
  hid_t dataset = H5Dcreate(file, "values", H5T_IEEE_F64LE, space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(dataset >= 0);
  H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &payload);
  H5Dclose(dataset);
  H5Sclose(space);

  CreateAxisDataset(file, "axis0", {0.0}, "Linear");

  H5Fclose(file);

  WeakLibReader::Hdf5Table table;
  const auto status = WeakLibReader::LoadHdf5Table(filePath.string(), table);

  CHECK(status == WeakLibReader::Hdf5LoadStatus::AxisNotMonotone);

  std::filesystem::remove(filePath);
}

TEST_CASE("HDF5 loader returns FileOpenFailed for non-existent file", "[hdf5][loader][error]")
{
  AmrexGuard amrex{};
  ScopedHdf5ErrorSilencer silencer{};

  WeakLibReader::Hdf5Table table;
  const auto status = WeakLibReader::LoadHdf5Table("/nonexistent/path/to/file.h5", table);

  CHECK(status == WeakLibReader::Hdf5LoadStatus::FileOpenFailed);
}

TEST_CASE("HDF5 loader returns DatasetOpenFailed for missing values dataset", "[hdf5][loader][error]")
{
  AmrexGuard amrex{};
  ScopedHdf5ErrorSilencer silencer{};

  const std::filesystem::path filePath =
      std::filesystem::temp_directory_path() / "weaklibreader_hdf5_no_values.h5";

  // Create file with no "values" dataset
  hid_t file = H5Fcreate(filePath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(file >= 0);

  // Just create an axis dataset, no values
  CreateAxisDataset(file, "axis0", {0.0, 1.0}, "Linear");

  H5Fclose(file);

  WeakLibReader::Hdf5Table table;
  const auto status = WeakLibReader::LoadHdf5Table(filePath.string(), table);

  CHECK(status == WeakLibReader::Hdf5LoadStatus::DatasetOpenFailed);

  std::filesystem::remove(filePath);
}

TEST_CASE("HDF5 loader returns AxisDatasetOpenFailed for missing axis", "[hdf5][loader][error]")
{
  AmrexGuard amrex{};
  ScopedHdf5ErrorSilencer silencer{};

  const std::filesystem::path filePath =
      std::filesystem::temp_directory_path() / "weaklibreader_hdf5_missing_axis.h5";

  // Create file with values dataset but missing axis1
  const hsize_t dims[2] = {3, 2}; // 2D dataset needs axis0 and axis1
  std::vector<double> rawData(6, 1.0);

  hid_t file = H5Fcreate(filePath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(file >= 0);

  hid_t space = H5Screate_simple(2, dims, nullptr);
  hid_t dataset = H5Dcreate(file, "values", H5T_IEEE_F64LE, space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, rawData.data());
  H5Dclose(dataset);
  H5Sclose(space);

  // Only create axis0, not axis1
  CreateAxisDataset(file, "axis0", {0.0, 1.0}, "Linear");

  H5Fclose(file);

  WeakLibReader::Hdf5Table table;
  const auto status = WeakLibReader::LoadHdf5Table(filePath.string(), table);

  CHECK(status == WeakLibReader::Hdf5LoadStatus::AxisDatasetOpenFailed);

  std::filesystem::remove(filePath);
}

TEST_CASE("HDF5 loader returns AxisExtentMismatch when axis size differs from values", "[hdf5][loader][error]")
{
  AmrexGuard amrex{};
  ScopedHdf5ErrorSilencer silencer{};

  const std::filesystem::path filePath =
      std::filesystem::temp_directory_path() / "weaklibreader_hdf5_extent_mismatch.h5";

  const hsize_t dims[2] = {3, 2}; // values has shape 3x2
  std::vector<double> rawData(6, 1.0);

  hid_t file = H5Fcreate(filePath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(file >= 0);

  hid_t space = H5Screate_simple(2, dims, nullptr);
  hid_t dataset = H5Dcreate(file, "values", H5T_IEEE_F64LE, space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, rawData.data());
  H5Dclose(dataset);
  H5Sclose(space);

  // axis0 should have 2 elements, but we create 3
  CreateAxisDataset(file, "axis0", {0.0, 1.0, 2.0}, "Linear"); // Wrong size!
  CreateAxisDataset(file, "axis1", {0.0, 1.0, 2.0}, "Linear");

  H5Fclose(file);

  WeakLibReader::Hdf5Table table;
  const auto status = WeakLibReader::LoadHdf5Table(filePath.string(), table);

  CHECK(status == WeakLibReader::Hdf5LoadStatus::AxisExtentMismatch);

  std::filesystem::remove(filePath);
}

TEST_CASE("HDF5 loader returns AxisInvalidScale for unknown scale attribute", "[hdf5][loader][error]")
{
  AmrexGuard amrex{};
  ScopedHdf5ErrorSilencer silencer{};

  const std::filesystem::path filePath =
      std::filesystem::temp_directory_path() / "weaklibreader_hdf5_invalid_scale.h5";

  const hsize_t dims[1] = {3};
  std::vector<double> rawData(3, 1.0);

  hid_t file = H5Fcreate(filePath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(file >= 0);

  hid_t space = H5Screate_simple(1, dims, nullptr);
  hid_t dataset = H5Dcreate(file, "values", H5T_IEEE_F64LE, space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, rawData.data());
  H5Dclose(dataset);
  H5Sclose(space);

  // Use invalid scale attribute
  CreateAxisDataset(file, "axis0", {0.0, 1.0, 2.0}, "InvalidScaleType");

  H5Fclose(file);

  WeakLibReader::Hdf5Table table;
  const auto status = WeakLibReader::LoadHdf5Table(filePath.string(), table);

  CHECK(status == WeakLibReader::Hdf5LoadStatus::AxisInvalidScale);

  std::filesystem::remove(filePath);
}

TEST_CASE("HDF5 loader returns AxisNotMonotone for non-monotonic axis", "[hdf5][loader][error]")
{
  AmrexGuard amrex{};
  ScopedHdf5ErrorSilencer silencer{};

  const std::filesystem::path filePath =
      std::filesystem::temp_directory_path() / "weaklibreader_hdf5_nonmonotonic.h5";

  const hsize_t dims[1] = {3};
  std::vector<double> rawData(3, 1.0);

  hid_t file = H5Fcreate(filePath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(file >= 0);

  hid_t space = H5Screate_simple(1, dims, nullptr);
  hid_t dataset = H5Dcreate(file, "values", H5T_IEEE_F64LE, space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, rawData.data());
  H5Dclose(dataset);
  H5Sclose(space);

  // Non-monotonic axis (1.0, 3.0, 2.0) - not ascending
  CreateAxisDataset(file, "axis0", {1.0, 3.0, 2.0}, "Linear");

  H5Fclose(file);

  WeakLibReader::Hdf5Table table;
  const auto status = WeakLibReader::LoadHdf5Table(filePath.string(), table);

  CHECK(status == WeakLibReader::Hdf5LoadStatus::AxisNotMonotone);

  std::filesystem::remove(filePath);
}

TEST_CASE("HDF5 loader returns AxisNotMonotone for descending axis", "[hdf5][loader][error]")
{
  AmrexGuard amrex{};
  ScopedHdf5ErrorSilencer silencer{};

  const std::filesystem::path filePath =
      std::filesystem::temp_directory_path() / "weaklibreader_hdf5_descending.h5";

  const hsize_t dims[1] = {3};
  std::vector<double> rawData(3, 1.0);

  hid_t file = H5Fcreate(filePath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(file >= 0);

  hid_t space = H5Screate_simple(1, dims, nullptr);
  hid_t dataset = H5Dcreate(file, "values", H5T_IEEE_F64LE, space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, rawData.data());
  H5Dclose(dataset);
  H5Sclose(space);

  // Descending axis (3.0, 2.0, 1.0) - not ascending
  CreateAxisDataset(file, "axis0", {3.0, 2.0, 1.0}, "Linear");

  H5Fclose(file);

  WeakLibReader::Hdf5Table table;
  const auto status = WeakLibReader::LoadHdf5Table(filePath.string(), table);

  CHECK(status == WeakLibReader::Hdf5LoadStatus::AxisNotMonotone);

  std::filesystem::remove(filePath);
}

TEST_CASE("HDF5 loader returns AxisNotMonotone for Log10 axis with non-positive values", "[hdf5][loader][error]")
{
  AmrexGuard amrex{};
  ScopedHdf5ErrorSilencer silencer{};

  const std::filesystem::path filePath =
      std::filesystem::temp_directory_path() / "weaklibreader_hdf5_log_nonpositive.h5";

  const hsize_t dims[1] = {3};
  std::vector<double> rawData(3, 1.0);

  hid_t file = H5Fcreate(filePath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(file >= 0);

  hid_t space = H5Screate_simple(1, dims, nullptr);
  hid_t dataset = H5Dcreate(file, "values", H5T_IEEE_F64LE, space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, rawData.data());
  H5Dclose(dataset);
  H5Sclose(space);

  // Log10 axis with non-positive value (includes 0)
  CreateAxisDataset(file, "axis0", {0.0, 1.0, 2.0}, "Log10");

  H5Fclose(file);

  WeakLibReader::Hdf5Table table;
  const auto status = WeakLibReader::LoadHdf5Table(filePath.string(), table);

  CHECK(status == WeakLibReader::Hdf5LoadStatus::AxisNotMonotone);

  std::filesystem::remove(filePath);
}

TEST_CASE("HDF5 loader returns DatasetRankInvalid for 0D dataset", "[hdf5][loader][error]")
{
  AmrexGuard amrex{};
  ScopedHdf5ErrorSilencer silencer{};

  const std::filesystem::path filePath =
      std::filesystem::temp_directory_path() / "weaklibreader_hdf5_0d.h5";

  hid_t file = H5Fcreate(filePath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(file >= 0);

  // Create scalar (0D) dataset
  hid_t space = H5Screate(H5S_SCALAR);
  hid_t dataset = H5Dcreate(file, "values", H5T_IEEE_F64LE, space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  double scalar = 1.0;
  H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &scalar);
  H5Dclose(dataset);
  H5Sclose(space);

  H5Fclose(file);

  WeakLibReader::Hdf5Table table;
  const auto status = WeakLibReader::LoadHdf5Table(filePath.string(), table);

  CHECK(status == WeakLibReader::Hdf5LoadStatus::DatasetRankInvalid);

  std::filesystem::remove(filePath);
}

TEST_CASE("HDF5 loader handles 5D tables correctly", "[hdf5][loader][5d]")
{
  AmrexGuard amrex{};

  const std::filesystem::path filePath =
      std::filesystem::temp_directory_path() / "weaklibreader_hdf5_5d.h5";

  // 5D table with shape (2, 3, 4, 5, 6) in C-order
  // HDF5 stores in Fortran-order, so dims are reversed: (6, 5, 4, 3, 2)
  const hsize_t dims[5] = {6, 5, 4, 3, 2};
  const std::size_t totalSize = 2 * 3 * 4 * 5 * 6; // 720 elements

  std::vector<double> rawData;
  rawData.reserve(totalSize);
  // Fill with unique values based on multi-index
  for (hsize_t i4 = 0; i4 < dims[0]; ++i4) {
    for (hsize_t i3 = 0; i3 < dims[1]; ++i3) {
      for (hsize_t i2 = 0; i2 < dims[2]; ++i2) {
        for (hsize_t i1 = 0; i1 < dims[3]; ++i1) {
          for (hsize_t i0 = 0; i0 < dims[4]; ++i0) {
            rawData.push_back(
                10000.0 * static_cast<double>(i4) +
                1000.0 * static_cast<double>(i3) +
                100.0 * static_cast<double>(i2) +
                10.0 * static_cast<double>(i1) +
                static_cast<double>(i0));
          }
        }
      }
    }
  }

  hid_t file = H5Fcreate(filePath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(file >= 0);

  hid_t space = H5Screate_simple(5, dims, nullptr);
  hid_t dataset = H5Dcreate(file, "values", H5T_IEEE_F64LE, space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(dataset >= 0);
  H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, rawData.data());
  H5Dclose(dataset);
  H5Sclose(space);

  // Create 5 axes
  CreateAxisDataset(file, "axis0", {0.0, 1.0}, "Linear");
  CreateAxisDataset(file, "axis1", {0.0, 1.0, 2.0}, "Linear");
  CreateAxisDataset(file, "axis2", {1.0, 2.0, 3.0, 4.0}, "Log10");
  CreateAxisDataset(file, "axis3", {0.0, 0.25, 0.5, 0.75, 1.0}, "Linear");
  CreateAxisDataset(file, "axis4", {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, "Log10");

  H5Fclose(file);

  // Load the table
  WeakLibReader::Hdf5Table table;
  const auto status = WeakLibReader::LoadHdf5Table(filePath.string(), table);

  REQUIRE(status == WeakLibReader::Hdf5LoadStatus::Success);
  REQUIRE(table.nd == 5);

  // Verify extents (C-order)
  CHECK(table.extents[0] == 2);
  CHECK(table.extents[1] == 3);
  CHECK(table.extents[2] == 4);
  CHECK(table.extents[3] == 5);
  CHECK(table.extents[4] == 6);

  // Verify layout
  CHECK(table.layout.nd == 5);
  CHECK(table.layout.n[0] == 2);
  CHECK(table.layout.n[1] == 3);
  CHECK(table.layout.n[2] == 4);
  CHECK(table.layout.n[3] == 5);
  CHECK(table.layout.n[4] == 6);

  // Verify axes
  CHECK(table.axes[0].scale == WeakLibReader::AxisScale::Linear);
  CHECK(table.axes[0].n == 2);
  CHECK(table.axes[2].scale == WeakLibReader::AxisScale::Log10);
  CHECK(table.axes[2].n == 4);
  CHECK(table.axes[4].scale == WeakLibReader::AxisScale::Log10);
  CHECK(table.axes[4].n == 6);

  // Verify a specific data point using Layout::Offset
  // Index (1, 2, 3, 4, 5) should give value 10000*5 + 1000*4 + 100*3 + 10*2 + 1 = 54321
  const std::size_t offset = table.layout.Offset(1, 2, 3, 4, 5);
  const double* data = table.DataPtr();
  CHECK(data[offset] == Catch::Approx(54321.0));

  // Test MakeDeviceCopy for 5D table
  const auto deviceTable = WeakLibReader::MakeDeviceCopy(table);
  CHECK(deviceTable.nd == 5);
  CHECK(deviceTable.layout.nd == 5);

  // Round-trip: copy device data back to host and verify
  amrex::Gpu::PinnedVector<double> roundtrip(deviceTable.values.size());
  amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                   deviceTable.values.begin(), deviceTable.values.end(),
                   roundtrip.begin());

  // Verify all data matches
  const double* originalPtr = table.DataPtr();
  const double* roundtripPtr = roundtrip.data();
  for (std::size_t i = 0; i < totalSize; ++i) {
    CHECK(roundtripPtr[i] == Catch::Approx(originalPtr[i]).margin(kTol));
  }

  // Verify device axis data
  for (int dim = 0; dim < table.nd; ++dim) {
    const auto& deviceAxis = deviceTable.axisStorage[dim];
    std::vector<double> hostAxis(deviceAxis.size());
    amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                     deviceAxis.begin(), deviceAxis.end(),
                     hostAxis.begin());
    REQUIRE(hostAxis.size() == table.axisStorage[dim].size());
    for (std::size_t i = 0; i < hostAxis.size(); ++i) {
      CHECK(hostAxis[i] == Catch::Approx(table.axisStorage[dim][i]).margin(kTol));
    }
  }

  std::filesystem::remove(filePath);
}
