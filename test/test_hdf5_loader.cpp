#define SIMPLE_CATCH_NO_MAIN
#include <catch2/catch_test_macros.hpp>

#include "Hdf5Loader.hpp"

#include <AMReX.H>
#include <hdf5.h>

#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {

struct AmrexGuard {
  AmrexGuard()
  {
    int argc = 0;
    char** argv = nullptr;
    amrex::Initialize(argc, argv);
  }
  ~AmrexGuard() { amrex::Finalize(); }
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

  const auto view = table.View();
  double coords[5] = {0.3, 1.5, 2.5, 0.0, 0.0};
  WeakLibReader::InterpConfig cfg;
  const double interp = WeakLibReader::InterpLinearND(
      view.data, view.layout, view.axes, coords, cfg, view.nd);

  int idx0 = 0;
  int idx1 = 0;
  int idx2 = 0;
  double frac0 = 0.0;
  double frac1 = 0.0;
  double frac2 = 0.0;

  REQUIRE_FALSE(WeakLibReader::detail::IndexAndDelta(view.axes[0], coords[0], idx0, frac0));
  REQUIRE_FALSE(WeakLibReader::detail::IndexAndDelta(view.axes[1], coords[1], idx1, frac1));
  REQUIRE_FALSE(WeakLibReader::detail::IndexAndDelta(view.axes[2], coords[2], idx2, frac2));

  const double expected =
      (static_cast<double>(idx0) + frac0) +
      10.0 * (static_cast<double>(idx1) + frac1) +
      100.0 * (static_cast<double>(idx2) + frac2);
  CHECK(interp == Catch::Approx(expected).margin(1.0e-12));

  std::filesystem::remove(filePath);
}
