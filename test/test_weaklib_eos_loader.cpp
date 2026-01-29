#define SIMPLE_CATCH_NO_MAIN
#include <catch2/catch_test_macros.hpp>

#include "WeakLibReader_Hdf5Loader.hpp"

#include <AMReX.H>
#include <hdf5.h>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using namespace WeakLibReader;

namespace {

constexpr std::size_t kRhoCount = 3;
constexpr std::size_t kTempCount = 2;
constexpr std::size_t kYeCount = 2;

const double kDensityAxis[kRhoCount] = {1.0e3, 1.0e6, 1.0e9};
const double kTemperatureAxis[kTempCount] = {0.1, 1.0};
const double kYeAxis[kYeCount] = {0.1, 0.3};

/// RAII helper that temporarily silences HDF5 automatic error printing.
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

void CreateAxisDataset(hid_t parent, const char* name, const double* values, std::size_t count)
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

void CreateIntDataset(hid_t parent, const char* name, const int* values, std::size_t count)
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

std::vector<double> MakeDependentValues(double offset)
{
  std::vector<double> values;
  values.reserve(kRhoCount * kTempCount * kYeCount);
  for (std::size_t ye = 0; ye < kYeCount; ++ye) {
    for (std::size_t temp = 0; temp < kTempCount; ++temp) {
      for (std::size_t rho = 0; rho < kRhoCount; ++rho) {
        values.push_back(offset +
                         1.0 +
                         100.0 * static_cast<double>(ye) +
                         10.0 * static_cast<double>(temp) +
                         static_cast<double>(rho));
      }
    }
  }
  return values;
}

void CreateDependentVariable(hid_t parent, const char* name, const std::vector<double>& values)
{
  const hsize_t dims[3] = {
      static_cast<hsize_t>(kYeCount),
      static_cast<hsize_t>(kTempCount),
      static_cast<hsize_t>(kRhoCount)};
  hid_t space = H5Screate_simple(3, dims, nullptr);
  REQUIRE(space >= 0);

  hid_t dataset = H5Dcreate(parent, name, H5T_IEEE_F64LE, space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(dataset >= 0);
  REQUIRE(H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                   H5P_DEFAULT, values.data()) >= 0);

  H5Dclose(dataset);
  H5Sclose(space);
}

void CreateWeakLibEosTestFile(const std::filesystem::path& path)
{
  hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(file >= 0);

  hid_t thermoGroup = H5Gcreate(file, "ThermoState", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(thermoGroup >= 0);

  const int logInterp[3] = {1, 1, 0};
  CreateIntDataset(thermoGroup, "LogInterp", logInterp, 3);

  CreateAxisDataset(thermoGroup, "Density", kDensityAxis, kRhoCount);
  CreateAxisDataset(thermoGroup, "Temperature", kTemperatureAxis, kTempCount);
  CreateAxisDataset(thermoGroup, "Electron Fraction", kYeAxis, kYeCount);

  H5Gclose(thermoGroup);

  hid_t dependentGroup =
      H5Gcreate(file, "DependentVariables", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(dependentGroup >= 0);

  const std::vector<double> pressure = MakeDependentValues(0.0);
  const std::vector<double> entropy = MakeDependentValues(1000.0);
  CreateDependentVariable(dependentGroup, "Pressure", pressure);
  CreateDependentVariable(dependentGroup, "Entropy Per Baryon", entropy);

  H5Gclose(dependentGroup);
  H5Fclose(file);
}

struct TempWeakLibEosFile {
  std::filesystem::path path;
  TempWeakLibEosFile()
  {
    path = std::filesystem::temp_directory_path() / "weaklibreader_eos_test.h5";
    CreateWeakLibEosTestFile(path);
  }
  ~TempWeakLibEosFile() { std::filesystem::remove(path); }
};

/// Initialize AMReX if not already initialized.
/// This relies on AMReX's internal tracking to avoid double-initialization.
void EnsureAmrexInitialized()
{
  if (!amrex::Initialized()) {
    int argc = 0;
    char** argv = nullptr;
    amrex::Initialize(argc, argv);
    // Register finalize at exit - but only if we initialized
    std::atexit([]() {
      if (amrex::Initialized()) {
        amrex::Finalize();
      }
    });
  }
}

} // namespace

TEST_CASE("LoadWeakLibEosTable loads Pressure", "[weaklib][eos]")
{
  EnsureAmrexInitialized();

  TempWeakLibEosFile tempFile{};
  const std::string path = tempFile.path.string();
  Hdf5Table table;
  const Hdf5LoadStatus status = LoadWeakLibEosTable(path, "Pressure", table);

  REQUIRE(status == Hdf5LoadStatus::Success);

  // Verify dimensions (C order: rho, T, Ye)
  REQUIRE(table.nd == 3);
  CHECK(table.extents[0] == static_cast<int>(kRhoCount));   // Density
  CHECK(table.extents[1] == static_cast<int>(kTempCount));  // Temperature
  CHECK(table.extents[2] == static_cast<int>(kYeCount));    // Electron Fraction

  // Verify axis scales
  CHECK(table.axes[0].scale == AxisScale::Log10);   // Density
  CHECK(table.axes[1].scale == AxisScale::Log10);   // Temperature
  CHECK(table.axes[2].scale == AxisScale::Linear);  // Ye

  // Verify axis grid sizes
  CHECK(table.axes[0].n == static_cast<int>(kRhoCount));
  CHECK(table.axes[1].n == static_cast<int>(kTempCount));
  CHECK(table.axes[2].n == static_cast<int>(kYeCount));

  // Verify axis grids are valid pointers
  REQUIRE(table.axes[0].grid != nullptr);
  REQUIRE(table.axes[1].grid != nullptr);
  REQUIRE(table.axes[2].grid != nullptr);

  // Verify data pointer is valid
  REQUIRE(table.DataPtr() != nullptr);

  // Verify layout
  CHECK(table.layout.nd == 3);
  CHECK(table.layout.n[0] == static_cast<int>(kRhoCount));
  CHECK(table.layout.n[1] == static_cast<int>(kTempCount));
  CHECK(table.layout.n[2] == static_cast<int>(kYeCount));
}

TEST_CASE("LoadWeakLibEosTable loads Entropy Per Baryon", "[weaklib][eos]")
{
  EnsureAmrexInitialized();

  TempWeakLibEosFile tempFile{};
  const std::string path = tempFile.path.string();
  Hdf5Table table;
  const Hdf5LoadStatus status = LoadWeakLibEosTable(path, "Entropy Per Baryon", table);

  REQUIRE(status == Hdf5LoadStatus::Success);
  CHECK(table.nd == 3);
  CHECK(table.extents[0] == static_cast<int>(kRhoCount));
  CHECK(table.extents[1] == static_cast<int>(kTempCount));
  CHECK(table.extents[2] == static_cast<int>(kYeCount));
}

TEST_CASE("LoadWeakLibEosTable fails for nonexistent variable", "[weaklib][eos]")
{
  EnsureAmrexInitialized();
  ScopedHdf5ErrorSilencer silencer{};

  TempWeakLibEosFile tempFile{};
  const std::string path = tempFile.path.string();
  Hdf5Table table;
  const Hdf5LoadStatus status = LoadWeakLibEosTable(path, "NonexistentVariable", table);

  CHECK(status == Hdf5LoadStatus::DatasetOpenFailed);
}

TEST_CASE("LoadWeakLibEosTable fails for nonexistent file", "[weaklib][eos]")
{
  EnsureAmrexInitialized();
  ScopedHdf5ErrorSilencer silencer{};

  Hdf5Table table;
  const Hdf5LoadStatus status = LoadWeakLibEosTable("/nonexistent/path.h5", "Pressure", table);

  CHECK(status == Hdf5LoadStatus::FileOpenFailed);
}

TEST_CASE("LoadWeakLibEosTable axis values are reasonable", "[weaklib][eos]")
{
  EnsureAmrexInitialized();

  TempWeakLibEosFile tempFile{};
  const std::string path = tempFile.path.string();
  Hdf5Table table;
  const Hdf5LoadStatus status = LoadWeakLibEosTable(path, "Pressure", table);

  REQUIRE(status == Hdf5LoadStatus::Success);

  // Density axis: should be positive (for log10), roughly 1e3 to 1e15 g/cm^3
  const double* rho = table.axes[0].grid;
  CHECK(rho[0] == Catch::Approx(kDensityAxis[0]));
  CHECK(rho[0] < rho[table.axes[0].n - 1]);  // Monotonically increasing

  // Temperature axis: should be positive, roughly 0.1 to 100 MeV
  const double* T = table.axes[1].grid;
  CHECK(T[0] == Catch::Approx(kTemperatureAxis[0]));
  CHECK(T[0] < T[table.axes[1].n - 1]);

  // Electron fraction: 0 to 0.6 typically
  const double* Ye = table.axes[2].grid;
  CHECK(Ye[0] == Catch::Approx(kYeAxis[0]));
  CHECK(Ye[0] < Ye[table.axes[2].n - 1]);
  CHECK(Ye[table.axes[2].n - 1] <= 1.0);
}

TEST_CASE("LoadWeakLibEosTable data values spot check", "[weaklib][eos][values]")
{
  EnsureAmrexInitialized();

  TempWeakLibEosFile tempFile{};
  const std::string path = tempFile.path.string();
  Hdf5Table table;
  const Hdf5LoadStatus status = LoadWeakLibEosTable(path, "Pressure", table);

  REQUIRE(status == Hdf5LoadStatus::Success);

  // Access data using layout
  const double* data = table.DataPtr();
  const Layout& layout = table.layout;

  // Check corner values (indices in C order: rho, T, Ye)
  // Value at (0, 0, 0) - first element
  const std::size_t idx000 = layout.Offset(0, 0, 0);
  CHECK(data[idx000] == Catch::Approx(1.0));

  // Value at last index
  const int lastRho = table.extents[0] - 1;
  const int lastT = table.extents[1] - 1;
  const int lastYe = table.extents[2] - 1;
  const std::size_t idxLast = layout.Offset(lastRho, lastT, lastYe);
  CHECK(data[idxLast] == Catch::Approx(113.0));

  // Total size check
  const std::size_t expectedSize = kRhoCount * kTempCount * kYeCount;
  CHECK(layout.Offset(lastRho, lastT, lastYe) == expectedSize - 1);
}
