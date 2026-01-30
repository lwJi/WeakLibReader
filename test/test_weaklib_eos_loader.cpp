#define SIMPLE_CATCH_NO_MAIN
#include <catch2/catch_test_macros.hpp>

#include "WeakLibReader_Hdf5Loader.hpp"

#include <AMReX.H>
#include <AMReX_GpuContainers.H>
#include <hdf5.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
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

void WriteIntArrayDataset(hid_t parent,
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

void WriteDoubleArrayDataset(hid_t parent,
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

void WriteStringArrayDataset(hid_t parent,
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

void WriteDoubleArray3dDataset(hid_t parent,
                               const std::string& name,
                               const std::array<hsize_t, 3>& dims,
                               const std::vector<double>& values)
{
  hid_t space = H5Screate_simple(3, dims.data(), nullptr);
  hid_t dataset = H5Dcreate(parent, name.c_str(), H5T_IEEE_F64LE, space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(dataset >= 0);
  H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data());
  H5Dclose(dataset);
  H5Sclose(space);
}

void WriteIntArray3dDataset(hid_t parent,
                            const std::string& name,
                            const std::array<hsize_t, 3>& dims,
                            const std::vector<int>& values)
{
  hid_t space = H5Screate_simple(3, dims.data(), nullptr);
  hid_t dataset = H5Dcreate(parent, name.c_str(), H5T_NATIVE_INT, space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(dataset >= 0);
  H5Dwrite(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data());
  H5Dclose(dataset);
  H5Sclose(space);
}

void CreateWeakLibEosTestFileFull(const std::filesystem::path& filePath)
{
  constexpr int nRho = 2;
  constexpr int nT = 3;
  constexpr int nYe = 2;
  constexpr int nVariables = 3;

  hid_t file = H5Fcreate(filePath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(file >= 0);

  hid_t thermoGroup = H5Gcreate(file, "ThermoState", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(thermoGroup >= 0);

  WriteIntArrayDataset(thermoGroup, "LogInterp", {1, 1, 0});
  WriteIntArrayDataset(thermoGroup, "Dimensions", {nRho, nT, nYe});
  WriteStringArrayDataset(thermoGroup, "Names",
                          {"Density", "Temperature", "Electron Fraction"});
  WriteStringArrayDataset(thermoGroup, "Units",
                          {"g/cm^3", "MeV", "dimensionless"});

  WriteDoubleArrayDataset(thermoGroup, "Density", {1.0, 2.0});
  WriteDoubleArrayDataset(thermoGroup, "Temperature", {0.5, 1.0, 2.0});
  WriteDoubleArrayDataset(thermoGroup, "Electron Fraction", {0.1, 0.2});

  H5Gclose(thermoGroup);

  hid_t dvGroup = H5Gcreate(file, "DependentVariables", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(dvGroup >= 0);

  WriteIntArrayDataset(dvGroup, "nVariables", {nVariables});

  const std::vector<std::string> varNames = {
      "Pressure", "Entropy Per Baryon", "Gamma1"};
  WriteStringArrayDataset(dvGroup, "Names", varNames);
  WriteStringArrayDataset(dvGroup, "Units",
                          {"dyn/cm^2", "kb/baryon", "dimensionless"});
  WriteDoubleArrayDataset(dvGroup, "Offsets", {0.0, 0.1, 0.2});

  const std::array<hsize_t, 3> fileDims = {
      static_cast<hsize_t>(nYe),
      static_cast<hsize_t>(nT),
      static_cast<hsize_t>(nRho)};
  const std::size_t totalSize = static_cast<std::size_t>(nRho * nT * nYe);

  for (int iVar = 0; iVar < nVariables; ++iVar) {
    std::vector<double> data(totalSize);
    for (std::size_t i = 0; i < totalSize; ++i) {
      data[i] = static_cast<double>(iVar) + 0.01 * static_cast<double>(i);
    }
    WriteDoubleArray3dDataset(dvGroup, varNames[iVar], fileDims, data);
  }

  std::vector<int> repaired(totalSize, 0);
  WriteIntArray3dDataset(dvGroup, "Repaired", fileDims, repaired);

  WriteIntArrayDataset(dvGroup, "iPressure", {1});
  WriteIntArrayDataset(dvGroup, "iEntropyPerBaryon", {2});
  WriteIntArrayDataset(dvGroup, "iInternalEnergyDensity", {1});
  WriteIntArrayDataset(dvGroup, "iElectronChemicalPotential", {1});
  WriteIntArrayDataset(dvGroup, "iProtonChemicalPotential", {1});
  WriteIntArrayDataset(dvGroup, "iNeutronChemicalPotential", {1});
  WriteIntArrayDataset(dvGroup, "iProtonMassFraction", {1});
  WriteIntArrayDataset(dvGroup, "iNeutronMassFraction", {1});
  WriteIntArrayDataset(dvGroup, "iAlphaMassFraction", {1});
  WriteIntArrayDataset(dvGroup, "iHeavyMassFraction", {1});
  WriteIntArrayDataset(dvGroup, "iHeavyChargeNumber", {1});
  WriteIntArrayDataset(dvGroup, "iHeavyMassNumber", {1});
  WriteIntArrayDataset(dvGroup, "iHeavyBindingEnergy", {1});
  WriteIntArrayDataset(dvGroup, "iThermalEnergy", {1});
  WriteIntArrayDataset(dvGroup, "iGamma1", {3});

  H5Gclose(dvGroup);
  H5Fclose(file);
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

TEST_CASE("LoadWeakLibEosTableFull reads complete EOS table", "[hdf5][weaklib]")
{
  EnsureAmrexInitialized();

  const std::filesystem::path eosPath =
      std::filesystem::temp_directory_path() / "weaklibreader_eos_small_full.h5";
  CreateWeakLibEosTestFileFull(eosPath);

  WeakLibEosTable table;
  const auto status = LoadWeakLibEosTableFull(eosPath.string(), table);
  REQUIRE(status == Hdf5LoadStatus::Success);

  // Dimensions are correct
  CHECK(table.dimensions[0] == 2);  // nRho
  CHECK(table.dimensions[1] == 3);  // nT
  CHECK(table.dimensions[2] == 2);  // nYe

  // nVariables is correct
  CHECK(table.nVariables == 3);

  // Axis scales match LogInterp
  CHECK(table.scales[0] == AxisScale::Log10);  // Density
  CHECK(table.scales[1] == AxisScale::Log10);  // Temperature
  CHECK(table.scales[2] == AxisScale::Linear); // Ye

  // Axis names are read correctly
  CHECK(table.axisNames[0] == "Density");
  CHECK(table.axisNames[1] == "Temperature");
  CHECK(table.axisNames[2] == "Electron Fraction");

  // Variable names are read correctly
  REQUIRE(table.variableNames.size() == 3);
  CHECK(table.variableNames[0] == "Pressure");
  CHECK(table.variableNames[1] == "Entropy Per Baryon");

  // Index mappings are valid
  // Indices are 1-based in Fortran, should be positive
  CHECK(table.indices.iPressure >= 1);
  CHECK(table.indices.iEntropyPerBaryon >= 1);
  CHECK(table.indices.iGamma1 >= 1);

  // Variable data has correct size
  for (int iVar = 0; iVar < table.nVariables; ++iVar) {
    const auto& var = table.variables[iVar];
    CHECK(var.const_table().p != nullptr);
  }

  // Repaired mask is loaded
  CHECK(table.repaired.const_table().p != nullptr);

  // Layout is computed correctly (row-major)
  // stride[0] = 1, stride[1] = n[0], stride[2] = n[0]*n[1]
  CHECK(table.layout.stride[0] == 1);
  CHECK(table.layout.stride[1] == static_cast<std::size_t>(table.dimensions[0]));
  CHECK(table.layout.stride[2] == static_cast<std::size_t>(table.dimensions[0] * table.dimensions[1]));

  std::filesystem::remove(eosPath);
}

TEST_CASE("MakeDeviceCopy works for WeakLibEosTable", "[hdf5][weaklib][gpu]")
{
  EnsureAmrexInitialized();

  const std::filesystem::path eosPath =
      std::filesystem::temp_directory_path() / "weaklibreader_eos_small_device.h5";
  CreateWeakLibEosTestFileFull(eosPath);

  WeakLibEosTable hostTable;
  const auto status = LoadWeakLibEosTableFull(eosPath.string(), hostTable);
  REQUIRE(status == Hdf5LoadStatus::Success);

  // This will only do meaningful work on GPU builds
  auto deviceTable = MakeDeviceCopy(hostTable);

  CHECK(deviceTable.nVariables == hostTable.nVariables);
  CHECK(deviceTable.dimensions == hostTable.dimensions);
  CHECK(deviceTable.indices.iPressure == hostTable.indices.iPressure);

  std::filesystem::remove(eosPath);
}
