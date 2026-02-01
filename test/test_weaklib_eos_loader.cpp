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

void CreateWeakLibEosTestFileFullCOrder(const std::filesystem::path& filePath)
{
  constexpr int nRho = 2;
  constexpr int nT = 3;
  constexpr int nYe = 4;
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
  WriteDoubleArrayDataset(thermoGroup, "Electron Fraction", {0.1, 0.2, 0.3, 0.4});

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
      static_cast<hsize_t>(nRho),
      static_cast<hsize_t>(nT),
      static_cast<hsize_t>(nYe)};
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
  // Indices are normalized to 0-based in C++
  CHECK(table.indices.iPressure == 0);
  CHECK(table.indices.iEntropyPerBaryon == 1);
  CHECK(table.indices.iGamma1 == 2);

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

TEST_CASE("LoadWeakLibEosTableFullParallel loads complete table", "[weaklib][eos][parallel]")
{
  EnsureAmrexInitialized();

  const std::filesystem::path eosPath =
      std::filesystem::temp_directory_path() / "weaklibreader_eos_parallel_full.h5";
  CreateWeakLibEosTestFileFull(eosPath);

  WeakLibEosTable table;
  const auto status = LoadWeakLibEosTableFullParallel(eosPath.string(), table);
  REQUIRE(status == Hdf5LoadStatus::Success);

  // Load sequential for comparison
  WeakLibEosTable seqTable;
  REQUIRE(LoadWeakLibEosTableFull(eosPath.string(), seqTable) == Hdf5LoadStatus::Success);

  // Verify dimensions match
  CHECK(table.nVariables == seqTable.nVariables);
  CHECK(table.dimensions == seqTable.dimensions);
  CHECK(table.scales == seqTable.scales);

  // Verify axis names and units
  CHECK(table.axisNames == seqTable.axisNames);
  CHECK(table.axisUnits == seqTable.axisUnits);

  // Verify variable names and units
  CHECK(table.variableNames == seqTable.variableNames);
  CHECK(table.variableUnits == seqTable.variableUnits);

  // Verify offsets
  REQUIRE(table.offsets.size() == seqTable.offsets.size());
  for (std::size_t i = 0; i < table.offsets.size(); ++i) {
    CHECK(table.offsets[i] == seqTable.offsets[i]);
  }

  // Verify indices
  CHECK(table.indices.iPressure == seqTable.indices.iPressure);
  CHECK(table.indices.iEntropyPerBaryon == seqTable.indices.iEntropyPerBaryon);
  CHECK(table.indices.iGamma1 == seqTable.indices.iGamma1);

  // Verify variable data matches
  const std::size_t varSize = static_cast<std::size_t>(table.dimensions[0]) *
                              table.dimensions[1] * table.dimensions[2];
  for (int iVar = 0; iVar < table.nVariables; ++iVar) {
    const double* parData = table.variables[iVar].const_table().p;
    const double* seqData = seqTable.variables[iVar].const_table().p;
    for (std::size_t i = 0; i < varSize; ++i) {
      CHECK(parData[i] == seqData[i]);
    }
  }

  // Verify repaired mask matches
  const int* parRepaired = table.repaired.const_table().p;
  const int* seqRepaired = seqTable.repaired.const_table().p;
  for (std::size_t i = 0; i < varSize; ++i) {
    CHECK(parRepaired[i] == seqRepaired[i]);
  }

  std::filesystem::remove(eosPath);
}

TEST_CASE("LoadWeakLibEosTableFullParallel fails for nonexistent file", "[weaklib][eos][parallel]")
{
  EnsureAmrexInitialized();
  ScopedHdf5ErrorSilencer silencer{};

  WeakLibEosTable table;
  const Hdf5LoadStatus status = LoadWeakLibEosTableFullParallel("/nonexistent/path.h5", table);

  CHECK(status == Hdf5LoadStatus::FileOpenFailed);
}

TEST_CASE("LoadWeakLibEosTableFull rejects C-ordered dependent variables", "[weaklib][eos][validation]")
{
  EnsureAmrexInitialized();

  const std::filesystem::path eosPath =
      std::filesystem::temp_directory_path() / "weaklibreader_eos_c_order.h5";
  CreateWeakLibEosTestFileFullCOrder(eosPath);

  WeakLibEosTable table;
  const auto status = LoadWeakLibEosTableFull(eosPath.string(), table);
  CHECK(status == Hdf5LoadStatus::DatasetReadFailed);

  std::filesystem::remove(eosPath);
}
