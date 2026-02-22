#define SIMPLE_CATCH_NO_MAIN
#include <catch2/catch_test_macros.hpp>

#include "hdf5/WeakLibReader_Hdf5Loader.hpp"
#include "test_amrex_guard.hpp"
#include "test_hdf5_helpers.hpp"

#include <AMReX_GpuContainers.H>
#include <hdf5.h>

#include <array>
#include <filesystem>
#include <string>
#include <vector>

namespace {

using namespace test_helpers;

// Write the EOS variable index datasets shared by all test EOS files.
void WriteEosVariableIndices(hid_t dvGroup)
{
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
}

// Write a complete EOS test file. The fileDimOrder controls whether the
// dependent variable datasets are written in Fortran order {nYe,nT,nRho}
// (the correct WeakLib layout) or C order {nRho,nT,nYe} (for rejection tests).
enum class DimOrder { Fortran, COrder };

void CreateWeakLibEosTestFileImpl(const std::filesystem::path& filePath,
                                  int nRho, int nT, int nYe,
                                  const std::vector<double>& yeValues,
                                  DimOrder order)
{
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
  WriteDoubleArrayDataset(thermoGroup, "Electron Fraction", yeValues);

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

  std::array<hsize_t, 3> fileDims{};
  if (order == DimOrder::Fortran) {
    fileDims = {static_cast<hsize_t>(nYe),
                static_cast<hsize_t>(nT),
                static_cast<hsize_t>(nRho)};
  } else {
    fileDims = {static_cast<hsize_t>(nRho),
                static_cast<hsize_t>(nT),
                static_cast<hsize_t>(nYe)};
  }
  const std::size_t totalSize = static_cast<std::size_t>(nRho * nT * nYe);

  for (int iVar = 0; iVar < nVariables; ++iVar) {
    std::vector<double> data(totalSize);
    for (std::size_t i = 0; i < totalSize; ++i) {
      data[i] = static_cast<double>(iVar) + 0.01 * static_cast<double>(i);
    }
    WriteDoubleNdDataset<3>(dvGroup, varNames[iVar], fileDims, data);
  }

  std::vector<int> repaired(totalSize, 0);
  WriteIntNdDataset<3>(dvGroup, "Repaired", fileDims, repaired);

  WriteEosVariableIndices(dvGroup);

  H5Gclose(dvGroup);
  H5Fclose(file);
}

void CreateWeakLibEosTestFileFull(const std::filesystem::path& filePath)
{
  CreateWeakLibEosTestFileImpl(filePath, 2, 3, 2, {0.1, 0.2}, DimOrder::Fortran);
}

void CreateWeakLibEosTestFileFullCOrder(const std::filesystem::path& filePath)
{
  CreateWeakLibEosTestFileImpl(filePath, 2, 3, 4, {0.1, 0.2, 0.3, 0.4}, DimOrder::COrder);
}

} // namespace

TEST_CASE("LoadWeakLibEosTableFull reads complete EOS table", "[hdf5][weaklib]")
{
  using namespace WeakLibReader;
  AmrexGuard amrex{};

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
  CHECK(table.axes[0].scale == AxisScale::Log10);  // Density
  CHECK(table.axes[1].scale == AxisScale::Log10);  // Temperature
  CHECK(table.axes[2].scale == AxisScale::Linear); // Ye

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
    CHECK(var.data() != nullptr);
  }

  // Repaired mask is loaded
  CHECK(table.repaired.data() != nullptr);

  // Layout is computed correctly (row-major)
  // stride[0] = 1, stride[1] = n[0], stride[2] = n[0]*n[1]
  CHECK(table.layout.stride[0] == 1);
  CHECK(table.layout.stride[1] == static_cast<std::size_t>(table.dimensions[0]));
  CHECK(table.layout.stride[2] == static_cast<std::size_t>(table.dimensions[0] * table.dimensions[1]));

  std::filesystem::remove(eosPath);
}

TEST_CASE("MakeDeviceCopy works for WeakLibEosTable", "[hdf5][weaklib][gpu]")
{
  using namespace WeakLibReader;
  AmrexGuard amrex{};

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

  // Verify axis data round-trips correctly
  for (int dim = 0; dim < 3; ++dim) {
    auto hostCopy = test_helpers::CopyDeviceToHost(deviceTable.axisStorage[dim]);
    REQUIRE(hostCopy.size() == hostTable.axisStorage[dim].size());
    for (std::size_t i = 0; i < hostCopy.size(); ++i) {
      CHECK(hostCopy[i] == hostTable.axisStorage[dim][i]);
    }
  }

  // Verify variable data round-trips correctly
  for (int iVar = 0; iVar < hostTable.nVariables; ++iVar) {
    auto hostCopy = test_helpers::CopyDeviceToHost(deviceTable.variables[iVar]);
    REQUIRE(hostCopy.size() == hostTable.variables[iVar].size());
    for (std::size_t i = 0; i < hostCopy.size(); ++i) {
      CHECK(hostCopy[i] == hostTable.variables[iVar][i]);
    }
  }

  // Verify repaired mask round-trips correctly
  {
    auto hostCopy = test_helpers::CopyDeviceToHost(deviceTable.repaired);
    REQUIRE(hostCopy.size() == hostTable.repaired.size());
    for (std::size_t i = 0; i < hostCopy.size(); ++i) {
      CHECK(hostCopy[i] == hostTable.repaired[i]);
    }
  }

  std::filesystem::remove(eosPath);
}

TEST_CASE("LoadWeakLibEosTableFullParallel loads complete table", "[weaklib][eos][parallel]")
{
  using namespace WeakLibReader;
  AmrexGuard amrex{};

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
  for (int dim = 0; dim < 3; ++dim) {
    CHECK(table.axes[dim].scale == seqTable.axes[dim].scale);
  }

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
    const double* parData = table.variables[iVar].data();
    const double* seqData = seqTable.variables[iVar].data();
    for (std::size_t i = 0; i < varSize; ++i) {
      CHECK(parData[i] == seqData[i]);
    }
  }

  // Verify repaired mask matches
  const int* parRepaired = table.repaired.data();
  const int* seqRepaired = seqTable.repaired.data();
  for (std::size_t i = 0; i < varSize; ++i) {
    CHECK(parRepaired[i] == seqRepaired[i]);
  }

  std::filesystem::remove(eosPath);
}

TEST_CASE("LoadWeakLibEosTableFullParallel fails for nonexistent file", "[weaklib][eos][parallel]")
{
  using namespace WeakLibReader;
  AmrexGuard amrex{};
  detail::ScopedH5ErrorSuppressor silencer{};

  WeakLibEosTable table;
  const Hdf5LoadStatus status = LoadWeakLibEosTableFullParallel("/nonexistent/path.h5", table);

  CHECK(status == Hdf5LoadStatus::FileOpenFailed);
}

TEST_CASE("LoadWeakLibEosTableFull rejects C-ordered dependent variables", "[weaklib][eos][validation]")
{
  using namespace WeakLibReader;
  AmrexGuard amrex{};

  const std::filesystem::path eosPath =
      std::filesystem::temp_directory_path() / "weaklibreader_eos_c_order.h5";
  CreateWeakLibEosTestFileFullCOrder(eosPath);

  WeakLibEosTable table;
  const auto status = LoadWeakLibEosTableFull(eosPath.string(), table);
  CHECK(status == Hdf5LoadStatus::DatasetReadFailed);

  std::filesystem::remove(eosPath);
}
