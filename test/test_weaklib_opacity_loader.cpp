#define SIMPLE_CATCH_NO_MAIN
#include <catch2/catch_test_macros.hpp>

#include "WeakLibReader_Hdf5Loader.hpp"
#include "WeakLibReader_Hdf5Types.hpp"

#include <AMReX.H>
#include <AMReX_GpuContainers.H>
#include <hdf5.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

using namespace WeakLibReader;

namespace {

// Test file paths (adjust as needed for your environment)
const std::string kTestDataDir = "/Users/liwei/docker-workspace/repos/weaklib-tables/SFHo/LowRes/";
const std::string kEmAbFile = kTestDataDir + "wl-Op-SFHo-15-25-50-E40-B85-AbEm.h5";
const std::string kIsoFile = kTestDataDir + "wl-Op-SFHo-15-25-50-E40-B85-Iso.h5";
const std::string kNESFile = kTestDataDir + "wl-Op-SFHo-15-25-50-E40-B85-NES.h5";
const std::string kPairFile = kTestDataDir + "wl-Op-SFHo-15-25-50-E40-B85-Pair.h5";
const std::string kBremFile = kTestDataDir + "wl-Op-SFHo-15-25-50-E40-HR98-Brem.h5";

/// Initialize AMReX if not already initialized.
void EnsureAmrexInitialized()
{
  if (!amrex::Initialized()) {
    int argc = 0;
    char** argv = nullptr;
    amrex::Initialize(argc, argv);
    std::atexit([]() {
      if (amrex::Initialized()) {
        amrex::Finalize();
      }
    });
  }
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

void WriteDoubleArray4dDataset(hid_t parent,
                               const std::string& name,
                               const std::array<hsize_t, 4>& dims,
                               const std::vector<double>& values)
{
  hid_t space = H5Screate_simple(4, dims.data(), nullptr);
  hid_t dataset = H5Dcreate(parent, name.c_str(), H5T_IEEE_F64LE, space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(dataset >= 0);
  H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data());
  H5Dclose(dataset);
  H5Sclose(space);
}

void CreateWeakLibEmAbTestFile(const std::filesystem::path& filePath)
{
  constexpr int nE = 4;
  constexpr int nRho = 2;
  constexpr int nT = 3;
  constexpr int nYe = 2;
  constexpr int nOpacities = 2;

  hid_t file = H5Fcreate(filePath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(file >= 0);

  hid_t energyGroup = H5Gcreate(file, "EnergyGrid", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(energyGroup >= 0);
  WriteStringArrayDataset(energyGroup, "Name", {"Test Energy"});
  WriteStringArrayDataset(energyGroup, "Unit", {"MeV"});
  WriteIntArrayDataset(energyGroup, "nPoints", {nE});
  WriteIntArrayDataset(energyGroup, "LogInterp", {0});
  WriteDoubleArrayDataset(energyGroup, "Values", {1.0, 2.0, 3.0, 4.0});
  H5Gclose(energyGroup);

  hid_t thermoGroup = H5Gcreate(file, "ThermoState", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(thermoGroup >= 0);
  WriteIntArrayDataset(thermoGroup, "LogInterp", {1, 1, 0});
  WriteIntArrayDataset(thermoGroup, "Dimensions", {nRho, nT, nYe});
  WriteStringArrayDataset(thermoGroup, "Names",
                          {"Density", "Temperature", "Electron Fraction"});
  WriteStringArrayDataset(thermoGroup, "Units",
                          {"g/cm^3", "MeV", "dimensionless"});
  WriteDoubleArrayDataset(thermoGroup, "Density", {1.0e3, 1.0e6});
  WriteDoubleArrayDataset(thermoGroup, "Temperature", {0.5, 1.0, 2.0});
  WriteDoubleArrayDataset(thermoGroup, "Electron Fraction", {0.1, 0.2});
  H5Gclose(thermoGroup);

  hid_t emabGroup =
      H5Gcreate(file, "EmAb_CorrectedAbsorption", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(emabGroup >= 0);
  WriteIntArrayDataset(emabGroup, "nOpacities", {nOpacities});
  WriteStringArrayDataset(emabGroup, "Units", {"1/cm", "1/cm"});
  WriteDoubleArrayDataset(emabGroup, "Offsets", {0.0, 0.1});

  const std::array<hsize_t, 4> dims = {
      static_cast<hsize_t>(nYe),
      static_cast<hsize_t>(nT),
      static_cast<hsize_t>(nRho),
      static_cast<hsize_t>(nE)};
  const std::size_t totalSize =
      static_cast<std::size_t>(nE) * nRho * nT * nYe;

  std::vector<double> data(totalSize, 0.0);
  std::size_t idx = 0;
  for (int ye = 0; ye < nYe; ++ye) {
    for (int temp = 0; temp < nT; ++temp) {
      for (int rho = 0; rho < nRho; ++rho) {
        for (int e = 0; e < nE; ++e) {
          data[idx++] =
              1.0 + 10.0 * static_cast<double>(ye) +
              1.0 * static_cast<double>(temp) +
              0.1 * static_cast<double>(rho) +
              0.01 * static_cast<double>(e);
        }
      }
    }
  }

  WriteDoubleArray4dDataset(emabGroup, "Electron Neutrino", dims, data);
  for (double& value : data) {
    value += 100.0;
  }
  WriteDoubleArray4dDataset(emabGroup, "Electron Antineutrino", dims, data);

  H5Gclose(emabGroup);
  H5Fclose(file);
}

} // namespace

TEST_CASE("LoadWeakLibEmAbTable loads legacy EmAb opacity table", "[weaklib][opacity][emab]")
{
  EnsureAmrexInitialized();

  const std::filesystem::path filePath =
      std::filesystem::temp_directory_path() / "weaklibreader_emab_legacy.h5";
  CreateWeakLibEmAbTestFile(filePath);

  hid_t file = H5Fopen(filePath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  REQUIRE(file >= 0);

  // Load energy grid
  WeakLibOpacityGrid energyGrid;
  auto status = detail::LoadWeakLibOpacityGrid(file, "EnergyGrid", energyGrid);
  REQUIRE(status == Hdf5LoadStatus::Success);
  CHECK(energyGrid.nPoints == 4);
  CHECK(energyGrid.name == "Test Energy");
  CHECK(energyGrid.unit == "MeV");

  // Load thermo state
  WeakLibOpacityThermoState thermoState;
  status = detail::LoadWeakLibOpacityThermoState(file, thermoState);
  REQUIRE(status == Hdf5LoadStatus::Success);
  CHECK(thermoState.dimensions[0] == 2);  // nRho
  CHECK(thermoState.dimensions[1] == 3);  // nT
  CHECK(thermoState.dimensions[2] == 2);  // nYe
  CHECK(thermoState.scales[0] == AxisScale::Log10);
  CHECK(thermoState.scales[1] == AxisScale::Log10);
  CHECK(thermoState.scales[2] == AxisScale::Linear);

  // Load EmAb table
  WeakLibEmAbTable emAb;
  status = LoadWeakLibEmAbTable(file, emAb, energyGrid, thermoState);
  REQUIRE(status == Hdf5LoadStatus::Success);

  H5Fclose(file);
  std::filesystem::remove(filePath);

  // Verify EmAb table was loaded correctly
  CHECK(emAb.IsLoaded());
  CHECK(emAb.nOpacities == 2);

  // Check dimensions: [nE, nRho, nT, nYe]
  CHECK(emAb.dimensions[0] == 4);  // nE
  CHECK(emAb.dimensions[1] == 2);  // nRho
  CHECK(emAb.dimensions[2] == 3);  // nT
  CHECK(emAb.dimensions[3] == 2);  // nYe

  // Legacy format detection
  CHECK(emAb.parameters.IsLegacy());

  // Species names
  CHECK(emAb.names[0] == "Electron Neutrino");
  CHECK(emAb.names[1] == "Electron Antineutrino");

  // Data pointers should be valid
  const double* data0 = emAb.OpacityData(0);
  const double* data1 = emAb.OpacityData(1);
  REQUIRE(data0 != nullptr);
  REQUIRE(data1 != nullptr);

  // Layout should be computed
  CHECK(emAb.layout.stride[0] == 1);  // nE stride
  CHECK(emAb.layout.stride[1] == static_cast<std::size_t>(emAb.dimensions[0]));  // nRho stride
}

TEST_CASE("LoadWeakLibOpacityTableFull loads EmAb table", "[hdf5][weaklib][opacity]")
{
  EnsureAmrexInitialized();

  // Require test data file exists
  REQUIRE(std::filesystem::exists(kEmAbFile));

  WeakLibOpacityTable table;
  auto status = LoadWeakLibOpacityTableFull(table, kEmAbFile);

  REQUIRE(status == Hdf5LoadStatus::Success);
  REQUIRE(table.HasEmAb());

  // Check EnergyGrid
  CHECK(table.energyGrid.nPoints == 40);
  CHECK(table.energyGrid.scale == AxisScale::Log10);

  // Check ThermoState
  CHECK(table.thermoState.dimensions[0] == 185);  // nRho
  CHECK(table.thermoState.dimensions[1] == 81);   // nT
  CHECK(table.thermoState.dimensions[2] == 30);   // nYe

  // Check EmAb
  CHECK(table.emAb.nOpacities == 2);
  CHECK(table.emAb.dimensions[0] == 40);   // nE
  CHECK(table.emAb.dimensions[1] == 185);  // nRho
  CHECK(table.emAb.dimensions[2] == 81);   // nT
  CHECK(table.emAb.dimensions[3] == 30);   // nYe

  CHECK(table.emAb.names[0] == "Electron Neutrino");
  CHECK(table.emAb.names[1] == "Electron Antineutrino");
}

TEST_CASE("LoadWeakLibOpacityTableFull loads Iso table", "[hdf5][weaklib][opacity]")
{
  EnsureAmrexInitialized();
  REQUIRE(std::filesystem::exists(kIsoFile));

  WeakLibOpacityTable table;
  auto status = LoadWeakLibOpacityTableFull(table, "", kIsoFile);

  REQUIRE(status == Hdf5LoadStatus::Success);
  REQUIRE(table.HasScatIso());

  CHECK(table.scatIso.nOpacities == 2);
  CHECK(table.scatIso.nMoments == 2);
  CHECK(table.scatIso.dimensions[0] == 40);   // nE
  CHECK(table.scatIso.dimensions[1] == 2);    // nMom
  CHECK(table.scatIso.dimensions[2] == 185);  // nRho
  CHECK(table.scatIso.dimensions[3] == 81);   // nT
  CHECK(table.scatIso.dimensions[4] == 30);   // nYe
}

TEST_CASE("LoadWeakLibOpacityTableFull loads NES table", "[hdf5][weaklib][opacity]")
{
  EnsureAmrexInitialized();
  REQUIRE(std::filesystem::exists(kNESFile));

  WeakLibOpacityTable table;
  auto status = LoadWeakLibOpacityTableFull(table, "", "", kNESFile);

  REQUIRE(status == Hdf5LoadStatus::Success);
  REQUIRE(table.HasScatNES());

  // Check EtaGrid was loaded
  CHECK(table.etaGrid.nPoints == 60);

  CHECK(table.scatNES.nOpacities == 1);
  CHECK(table.scatNES.nMoments == 4);
  CHECK(table.scatNES.dimensions[0] == 40);  // nE_in
  CHECK(table.scatNES.dimensions[1] == 40);  // nE_out
  CHECK(table.scatNES.dimensions[2] == 4);   // nMom
  CHECK(table.scatNES.dimensions[3] == 81);  // nT
  CHECK(table.scatNES.dimensions[4] == 60);  // nEta
}

TEST_CASE("LoadWeakLibOpacityTableFull loads Pair table", "[hdf5][weaklib][opacity]")
{
  EnsureAmrexInitialized();
  REQUIRE(std::filesystem::exists(kPairFile));

  WeakLibOpacityTable table;
  auto status = LoadWeakLibOpacityTableFull(table, "", "", "", kPairFile);

  REQUIRE(status == Hdf5LoadStatus::Success);
  REQUIRE(table.HasScatPair());

  CHECK(table.scatPair.nOpacities == 1);
  CHECK(table.scatPair.nMoments == 4);
}

TEST_CASE("LoadWeakLibOpacityTableFull loads Brem table", "[hdf5][weaklib][opacity]")
{
  EnsureAmrexInitialized();
  REQUIRE(std::filesystem::exists(kBremFile));

  WeakLibOpacityTable table;
  auto status = LoadWeakLibOpacityTableFull(table, "", "", "", "", kBremFile);

  REQUIRE(status == Hdf5LoadStatus::Success);
  REQUIRE(table.HasScatBrem());

  CHECK(table.scatBrem.nOpacities == 1);
  CHECK(table.scatBrem.nMoments == 1);
  CHECK(table.scatBrem.name == "S_sigma");
}

TEST_CASE("LoadWeakLibOpacityTableFull loads multiple types", "[hdf5][weaklib][opacity]")
{
  EnsureAmrexInitialized();
  REQUIRE(std::filesystem::exists(kEmAbFile));
  REQUIRE(std::filesystem::exists(kIsoFile));

  WeakLibOpacityTable table;
  auto status = LoadWeakLibOpacityTableFull(table, kEmAbFile, kIsoFile);

  REQUIRE(status == Hdf5LoadStatus::Success);
  CHECK(table.HasEmAb());
  CHECK(table.HasScatIso());
  CHECK_FALSE(table.HasScatNES());
  CHECK_FALSE(table.HasScatPair());
  CHECK_FALSE(table.HasScatBrem());
}

TEST_CASE("LoadWeakLibOpacityTableFull returns error for missing files", "[hdf5][weaklib][opacity]")
{
  EnsureAmrexInitialized();

  WeakLibOpacityTable table;
  auto status = LoadWeakLibOpacityTableFull(table, "nonexistent.h5");

  CHECK(status == Hdf5LoadStatus::FileOpenFailed);
}

TEST_CASE("LoadWeakLibOpacityTableFull returns error for no files", "[hdf5][weaklib][opacity]")
{
  EnsureAmrexInitialized();

  WeakLibOpacityTable table;
  auto status = LoadWeakLibOpacityTableFull(table);

  CHECK(status == Hdf5LoadStatus::FileOpenFailed);
}

TEST_CASE("MakeDeviceCopy creates device copy of opacity table", "[hdf5][weaklib][opacity][gpu]")
{
  EnsureAmrexInitialized();
  REQUIRE(std::filesystem::exists(kEmAbFile));

  WeakLibOpacityTable hostTable;
  auto status = LoadWeakLibOpacityTableFull(hostTable, kEmAbFile);
  REQUIRE(status == Hdf5LoadStatus::Success);

  auto deviceTable = MakeDeviceCopy(hostTable);

  CHECK(deviceTable.HasEmAb());
  CHECK(deviceTable.emAb.nOpacities == hostTable.emAb.nOpacities);
  CHECK(deviceTable.emAb.dimensions == hostTable.emAb.dimensions);
  CHECK(deviceTable.energyGrid.nPoints == hostTable.energyGrid.nPoints);
}
