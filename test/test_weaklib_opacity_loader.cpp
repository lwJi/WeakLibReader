#define SIMPLE_CATCH_NO_MAIN
#include <catch2/catch_test_macros.hpp>

#include "hdf5/WeakLibReader_Hdf5Loader.hpp"
#include "hdf5/WeakLibReader_Hdf5Types.hpp"
#include "test_amrex_guard.hpp"
#include "test_hdf5_helpers.hpp"

#include <AMReX_GpuContainers.H>
#include <hdf5.h>

#include <array>
#include <filesystem>
#include <string>
#include <vector>

namespace {

using namespace TestHelpers;

constexpr int TestNE = 4;
constexpr int TestNRho = 2;
constexpr int TestNT = 3;
constexpr int TestNYe = 2;
constexpr int TestNOpacities = 2;
constexpr int TestNMoments = 2;
constexpr int TestNEta = 3;

double Encode5dIndexValue(int i0, int i1, int i2, int i3, int i4,
                          double base = 0.0)
{
  return base +
         static_cast<double>(i0) +
         10.0 * static_cast<double>(i1) +
         100.0 * static_cast<double>(i2) +
         1000.0 * static_cast<double>(i3) +
         10000.0 * static_cast<double>(i4);
}

std::vector<double> BuildFortranOrdered5dData(const std::array<int, 5>& cOrderDims,
                                              double base = 0.0)
{
  const std::size_t totalSize =
      static_cast<std::size_t>(cOrderDims[0]) *
      static_cast<std::size_t>(cOrderDims[1]) *
      static_cast<std::size_t>(cOrderDims[2]) *
      static_cast<std::size_t>(cOrderDims[3]) *
      static_cast<std::size_t>(cOrderDims[4]);

  std::vector<double> data;
  data.reserve(totalSize);

  // Build linear data with i0 fastest, matching expected C-order indexing.
  for (int i4 = 0; i4 < cOrderDims[4]; ++i4) {
    for (int i3 = 0; i3 < cOrderDims[3]; ++i3) {
      for (int i2 = 0; i2 < cOrderDims[2]; ++i2) {
        for (int i1 = 0; i1 < cOrderDims[1]; ++i1) {
          for (int i0 = 0; i0 < cOrderDims[0]; ++i0) {
            data.push_back(Encode5dIndexValue(i0, i1, i2, i3, i4, base));
          }
        }
      }
    }
  }

  return data;
}

void WriteOpacityGridGroup(hid_t file,
                           const std::string& groupName,
                           const std::string& name,
                           const std::string& unit,
                           const std::vector<double>& values,
                           int logInterp)
{
  hid_t group = H5Gcreate(file, groupName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(group >= 0);
  WriteStringArrayDataset(group, "Name", {name});
  WriteStringArrayDataset(group, "Unit", {unit});
  WriteIntArrayDataset(group, "nPoints", {static_cast<int>(values.size())});
  WriteIntArrayDataset(group, "LogInterp", {logInterp});
  WriteDoubleArrayDataset(group, "Values", values);
  H5Gclose(group);
}

void WriteThermoStateGroup(hid_t file)
{
  hid_t thermoGroup = H5Gcreate(file, "ThermoState", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(thermoGroup >= 0);
  WriteIntArrayDataset(thermoGroup, "LogInterp", {1, 1, 0});
  WriteIntArrayDataset(thermoGroup, "Dimensions", {TestNRho, TestNT, TestNYe});
  WriteStringArrayDataset(thermoGroup, "Names",
                          {"Density", "Temperature", "Electron Fraction"});
  WriteStringArrayDataset(thermoGroup, "Units",
                          {"g/cm^3", "MeV", "dimensionless"});
  WriteDoubleArrayDataset(thermoGroup, "Density", {1.0e3, 1.0e6});
  WriteDoubleArrayDataset(thermoGroup, "Temperature", {0.5, 1.0, 2.0});
  WriteDoubleArrayDataset(thermoGroup, "Electron Fraction", {0.1, 0.2});
  H5Gclose(thermoGroup);
}

void WriteOpacityBaseGroups(hid_t file, bool includeEtaGrid)
{
  WriteOpacityGridGroup(file, "EnergyGrid", "Test Energy", "MeV",
                        {1.0, 2.0, 3.0, 4.0}, 0);
  WriteThermoStateGroup(file);
  if (includeEtaGrid) {
    WriteOpacityGridGroup(file, "EtaGrid", "Test Eta", "dimensionless",
                          {0.1, 0.2, 0.3}, 0);
  }
}

void CreateWeakLibEmAbTestFile(const std::filesystem::path& filePath)
{
  constexpr int nE = TestNE;
  constexpr int nRho = TestNRho;
  constexpr int nT = TestNT;
  constexpr int nYe = TestNYe;
  constexpr int nOpacities = TestNOpacities;

  hid_t file = H5Fcreate(filePath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(file >= 0);

  WriteOpacityBaseGroups(file, false);

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

void CreateWeakLibIsoTestFile(const std::filesystem::path& filePath)
{
  constexpr int nMoments = TestNMoments;
  constexpr int nOpacities = TestNOpacities;
  constexpr double IsoNuBase = 100000.0;
  constexpr double IsoNuBarBase = 200000.0;

  hid_t file = H5Fcreate(filePath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(file >= 0);

  WriteOpacityBaseGroups(file, false);

  hid_t group = H5Gcreate(file, "Scat_Iso_Kernels", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(group >= 0);
  WriteIntArrayDataset(group, "nOpacities", {nOpacities});
  WriteIntArrayDataset(group, "nMoments", {nMoments});
  WriteStringArrayDataset(group, "Units", {"1/cm", "1/cm"});

  const std::array<hsize_t, 2> offsetDims = {
      static_cast<hsize_t>(nMoments),
      static_cast<hsize_t>(nOpacities)};
  std::vector<double> offsets(static_cast<std::size_t>(nMoments) * nOpacities, 0.0);
  WriteDoubleArray2dDataset(group, "Offsets", offsetDims, offsets);

  const std::array<int, 5> kernelDimsC = {
      TestNE, nMoments, TestNRho, TestNT, TestNYe};
  const std::array<hsize_t, 5> kernelDims = {
      static_cast<hsize_t>(kernelDimsC[4]),
      static_cast<hsize_t>(kernelDimsC[3]),
      static_cast<hsize_t>(kernelDimsC[2]),
      static_cast<hsize_t>(kernelDimsC[1]),
      static_cast<hsize_t>(kernelDimsC[0])};
  const std::vector<double> nuData =
      BuildFortranOrdered5dData(kernelDimsC, IsoNuBase);
  const std::vector<double> nuBarData =
      BuildFortranOrdered5dData(kernelDimsC, IsoNuBarBase);
  WriteDoubleArray5dDataset(group, "Electron Neutrino", kernelDims, nuData);
  WriteDoubleArray5dDataset(group, "Electron Antineutrino", kernelDims, nuBarData);

  H5Gclose(group);
  H5Fclose(file);
}

// Write scattering kernel metadata (nOpacities, nMoments, Units, Offsets) into an HDF5 group.
void WriteScatKernelMetadata(hid_t group, int nOpacities, int nMoments,
                             const std::vector<std::string>& units)
{
  WriteIntArrayDataset(group, "nOpacities", {nOpacities});
  WriteIntArrayDataset(group, "nMoments", {nMoments});
  WriteStringArrayDataset(group, "Units", units);

  const std::array<hsize_t, 2> offsetDims = {
      static_cast<hsize_t>(nMoments),
      static_cast<hsize_t>(nOpacities)};
  std::vector<double> offsets(static_cast<std::size_t>(nMoments) * nOpacities, 0.0);
  WriteDoubleArray2dDataset(group, "Offsets", offsetDims, offsets);
}

// Write a 5D kernel dataset from C-order dimensions (stored Fortran-reversed in HDF5).
void WriteKernel5d(hid_t group, const char* name,
                   const std::array<int, 5>& cDims, double base)
{
  const std::array<hsize_t, 5> hdfDims = {
      static_cast<hsize_t>(cDims[4]),
      static_cast<hsize_t>(cDims[3]),
      static_cast<hsize_t>(cDims[2]),
      static_cast<hsize_t>(cDims[1]),
      static_cast<hsize_t>(cDims[0])};
  const std::vector<double> data = BuildFortranOrdered5dData(cDims, base);
  WriteDoubleArray5dDataset(group, name, hdfDims, data);
}

// NES and Pair share the same structure: eta grid, 2 moments, 1 opacity, dataset name "Kernels".
void CreateWeakLibEtaScatTestFile(const std::filesystem::path& filePath,
                                  const char* groupName, double base)
{
  hid_t file = H5Fcreate(filePath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(file >= 0);

  WriteOpacityBaseGroups(file, true);

  hid_t group = H5Gcreate(file, groupName, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(group >= 0);
  WriteScatKernelMetadata(group, 1, 2, {"1/cm"});
  WriteKernel5d(group, "Kernels", {TestNE, TestNE, 2, TestNT, TestNEta}, base);

  H5Gclose(group);
  H5Fclose(file);
}

void CreateWeakLibNESTestFile(const std::filesystem::path& filePath)
{
  CreateWeakLibEtaScatTestFile(filePath, "Scat_NES_Kernels", 300000.0);
}

void CreateWeakLibPairTestFile(const std::filesystem::path& filePath)
{
  CreateWeakLibEtaScatTestFile(filePath, "Scat_Pair_Kernels", 400000.0);
}

void CreateWeakLibBremTestFile(const std::filesystem::path& filePath)
{
  hid_t file = H5Fcreate(filePath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(file >= 0);

  WriteOpacityBaseGroups(file, false);

  hid_t group = H5Gcreate(file, "Scat_Brem_Kernels", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(group >= 0);
  WriteScatKernelMetadata(group, 1, 1, {"1/cm"});
  WriteKernel5d(group, "S_sigma", {TestNE, TestNE, 1, TestNRho, TestNT}, 500000.0);

  H5Gclose(group);
  H5Fclose(file);
}

} // namespace

TEST_CASE("LoadWeakLibEmAbTable loads legacy EmAb opacity table", "[weaklib][opacity][emab]")
{
  using namespace WeakLibReader;
  AmrexGuard amrex{};

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
  using namespace WeakLibReader;
  AmrexGuard amrex{};

  const std::filesystem::path filePath =
      std::filesystem::temp_directory_path() / "weaklibreader_opacity_emab_full.h5";
  CreateWeakLibEmAbTestFile(filePath);

  WeakLibOpacityTable table;
  auto status = LoadWeakLibOpacityTableFull(table, filePath.string());

  std::filesystem::remove(filePath);

  REQUIRE(status == Hdf5LoadStatus::Success);
  REQUIRE(table.HasEmAb());

  // Check EnergyGrid
  CHECK(table.energyGrid.nPoints == TestNE);
  CHECK(table.energyGrid.scale == AxisScale::Linear);

  // Check ThermoState
  CHECK(table.thermoState.dimensions[0] == TestNRho);  // nRho
  CHECK(table.thermoState.dimensions[1] == TestNT);    // nT
  CHECK(table.thermoState.dimensions[2] == TestNYe);   // nYe

  // Check EmAb
  CHECK(table.emAb.nOpacities == TestNOpacities);
  CHECK(table.emAb.dimensions[0] == TestNE);    // nE
  CHECK(table.emAb.dimensions[1] == TestNRho);  // nRho
  CHECK(table.emAb.dimensions[2] == TestNT);    // nT
  CHECK(table.emAb.dimensions[3] == TestNYe);   // nYe

  CHECK(table.emAb.names[0] == "Electron Neutrino");
  CHECK(table.emAb.names[1] == "Electron Antineutrino");
}

TEST_CASE("LoadWeakLibOpacityTableFull loads Iso table", "[hdf5][weaklib][opacity]")
{
  using namespace WeakLibReader;
  constexpr double IsoNuBase = 100000.0;
  constexpr double IsoNuBarBase = 200000.0;

  AmrexGuard amrex{};
  const std::filesystem::path filePath =
      std::filesystem::temp_directory_path() / "weaklibreader_opacity_iso_full.h5";
  CreateWeakLibIsoTestFile(filePath);

  WeakLibOpacityTable table;
  auto status = LoadWeakLibOpacityTableFull(table, "", filePath.string());

  std::filesystem::remove(filePath);

  REQUIRE(status == Hdf5LoadStatus::Success);
  REQUIRE(table.HasScatIso());

  CHECK(table.scatIso.nOpacities == TestNOpacities);
  CHECK(table.scatIso.nMoments == TestNMoments);
  CHECK(table.scatIso.dimensions[0] == TestNE);    // nE
  CHECK(table.scatIso.dimensions[1] == TestNMoments);  // nMom
  CHECK(table.scatIso.dimensions[2] == TestNRho);  // nRho
  CHECK(table.scatIso.dimensions[3] == TestNT);    // nT
  CHECK(table.scatIso.dimensions[4] == TestNYe);   // nYe

  const double* nuData = table.scatIso.KernelData(0);
  const double* nuBarData = table.scatIso.KernelData(1);
  REQUIRE(nuData != nullptr);
  REQUIRE(nuBarData != nullptr);

  const auto& layout = table.scatIso.layout;
  CHECK(nuData[layout.Offset(0, 0, 0, 0, 0)] ==
        Catch::Approx(Encode5dIndexValue(0, 0, 0, 0, 0, IsoNuBase)));
  CHECK(nuData[layout.Offset(3, 1, 1, 2, 1)] ==
        Catch::Approx(Encode5dIndexValue(3, 1, 1, 2, 1, IsoNuBase)));
  CHECK(nuBarData[layout.Offset(2, 0, 1, 1, 1)] ==
        Catch::Approx(Encode5dIndexValue(2, 0, 1, 1, 1, IsoNuBarBase)));
}

TEST_CASE("LoadWeakLibOpacityTableFull loads NES table", "[hdf5][weaklib][opacity]")
{
  using namespace WeakLibReader;
  constexpr double NesBase = 300000.0;

  AmrexGuard amrex{};
  const std::filesystem::path filePath =
      std::filesystem::temp_directory_path() / "weaklibreader_opacity_nes_full.h5";
  CreateWeakLibNESTestFile(filePath);

  WeakLibOpacityTable table;
  auto status = LoadWeakLibOpacityTableFull(table, "", "", filePath.string());

  std::filesystem::remove(filePath);

  REQUIRE(status == Hdf5LoadStatus::Success);
  REQUIRE(table.HasScatNES());

  // Check EtaGrid was loaded
  CHECK(table.etaGrid.nPoints == TestNEta);

  CHECK(table.scatNES.nOpacities == 1);
  CHECK(table.scatNES.nMoments == 2);
  CHECK(table.scatNES.dimensions[0] == TestNE);    // nE_in
  CHECK(table.scatNES.dimensions[1] == TestNE);    // nE_out
  CHECK(table.scatNES.dimensions[2] == 2);          // nMom
  CHECK(table.scatNES.dimensions[3] == TestNT);    // nT
  CHECK(table.scatNES.dimensions[4] == TestNEta);  // nEta

  const double* kernel = table.scatNES.KernelData();
  REQUIRE(kernel != nullptr);

  const auto& layout = table.scatNES.layout;
  CHECK(kernel[layout.Offset(0, 0, 0, 0, 0)] ==
        Catch::Approx(Encode5dIndexValue(0, 0, 0, 0, 0, NesBase)));
  CHECK(kernel[layout.Offset(3, 2, 1, 2, 2)] ==
        Catch::Approx(Encode5dIndexValue(3, 2, 1, 2, 2, NesBase)));
}

TEST_CASE("LoadWeakLibOpacityTableFull loads Pair table", "[hdf5][weaklib][opacity]")
{
  using namespace WeakLibReader;
  constexpr double PairBase = 400000.0;

  AmrexGuard amrex{};
  const std::filesystem::path filePath =
      std::filesystem::temp_directory_path() / "weaklibreader_opacity_pair_full.h5";
  CreateWeakLibPairTestFile(filePath);

  WeakLibOpacityTable table;
  auto status = LoadWeakLibOpacityTableFull(table, "", "", "", filePath.string());

  std::filesystem::remove(filePath);

  REQUIRE(status == Hdf5LoadStatus::Success);
  REQUIRE(table.HasScatPair());

  CHECK(table.scatPair.nOpacities == 1);
  CHECK(table.scatPair.nMoments == 2);

  const double* kernel = table.scatPair.KernelData();
  REQUIRE(kernel != nullptr);

  const auto& layout = table.scatPair.layout;
  CHECK(kernel[layout.Offset(1, 0, 0, 0, 0)] ==
        Catch::Approx(Encode5dIndexValue(1, 0, 0, 0, 0, PairBase)));
  CHECK(kernel[layout.Offset(3, 3, 1, 2, 2)] ==
        Catch::Approx(Encode5dIndexValue(3, 3, 1, 2, 2, PairBase)));
}

TEST_CASE("LoadWeakLibOpacityTableFull loads Brem table", "[hdf5][weaklib][opacity]")
{
  using namespace WeakLibReader;
  constexpr double BremBase = 500000.0;

  AmrexGuard amrex{};
  const std::filesystem::path filePath =
      std::filesystem::temp_directory_path() / "weaklibreader_opacity_brem_full.h5";
  CreateWeakLibBremTestFile(filePath);

  WeakLibOpacityTable table;
  auto status = LoadWeakLibOpacityTableFull(table, "", "", "", "", filePath.string());

  std::filesystem::remove(filePath);

  REQUIRE(status == Hdf5LoadStatus::Success);
  REQUIRE(table.HasScatBrem());

  CHECK(table.scatBrem.nOpacities == 1);
  CHECK(table.scatBrem.nMoments == 1);
  CHECK(table.scatBrem.name == "S_sigma");

  const double* kernel = table.scatBrem.KernelData();
  REQUIRE(kernel != nullptr);

  const auto& layout = table.scatBrem.layout;
  CHECK(kernel[layout.Offset(0, 0, 0, 0, 0)] ==
        Catch::Approx(Encode5dIndexValue(0, 0, 0, 0, 0, BremBase)));
  CHECK(kernel[layout.Offset(3, 1, 0, 1, 2)] ==
        Catch::Approx(Encode5dIndexValue(3, 1, 0, 1, 2, BremBase)));
}

TEST_CASE("LoadWeakLibOpacityTableFull loads multiple types", "[hdf5][weaklib][opacity]")
{
  using namespace WeakLibReader;
  AmrexGuard amrex{};
  const std::filesystem::path emabPath =
      std::filesystem::temp_directory_path() / "weaklibreader_opacity_emab_multi.h5";
  const std::filesystem::path isoPath =
      std::filesystem::temp_directory_path() / "weaklibreader_opacity_iso_multi.h5";
  CreateWeakLibEmAbTestFile(emabPath);
  CreateWeakLibIsoTestFile(isoPath);

  WeakLibOpacityTable table;
  auto status = LoadWeakLibOpacityTableFull(table, emabPath.string(), isoPath.string());

  std::filesystem::remove(emabPath);
  std::filesystem::remove(isoPath);

  REQUIRE(status == Hdf5LoadStatus::Success);
  CHECK(table.HasEmAb());
  CHECK(table.HasScatIso());
  CHECK_FALSE(table.HasScatNES());
  CHECK_FALSE(table.HasScatPair());
  CHECK_FALSE(table.HasScatBrem());
}

TEST_CASE("LoadWeakLibOpacityTableFull returns error for missing files", "[hdf5][weaklib][opacity]")
{
  using namespace WeakLibReader;
  AmrexGuard amrex{};
  detail::ScopedH5ErrorSuppressor silencer{};

  WeakLibOpacityTable table;
  auto status = LoadWeakLibOpacityTableFull(table, "nonexistent.h5");

  CHECK(status == Hdf5LoadStatus::FileOpenFailed);
}

TEST_CASE("LoadWeakLibOpacityTableFull returns error for no files", "[hdf5][weaklib][opacity]")
{
  using namespace WeakLibReader;
  AmrexGuard amrex{};
  detail::ScopedH5ErrorSuppressor silencer{};

  WeakLibOpacityTable table;
  auto status = LoadWeakLibOpacityTableFull(table);

  CHECK(status == Hdf5LoadStatus::FileOpenFailed);
}

TEST_CASE("MakeDeviceCopy creates device copy of opacity table", "[hdf5][weaklib][opacity][gpu]")
{
  using namespace WeakLibReader;
  AmrexGuard amrex{};
  const std::filesystem::path filePath =
      std::filesystem::temp_directory_path() / "weaklibreader_opacity_emab_device.h5";
  CreateWeakLibEmAbTestFile(filePath);

  WeakLibOpacityTable hostTable;
  auto status = LoadWeakLibOpacityTableFull(hostTable, filePath.string());
  REQUIRE(status == Hdf5LoadStatus::Success);

  std::filesystem::remove(filePath);

  auto deviceTable = MakeDeviceCopy(hostTable);

  CHECK(deviceTable.HasEmAb());
  CHECK(deviceTable.emAb.nOpacities == hostTable.emAb.nOpacities);
  CHECK(deviceTable.emAb.dimensions == hostTable.emAb.dimensions);
  CHECK(deviceTable.energyGrid.nPoints == hostTable.energyGrid.nPoints);
}

TEST_CASE("LoadWeakLibOpacityTableFullParallel loads EmAb and Iso",
          "[weaklib][opacity][parallel]")
{
  using namespace WeakLibReader;
  AmrexGuard amrex{};

  const auto emabPath =
      std::filesystem::temp_directory_path() / "weaklibreader_opacity_emab_parallel.h5";
  const auto isoPath =
      std::filesystem::temp_directory_path() / "weaklibreader_opacity_iso_parallel.h5";
  CreateWeakLibEmAbTestFile(emabPath);
  CreateWeakLibIsoTestFile(isoPath);

  // Load via parallel
  WeakLibOpacityTable parTable;
  auto parStatus = LoadWeakLibOpacityTableFullParallel(
      parTable, emabPath.string(), isoPath.string());

  // Load via serial for comparison
  WeakLibOpacityTable seqTable;
  auto seqStatus = LoadWeakLibOpacityTableFull(
      seqTable, emabPath.string(), isoPath.string());

  std::filesystem::remove(emabPath);
  std::filesystem::remove(isoPath);

  REQUIRE(parStatus == Hdf5LoadStatus::Success);
  REQUIRE(seqStatus == Hdf5LoadStatus::Success);

  // Verify sub-table presence
  CHECK(parTable.HasEmAb() == seqTable.HasEmAb());
  CHECK(parTable.HasScatIso() == seqTable.HasScatIso());
  CHECK_FALSE(parTable.HasScatNES());
  CHECK_FALSE(parTable.HasScatPair());
  CHECK_FALSE(parTable.HasScatBrem());

  // Verify energyGrid
  CHECK(parTable.energyGrid.nPoints == seqTable.energyGrid.nPoints);
  CHECK(parTable.energyGrid.scale == seqTable.energyGrid.scale);
  CHECK(parTable.energyGrid.name == seqTable.energyGrid.name);
  CHECK(parTable.energyGrid.unit == seqTable.energyGrid.unit);
  for (int i = 0; i < parTable.energyGrid.nPoints; ++i) {
    CHECK(parTable.energyGrid.values[i] == seqTable.energyGrid.values[i]);
  }

  // Verify thermoState
  CHECK(parTable.thermoState.dimensions == seqTable.thermoState.dimensions);
  CHECK(parTable.thermoState.scales == seqTable.thermoState.scales);
  CHECK(parTable.thermoState.names == seqTable.thermoState.names);
  CHECK(parTable.thermoState.units == seqTable.thermoState.units);
  for (int i = 0; i < 3; ++i) {
    REQUIRE(parTable.thermoState.axisStorage[i].size() ==
            seqTable.thermoState.axisStorage[i].size());
    for (std::size_t j = 0; j < parTable.thermoState.axisStorage[i].size(); ++j) {
      CHECK(parTable.thermoState.axisStorage[i][j] ==
            seqTable.thermoState.axisStorage[i][j]);
    }
  }

  // Verify EmAb
  CHECK(parTable.emAb.nOpacities == seqTable.emAb.nOpacities);
  CHECK(parTable.emAb.dimensions == seqTable.emAb.dimensions);
  CHECK(parTable.emAb.offsets == seqTable.emAb.offsets);
  CHECK(parTable.emAb.names == seqTable.emAb.names);
  CHECK(parTable.emAb.units == seqTable.emAb.units);
  const auto emabSize = static_cast<std::size_t>(seqTable.emAb.dimensions[0]) *
                         seqTable.emAb.dimensions[1] *
                         seqTable.emAb.dimensions[2] *
                         seqTable.emAb.dimensions[3];
  for (int s = 0; s < seqTable.emAb.nOpacities; ++s) {
    REQUIRE(parTable.emAb.opacities[s].size() == emabSize);
    for (std::size_t i = 0; i < emabSize; ++i) {
      CHECK(parTable.emAb.opacities[s][i] == seqTable.emAb.opacities[s][i]);
    }
  }

  // Verify ScatIso
  CHECK(parTable.scatIso.nOpacities == seqTable.scatIso.nOpacities);
  CHECK(parTable.scatIso.nMoments == seqTable.scatIso.nMoments);
  CHECK(parTable.scatIso.dimensions == seqTable.scatIso.dimensions);
  CHECK(parTable.scatIso.weak_magnetism_corrections ==
        seqTable.scatIso.weak_magnetism_corrections);
  CHECK(parTable.scatIso.ion_ion_corrections ==
        seqTable.scatIso.ion_ion_corrections);
  CHECK(parTable.scatIso.many_body_corrections ==
        seqTable.scatIso.many_body_corrections);
  CHECK(parTable.scatIso.ga_strange == seqTable.scatIso.ga_strange);
  const auto isoOffsetSize = static_cast<std::size_t>(
      seqTable.scatIso.nOpacities) * seqTable.scatIso.nMoments;
  REQUIRE(parTable.scatIso.offsets.size() == isoOffsetSize);
  for (std::size_t i = 0; i < isoOffsetSize; ++i) {
    CHECK(parTable.scatIso.offsets[i] == seqTable.scatIso.offsets[i]);
  }
  const auto isoDataSize = static_cast<std::size_t>(seqTable.scatIso.dimensions[0]) *
                             seqTable.scatIso.dimensions[1] *
                             seqTable.scatIso.dimensions[2] *
                             seqTable.scatIso.dimensions[3] *
                             seqTable.scatIso.dimensions[4];
  for (int s = 0; s < seqTable.scatIso.nOpacities; ++s) {
    REQUIRE(parTable.scatIso.kernels[s].size() == isoDataSize);
    for (std::size_t i = 0; i < isoDataSize; ++i) {
      CHECK(parTable.scatIso.kernels[s][i] == seqTable.scatIso.kernels[s][i]);
    }
  }
  CHECK(parTable.scatIso.names == seqTable.scatIso.names);
  CHECK(parTable.scatIso.units == seqTable.scatIso.units);
}

TEST_CASE("LoadWeakLibOpacityTableFullParallel fails for nonexistent file",
          "[weaklib][opacity][parallel]")
{
  using namespace WeakLibReader;
  AmrexGuard amrex{};
  detail::ScopedH5ErrorSuppressor silencer{};

  WeakLibOpacityTable table;
  auto status = LoadWeakLibOpacityTableFullParallel(table, "nonexistent.h5");
  CHECK(status == Hdf5LoadStatus::FileOpenFailed);
}
