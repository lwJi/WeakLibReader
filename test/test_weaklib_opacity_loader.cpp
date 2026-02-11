#define SIMPLE_CATCH_NO_MAIN
#include <catch2/catch_test_macros.hpp>

#include "hdf5/WeakLibReader_Hdf5Loader.hpp"
#include "hdf5/WeakLibReader_Hdf5Types.hpp"

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

constexpr int kTestNE = 4;
constexpr int kTestNRho = 2;
constexpr int kTestNT = 3;
constexpr int kTestNYe = 2;
constexpr int kTestNOpacities = 2;
constexpr int kTestNMoments = 2;
constexpr int kTestNEta = 3;

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

void WriteDoubleArray2dDataset(hid_t parent,
                               const std::string& name,
                               const std::array<hsize_t, 2>& dims,
                               const std::vector<double>& values)
{
  hid_t space = H5Screate_simple(2, dims.data(), nullptr);
  hid_t dataset = H5Dcreate(parent, name.c_str(), H5T_IEEE_F64LE, space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(dataset >= 0);
  H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data());
  H5Dclose(dataset);
  H5Sclose(space);
}

void WriteDoubleArray5dDataset(hid_t parent,
                               const std::string& name,
                               const std::array<hsize_t, 5>& dims,
                               const std::vector<double>& values)
{
  hid_t space = H5Screate_simple(5, dims.data(), nullptr);
  hid_t dataset = H5Dcreate(parent, name.c_str(), H5T_IEEE_F64LE, space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(dataset >= 0);
  H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data());
  H5Dclose(dataset);
  H5Sclose(space);
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
  WriteIntArrayDataset(thermoGroup, "Dimensions", {kTestNRho, kTestNT, kTestNYe});
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
  constexpr int nE = kTestNE;
  constexpr int nRho = kTestNRho;
  constexpr int nT = kTestNT;
  constexpr int nYe = kTestNYe;
  constexpr int nOpacities = kTestNOpacities;

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
  constexpr int nMoments = kTestNMoments;
  constexpr int nOpacities = kTestNOpacities;
  constexpr double kIsoNuBase = 100000.0;
  constexpr double kIsoNuBarBase = 200000.0;

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
      kTestNE, nMoments, kTestNRho, kTestNT, kTestNYe};
  const std::array<hsize_t, 5> kernelDims = {
      static_cast<hsize_t>(kernelDimsC[4]),
      static_cast<hsize_t>(kernelDimsC[3]),
      static_cast<hsize_t>(kernelDimsC[2]),
      static_cast<hsize_t>(kernelDimsC[1]),
      static_cast<hsize_t>(kernelDimsC[0])};
  const std::vector<double> nuData =
      BuildFortranOrdered5dData(kernelDimsC, kIsoNuBase);
  const std::vector<double> nuBarData =
      BuildFortranOrdered5dData(kernelDimsC, kIsoNuBarBase);
  WriteDoubleArray5dDataset(group, "Electron Neutrino", kernelDims, nuData);
  WriteDoubleArray5dDataset(group, "Electron Antineutrino", kernelDims, nuBarData);

  H5Gclose(group);
  H5Fclose(file);
}

void CreateWeakLibNESTestFile(const std::filesystem::path& filePath)
{
  constexpr int nMoments = 2;
  constexpr int nOpacities = 1;
  constexpr double kNesBase = 300000.0;

  hid_t file = H5Fcreate(filePath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(file >= 0);

  WriteOpacityBaseGroups(file, true);

  hid_t group = H5Gcreate(file, "Scat_NES_Kernels", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(group >= 0);
  WriteIntArrayDataset(group, "nOpacities", {nOpacities});
  WriteIntArrayDataset(group, "nMoments", {nMoments});
  WriteStringArrayDataset(group, "Units", {"1/cm"});

  const std::array<hsize_t, 2> offsetDims = {
      static_cast<hsize_t>(nMoments),
      static_cast<hsize_t>(nOpacities)};
  std::vector<double> offsets(static_cast<std::size_t>(nMoments) * nOpacities, 0.0);
  WriteDoubleArray2dDataset(group, "Offsets", offsetDims, offsets);

  const std::array<int, 5> kernelDimsC = {
      kTestNE, kTestNE, nMoments, kTestNT, kTestNEta};
  const std::array<hsize_t, 5> kernelDims = {
      static_cast<hsize_t>(kernelDimsC[4]),
      static_cast<hsize_t>(kernelDimsC[3]),
      static_cast<hsize_t>(kernelDimsC[2]),
      static_cast<hsize_t>(kernelDimsC[1]),
      static_cast<hsize_t>(kernelDimsC[0])};
  const std::vector<double> data =
      BuildFortranOrdered5dData(kernelDimsC, kNesBase);
  WriteDoubleArray5dDataset(group, "Kernels", kernelDims, data);

  H5Gclose(group);
  H5Fclose(file);
}

void CreateWeakLibPairTestFile(const std::filesystem::path& filePath)
{
  constexpr int nMoments = 2;
  constexpr int nOpacities = 1;
  constexpr double kPairBase = 400000.0;

  hid_t file = H5Fcreate(filePath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(file >= 0);

  WriteOpacityBaseGroups(file, true);

  hid_t group = H5Gcreate(file, "Scat_Pair_Kernels", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(group >= 0);
  WriteIntArrayDataset(group, "nOpacities", {nOpacities});
  WriteIntArrayDataset(group, "nMoments", {nMoments});
  WriteStringArrayDataset(group, "Units", {"1/cm"});

  const std::array<hsize_t, 2> offsetDims = {
      static_cast<hsize_t>(nMoments),
      static_cast<hsize_t>(nOpacities)};
  std::vector<double> offsets(static_cast<std::size_t>(nMoments) * nOpacities, 0.0);
  WriteDoubleArray2dDataset(group, "Offsets", offsetDims, offsets);

  const std::array<int, 5> kernelDimsC = {
      kTestNE, kTestNE, nMoments, kTestNT, kTestNEta};
  const std::array<hsize_t, 5> kernelDims = {
      static_cast<hsize_t>(kernelDimsC[4]),
      static_cast<hsize_t>(kernelDimsC[3]),
      static_cast<hsize_t>(kernelDimsC[2]),
      static_cast<hsize_t>(kernelDimsC[1]),
      static_cast<hsize_t>(kernelDimsC[0])};
  const std::vector<double> data =
      BuildFortranOrdered5dData(kernelDimsC, kPairBase);
  WriteDoubleArray5dDataset(group, "Kernels", kernelDims, data);

  H5Gclose(group);
  H5Fclose(file);
}

void CreateWeakLibBremTestFile(const std::filesystem::path& filePath)
{
  constexpr int nMoments = 1;
  constexpr int nOpacities = 1;
  constexpr double kBremBase = 500000.0;

  hid_t file = H5Fcreate(filePath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(file >= 0);

  WriteOpacityBaseGroups(file, false);

  hid_t group = H5Gcreate(file, "Scat_Brem_Kernels", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(group >= 0);
  WriteIntArrayDataset(group, "nOpacities", {nOpacities});
  WriteIntArrayDataset(group, "nMoments", {nMoments});
  WriteStringArrayDataset(group, "Units", {"1/cm"});

  const std::array<hsize_t, 2> offsetDims = {
      static_cast<hsize_t>(nMoments),
      static_cast<hsize_t>(nOpacities)};
  std::vector<double> offsets(static_cast<std::size_t>(nMoments) * nOpacities, 0.0);
  WriteDoubleArray2dDataset(group, "Offsets", offsetDims, offsets);

  const std::array<int, 5> kernelDimsC = {
      kTestNE, kTestNE, nMoments, kTestNRho, kTestNT};
  const std::array<hsize_t, 5> kernelDims = {
      static_cast<hsize_t>(kernelDimsC[4]),
      static_cast<hsize_t>(kernelDimsC[3]),
      static_cast<hsize_t>(kernelDimsC[2]),
      static_cast<hsize_t>(kernelDimsC[1]),
      static_cast<hsize_t>(kernelDimsC[0])};
  const std::vector<double> data =
      BuildFortranOrdered5dData(kernelDimsC, kBremBase);
  WriteDoubleArray5dDataset(group, "S_sigma", kernelDims, data);

  H5Gclose(group);
  H5Fclose(file);
}

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

  const std::filesystem::path filePath =
      std::filesystem::temp_directory_path() / "weaklibreader_opacity_emab_full.h5";
  CreateWeakLibEmAbTestFile(filePath);

  WeakLibOpacityTable table;
  auto status = LoadWeakLibOpacityTableFull(table, filePath.string());

  std::filesystem::remove(filePath);

  REQUIRE(status == Hdf5LoadStatus::Success);
  REQUIRE(table.HasEmAb());

  // Check EnergyGrid
  CHECK(table.energyGrid.nPoints == kTestNE);
  CHECK(table.energyGrid.scale == AxisScale::Linear);

  // Check ThermoState
  CHECK(table.thermoState.dimensions[0] == kTestNRho);  // nRho
  CHECK(table.thermoState.dimensions[1] == kTestNT);    // nT
  CHECK(table.thermoState.dimensions[2] == kTestNYe);   // nYe

  // Check EmAb
  CHECK(table.emAb.nOpacities == kTestNOpacities);
  CHECK(table.emAb.dimensions[0] == kTestNE);    // nE
  CHECK(table.emAb.dimensions[1] == kTestNRho);  // nRho
  CHECK(table.emAb.dimensions[2] == kTestNT);    // nT
  CHECK(table.emAb.dimensions[3] == kTestNYe);   // nYe

  CHECK(table.emAb.names[0] == "Electron Neutrino");
  CHECK(table.emAb.names[1] == "Electron Antineutrino");
}

TEST_CASE("LoadWeakLibOpacityTableFull loads Iso table", "[hdf5][weaklib][opacity]")
{
  constexpr double kIsoNuBase = 100000.0;
  constexpr double kIsoNuBarBase = 200000.0;

  EnsureAmrexInitialized();
  const std::filesystem::path filePath =
      std::filesystem::temp_directory_path() / "weaklibreader_opacity_iso_full.h5";
  CreateWeakLibIsoTestFile(filePath);

  WeakLibOpacityTable table;
  auto status = LoadWeakLibOpacityTableFull(table, "", filePath.string());

  std::filesystem::remove(filePath);

  REQUIRE(status == Hdf5LoadStatus::Success);
  REQUIRE(table.HasScatIso());

  CHECK(table.scatIso.nOpacities == kTestNOpacities);
  CHECK(table.scatIso.nMoments == kTestNMoments);
  CHECK(table.scatIso.dimensions[0] == kTestNE);    // nE
  CHECK(table.scatIso.dimensions[1] == kTestNMoments);  // nMom
  CHECK(table.scatIso.dimensions[2] == kTestNRho);  // nRho
  CHECK(table.scatIso.dimensions[3] == kTestNT);    // nT
  CHECK(table.scatIso.dimensions[4] == kTestNYe);   // nYe

  const double* nuData = table.scatIso.KernelData(0);
  const double* nuBarData = table.scatIso.KernelData(1);
  REQUIRE(nuData != nullptr);
  REQUIRE(nuBarData != nullptr);

  const auto& layout = table.scatIso.layout;
  CHECK(nuData[layout.Offset(0, 0, 0, 0, 0)] ==
        Catch::Approx(Encode5dIndexValue(0, 0, 0, 0, 0, kIsoNuBase)));
  CHECK(nuData[layout.Offset(3, 1, 1, 2, 1)] ==
        Catch::Approx(Encode5dIndexValue(3, 1, 1, 2, 1, kIsoNuBase)));
  CHECK(nuBarData[layout.Offset(2, 0, 1, 1, 1)] ==
        Catch::Approx(Encode5dIndexValue(2, 0, 1, 1, 1, kIsoNuBarBase)));
}

TEST_CASE("LoadWeakLibOpacityTableFull loads NES table", "[hdf5][weaklib][opacity]")
{
  constexpr double kNesBase = 300000.0;

  EnsureAmrexInitialized();
  const std::filesystem::path filePath =
      std::filesystem::temp_directory_path() / "weaklibreader_opacity_nes_full.h5";
  CreateWeakLibNESTestFile(filePath);

  WeakLibOpacityTable table;
  auto status = LoadWeakLibOpacityTableFull(table, "", "", filePath.string());

  std::filesystem::remove(filePath);

  REQUIRE(status == Hdf5LoadStatus::Success);
  REQUIRE(table.HasScatNES());

  // Check EtaGrid was loaded
  CHECK(table.etaGrid.nPoints == kTestNEta);

  CHECK(table.scatNES.nOpacities == 1);
  CHECK(table.scatNES.nMoments == 2);
  CHECK(table.scatNES.dimensions[0] == kTestNE);    // nE_in
  CHECK(table.scatNES.dimensions[1] == kTestNE);    // nE_out
  CHECK(table.scatNES.dimensions[2] == 2);          // nMom
  CHECK(table.scatNES.dimensions[3] == kTestNT);    // nT
  CHECK(table.scatNES.dimensions[4] == kTestNEta);  // nEta

  const double* kernel = table.scatNES.KernelData();
  REQUIRE(kernel != nullptr);

  const auto& layout = table.scatNES.layout;
  CHECK(kernel[layout.Offset(0, 0, 0, 0, 0)] ==
        Catch::Approx(Encode5dIndexValue(0, 0, 0, 0, 0, kNesBase)));
  CHECK(kernel[layout.Offset(3, 2, 1, 2, 2)] ==
        Catch::Approx(Encode5dIndexValue(3, 2, 1, 2, 2, kNesBase)));
}

TEST_CASE("LoadWeakLibOpacityTableFull loads Pair table", "[hdf5][weaklib][opacity]")
{
  constexpr double kPairBase = 400000.0;

  EnsureAmrexInitialized();
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
        Catch::Approx(Encode5dIndexValue(1, 0, 0, 0, 0, kPairBase)));
  CHECK(kernel[layout.Offset(3, 3, 1, 2, 2)] ==
        Catch::Approx(Encode5dIndexValue(3, 3, 1, 2, 2, kPairBase)));
}

TEST_CASE("LoadWeakLibOpacityTableFull loads Brem table", "[hdf5][weaklib][opacity]")
{
  constexpr double kBremBase = 500000.0;

  EnsureAmrexInitialized();
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
        Catch::Approx(Encode5dIndexValue(0, 0, 0, 0, 0, kBremBase)));
  CHECK(kernel[layout.Offset(3, 1, 0, 1, 2)] ==
        Catch::Approx(Encode5dIndexValue(3, 1, 0, 1, 2, kBremBase)));
}

TEST_CASE("LoadWeakLibOpacityTableFull loads multiple types", "[hdf5][weaklib][opacity]")
{
  EnsureAmrexInitialized();
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
  EnsureAmrexInitialized();
  ScopedHdf5ErrorSilencer silencer{};

  WeakLibOpacityTable table;
  auto status = LoadWeakLibOpacityTableFull(table, "nonexistent.h5");

  CHECK(status == Hdf5LoadStatus::FileOpenFailed);
}

TEST_CASE("LoadWeakLibOpacityTableFull returns error for no files", "[hdf5][weaklib][opacity]")
{
  EnsureAmrexInitialized();
  ScopedHdf5ErrorSilencer silencer{};

  WeakLibOpacityTable table;
  auto status = LoadWeakLibOpacityTableFull(table);

  CHECK(status == Hdf5LoadStatus::FileOpenFailed);
}

TEST_CASE("MakeDeviceCopy creates device copy of opacity table", "[hdf5][weaklib][opacity][gpu]")
{
  EnsureAmrexInitialized();
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
  EnsureAmrexInitialized();

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
  EnsureAmrexInitialized();
  ScopedHdf5ErrorSilencer silencer{};

  WeakLibOpacityTable table;
  auto status = LoadWeakLibOpacityTableFullParallel(table, "nonexistent.h5");
  CHECK(status == Hdf5LoadStatus::FileOpenFailed);
}
