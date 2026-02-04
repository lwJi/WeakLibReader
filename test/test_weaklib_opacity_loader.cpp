#define SIMPLE_CATCH_NO_MAIN
#include <catch2/catch_test_macros.hpp>

#include "WeakLibReader_Hdf5Loader.hpp"
#include "WeakLibReader_Hdf5Types.hpp"

#include <AMReX.H>
#include <AMReX_GpuContainers.H>
#include <hdf5.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>

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

} // namespace

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
