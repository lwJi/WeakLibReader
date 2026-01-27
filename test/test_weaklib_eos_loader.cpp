#define SIMPLE_CATCH_NO_MAIN
#include <catch2/catch_test_macros.hpp>

#include "WeakLibReader_WeakLibEosLoader.hpp"
#include "test_amrex_guard.hpp"

#include <AMReX_Arena.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_TableData.H>
#include <hdf5.h>

#include <filesystem>
#include <string>
#include <vector>

namespace {

constexpr double kTol = 1.0e-12;

// Path to the real WeakLib EOS table file
// This file must exist for the tests to run
const std::string kEosTablePath =
    "/Users/liwei/docker-workspace/repos/weaklib-tables/SFHo/LowRes/wl-EOS-SFHo-15-25-50.h5";

using test_utils::AmrexGuard;

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

bool EosTableFileExists()
{
  return std::filesystem::exists(kEosTablePath);
}

} // namespace

TEST_CASE("WeakLib EOS loader loads axes correctly", "[weaklib][eos][loader]")
{
  AmrexGuard amrex{};

  if (!EosTableFileExists()) {
    return;
  }

  WeakLibReader::WeakLibEosTable table;

  // Load only one variable to minimize memory usage
  WeakLibReader::WeakLibEosLoadConfig cfg;
  cfg.variablesToLoad = {WeakLibReader::EosVariable::Pressure};

  const auto status = WeakLibReader::LoadWeakLibEosTable(kEosTablePath, table, cfg);

  REQUIRE(status == WeakLibReader::Hdf5LoadStatus::Success);

  // Verify axis dimensions (expected: 185, 81, 30)
  CHECK(table.extents[0] == 185);  // Density
  CHECK(table.extents[1] == 81);   // Temperature
  CHECK(table.extents[2] == 30);   // Electron Fraction

  // Verify axis counts match extents
  CHECK(table.axes[0].n == 185);
  CHECK(table.axes[1].n == 81);
  CHECK(table.axes[2].n == 30);

  // Verify axis scales (expected: Log10, Log10, Linear based on LogInterp)
  CHECK(table.axes[0].scale == WeakLibReader::AxisScale::Log10);  // Density
  CHECK(table.axes[1].scale == WeakLibReader::AxisScale::Log10);  // Temperature
  CHECK(table.axes[2].scale == WeakLibReader::AxisScale::Linear); // Electron Fraction

  // Verify axis grid pointers are valid
  REQUIRE(table.axes[0].grid != nullptr);
  REQUIRE(table.axes[1].grid != nullptr);
  REQUIRE(table.axes[2].grid != nullptr);

  // Verify axis values are monotonically increasing
  for (int i = 1; i < table.axes[0].n; ++i) {
    CHECK(table.axes[0].grid[i] > table.axes[0].grid[i - 1]);
  }
  for (int i = 1; i < table.axes[1].n; ++i) {
    CHECK(table.axes[1].grid[i] > table.axes[1].grid[i - 1]);
  }
  for (int i = 1; i < table.axes[2].n; ++i) {
    CHECK(table.axes[2].grid[i] > table.axes[2].grid[i - 1]);
  }

  // Verify Log10 axes have positive values
  for (int i = 0; i < table.axes[0].n; ++i) {
    CHECK(table.axes[0].grid[i] > 0.0);
  }
  for (int i = 0; i < table.axes[1].n; ++i) {
    CHECK(table.axes[1].grid[i] > 0.0);
  }
}

TEST_CASE("WeakLib EOS loader loads single variable correctly", "[weaklib][eos][loader]")
{
  AmrexGuard amrex{};

  if (!EosTableFileExists()) {
    return;
  }

  WeakLibReader::WeakLibEosTable table;

  WeakLibReader::WeakLibEosLoadConfig cfg;
  cfg.variablesToLoad = {WeakLibReader::EosVariable::Pressure};

  const auto status = WeakLibReader::LoadWeakLibEosTable(kEosTablePath, table, cfg);

  REQUIRE(status == WeakLibReader::Hdf5LoadStatus::Success);

  // Verify only one variable was loaded
  CHECK(table.loadedVariables.size() == 1);
  CHECK(table.loadedVariables[0] == WeakLibReader::EosVariable::Pressure);
  CHECK(table.variables.size() == 1);

  // Verify data pointer is valid
  const double* data = table.VariableDataPtr(0);
  REQUIRE(data != nullptr);

  // Verify layout
  CHECK(table.layout.nd == 3);
  CHECK(table.layout.n[0] == 185);
  CHECK(table.layout.n[1] == 81);
  CHECK(table.layout.n[2] == 30);

  // Verify we can create a TableView
  const auto view = table.ViewForVariable(0);
  CHECK(view.nd == 3);
  CHECK(view.data == data);
}

TEST_CASE("WeakLib EOS loader loads multiple variables", "[weaklib][eos][loader]")
{
  AmrexGuard amrex{};

  if (!EosTableFileExists()) {
    return;
  }

  WeakLibReader::WeakLibEosTable table;

  WeakLibReader::WeakLibEosLoadConfig cfg;
  cfg.variablesToLoad = {
    WeakLibReader::EosVariable::Pressure,
    WeakLibReader::EosVariable::EntropyPerBaryon,
    WeakLibReader::EosVariable::InternalEnergyDensity
  };

  const auto status = WeakLibReader::LoadWeakLibEosTable(kEosTablePath, table, cfg);

  REQUIRE(status == WeakLibReader::Hdf5LoadStatus::Success);

  // Verify all three variables were loaded
  CHECK(table.loadedVariables.size() == 3);
  CHECK(table.loadedVariables[0] == WeakLibReader::EosVariable::Pressure);
  CHECK(table.loadedVariables[1] == WeakLibReader::EosVariable::EntropyPerBaryon);
  CHECK(table.loadedVariables[2] == WeakLibReader::EosVariable::InternalEnergyDensity);
  CHECK(table.variables.size() == 3);

  // Verify each variable has its own data
  const double* data0 = table.VariableDataPtr(0);
  const double* data1 = table.VariableDataPtr(1);
  const double* data2 = table.VariableDataPtr(2);
  REQUIRE(data0 != nullptr);
  REQUIRE(data1 != nullptr);
  REQUIRE(data2 != nullptr);

  // Data pointers should be different
  CHECK(data0 != data1);
  CHECK(data1 != data2);
  CHECK(data0 != data2);
}

TEST_CASE("WeakLib EOS loader loads offsets", "[weaklib][eos][loader]")
{
  AmrexGuard amrex{};

  if (!EosTableFileExists()) {
    return;
  }

  WeakLibReader::WeakLibEosTable table;

  WeakLibReader::WeakLibEosLoadConfig cfg;
  cfg.variablesToLoad = {WeakLibReader::EosVariable::Pressure};

  const auto status = WeakLibReader::LoadWeakLibEosTable(kEosTablePath, table, cfg);

  REQUIRE(status == WeakLibReader::Hdf5LoadStatus::Success);

  // Verify offsets were loaded (should be 15 for all variables)
  CHECK(table.offsets.size() == 15);
}

TEST_CASE("WeakLib EOS loader device copy works", "[weaklib][eos][loader][device]")
{
  AmrexGuard amrex{};

  if (!EosTableFileExists()) {
    return;
  }

  WeakLibReader::WeakLibEosTable table;

  WeakLibReader::WeakLibEosLoadConfig cfg;
  cfg.variablesToLoad = {WeakLibReader::EosVariable::Pressure};

  const auto status = WeakLibReader::LoadWeakLibEosTable(kEosTablePath, table, cfg);
  REQUIRE(status == WeakLibReader::Hdf5LoadStatus::Success);

  // Create device copy
  const auto deviceTable = WeakLibReader::MakeDeviceCopy(table);

  // Verify device table properties
  CHECK(deviceTable.layout.nd == 3);
  CHECK(deviceTable.loadedVariables.size() == 1);
  CHECK(deviceTable.loadedVariables[0] == WeakLibReader::EosVariable::Pressure);
  CHECK(deviceTable.offsets.size() == 15);

  // Verify device axes
  CHECK(deviceTable.axes[0].n == 185);
  CHECK(deviceTable.axes[1].n == 81);
  CHECK(deviceTable.axes[2].n == 30);
  CHECK(deviceTable.axes[0].scale == WeakLibReader::AxisScale::Log10);
  CHECK(deviceTable.axes[1].scale == WeakLibReader::AxisScale::Log10);
  CHECK(deviceTable.axes[2].scale == WeakLibReader::AxisScale::Linear);

  // Round-trip variable data back to host and verify
  const std::size_t totalSize = 185 * 81 * 30;
  amrex::TableData<double, 4> roundtrip;
  const amrex::Array<int, 4> lo{{0, 0, 0, 0}};
  const amrex::Array<int, 4> hi{{184, 80, 29, 0}};
  roundtrip.resize(lo, hi, amrex::The_Pinned_Arena());
  roundtrip.copy(deviceTable.variables[0]);

  const double* originalPtr = table.VariableDataPtr(0);
  const double* roundtripPtr = roundtrip.const_table().p;
  for (std::size_t i = 0; i < totalSize; ++i) {
    CHECK(roundtripPtr[i] == Catch::Approx(originalPtr[i]).margin(kTol));
  }

  // Round-trip axis data and verify
  for (int dim = 0; dim < 3; ++dim) {
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
}

TEST_CASE("WeakLib EOS loader ViewForVariable works", "[weaklib][eos][loader]")
{
  AmrexGuard amrex{};

  if (!EosTableFileExists()) {
    return;
  }

  WeakLibReader::WeakLibEosTable table;

  WeakLibReader::WeakLibEosLoadConfig cfg;
  cfg.variablesToLoad = {WeakLibReader::EosVariable::Pressure};

  const auto status = WeakLibReader::LoadWeakLibEosTable(kEosTablePath, table, cfg);
  REQUIRE(status == WeakLibReader::Hdf5LoadStatus::Success);

  // Get view for the pressure variable
  const auto view = table.ViewForVariable(0);

  CHECK(view.nd == 3);
  CHECK(view.data == table.VariableDataPtr(0));

  // Verify axes are copied correctly
  CHECK(view.axes[0].n == table.axes[0].n);
  CHECK(view.axes[0].scale == table.axes[0].scale);
  CHECK(view.axes[1].n == table.axes[1].n);
  CHECK(view.axes[1].scale == table.axes[1].scale);
  CHECK(view.axes[2].n == table.axes[2].n);
  CHECK(view.axes[2].scale == table.axes[2].scale);

  // Verify layout is copied
  CHECK(view.layout.nd == table.layout.nd);
  CHECK(view.layout.n[0] == table.layout.n[0]);
  CHECK(view.layout.n[1] == table.layout.n[1]);
  CHECK(view.layout.n[2] == table.layout.n[2]);
}

TEST_CASE("WeakLib EOS loader returns error for non-existent file", "[weaklib][eos][loader][error]")
{
  AmrexGuard amrex{};
  ScopedHdf5ErrorSilencer silencer{};

  WeakLibReader::WeakLibEosTable table;
  const auto status = WeakLibReader::LoadWeakLibEosTable("/nonexistent/path.h5", table);

  CHECK(status == WeakLibReader::Hdf5LoadStatus::FileOpenFailed);
}

TEST_CASE("WeakLib EOS loader parallel load works in single-process mode", "[weaklib][eos][loader][parallel]")
{
  AmrexGuard amrex{};

  if (!EosTableFileExists()) {
    return;
  }

  WeakLibReader::WeakLibEosTable table;

  WeakLibReader::WeakLibEosLoadConfig cfg;
  cfg.variablesToLoad = {WeakLibReader::EosVariable::Pressure};

  const auto status = WeakLibReader::LoadWeakLibEosTableParallel(kEosTablePath, table, cfg);

  REQUIRE(status == WeakLibReader::Hdf5LoadStatus::Success);

  // Verify same results as non-parallel load
  CHECK(table.extents[0] == 185);
  CHECK(table.extents[1] == 81);
  CHECK(table.extents[2] == 30);
  CHECK(table.loadedVariables.size() == 1);
  CHECK(table.loadedVariables[0] == WeakLibReader::EosVariable::Pressure);
}
