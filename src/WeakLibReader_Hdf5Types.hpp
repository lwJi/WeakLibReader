#pragma once

#include <AMReX_Array.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_TableData.H>
#include <AMReX_Vector.H>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "WeakLibReader_Layout.hpp"
#include "WeakLibReader_AxisTypes.hpp"

namespace WeakLibReader {

enum class Hdf5LoadStatus : std::uint8_t {
  Success = 0,
  FileOpenFailed,
  DatasetOpenFailed,
  DatasetRankInvalid,
  DatasetReadFailed,
  AxisDatasetOpenFailed,
  AxisExtentMismatch,
  AxisNotMonotone,
  AxisInvalidScale,
  AxisReadFailed,
  IncompatibleDatasetExtent
};

struct Hdf5LoadConfig {
  std::string valueDataset = "values";
  std::string axisPrefix = "axis";
  std::string axisScaleAttribute = "scale";
};

struct TableView {
  int nd = 0;
  Layout layout{};
  Axis axes[5]{};
  const double* data = nullptr;
};

struct TableDevice {
  int nd = 0;
  Layout layout{};
  Axis axes[5]{};
  amrex::TableData<double, 4> values{};
  std::array<amrex::Gpu::DeviceVector<double>, 5> axisStorage{};

  [[nodiscard]] TableView View() const noexcept
  {
    TableView view{};
    view.nd = nd;
    view.layout = layout;
    view.data = values.const_table().p;
    for (int dim = 0; dim < 5; ++dim) {
      view.axes[dim] = axes[dim];
    }
    return view;
  }
};

struct Hdf5Table {
  int nd = 0;
  std::array<int, 5> extents{{1, 1, 1, 1, 1}};
  Layout layout{};
  Axis axes[5]{};
  amrex::TableData<double, 4> values{};
  std::array<amrex::Vector<double>, 5> axisStorage{};

  Hdf5Table() = default;
  Hdf5Table(Hdf5Table&&) = default;
  Hdf5Table& operator=(Hdf5Table&&) = default;
  Hdf5Table(const Hdf5Table&) = delete;
  Hdf5Table& operator=(const Hdf5Table&) = delete;

  [[nodiscard]] double* DataPtr() noexcept { return values.table().p; }
  [[nodiscard]] const double* DataPtr() const noexcept { return values.const_table().p; }

  [[nodiscard]] TableView View() const noexcept
  {
    TableView view{};
    view.nd = nd;
    view.layout = layout;
    view.data = values.const_table().p;
    for (int dim = 0; dim < 5; ++dim) {
      view.axes[dim] = axes[dim];
    }
    return view;
  }
};

// Index mappings for dependent variables (matches Fortran DV % Indices)
struct WeakLibEosIndices {
  int iPressure = -1;
  int iEntropyPerBaryon = -1;
  int iInternalEnergyDensity = -1;
  int iElectronChemicalPotential = -1;
  int iProtonChemicalPotential = -1;
  int iNeutronChemicalPotential = -1;
  int iProtonMassFraction = -1;
  int iNeutronMassFraction = -1;
  int iAlphaMassFraction = -1;
  int iHeavyMassFraction = -1;
  int iHeavyChargeNumber = -1;
  int iHeavyMassNumber = -1;
  int iHeavyBindingEnergy = -1;
  int iThermalEnergy = -1;
  int iGamma1 = -1;
};

// Full WeakLib EOS table (host memory)
struct WeakLibEosTable {
  // Dimensions
  int nVariables = 0;
  std::array<int, 3> dimensions{{0, 0, 0}};  // [nRho, nT, nYe] after reversal

  // ThermoState data
  std::array<amrex::Vector<double>, 3> axisStorage;  // Density, Temperature, Ye
  Axis axes[3]{};
  std::array<std::string, 3> axisNames;
  std::array<std::string, 3> axisUnits;

  // DependentVariables data
  std::vector<std::string> variableNames;
  std::vector<std::string> variableUnits;
  std::vector<double> offsets;
  std::vector<amrex::TableData<double, 3>> variables;
  amrex::TableData<int, 3> repaired;
  WeakLibEosIndices indices;

  // Layout for interpolation
  Layout layout{};

  WeakLibEosTable() = default;
  WeakLibEosTable(WeakLibEosTable&&) = default;
  WeakLibEosTable& operator=(WeakLibEosTable&&) = default;
  WeakLibEosTable(const WeakLibEosTable&) = delete;
  WeakLibEosTable& operator=(const WeakLibEosTable&) = delete;

  // Get data pointer for a specific variable
  [[nodiscard]] const double* VariableData(int varIndex) const noexcept {
    return variables[varIndex].const_table().p;
  }
};

// Full WeakLib EOS table (device memory)
struct WeakLibEosTableDevice {
  int nVariables = 0;
  std::array<int, 3> dimensions{{0, 0, 0}};
  std::array<amrex::Gpu::DeviceVector<double>, 3> axisStorage;
  Axis axes[3]{};
  Layout layout{};

  std::vector<double> offsets;
  std::vector<amrex::TableData<double, 3>> variables;
  amrex::TableData<int, 3> repaired;
  WeakLibEosIndices indices;

  // Get device data pointer for a specific variable
  [[nodiscard]] const double* VariableData(int varIndex) const noexcept {
    return variables[varIndex].const_table().p;
  }
};

} // namespace WeakLibReader
