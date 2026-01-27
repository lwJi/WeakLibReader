#pragma once

#include <AMReX_Array.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_TableData.H>
#include <AMReX_Vector.H>

#include <array>
#include <cstdint>
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
  IncompatibleDatasetExtent,
  GroupOpenFailed,
  LogInterpReadFailed,
  OffsetsReadFailed,
  VariableCountMismatch
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

// WeakLib EOS table variable indices (matching WeakLib convention)
enum class EosVariable : int {
  Pressure = 0,
  EntropyPerBaryon = 1,
  InternalEnergyDensity = 2,
  ElectronChemicalPotential = 3,
  ProtonChemicalPotential = 4,
  NeutronChemicalPotential = 5,
  ProtonMassFraction = 6,
  NeutronMassFraction = 7,
  AlphaMassFraction = 8,
  HeavyMassFraction = 9,
  HeavyChargeNumber = 10,
  HeavyMassNumber = 11,
  HeavyBindingEnergy = 12,
  ThermalEnergy = 13,
  Gamma1 = 14
};

constexpr int kEosVariableCount = 15;

// Single WeakLib EOS table (3D: rho, T, Ye)
struct WeakLibEosTable {
  static constexpr int nd = 3;
  Layout layout{};
  Axis axes[3]{};  // [0]=rho, [1]=T, [2]=Ye
  std::array<int, 3> extents{{1, 1, 1}};

  // Selected variables (user chooses which to load)
  std::vector<amrex::TableData<double, 4>> variables{};
  std::vector<EosVariable> loadedVariables{};
  std::vector<double> offsets{};

  // Axis backing storage
  std::array<amrex::Vector<double>, 3> axisStorage{};

  WeakLibEosTable() = default;
  WeakLibEosTable(WeakLibEosTable&&) = default;
  WeakLibEosTable& operator=(WeakLibEosTable&&) = default;
  WeakLibEosTable(const WeakLibEosTable&) = delete;
  WeakLibEosTable& operator=(const WeakLibEosTable&) = delete;

  [[nodiscard]] double* VariableDataPtr(int idx) noexcept
  {
    return variables[static_cast<std::size_t>(idx)].table().p;
  }

  [[nodiscard]] const double* VariableDataPtr(int idx) const noexcept
  {
    return variables[static_cast<std::size_t>(idx)].const_table().p;
  }

  [[nodiscard]] TableView ViewForVariable(int idx) const noexcept
  {
    TableView view{};
    view.nd = nd;
    view.layout = layout;
    view.data = VariableDataPtr(idx);
    for (int dim = 0; dim < nd; ++dim) {
      view.axes[dim] = axes[dim];
    }
    return view;
  }
};

// Device-side copy of WeakLib EOS table
struct WeakLibEosTableDevice {
  static constexpr int nd = 3;
  Layout layout{};
  Axis axes[3]{};

  std::vector<amrex::TableData<double, 4>> variables{};
  std::vector<EosVariable> loadedVariables{};
  std::vector<double> offsets{};
  std::array<amrex::Gpu::DeviceVector<double>, 3> axisStorage{};

  [[nodiscard]] const double* VariableDataPtr(int idx) const noexcept
  {
    return variables[static_cast<std::size_t>(idx)].const_table().p;
  }

  [[nodiscard]] TableView ViewForVariable(int idx) const noexcept
  {
    TableView view{};
    view.nd = nd;
    view.layout = layout;
    view.data = VariableDataPtr(idx);
    for (int dim = 0; dim < nd; ++dim) {
      view.axes[dim] = axes[dim];
    }
    return view;
  }
};

} // namespace WeakLibReader
