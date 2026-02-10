#pragma once

#include <AMReX_Array.H>
#include <AMReX_GpuContainers.H>
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
  amrex::Gpu::DeviceVector<double> values;
  std::array<amrex::Gpu::DeviceVector<double>, 5> axisStorage{};

  [[nodiscard]] TableView View() const noexcept
  {
    TableView view{};
    view.nd = nd;
    view.layout = layout;
    view.data = values.data();
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
  amrex::Gpu::PinnedVector<double> values;
  std::array<amrex::Vector<double>, 5> axisStorage{};

  Hdf5Table() = default;
  Hdf5Table(Hdf5Table&&) = default;
  Hdf5Table& operator=(Hdf5Table&&) = default;
  Hdf5Table(const Hdf5Table&) = delete;
  Hdf5Table& operator=(const Hdf5Table&) = delete;

  [[nodiscard]] double* DataPtr() noexcept { return values.data(); }
  [[nodiscard]] const double* DataPtr() const noexcept { return values.data(); }

  [[nodiscard]] TableView View() const noexcept
  {
    TableView view{};
    view.nd = nd;
    view.layout = layout;
    view.data = values.data();
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
  std::vector<amrex::Gpu::PinnedVector<double>> variables;
  amrex::Gpu::PinnedVector<int> repaired;
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
    return variables[varIndex].data();
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
  std::vector<amrex::Gpu::DeviceVector<double>> variables;
  amrex::Gpu::DeviceVector<int> repaired;
  WeakLibEosIndices indices;

  // Get device data pointer for a specific variable
  [[nodiscard]] const double* VariableData(int varIndex) const noexcept {
    return variables[varIndex].data();
  }
};

// ============================================================================
// Opacity Table Structures
// ============================================================================

// Grid for opacity axes (EnergyGrid, EtaGrid)
struct WeakLibOpacityGrid {
  std::string name;
  std::string unit;
  int nPoints = 0;
  AxisScale scale = AxisScale::Linear;
  amrex::Vector<double> values;

  // For geometric grids (optional)
  double zoom = 0.0;
  double minValue = 0.0;
  double maxValue = 0.0;

  [[nodiscard]] Axis MakeAxis() const noexcept {
    return Axis{values.data(), nPoints, scale};
  }
};

struct WeakLibOpacityGridDevice {
  int nPoints = 0;
  AxisScale scale = AxisScale::Linear;
  amrex::Gpu::DeviceVector<double> values;

  [[nodiscard]] Axis MakeAxis() const noexcept {
    return Axis{values.data(), nPoints, scale};
  }
};

// EC table for electron capture on nuclei (optional in EmAb)
struct WeakLibECTable {
  int nE = 0;
  int nRho = 0;
  int nT = 0;
  int nYe = 0;

  amrex::Vector<double> energyValues;
  amrex::Vector<double> rhoValues;
  amrex::Vector<double> tempValues;
  amrex::Vector<double> yeValues;

  double rhoMin = 0.0, rhoMax = 0.0;
  double tempMin = 0.0, tempMax = 0.0;
  double yeMin = 0.0, yeMax = 0.0;

  std::string unit;
  double specOffset = 0.0;
  double rateOffset = 0.0;

  // Spectrum: 4D [nRho, nT, nYe, nE]
  amrex::Gpu::PinnedVector<double> spectrum;
  // Rate: 3D [nRho, nT, nYe]
  amrex::Gpu::PinnedVector<double> rate;

  [[nodiscard]] bool IsPresent() const noexcept { return nE > 0; }
};

struct WeakLibECTableDevice {
  int nE = 0;
  int nRho = 0;
  int nT = 0;
  int nYe = 0;

  amrex::Gpu::DeviceVector<double> energyValues;
  amrex::Gpu::DeviceVector<double> rhoValues;
  amrex::Gpu::DeviceVector<double> tempValues;
  amrex::Gpu::DeviceVector<double> yeValues;

  double rhoMin = 0.0, rhoMax = 0.0;
  double tempMin = 0.0, tempMax = 0.0;
  double yeMin = 0.0, yeMax = 0.0;

  double specOffset = 0.0;
  double rateOffset = 0.0;

  amrex::Gpu::DeviceVector<double> spectrum;
  amrex::Gpu::DeviceVector<double> rate;

  [[nodiscard]] bool IsPresent() const noexcept { return nE > 0; }
};

// EmAb physics parameters (new format only, -1 for legacy)
struct WeakLibEmAbParameters {
  int np_FK = -1;
  int np_FK_inv_n_decay = -1;
  int np_isoenergetic = -1;
  int np_non_isoenergetic = -1;
  int np_weak_magnetism = -1;
  int nuclei_EC_FFN = -1;
  int nuclei_EC_table = -1;

  [[nodiscard]] bool IsLegacy() const noexcept { return nuclei_EC_table == -1; }
};

// EmAb opacity table (4D: nE x nRho x nT x nYe)
struct WeakLibEmAbTable {
  static constexpr int kNumSpecies = 2;  // nu_e, nu_e_bar

  int nOpacities = 0;
  std::array<int, 4> dimensions{{0, 0, 0, 0}};  // [nE, nRho, nT, nYe]

  std::array<std::string, kNumSpecies> names;
  std::array<std::string, kNumSpecies> units;
  std::array<double, kNumSpecies> offsets{{0.0, 0.0}};

  // Opacity data: 4D [nE, nRho, nT, nYe] per species
  std::array<amrex::Gpu::PinnedVector<double>, kNumSpecies> opacities;

  WeakLibEmAbParameters parameters;
  WeakLibECTable ecTable;

  Layout layout{};

  WeakLibEmAbTable() = default;
  WeakLibEmAbTable(WeakLibEmAbTable&&) = default;
  WeakLibEmAbTable& operator=(WeakLibEmAbTable&&) = default;
  WeakLibEmAbTable(const WeakLibEmAbTable&) = delete;
  WeakLibEmAbTable& operator=(const WeakLibEmAbTable&) = delete;

  [[nodiscard]] bool IsLoaded() const noexcept { return nOpacities > 0; }
  [[nodiscard]] const double* OpacityData(int species) const noexcept {
    return opacities[species].data();
  }
};

struct WeakLibEmAbTableDevice {
  static constexpr int kNumSpecies = 2;

  int nOpacities = 0;
  std::array<int, 4> dimensions{{0, 0, 0, 0}};
  std::array<double, kNumSpecies> offsets{{0.0, 0.0}};
  std::array<amrex::Gpu::DeviceVector<double>, kNumSpecies> opacities;
  WeakLibEmAbParameters parameters;
  WeakLibECTableDevice ecTable;
  Layout layout{};

  [[nodiscard]] bool IsLoaded() const noexcept { return nOpacities > 0; }
  [[nodiscard]] const double* OpacityData(int species) const noexcept {
    return opacities[species].data();
  }
};

// Iso scattering table (5D: nE x nMom x nRho x nT x nYe)
struct WeakLibScatIsoTable {
  static constexpr int kNumSpecies = 2;

  int nOpacities = 0;
  int nMoments = 0;
  std::array<int, 5> dimensions{{0, 0, 0, 0, 0}};  // [nE, nMom, nRho, nT, nYe]

  std::array<std::string, kNumSpecies> names;
  std::array<std::string, kNumSpecies> units;
  // Offsets: 2D [nOpacities, nMoments] — column-major (species stride=1)
  amrex::Gpu::PinnedVector<double> offsets;

  // Kernel data: 5D per species (stored as flat arrays)
  std::array<amrex::Vector<double>, kNumSpecies> kernels;

  // Correction flags
  int weak_magnetism_corrections = -1;
  int ion_ion_corrections = -1;
  int many_body_corrections = -1;
  double ga_strange = 0.0;

  Layout layout{};

  WeakLibScatIsoTable() = default;
  WeakLibScatIsoTable(WeakLibScatIsoTable&&) = default;
  WeakLibScatIsoTable& operator=(WeakLibScatIsoTable&&) = default;
  WeakLibScatIsoTable(const WeakLibScatIsoTable&) = delete;
  WeakLibScatIsoTable& operator=(const WeakLibScatIsoTable&) = delete;

  [[nodiscard]] bool IsLoaded() const noexcept { return nOpacities > 0; }
  [[nodiscard]] const double* KernelData(int species) const noexcept {
    return kernels[species].data();
  }
  [[nodiscard]] double OffsetValue(int species, int moment) const noexcept {
    return offsets[static_cast<std::size_t>(species)
                   + static_cast<std::size_t>(moment) * nOpacities];
  }
};

struct WeakLibScatIsoTableDevice {
  static constexpr int kNumSpecies = 2;

  int nOpacities = 0;
  int nMoments = 0;
  std::array<int, 5> dimensions{{0, 0, 0, 0, 0}};
  amrex::Gpu::DeviceVector<double> offsets;
  std::array<amrex::Gpu::DeviceVector<double>, kNumSpecies> kernels;

  int weak_magnetism_corrections = -1;
  int ion_ion_corrections = -1;
  int many_body_corrections = -1;
  double ga_strange = 0.0;

  Layout layout{};

  [[nodiscard]] bool IsLoaded() const noexcept { return nOpacities > 0; }
  [[nodiscard]] const double* KernelData(int species) const noexcept {
    return kernels[species].data();
  }
  [[nodiscard]] double OffsetValue(int species, int moment) const noexcept {
    return offsets[static_cast<std::size_t>(species)
                   + static_cast<std::size_t>(moment) * nOpacities];
  }
};

// Unified scattering kernel table (NES, Pair, Brem)
// NES: 5D [nE_in, nE_out, nMom, nT, nEta]
// Pair: 5D [nE_in, nE_out, nMom, nT, nEta]
// Brem: 5D [nE_in, nE_out, nMom, nRho, nT]
struct WeakLibScatKernelTable {
  int nOpacities = 0;
  int nMoments = 0;
  std::array<int, 5> dimensions{{0, 0, 0, 0, 0}};

  std::string name;
  std::string unit;
  // Offsets: 2D [nOpacities, nMoments] — column-major (species stride=1)
  amrex::Gpu::PinnedVector<double> offsets;
  amrex::Vector<double> kernel;  // Stored as flat array

  int NPS = -1;  // Neutrino-positron scattering flag (NES only; -1 = not set)

  Layout layout{};

  WeakLibScatKernelTable() = default;
  WeakLibScatKernelTable(WeakLibScatKernelTable&&) = default;
  WeakLibScatKernelTable& operator=(WeakLibScatKernelTable&&) = default;
  WeakLibScatKernelTable(const WeakLibScatKernelTable&) = delete;
  WeakLibScatKernelTable& operator=(const WeakLibScatKernelTable&) = delete;

  [[nodiscard]] bool IsLoaded() const noexcept { return nOpacities > 0; }
  [[nodiscard]] const double* KernelData() const noexcept { return kernel.data(); }
  [[nodiscard]] double OffsetValue(int species, int moment) const noexcept {
    return offsets[static_cast<std::size_t>(species)
                   + static_cast<std::size_t>(moment) * nOpacities];
  }
};

struct WeakLibScatKernelTableDevice {
  int nOpacities = 0;
  int nMoments = 0;
  std::array<int, 5> dimensions{{0, 0, 0, 0, 0}};
  amrex::Gpu::DeviceVector<double> offsets;
  amrex::Gpu::DeviceVector<double> kernel;
  int NPS = -1;
  Layout layout{};

  [[nodiscard]] bool IsLoaded() const noexcept { return nOpacities > 0; }
  [[nodiscard]] const double* KernelData() const noexcept { return kernel.data(); }
  [[nodiscard]] double OffsetValue(int species, int moment) const noexcept {
    return offsets[static_cast<std::size_t>(species)
                   + static_cast<std::size_t>(moment) * nOpacities];
  }
};

using WeakLibScatNESTable = WeakLibScatKernelTable;
using WeakLibScatPairTable = WeakLibScatKernelTable;
using WeakLibScatBremTable = WeakLibScatKernelTable;
using WeakLibScatNESTableDevice = WeakLibScatKernelTableDevice;
using WeakLibScatPairTableDevice = WeakLibScatKernelTableDevice;
using WeakLibScatBremTableDevice = WeakLibScatKernelTableDevice;

// ThermoState for opacity tables (shared across types)
struct WeakLibOpacityThermoState {
  std::array<int, 3> dimensions{{0, 0, 0}};  // [nRho, nT, nYe]
  std::array<AxisScale, 3> scales{{AxisScale::Log10, AxisScale::Log10, AxisScale::Linear}};
  std::array<amrex::Vector<double>, 3> axisStorage;  // Density, Temperature, Ye
  Axis axes[3]{};
  std::array<std::string, 3> names;
  std::array<std::string, 3> units;
};

struct WeakLibOpacityThermoStateDevice {
  std::array<int, 3> dimensions{{0, 0, 0}};
  std::array<AxisScale, 3> scales{{AxisScale::Log10, AxisScale::Log10, AxisScale::Linear}};
  std::array<amrex::Gpu::DeviceVector<double>, 3> axisStorage;
  Axis axes[3]{};
};

// Unified opacity table containing all types
struct WeakLibOpacityTable {
  // Grids
  WeakLibOpacityGrid energyGrid;
  WeakLibOpacityGrid etaGrid;  // Only used by NES/Pair
  WeakLibOpacityThermoState thermoState;

  // Opacity types
  WeakLibEmAbTable emAb;
  WeakLibScatIsoTable scatIso;
  WeakLibScatNESTable scatNES;
  WeakLibScatPairTable scatPair;
  WeakLibScatBremTable scatBrem;

  WeakLibOpacityTable() = default;
  WeakLibOpacityTable(WeakLibOpacityTable&&) = default;
  WeakLibOpacityTable& operator=(WeakLibOpacityTable&&) = default;
  WeakLibOpacityTable(const WeakLibOpacityTable&) = delete;
  WeakLibOpacityTable& operator=(const WeakLibOpacityTable&) = delete;

  [[nodiscard]] bool HasEmAb() const noexcept { return emAb.IsLoaded(); }
  [[nodiscard]] bool HasScatIso() const noexcept { return scatIso.IsLoaded(); }
  [[nodiscard]] bool HasScatNES() const noexcept { return scatNES.IsLoaded(); }
  [[nodiscard]] bool HasScatPair() const noexcept { return scatPair.IsLoaded(); }
  [[nodiscard]] bool HasScatBrem() const noexcept { return scatBrem.IsLoaded(); }
};

struct WeakLibOpacityTableDevice {
  WeakLibOpacityGridDevice energyGrid;
  WeakLibOpacityGridDevice etaGrid;
  WeakLibOpacityThermoStateDevice thermoState;

  WeakLibEmAbTableDevice emAb;
  WeakLibScatIsoTableDevice scatIso;
  WeakLibScatNESTableDevice scatNES;
  WeakLibScatPairTableDevice scatPair;
  WeakLibScatBremTableDevice scatBrem;

  [[nodiscard]] bool HasEmAb() const noexcept { return emAb.IsLoaded(); }
  [[nodiscard]] bool HasScatIso() const noexcept { return scatIso.IsLoaded(); }
  [[nodiscard]] bool HasScatNES() const noexcept { return scatNES.IsLoaded(); }
  [[nodiscard]] bool HasScatPair() const noexcept { return scatPair.IsLoaded(); }
  [[nodiscard]] bool HasScatBrem() const noexcept { return scatBrem.IsLoaded(); }
};

} // namespace WeakLibReader
