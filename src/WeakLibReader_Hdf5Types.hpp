#pragma once

#include <AMReX_Array.H>
#include <AMReX_BLassert.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Vector.H>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "WeakLibReader_Layout.hpp"
#include "WeakLibReader_AxisTypes.hpp"

namespace WeakLibReader {

namespace detail {

template <std::size_t N>
[[nodiscard]] inline std::size_t ExpectedSizeOrZero(
    const std::array<int, N>& dims) noexcept
{
  std::size_t total = 1;
  for (std::size_t i = 0; i < N; ++i) {
    const int extent = dims[i];
    if (extent <= 0) {
      return 0;
    }
    total *= static_cast<std::size_t>(extent);
  }
  return total;
}

} // namespace detail

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

  TableDevice() = default;
  TableDevice(TableDevice&&) = default;
  TableDevice& operator=(TableDevice&&) = default;
  TableDevice(const TableDevice&) = delete;
  TableDevice& operator=(const TableDevice&) = delete;

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

  [[nodiscard]] bool HasVariable(int varIndex) const noexcept {
    return varIndex >= 0
        && varIndex < nVariables
        && static_cast<std::size_t>(varIndex) < variables.size();
  }

  [[nodiscard]] const double* TryVariableData(int varIndex) const noexcept {
    return HasVariable(varIndex) ? variables[static_cast<std::size_t>(varIndex)].data()
                                 : nullptr;
  }

  // Get data pointer for a specific variable (asserts on invalid index)
  [[nodiscard]] const double* VariableData(int varIndex) const noexcept {
    AMREX_ASSERT(HasVariable(varIndex));
    return TryVariableData(varIndex);
  }
};

struct WeakLibEosTableDeviceView {
  int nVariables = 0;
  std::array<int, 3> dimensions{{0, 0, 0}};
  Axis axes[3]{};
  Layout layout{};
  const double* offsets = nullptr;
  double* const* variablePointers = nullptr;
  const int* repaired = nullptr;
  std::size_t variableBlockSize = 0;

  AMREX_GPU_HOST_DEVICE
  [[nodiscard]] bool HasVariable(int varIndex) const noexcept {
    return varIndex >= 0 && varIndex < nVariables
        && offsets != nullptr
        && variablePointers != nullptr;
  }

  AMREX_GPU_HOST_DEVICE
  [[nodiscard]] const double* TryVariableData(int varIndex) const noexcept {
    return HasVariable(varIndex) ? variablePointers[varIndex] : nullptr;
  }

  AMREX_GPU_HOST_DEVICE
  [[nodiscard]] const double* VariableData(int varIndex) const noexcept {
    AMREX_ASSERT(HasVariable(varIndex));
    return TryVariableData(varIndex);
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
  amrex::Gpu::DeviceVector<double> offsetsDevice;
  std::size_t variableBlockSize = 0;
  amrex::Gpu::DeviceVector<double> variableData;
  amrex::Gpu::DeviceVector<double*> variablePointers;
  amrex::Gpu::DeviceVector<int> repaired;
  WeakLibEosIndices indices;

  WeakLibEosTableDevice() = default;
  WeakLibEosTableDevice(WeakLibEosTableDevice&&) = default;
  WeakLibEosTableDevice& operator=(WeakLibEosTableDevice&&) = default;
  WeakLibEosTableDevice(const WeakLibEosTableDevice&) = delete;
  WeakLibEosTableDevice& operator=(const WeakLibEosTableDevice&) = delete;

  [[nodiscard]] bool HasVariable(int varIndex) const noexcept {
    return varIndex >= 0 && varIndex < nVariables
        && variableBlockSize > 0
        && variableData.size() >= variableBlockSize * static_cast<std::size_t>(nVariables);
  }

  [[nodiscard]] const double* TryVariableData(int varIndex) const noexcept {
    if (!HasVariable(varIndex)) {
      return nullptr;
    }
    return variableData.data() + static_cast<std::size_t>(varIndex) * variableBlockSize;
  }

  // Get device data pointer for a specific variable (asserts on invalid index)
  [[nodiscard]] const double* VariableData(int varIndex) const noexcept {
    AMREX_ASSERT(HasVariable(varIndex));
    return TryVariableData(varIndex);
  }

  [[nodiscard]] WeakLibEosTableDeviceView View() const noexcept
  {
    WeakLibEosTableDeviceView view{};
    view.nVariables = nVariables;
    view.dimensions = dimensions;
    for (int dim = 0; dim < 3; ++dim) {
      view.axes[dim] = axes[dim];
    }
    view.layout = layout;
    view.offsets = offsetsDevice.data();
    view.variablePointers = variablePointers.data();
    view.repaired = repaired.data();
    view.variableBlockSize = variableBlockSize;
    return view;
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
    AMREX_ASSERT(nPoints >= 0);
    AMREX_ASSERT(static_cast<std::size_t>(nPoints) <= values.size());
    return Axis{values.data(), nPoints, scale};
  }
};

struct WeakLibOpacityGridDevice {
  int nPoints = 0;
  AxisScale scale = AxisScale::Linear;
  amrex::Gpu::DeviceVector<double> values;

  [[nodiscard]] Axis MakeAxis() const noexcept {
    AMREX_ASSERT(nPoints >= 0);
    AMREX_ASSERT(static_cast<std::size_t>(nPoints) <= values.size());
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

  [[nodiscard]] bool IsPresent() const noexcept {
    if (nE <= 0 || nRho <= 0 || nT <= 0 || nYe <= 0) {
      return false;
    }
    const std::array<int, 4> specDims{{nRho, nT, nYe, nE}};
    const std::array<int, 3> rateDims{{nRho, nT, nYe}};
    return energyValues.size() == static_cast<std::size_t>(nE)
        && rhoValues.size() == static_cast<std::size_t>(nRho)
        && tempValues.size() == static_cast<std::size_t>(nT)
        && yeValues.size() == static_cast<std::size_t>(nYe)
        && spectrum.size() == detail::ExpectedSizeOrZero(specDims)
        && rate.size() == detail::ExpectedSizeOrZero(rateDims);
  }
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

  [[nodiscard]] bool IsPresent() const noexcept {
    if (nE <= 0 || nRho <= 0 || nT <= 0 || nYe <= 0) {
      return false;
    }
    const std::array<int, 4> specDims{{nRho, nT, nYe, nE}};
    const std::array<int, 3> rateDims{{nRho, nT, nYe}};
    return energyValues.size() == static_cast<std::size_t>(nE)
        && rhoValues.size() == static_cast<std::size_t>(nRho)
        && tempValues.size() == static_cast<std::size_t>(nT)
        && yeValues.size() == static_cast<std::size_t>(nYe)
        && spectrum.size() == detail::ExpectedSizeOrZero(specDims)
        && rate.size() == detail::ExpectedSizeOrZero(rateDims);
  }
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

  [[nodiscard]] bool IsLoaded() const noexcept {
    if (nOpacities <= 0 || nOpacities > kNumSpecies || layout.nd != 4) {
      return false;
    }
    const std::size_t expected = detail::ExpectedSizeOrZero(dimensions);
    if (expected == 0) {
      return false;
    }
    for (int species = 0; species < nOpacities; ++species) {
      if (opacities[species].size() != expected) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool HasSpecies(int species) const noexcept {
    return species >= 0
        && species < nOpacities
        && species < kNumSpecies
        && opacities[static_cast<std::size_t>(species)].data() != nullptr;
  }

  [[nodiscard]] const double* TryOpacityData(int species) const noexcept {
    return HasSpecies(species) ? opacities[static_cast<std::size_t>(species)].data()
                               : nullptr;
  }

  [[nodiscard]] const double* OpacityData(int species) const noexcept {
    AMREX_ASSERT(HasSpecies(species));
    return TryOpacityData(species);
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

  [[nodiscard]] bool IsLoaded() const noexcept {
    if (nOpacities <= 0 || nOpacities > kNumSpecies || layout.nd != 4) {
      return false;
    }
    const std::size_t expected = detail::ExpectedSizeOrZero(dimensions);
    if (expected == 0) {
      return false;
    }
    for (int species = 0; species < nOpacities; ++species) {
      if (opacities[species].size() != expected) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool HasSpecies(int species) const noexcept {
    return species >= 0
        && species < nOpacities
        && species < kNumSpecies
        && opacities[static_cast<std::size_t>(species)].data() != nullptr;
  }

  [[nodiscard]] const double* TryOpacityData(int species) const noexcept {
    return HasSpecies(species) ? opacities[static_cast<std::size_t>(species)].data()
                               : nullptr;
  }

  [[nodiscard]] const double* OpacityData(int species) const noexcept {
    AMREX_ASSERT(HasSpecies(species));
    return TryOpacityData(species);
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
  std::array<amrex::Gpu::PinnedVector<double>, kNumSpecies> kernels;

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

  [[nodiscard]] bool IsLoaded() const noexcept {
    if (nOpacities <= 0 || nOpacities > kNumSpecies || nMoments <= 0 || layout.nd != 5) {
      return false;
    }
    const std::size_t expectedOffsets =
        static_cast<std::size_t>(nOpacities) * static_cast<std::size_t>(nMoments);
    if (offsets.size() != expectedOffsets) {
      return false;
    }
    if (dimensions[1] != nMoments) {
      return false;
    }
    const std::size_t expectedKernel = detail::ExpectedSizeOrZero(dimensions);
    if (expectedKernel == 0) {
      return false;
    }
    for (int species = 0; species < nOpacities; ++species) {
      if (kernels[species].size() != expectedKernel) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool HasSpecies(int species) const noexcept {
    return species >= 0
        && species < nOpacities
        && species < kNumSpecies
        && kernels[static_cast<std::size_t>(species)].data() != nullptr;
  }

  [[nodiscard]] bool HasMoment(int moment) const noexcept {
    return moment >= 0 && moment < nMoments;
  }

  [[nodiscard]] const double* TryKernelData(int species) const noexcept {
    return HasSpecies(species) ? kernels[static_cast<std::size_t>(species)].data()
                               : nullptr;
  }

  [[nodiscard]] const double* KernelData(int species) const noexcept {
    AMREX_ASSERT(HasSpecies(species));
    return TryKernelData(species);
  }

  [[nodiscard]] bool TryOffsetValue(int species, int moment, double& value) const noexcept {
    if (!HasSpecies(species) || !HasMoment(moment)) {
      value = 0.0;
      return false;
    }
    const std::size_t idx =
        static_cast<std::size_t>(species)
        + static_cast<std::size_t>(moment) * static_cast<std::size_t>(nOpacities);
    if (idx >= offsets.size()) {
      value = 0.0;
      return false;
    }
    value = offsets[idx];
    return true;
  }

  [[nodiscard]] double OffsetValue(int species, int moment) const noexcept {
    double value = 0.0;
    const bool ok = TryOffsetValue(species, moment, value);
    AMREX_ASSERT(ok);
    return value;
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

  [[nodiscard]] bool IsLoaded() const noexcept {
    if (nOpacities <= 0 || nOpacities > kNumSpecies || nMoments <= 0 || layout.nd != 5) {
      return false;
    }
    const std::size_t expectedOffsets =
        static_cast<std::size_t>(nOpacities) * static_cast<std::size_t>(nMoments);
    if (offsets.size() != expectedOffsets) {
      return false;
    }
    if (dimensions[1] != nMoments) {
      return false;
    }
    const std::size_t expectedKernel = detail::ExpectedSizeOrZero(dimensions);
    if (expectedKernel == 0) {
      return false;
    }
    for (int species = 0; species < nOpacities; ++species) {
      if (kernels[species].size() != expectedKernel) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool HasSpecies(int species) const noexcept {
    return species >= 0
        && species < nOpacities
        && species < kNumSpecies
        && kernels[static_cast<std::size_t>(species)].data() != nullptr;
  }

  [[nodiscard]] bool HasMoment(int moment) const noexcept {
    return moment >= 0 && moment < nMoments;
  }

  [[nodiscard]] const double* TryKernelData(int species) const noexcept {
    return HasSpecies(species) ? kernels[static_cast<std::size_t>(species)].data()
                               : nullptr;
  }

  [[nodiscard]] const double* KernelData(int species) const noexcept {
    AMREX_ASSERT(HasSpecies(species));
    return TryKernelData(species);
  }

  [[nodiscard]] bool TryOffsetValue(int species, int moment, double& value) const noexcept {
    if (!HasSpecies(species) || !HasMoment(moment)) {
      value = 0.0;
      return false;
    }
    const std::size_t idx =
        static_cast<std::size_t>(species)
        + static_cast<std::size_t>(moment) * static_cast<std::size_t>(nOpacities);
    if (idx >= offsets.size()) {
      value = 0.0;
      return false;
    }
    value = offsets[idx];
    return true;
  }

  [[nodiscard]] double OffsetValue(int species, int moment) const noexcept {
    double value = 0.0;
    const bool ok = TryOffsetValue(species, moment, value);
    AMREX_ASSERT(ok);
    return value;
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
  amrex::Gpu::PinnedVector<double> kernel;  // Stored as flat array

  int NPS = -1;  // Neutrino-positron scattering flag (NES only; -1 = not set)

  Layout layout{};

  WeakLibScatKernelTable() = default;
  WeakLibScatKernelTable(WeakLibScatKernelTable&&) = default;
  WeakLibScatKernelTable& operator=(WeakLibScatKernelTable&&) = default;
  WeakLibScatKernelTable(const WeakLibScatKernelTable&) = delete;
  WeakLibScatKernelTable& operator=(const WeakLibScatKernelTable&) = delete;

  [[nodiscard]] bool IsLoaded() const noexcept {
    if (nOpacities <= 0 || nMoments <= 0 || layout.nd != 5) {
      return false;
    }
    const std::size_t expectedOffsets =
        static_cast<std::size_t>(nOpacities) * static_cast<std::size_t>(nMoments);
    if (offsets.size() != expectedOffsets) {
      return false;
    }
    if (dimensions[2] != nMoments) {
      return false;
    }
    const std::size_t expectedKernel = detail::ExpectedSizeOrZero(dimensions);
    return expectedKernel > 0 && kernel.size() == expectedKernel;
  }

  [[nodiscard]] const double* KernelData() const noexcept { return kernel.data(); }
  [[nodiscard]] bool HasSpecies(int species) const noexcept {
    return species >= 0 && species < nOpacities;
  }
  [[nodiscard]] bool HasMoment(int moment) const noexcept {
    return moment >= 0 && moment < nMoments;
  }
  [[nodiscard]] bool TryOffsetValue(int species, int moment, double& value) const noexcept {
    if (!HasSpecies(species) || !HasMoment(moment)) {
      value = 0.0;
      return false;
    }
    const std::size_t idx =
        static_cast<std::size_t>(species)
        + static_cast<std::size_t>(moment) * static_cast<std::size_t>(nOpacities);
    if (idx >= offsets.size()) {
      value = 0.0;
      return false;
    }
    value = offsets[idx];
    return true;
  }
  [[nodiscard]] double OffsetValue(int species, int moment) const noexcept {
    double value = 0.0;
    const bool ok = TryOffsetValue(species, moment, value);
    AMREX_ASSERT(ok);
    return value;
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

  [[nodiscard]] bool IsLoaded() const noexcept {
    if (nOpacities <= 0 || nMoments <= 0 || layout.nd != 5) {
      return false;
    }
    const std::size_t expectedOffsets =
        static_cast<std::size_t>(nOpacities) * static_cast<std::size_t>(nMoments);
    if (offsets.size() != expectedOffsets) {
      return false;
    }
    if (dimensions[2] != nMoments) {
      return false;
    }
    const std::size_t expectedKernel = detail::ExpectedSizeOrZero(dimensions);
    return expectedKernel > 0 && kernel.size() == expectedKernel;
  }

  [[nodiscard]] const double* KernelData() const noexcept { return kernel.data(); }
  [[nodiscard]] bool HasSpecies(int species) const noexcept {
    return species >= 0 && species < nOpacities;
  }
  [[nodiscard]] bool HasMoment(int moment) const noexcept {
    return moment >= 0 && moment < nMoments;
  }
  [[nodiscard]] bool TryOffsetValue(int species, int moment, double& value) const noexcept {
    if (!HasSpecies(species) || !HasMoment(moment)) {
      value = 0.0;
      return false;
    }
    const std::size_t idx =
        static_cast<std::size_t>(species)
        + static_cast<std::size_t>(moment) * static_cast<std::size_t>(nOpacities);
    if (idx >= offsets.size()) {
      value = 0.0;
      return false;
    }
    value = offsets[idx];
    return true;
  }
  [[nodiscard]] double OffsetValue(int species, int moment) const noexcept {
    double value = 0.0;
    const bool ok = TryOffsetValue(species, moment, value);
    AMREX_ASSERT(ok);
    return value;
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

  WeakLibOpacityThermoState() = default;
  WeakLibOpacityThermoState(WeakLibOpacityThermoState&&) = default;
  WeakLibOpacityThermoState& operator=(WeakLibOpacityThermoState&&) = default;
  WeakLibOpacityThermoState(const WeakLibOpacityThermoState&) = delete;
  WeakLibOpacityThermoState& operator=(const WeakLibOpacityThermoState&) = delete;
};

struct WeakLibOpacityThermoStateDevice {
  std::array<int, 3> dimensions{{0, 0, 0}};
  std::array<AxisScale, 3> scales{{AxisScale::Log10, AxisScale::Log10, AxisScale::Linear}};
  std::array<amrex::Gpu::DeviceVector<double>, 3> axisStorage;
  Axis axes[3]{};

  WeakLibOpacityThermoStateDevice() = default;
  WeakLibOpacityThermoStateDevice(WeakLibOpacityThermoStateDevice&&) = default;
  WeakLibOpacityThermoStateDevice& operator=(WeakLibOpacityThermoStateDevice&&) = default;
  WeakLibOpacityThermoStateDevice(const WeakLibOpacityThermoStateDevice&) = delete;
  WeakLibOpacityThermoStateDevice& operator=(const WeakLibOpacityThermoStateDevice&) = delete;
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
