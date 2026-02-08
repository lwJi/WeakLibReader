#pragma once

#include <AMReX_Arena.H>
#include <AMReX_Array.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_ParallelDescriptor.H>

#include <limits>
#include <string>
#include <utility>

#include "WeakLibReader_Hdf5Types.hpp"
#include "detail/WeakLibReader_Hdf5LoaderDetail.hpp"

namespace WeakLibReader {

inline Hdf5LoadStatus LoadHdf5Table(const std::string& filePath,
                                    Hdf5Table& output,
                                    const Hdf5LoadConfig& cfg = Hdf5LoadConfig{})
{
  Hdf5Table result;

  detail::ScopedHandle file(H5Fopen(filePath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
  if (!file.Valid()) {
    return Hdf5LoadStatus::FileOpenFailed;
  }

  detail::ScopedHandle dataset(H5Dopen(file.Get(), cfg.valueDataset.c_str(), H5P_DEFAULT), H5Dclose);
  if (!dataset.Valid()) {
    return Hdf5LoadStatus::DatasetOpenFailed;
  }

  detail::ScopedHandle dataspace(H5Dget_space(dataset.Get()), H5Sclose);
  if (!dataspace.Valid()) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }

  const int rank = H5Sget_simple_extent_ndims(dataspace.Get());
  if (rank <= 0 || rank > 5) {
    return Hdf5LoadStatus::DatasetRankInvalid;
  }

  std::array<hsize_t, 5> nativeDims{{1, 1, 1, 1, 1}};
  if (H5Sget_simple_extent_dims(dataspace.Get(), nativeDims.data(), nullptr) < 0) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }

  result.nd = rank;
  std::array<int, 5> extents{{1, 1, 1, 1, 1}};
  for (int dim = 0; dim < rank; ++dim) {
    const hsize_t source = nativeDims[rank - 1 - dim];
    if (source == 0 || source > static_cast<hsize_t>(std::numeric_limits<int>::max())) {
      return Hdf5LoadStatus::IncompatibleDatasetExtent;
    }
    extents[dim] = static_cast<int>(source);
  }
  result.extents = extents;

  const std::size_t totalSize = detail::ComputeTotalSize(rank, extents);
  if (totalSize == 0) {
    return Hdf5LoadStatus::IncompatibleDatasetExtent;
  }

  const amrex::Array<int, 4> lo{{0, 0, 0, 0}};
  bool extentOverflow = false;
  amrex::Array<int, 4> hi = detail::MakeHiArray(rank, extents, extentOverflow);
  if (extentOverflow) {
    return Hdf5LoadStatus::IncompatibleDatasetExtent;
  }
  result.values.resize(lo, hi, amrex::The_Pinned_Arena());

  if (H5Dread(dataset.Get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
              result.values.table().p) < 0) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }

  const Hdf5LoadStatus axisStatus = detail::LoadAxes(file.Get(), rank, cfg, result);
  if (axisStatus != Hdf5LoadStatus::Success) {
    return axisStatus;
  }

  result.layout = MakeLayout(result.extents.data(), result.nd);
  output = std::move(result);
  return Hdf5LoadStatus::Success;
}

inline Hdf5LoadStatus LoadWeakLibEosTableFull(const std::string& filePath,
                                               WeakLibEosTable& output)
{
  WeakLibEosTable result;

  detail::ScopedHandle file(H5Fopen(filePath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
  if (!file.Valid()) {
    return Hdf5LoadStatus::FileOpenFailed;
  }

  // ========== Read ThermoState ==========
  detail::ScopedHandle thermoGroup(H5Gopen(file.Get(), "ThermoState", H5P_DEFAULT), H5Gclose);
  if (!thermoGroup.Valid()) {
    return Hdf5LoadStatus::DatasetOpenFailed;
  }

  // Read LogInterp to determine axis scales
  int logInterp[3] = {0, 0, 0};
  if (!detail::ReadIntArray(thermoGroup.Get(), "LogInterp", logInterp, 3)) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }
  const std::array<AxisScale, 3> scales{{
      (logInterp[0] == 1) ? AxisScale::Log10 : AxisScale::Linear,
      (logInterp[1] == 1) ? AxisScale::Log10 : AxisScale::Linear,
      (logInterp[2] == 1) ? AxisScale::Log10 : AxisScale::Linear
  }};

  // Read Dimensions (file order: [nRho, nT, nYe], same as C order)
  int fileDims[3] = {0, 0, 0};
  if (!detail::ReadIntArray(thermoGroup.Get(), "Dimensions", fileDims, 3)) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }
  // File order is already [nRho, nT, nYe], no reversal needed
  result.dimensions[0] = fileDims[0];  // nRho
  result.dimensions[1] = fileDims[1];  // nT
  result.dimensions[2] = fileDims[2];  // nYe

  // Read axis names and units
  std::vector<std::string> axisNamesVec, axisUnitsVec;
  if (!detail::ReadStringArray(thermoGroup.Get(), "Names", axisNamesVec) || axisNamesVec.size() != 3) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }
  if (!detail::ReadStringArray(thermoGroup.Get(), "Units", axisUnitsVec) || axisUnitsVec.size() != 3) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }
  for (int i = 0; i < 3; ++i) {
    result.axisNames[i] = axisNamesVec[i];
    result.axisUnits[i] = axisUnitsVec[i];
  }

  // Load axes using existing helper
  // Axis 0 = Density, Axis 1 = Temperature, Axis 2 = Electron Fraction
  Hdf5Table tempTable;
  tempTable.extents = {result.dimensions[0], result.dimensions[1], result.dimensions[2], 1, 1};

  Hdf5LoadStatus axisStatus;
  axisStatus = detail::LoadWeakLibAxis(thermoGroup.Get(), "Density",
                                       result.dimensions[0], scales[0], 0, tempTable);
  if (axisStatus != Hdf5LoadStatus::Success) return axisStatus;

  axisStatus = detail::LoadWeakLibAxis(thermoGroup.Get(), "Temperature",
                                       result.dimensions[1], scales[1], 1, tempTable);
  if (axisStatus != Hdf5LoadStatus::Success) return axisStatus;

  axisStatus = detail::LoadWeakLibAxis(thermoGroup.Get(), "Electron Fraction",
                                       result.dimensions[2], scales[2], 2, tempTable);
  if (axisStatus != Hdf5LoadStatus::Success) return axisStatus;

  // Transfer axis data from temp table
  for (int i = 0; i < 3; ++i) {
    result.axisStorage[i] = std::move(tempTable.axisStorage[i]);
    result.axes[i] = tempTable.axes[i];
    result.axes[i].grid = result.axisStorage[i].data();
  }

  // ========== Read DependentVariables ==========
  detail::ScopedHandle dvGroup(H5Gopen(file.Get(), "DependentVariables", H5P_DEFAULT), H5Gclose);
  if (!dvGroup.Valid()) {
    return Hdf5LoadStatus::DatasetOpenFailed;
  }

  // Read nVariables
  if (!detail::ReadScalarInt(dvGroup.Get(), "nVariables", result.nVariables)) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }

  // Read variable names and units
  if (!detail::ReadStringArray(dvGroup.Get(), "Names", result.variableNames)) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }
  if (!detail::ReadStringArray(dvGroup.Get(), "Units", result.variableUnits)) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }
  if (result.variableNames.size() != static_cast<std::size_t>(result.nVariables) ||
      result.variableUnits.size() != static_cast<std::size_t>(result.nVariables)) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }

  // Read offsets
  result.offsets.resize(result.nVariables);
  {
    detail::ScopedHandle offsetsDs(H5Dopen(dvGroup.Get(), "Offsets", H5P_DEFAULT), H5Dclose);
    if (!offsetsDs.Valid()) {
      return Hdf5LoadStatus::DatasetOpenFailed;
    }
    detail::ScopedHandle offsetsSpace(H5Dget_space(offsetsDs.Get()), H5Sclose);
    if (!offsetsSpace.Valid()) {
      return Hdf5LoadStatus::DatasetReadFailed;
    }
    const int rank = H5Sget_simple_extent_ndims(offsetsSpace.Get());
    if (rank != 1) {
      return Hdf5LoadStatus::DatasetReadFailed;
    }
    hsize_t length = 0;
    if (H5Sget_simple_extent_dims(offsetsSpace.Get(), &length, nullptr) < 0) {
      return Hdf5LoadStatus::DatasetReadFailed;
    }
    if (length != static_cast<hsize_t>(result.nVariables)) {
      return Hdf5LoadStatus::DatasetReadFailed;
    }
    if (H5Dread(offsetsDs.Get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                result.offsets.data()) < 0) {
      return Hdf5LoadStatus::DatasetReadFailed;
    }
  }

  // Read each variable's 3D data
  result.variables.resize(result.nVariables);
  for (int iVar = 0; iVar < result.nVariables; ++iVar) {
    const std::string& varName = result.variableNames[iVar];
    if (!detail::ReadDoubleArray3d(dvGroup.Get(), varName.c_str(),
                                   result.variables[iVar], result.dimensions)) {
      return Hdf5LoadStatus::DatasetReadFailed;
    }
  }

  // Read Repaired mask
  if (!detail::ReadIntArray3d(dvGroup.Get(), "Repaired", result.repaired, result.dimensions)) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }

  // Read index mappings
  auto readIndex = [&](const char* name, int& idx) -> bool {
    return detail::ReadScalarInt(dvGroup.Get(), name, idx);
  };

  if (!readIndex("iPressure", result.indices.iPressure) ||
      !readIndex("iEntropyPerBaryon", result.indices.iEntropyPerBaryon) ||
      !readIndex("iInternalEnergyDensity", result.indices.iInternalEnergyDensity) ||
      !readIndex("iElectronChemicalPotential", result.indices.iElectronChemicalPotential) ||
      !readIndex("iProtonChemicalPotential", result.indices.iProtonChemicalPotential) ||
      !readIndex("iNeutronChemicalPotential", result.indices.iNeutronChemicalPotential) ||
      !readIndex("iProtonMassFraction", result.indices.iProtonMassFraction) ||
      !readIndex("iNeutronMassFraction", result.indices.iNeutronMassFraction) ||
      !readIndex("iAlphaMassFraction", result.indices.iAlphaMassFraction) ||
      !readIndex("iHeavyMassFraction", result.indices.iHeavyMassFraction) ||
      !readIndex("iHeavyChargeNumber", result.indices.iHeavyChargeNumber) ||
      !readIndex("iHeavyMassNumber", result.indices.iHeavyMassNumber) ||
      !readIndex("iHeavyBindingEnergy", result.indices.iHeavyBindingEnergy) ||
      !readIndex("iThermalEnergy", result.indices.iThermalEnergy) ||
      !readIndex("iGamma1", result.indices.iGamma1)) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }

  // Fortran indices are 1-based; normalize to 0-based for C++ access.
  auto normalizeIndex = [](int& idx) {
    if (idx > 0) {
      --idx;
    }
  };
  normalizeIndex(result.indices.iPressure);
  normalizeIndex(result.indices.iEntropyPerBaryon);
  normalizeIndex(result.indices.iInternalEnergyDensity);
  normalizeIndex(result.indices.iElectronChemicalPotential);
  normalizeIndex(result.indices.iProtonChemicalPotential);
  normalizeIndex(result.indices.iNeutronChemicalPotential);
  normalizeIndex(result.indices.iProtonMassFraction);
  normalizeIndex(result.indices.iNeutronMassFraction);
  normalizeIndex(result.indices.iAlphaMassFraction);
  normalizeIndex(result.indices.iHeavyMassFraction);
  normalizeIndex(result.indices.iHeavyChargeNumber);
  normalizeIndex(result.indices.iHeavyMassNumber);
  normalizeIndex(result.indices.iHeavyBindingEnergy);
  normalizeIndex(result.indices.iThermalEnergy);
  normalizeIndex(result.indices.iGamma1);

  // Compute layout for interpolation
  std::array<int, 5> extents5{{result.dimensions[0], result.dimensions[1], result.dimensions[2], 1, 1}};
  result.layout = MakeLayout(extents5.data(), 3);

  output = std::move(result);
  return Hdf5LoadStatus::Success;
}

inline TableDevice MakeDeviceCopy(const Hdf5Table& host,
                                  amrex::Arena* arena = amrex::The_Device_Arena())
{
  TableDevice device{};
  device.nd = host.nd;
  device.layout = host.layout;

  const amrex::Array<int, 4> lo{{0, 0, 0, 0}};
  bool overflow = false;
  const amrex::Array<int, 4> hi = detail::MakeHiArray(host.nd, host.extents, overflow);
  AMREX_ASSERT(!overflow);

  device.values.resize(lo, hi, arena);
  device.values.copy(host.values);

  for (int dim = 0; dim < host.nd; ++dim) {
    const auto& hostAxis = host.axisStorage[dim];
    amrex::Gpu::DeviceVector<double>& deviceAxis = device.axisStorage[dim];
    deviceAxis.resize(hostAxis.size());
    if (!hostAxis.empty()) {
      amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                       hostAxis.begin(), hostAxis.end(),
                       deviceAxis.begin());
    }
    Axis axis{};
    axis.grid = deviceAxis.data();
    axis.n = static_cast<int>(deviceAxis.size());
    axis.scale = host.axes[dim].scale;
    device.axes[dim] = axis;
  }

  for (int dim = host.nd; dim < 5; ++dim) {
    device.axisStorage[dim].clear();
    device.axes[dim] = Axis{};
  }

  return device;
}

// ============================================================================
// Opacity Table Loaders
// ============================================================================

namespace detail {

// Load EmAb parameters (new format only)
inline bool LoadEmAbParameters(hid_t file, WeakLibEmAbParameters& params)
{
  if (!GroupExists(file, "EmAb Parameters")) {
    // Legacy format - set all to -1
    params = WeakLibEmAbParameters{};
    return true;
  }

  ScopedHandle group(H5Gopen(file, "EmAb Parameters", H5P_DEFAULT), H5Gclose);
  if (!group.Valid()) {
    return false;
  }

  ReadScalarInt(group.Get(), "np_FK", params.np_FK);
  ReadScalarInt(group.Get(), "np_FK_inv_n_decay", params.np_FK_inv_n_decay);
  ReadScalarInt(group.Get(), "np_isoenergetic", params.np_isoenergetic);
  ReadScalarInt(group.Get(), "np_non_isoenergetic", params.np_non_isoenergetic);
  ReadScalarInt(group.Get(), "np_weak_magnetism", params.np_weak_magnetism);
  ReadScalarInt(group.Get(), "nuclei_EC_FFN", params.nuclei_EC_FFN);
  ReadScalarInt(group.Get(), "nuclei_EC_table", params.nuclei_EC_table);

  return true;
}

// Load EC table (if present)
inline Hdf5LoadStatus LoadECTable(hid_t file, WeakLibECTable& ecTable)
{
  if (!GroupExists(file, "EC_table")) {
    ecTable = WeakLibECTable{};
    return Hdf5LoadStatus::Success;
  }

  ScopedHandle group(H5Gopen(file, "EC_table", H5P_DEFAULT), H5Gclose);
  if (!group.Valid()) {
    return Hdf5LoadStatus::DatasetOpenFailed;
  }

  // Read dimensions
  if (!ReadScalarInt(group.Get(), "nPointsE", ecTable.nE) ||
      !ReadScalarInt(group.Get(), "nPointsRho", ecTable.nRho) ||
      !ReadScalarInt(group.Get(), "nPointsT", ecTable.nT) ||
      !ReadScalarInt(group.Get(), "nPointsYe", ecTable.nYe)) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }

  // Read axis values
  ecTable.energyValues.resize(ecTable.nE);
  ecTable.rhoValues.resize(ecTable.nRho);
  ecTable.tempValues.resize(ecTable.nT);
  ecTable.yeValues.resize(ecTable.nYe);

  auto readArray = [&](const char* name, amrex::Vector<double>& arr) -> bool {
    ScopedHandle ds(H5Dopen(group.Get(), name, H5P_DEFAULT), H5Dclose);
    if (!ds.Valid()) return false;
    return H5Dread(ds.Get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                   H5P_DEFAULT, arr.data()) >= 0;
  };

  if (!readArray("nu_E", ecTable.energyValues) ||
      !readArray("rho", ecTable.rhoValues) ||
      !readArray("T", ecTable.tempValues) ||
      !readArray("Ye", ecTable.yeValues)) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }

  // Read min/max values
  auto readScalar = [&](const char* name, double& val) -> bool {
    ScopedHandle ds(H5Dopen(group.Get(), name, H5P_DEFAULT), H5Dclose);
    if (!ds.Valid()) return false;
    return H5Dread(ds.Get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                   H5P_DEFAULT, &val) >= 0;
  };

  readScalar("minRho", ecTable.rhoMin);
  readScalar("maxRho", ecTable.rhoMax);
  readScalar("minT", ecTable.tempMin);
  readScalar("maxT", ecTable.tempMax);
  readScalar("minYe", ecTable.yeMin);
  readScalar("maxYe", ecTable.yeMax);

  // Read units and offsets
  std::vector<std::string> unitVec;
  if (ReadStringArray(group.Get(), "Units", unitVec) && !unitVec.empty()) {
    ecTable.unit = unitVec[0];
  }

  {
    ScopedHandle ds(H5Dopen(group.Get(), "spec_Offsets", H5P_DEFAULT), H5Dclose);
    if (ds.Valid()) {
      H5Dread(ds.Get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
              H5P_DEFAULT, &ecTable.specOffset);
    }
  }
  {
    ScopedHandle ds(H5Dopen(group.Get(), "rate_Offsets", H5P_DEFAULT), H5Dclose);
    if (ds.Valid()) {
      H5Dread(ds.Get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
              H5P_DEFAULT, &ecTable.rateOffset);
    }
  }

  // Read Spectrum (4D: [nRho, nT, nYe, nE])
  std::array<int, 4> specDims{{ecTable.nRho, ecTable.nT, ecTable.nYe, ecTable.nE}};
  if (!ReadDoubleArray4d(group.Get(), "Spectrum", ecTable.spectrum, specDims)) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }

  // Read Rate (3D: [nRho, nT, nYe])
  std::array<int, 3> rateDims{{ecTable.nRho, ecTable.nT, ecTable.nYe}};
  if (!ReadDoubleArray3d(group.Get(), "Rate", ecTable.rate, rateDims)) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }

  return Hdf5LoadStatus::Success;
}

} // namespace detail

inline Hdf5LoadStatus LoadWeakLibEmAbTable(hid_t file,
                                            WeakLibEmAbTable& emAb,
                                            const WeakLibOpacityGrid& energyGrid,
                                            const WeakLibOpacityThermoState& thermoState)
{
  // Load parameters first (determines legacy vs new format)
  if (!detail::LoadEmAbParameters(file, emAb.parameters)) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }

  // Determine group name based on format
  const char* groupName = emAb.parameters.IsLegacy()
                              ? "EmAb_CorrectedAbsorption"
                              : "EmAb";

  if (!detail::GroupExists(file, groupName)) {
    return Hdf5LoadStatus::DatasetOpenFailed;
  }

  detail::ScopedHandle group(H5Gopen(file, groupName, H5P_DEFAULT), H5Gclose);
  if (!group.Valid()) {
    return Hdf5LoadStatus::DatasetOpenFailed;
  }

  // Read nOpacities
  if (!detail::ReadScalarInt(group.Get(), "nOpacities", emAb.nOpacities)) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }

  // Set dimensions: [nE, nRho, nT, nYe]
  emAb.dimensions[0] = energyGrid.nPoints;
  emAb.dimensions[1] = thermoState.dimensions[0];  // nRho
  emAb.dimensions[2] = thermoState.dimensions[1];  // nT
  emAb.dimensions[3] = thermoState.dimensions[2];  // nYe

  // Read Units
  std::vector<std::string> unitVec;
  if (!detail::ReadStringArray(group.Get(), "Units", unitVec) ||
      unitVec.size() < static_cast<size_t>(emAb.nOpacities)) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }
  for (int i = 0; i < WeakLibEmAbTable::kNumSpecies && i < emAb.nOpacities; ++i) {
    emAb.units[i] = unitVec[i];
  }

  // Read Offsets
  {
    detail::ScopedHandle ds(H5Dopen(group.Get(), "Offsets", H5P_DEFAULT), H5Dclose);
    if (!ds.Valid()) {
      return Hdf5LoadStatus::DatasetOpenFailed;
    }
    std::vector<double> offsetVec(emAb.nOpacities);
    if (H5Dread(ds.Get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                offsetVec.data()) < 0) {
      return Hdf5LoadStatus::DatasetReadFailed;
    }
    for (int i = 0; i < WeakLibEmAbTable::kNumSpecies && i < emAb.nOpacities; ++i) {
      emAb.offsets[i] = offsetVec[i];
    }
  }

  // Set species names (hardcoded like Fortran)
  emAb.names[0] = "Electron Neutrino";
  emAb.names[1] = "Electron Antineutrino";

  // Read opacity data for each species
  for (int iSpecies = 0; iSpecies < WeakLibEmAbTable::kNumSpecies && iSpecies < emAb.nOpacities; ++iSpecies) {
    if (!detail::ReadDoubleArray4d(group.Get(), emAb.names[iSpecies].c_str(),
                                   emAb.opacities[iSpecies], emAb.dimensions)) {
      return Hdf5LoadStatus::DatasetReadFailed;
    }
  }

  // Compute layout
  std::array<int, 5> extents5{{emAb.dimensions[0], emAb.dimensions[1],
                                emAb.dimensions[2], emAb.dimensions[3], 1}};
  emAb.layout = MakeLayout(extents5.data(), 4);

  // Load EC table if indicated
  if (emAb.parameters.nuclei_EC_table > 0) {
    Hdf5LoadStatus ecStatus = detail::LoadECTable(file, emAb.ecTable);
    if (ecStatus != Hdf5LoadStatus::Success) {
      return ecStatus;
    }
  }

  return Hdf5LoadStatus::Success;
}

inline Hdf5LoadStatus LoadWeakLibScatIsoTable(hid_t file,
                                               WeakLibScatIsoTable& scatIso,
                                               const WeakLibOpacityGrid& energyGrid,
                                               const WeakLibOpacityThermoState& thermoState)
{
  if (!detail::GroupExists(file, "Scat_Iso_Kernels")) {
    return Hdf5LoadStatus::DatasetOpenFailed;
  }

  detail::ScopedHandle group(H5Gopen(file, "Scat_Iso_Kernels", H5P_DEFAULT), H5Gclose);
  if (!group.Valid()) {
    return Hdf5LoadStatus::DatasetOpenFailed;
  }

  // Read nOpacities and nMoments
  if (!detail::ReadScalarInt(group.Get(), "nOpacities", scatIso.nOpacities) ||
      !detail::ReadScalarInt(group.Get(), "nMoments", scatIso.nMoments)) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }

  // Set dimensions: [nE, nMom, nRho, nT, nYe]
  scatIso.dimensions[0] = energyGrid.nPoints;
  scatIso.dimensions[1] = scatIso.nMoments;
  scatIso.dimensions[2] = thermoState.dimensions[0];  // nRho
  scatIso.dimensions[3] = thermoState.dimensions[1];  // nT
  scatIso.dimensions[4] = thermoState.dimensions[2];  // nYe

  // Read Units
  std::vector<std::string> unitVec;
  if (detail::ReadStringArray(group.Get(), "Units", unitVec)) {
    for (int i = 0; i < WeakLibScatIsoTable::kNumSpecies && i < static_cast<int>(unitVec.size()); ++i) {
      scatIso.units[i] = unitVec[i];
    }
  }

  // Read Offsets (2D: [nOpacities, nMoments])
  std::array<int, 2> offsetDims{{scatIso.nOpacities, scatIso.nMoments}};
  if (!detail::ReadDoubleArray2d(group.Get(), "Offsets", scatIso.offsets, offsetDims)) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }

  // Read correction flags and ga_strange (optional, may not exist in legacy files)
  // Suppress HDF5 error messages for optional fields
  {
    H5E_auto2_t oldFunc;
    void* oldClientData;
    H5Eget_auto2(H5E_DEFAULT, &oldFunc, &oldClientData);
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);

    detail::ReadScalarInt(group.Get(), "weak_magnetism_corr", scatIso.weak_magnetism_corrections);
    detail::ReadScalarInt(group.Get(), "ion_ion_corr", scatIso.ion_ion_corrections);
    detail::ReadScalarInt(group.Get(), "many_body_corr", scatIso.many_body_corrections);

    detail::ScopedHandle ds(H5Dopen(group.Get(), "ga_strange", H5P_DEFAULT), H5Dclose);
    if (ds.Valid()) {
      H5Dread(ds.Get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &scatIso.ga_strange);
    }

    H5Eset_auto2(H5E_DEFAULT, oldFunc, oldClientData);
  }

  // Set species names
  scatIso.names[0] = "Electron Neutrino";
  scatIso.names[1] = "Electron Antineutrino";

  // Read kernel data for each species
  for (int iSpecies = 0; iSpecies < WeakLibScatIsoTable::kNumSpecies && iSpecies < scatIso.nOpacities; ++iSpecies) {
    if (!detail::ReadDoubleArray5d(group.Get(), scatIso.names[iSpecies].c_str(),
                                   scatIso.kernels[iSpecies], scatIso.dimensions)) {
      return Hdf5LoadStatus::DatasetReadFailed;
    }
  }

  // Compute layout
  scatIso.layout = MakeLayout(scatIso.dimensions.data(), 5);

  return Hdf5LoadStatus::Success;
}

/// Extract a contiguous 4D slice [nE, nRho, nT, nYe] from a 5D Iso kernel
/// at a fixed moment index.
///
/// The 5D kernel has dimensions [nE, nMom, nRho, nT, nYe] with column-major
/// strides [1, nE, nE*nMom, ...]. A slice at fixed iMom is not contiguous
/// because nMom interleaves the Rho stride.
///
/// @param kernel5d   Flat 5D kernel data (column-major ordered)
/// @param dims       5D dimensions: [nE, nMom, nRho, nT, nYe]
/// @param iMom       Moment index to extract (0-based)
/// @return           Contiguous 4D array [nE, nRho, nT, nYe]
inline amrex::Vector<double> ExtractIsoMomentSlice4D(
    const double* kernel5d,
    const std::array<int, 5>& dims,
    int iMom)
{
  const int nE   = dims[0];
  const int nMom = dims[1];
  const int nRho = dims[2];
  const int nT   = dims[3];
  const int nYe  = dims[4];

  const Layout layout5d = MakeLayout(dims.data(), 5);
  const std::size_t size4d =
      static_cast<std::size_t>(nE) * nRho * nT * nYe;
  amrex::Vector<double> result(size4d);

  const std::array<int, 4> dims4d{{nE, nRho, nT, nYe}};
  const Layout layout4d = MakeLayout(dims4d.data(), 4);

  for (int iYe = 0; iYe < nYe; ++iYe) {
    for (int iT = 0; iT < nT; ++iT) {
      for (int iRho = 0; iRho < nRho; ++iRho) {
        for (int iE = 0; iE < nE; ++iE) {
          const int src[5] = {iE, iMom, iRho, iT, iYe};
          const int dst[4] = {iE, iRho, iT, iYe};
          result[layout4d.Offset(dst)] = kernel5d[layout5d.Offset(src)];
        }
      }
    }
  }

  return result;
}

inline Hdf5LoadStatus LoadWeakLibScatNESTable(hid_t file,
                                               WeakLibScatNESTable& scatNES,
                                               const WeakLibOpacityGrid& energyGrid,
                                               const WeakLibOpacityGrid& etaGrid,
                                               const WeakLibOpacityThermoState& thermoState)
{
  if (!detail::GroupExists(file, "Scat_NES_Kernels")) {
    return Hdf5LoadStatus::DatasetOpenFailed;
  }

  detail::ScopedHandle group(H5Gopen(file, "Scat_NES_Kernels", H5P_DEFAULT), H5Gclose);
  if (!group.Valid()) {
    return Hdf5LoadStatus::DatasetOpenFailed;
  }

  // Read nOpacities and nMoments
  if (!detail::ReadScalarInt(group.Get(), "nOpacities", scatNES.nOpacities) ||
      !detail::ReadScalarInt(group.Get(), "nMoments", scatNES.nMoments)) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }

  // Set dimensions: [nE_in, nE_out, nMom, nT, nEta]
  scatNES.dimensions[0] = energyGrid.nPoints;   // nE_in
  scatNES.dimensions[1] = energyGrid.nPoints;   // nE_out
  scatNES.dimensions[2] = scatNES.nMoments;
  scatNES.dimensions[3] = thermoState.dimensions[1];  // nT
  scatNES.dimensions[4] = etaGrid.nPoints;

  // Read Units
  std::vector<std::string> unitVec;
  if (detail::ReadStringArray(group.Get(), "Units", unitVec) && !unitVec.empty()) {
    scatNES.unit = unitVec[0];
  }

  // Read Offsets (2D: [nOpacities, nMoments] in C order)
  // File stores Fortran order (nMoments, nOpacities), so C order is reversed
  std::array<int, 2> offsetDims{{scatNES.nOpacities, scatNES.nMoments}};
  if (!detail::ReadDoubleArray2d(group.Get(), "Offsets", scatNES.offsets, offsetDims)) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }

  // Read NPS flag (optional — suppress HDF5 error if absent)
  {
    H5E_auto2_t oldFunc;
    void* oldClientData;
    H5Eget_auto2(H5E_DEFAULT, &oldFunc, &oldClientData);
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    detail::ReadScalarInt(group.Get(), "NPS", scatNES.NPS);
    H5Eset_auto2(H5E_DEFAULT, oldFunc, oldClientData);
  }

  // Set kernel name
  scatNES.name = "Kernels";

  // Read kernel data
  if (!detail::ReadDoubleArray5d(group.Get(), "Kernels", scatNES.kernel, scatNES.dimensions)) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }

  // Compute layout
  scatNES.layout = MakeLayout(scatNES.dimensions.data(), 5);

  return Hdf5LoadStatus::Success;
}

inline Hdf5LoadStatus LoadWeakLibScatPairTable(hid_t file,
                                                WeakLibScatPairTable& scatPair,
                                                const WeakLibOpacityGrid& energyGrid,
                                                const WeakLibOpacityGrid& etaGrid,
                                                const WeakLibOpacityThermoState& thermoState)
{
  if (!detail::GroupExists(file, "Scat_Pair_Kernels")) {
    return Hdf5LoadStatus::DatasetOpenFailed;
  }

  detail::ScopedHandle group(H5Gopen(file, "Scat_Pair_Kernels", H5P_DEFAULT), H5Gclose);
  if (!group.Valid()) {
    return Hdf5LoadStatus::DatasetOpenFailed;
  }

  // Read nOpacities and nMoments
  if (!detail::ReadScalarInt(group.Get(), "nOpacities", scatPair.nOpacities) ||
      !detail::ReadScalarInt(group.Get(), "nMoments", scatPair.nMoments)) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }

  // Set dimensions: [nE_in, nE_out, nMom, nT, nEta]
  scatPair.dimensions[0] = energyGrid.nPoints;
  scatPair.dimensions[1] = energyGrid.nPoints;
  scatPair.dimensions[2] = scatPair.nMoments;
  scatPair.dimensions[3] = thermoState.dimensions[1];  // nT
  scatPair.dimensions[4] = etaGrid.nPoints;

  // Read Units
  std::vector<std::string> unitVec;
  if (detail::ReadStringArray(group.Get(), "Units", unitVec) && !unitVec.empty()) {
    scatPair.unit = unitVec[0];
  }

  // Read Offsets (2D: [nOpacities, nMoments] in C order)
  // File stores Fortran order (nMoments, nOpacities), so C order is reversed
  std::array<int, 2> offsetDims{{scatPair.nOpacities, scatPair.nMoments}};
  if (!detail::ReadDoubleArray2d(group.Get(), "Offsets", scatPair.offsets, offsetDims)) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }

  // Set kernel name
  scatPair.name = "Kernels";

  // Read kernel data
  if (!detail::ReadDoubleArray5d(group.Get(), "Kernels", scatPair.kernel, scatPair.dimensions)) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }

  // Compute layout
  scatPair.layout = MakeLayout(scatPair.dimensions.data(), 5);

  return Hdf5LoadStatus::Success;
}

inline Hdf5LoadStatus LoadWeakLibScatBremTable(hid_t file,
                                                WeakLibScatBremTable& scatBrem,
                                                const WeakLibOpacityGrid& energyGrid,
                                                const WeakLibOpacityThermoState& thermoState)
{
  if (!detail::GroupExists(file, "Scat_Brem_Kernels")) {
    return Hdf5LoadStatus::DatasetOpenFailed;
  }

  detail::ScopedHandle group(H5Gopen(file, "Scat_Brem_Kernels", H5P_DEFAULT), H5Gclose);
  if (!group.Valid()) {
    return Hdf5LoadStatus::DatasetOpenFailed;
  }

  // Read nOpacities and nMoments
  if (!detail::ReadScalarInt(group.Get(), "nOpacities", scatBrem.nOpacities) ||
      !detail::ReadScalarInt(group.Get(), "nMoments", scatBrem.nMoments)) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }

  // Set dimensions: [nE_in, nE_out, nMom, nRho, nT]
  scatBrem.dimensions[0] = energyGrid.nPoints;
  scatBrem.dimensions[1] = energyGrid.nPoints;
  scatBrem.dimensions[2] = scatBrem.nMoments;
  scatBrem.dimensions[3] = thermoState.dimensions[0];  // nRho
  scatBrem.dimensions[4] = thermoState.dimensions[1];  // nT

  // Read Units
  std::vector<std::string> unitVec;
  if (detail::ReadStringArray(group.Get(), "Units", unitVec) && !unitVec.empty()) {
    scatBrem.unit = unitVec[0];
  }

  // Read Offsets (2D: [nOpacities, nMoments] in C order)
  // File stores Fortran order (nMoments, nOpacities), so C order is reversed
  std::array<int, 2> offsetDims{{scatBrem.nOpacities, scatBrem.nMoments}};
  if (!detail::ReadDoubleArray2d(group.Get(), "Offsets", scatBrem.offsets, offsetDims)) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }

  // Set kernel name
  scatBrem.name = "S_sigma";

  // Read kernel data
  if (!detail::ReadDoubleArray5d(group.Get(), "S_sigma", scatBrem.kernel, scatBrem.dimensions)) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }

  // Compute layout
  scatBrem.layout = MakeLayout(scatBrem.dimensions.data(), 5);

  return Hdf5LoadStatus::Success;
}

namespace detail {

inline Hdf5LoadStatus LoadWeakLibOpacityThermoState(hid_t file,
                                                     WeakLibOpacityThermoState& ts)
{
  ScopedHandle group(H5Gopen(file, "ThermoState", H5P_DEFAULT), H5Gclose);
  if (!group.Valid()) {
    return Hdf5LoadStatus::DatasetOpenFailed;
  }

  // Read LogInterp
  int logInterp[3] = {0, 0, 0};
  if (!ReadIntArray(group.Get(), "LogInterp", logInterp, 3)) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }
  ts.scales[0] = (logInterp[0] == 1) ? AxisScale::Log10 : AxisScale::Linear;
  ts.scales[1] = (logInterp[1] == 1) ? AxisScale::Log10 : AxisScale::Linear;
  ts.scales[2] = (logInterp[2] == 1) ? AxisScale::Log10 : AxisScale::Linear;

  // Read Dimensions
  int dims[3] = {0, 0, 0};
  if (!ReadIntArray(group.Get(), "Dimensions", dims, 3)) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }
  ts.dimensions[0] = dims[0];  // nRho
  ts.dimensions[1] = dims[1];  // nT
  ts.dimensions[2] = dims[2];  // nYe

  // Read Names and Units
  std::vector<std::string> namesVec, unitsVec;
  if (!ReadStringArray(group.Get(), "Names", namesVec) || namesVec.size() != 3) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }
  if (!ReadStringArray(group.Get(), "Units", unitsVec) || unitsVec.size() != 3) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }
  for (int i = 0; i < 3; ++i) {
    ts.names[i] = namesVec[i];
    ts.units[i] = unitsVec[i];
  }

  // Load axis data
  Hdf5Table tempTable;
  tempTable.extents = {ts.dimensions[0], ts.dimensions[1], ts.dimensions[2], 1, 1};

  Hdf5LoadStatus status;
  status = LoadWeakLibAxis(group.Get(), "Density", ts.dimensions[0], ts.scales[0], 0, tempTable);
  if (status != Hdf5LoadStatus::Success) return status;

  status = LoadWeakLibAxis(group.Get(), "Temperature", ts.dimensions[1], ts.scales[1], 1, tempTable);
  if (status != Hdf5LoadStatus::Success) return status;

  status = LoadWeakLibAxis(group.Get(), "Electron Fraction", ts.dimensions[2], ts.scales[2], 2, tempTable);
  if (status != Hdf5LoadStatus::Success) return status;

  // Transfer to thermoState
  for (int i = 0; i < 3; ++i) {
    ts.axisStorage[i] = std::move(tempTable.axisStorage[i]);
    ts.axes[i] = tempTable.axes[i];
    ts.axes[i].grid = ts.axisStorage[i].data();
  }

  return Hdf5LoadStatus::Success;
}

} // namespace detail

inline Hdf5LoadStatus LoadWeakLibOpacityTableFull(
    WeakLibOpacityTable& output,
    const std::string& fileEmAb = "",
    const std::string& fileIso = "",
    const std::string& fileNES = "",
    const std::string& filePair = "",
    const std::string& fileBrem = "")
{
  WeakLibOpacityTable result;
  Hdf5LoadStatus status;

  // Check that at least one file is provided
  const bool hasEmAb = !fileEmAb.empty();
  const bool hasIso = !fileIso.empty();
  const bool hasNES = !fileNES.empty();
  const bool hasPair = !filePair.empty();
  const bool hasBrem = !fileBrem.empty();

  if (!hasEmAb && !hasIso && !hasNES && !hasPair && !hasBrem) {
    return Hdf5LoadStatus::FileOpenFailed;
  }

  // Determine first available file for reading shared grids
  std::string firstFile;
  if (hasEmAb) firstFile = fileEmAb;
  else if (hasIso) firstFile = fileIso;
  else if (hasNES) firstFile = fileNES;
  else if (hasPair) firstFile = filePair;
  else firstFile = fileBrem;

  // Load EnergyGrid from first file
  {
    detail::ScopedHandle file(H5Fopen(firstFile.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
    if (!file.Valid()) {
      return Hdf5LoadStatus::FileOpenFailed;
    }

    status = detail::LoadWeakLibOpacityGrid(file.Get(), "EnergyGrid", result.energyGrid);
    if (status != Hdf5LoadStatus::Success) {
      return status;
    }

    // Load ThermoState from first file
    status = detail::LoadWeakLibOpacityThermoState(file.Get(), result.thermoState);
    if (status != Hdf5LoadStatus::Success) {
      return status;
    }
  }

  // Load EtaGrid if NES or Pair files are provided
  if (hasNES || hasPair) {
    std::string etaFile = hasNES ? fileNES : filePair;
    detail::ScopedHandle file(H5Fopen(etaFile.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
    if (!file.Valid()) {
      return Hdf5LoadStatus::FileOpenFailed;
    }

    status = detail::LoadWeakLibOpacityGrid(file.Get(), "EtaGrid", result.etaGrid);
    if (status != Hdf5LoadStatus::Success) {
      return status;
    }
  }

  // Load EmAb
  if (hasEmAb) {
    detail::ScopedHandle file(H5Fopen(fileEmAb.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
    if (!file.Valid()) {
      return Hdf5LoadStatus::FileOpenFailed;
    }

    status = LoadWeakLibEmAbTable(file.Get(), result.emAb, result.energyGrid, result.thermoState);
    if (status != Hdf5LoadStatus::Success) {
      return status;
    }
  }

  // Load Iso
  if (hasIso) {
    detail::ScopedHandle file(H5Fopen(fileIso.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
    if (!file.Valid()) {
      return Hdf5LoadStatus::FileOpenFailed;
    }

    status = LoadWeakLibScatIsoTable(file.Get(), result.scatIso, result.energyGrid, result.thermoState);
    if (status != Hdf5LoadStatus::Success) {
      return status;
    }
  }

  // Load NES
  if (hasNES) {
    detail::ScopedHandle file(H5Fopen(fileNES.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
    if (!file.Valid()) {
      return Hdf5LoadStatus::FileOpenFailed;
    }

    status = LoadWeakLibScatNESTable(file.Get(), result.scatNES, result.energyGrid, result.etaGrid, result.thermoState);
    if (status != Hdf5LoadStatus::Success) {
      return status;
    }
  }

  // Load Pair
  if (hasPair) {
    detail::ScopedHandle file(H5Fopen(filePair.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
    if (!file.Valid()) {
      return Hdf5LoadStatus::FileOpenFailed;
    }

    status = LoadWeakLibScatPairTable(file.Get(), result.scatPair, result.energyGrid, result.etaGrid, result.thermoState);
    if (status != Hdf5LoadStatus::Success) {
      return status;
    }
  }

  // Load Brem
  if (hasBrem) {
    detail::ScopedHandle file(H5Fopen(fileBrem.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
    if (!file.Valid()) {
      return Hdf5LoadStatus::FileOpenFailed;
    }

    status = LoadWeakLibScatBremTable(file.Get(), result.scatBrem, result.energyGrid, result.thermoState);
    if (status != Hdf5LoadStatus::Success) {
      return status;
    }
  }

  output = std::move(result);
  return Hdf5LoadStatus::Success;
}

inline WeakLibEosTableDevice MakeDeviceCopy(const WeakLibEosTable& host,
                                             amrex::Arena* arena = amrex::The_Device_Arena())
{
  WeakLibEosTableDevice device{};
  device.nVariables = host.nVariables;
  device.dimensions = host.dimensions;
  device.layout = host.layout;
  device.indices = host.indices;
  device.offsets = host.offsets;

  // Copy axes to device
  for (int dim = 0; dim < 3; ++dim) {
    const auto& hostAxis = host.axisStorage[dim];
    amrex::Gpu::DeviceVector<double>& deviceAxis = device.axisStorage[dim];
    deviceAxis.resize(hostAxis.size());
    if (!hostAxis.empty()) {
      amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                       hostAxis.begin(), hostAxis.end(),
                       deviceAxis.begin());
    }
    device.axes[dim].grid = deviceAxis.data();
    device.axes[dim].n = host.axes[dim].n;
    device.axes[dim].scale = host.axes[dim].scale;
  }

  // Copy all variable data to device
  const amrex::Array<int, 3> lo{{0, 0, 0}};
  const amrex::Array<int, 3> hi{{host.dimensions[0] - 1, host.dimensions[1] - 1, host.dimensions[2] - 1}};

  device.variables.resize(host.nVariables);
  for (int iVar = 0; iVar < host.nVariables; ++iVar) {
    device.variables[iVar].resize(lo, hi, arena);
    device.variables[iVar].copy(host.variables[iVar]);
  }

  // Copy Repaired mask to device
  device.repaired.resize(lo, hi, arena);
  device.repaired.copy(host.repaired);

  return device;
}

namespace detail {

inline void CopyGridToDevice(const WeakLibOpacityGrid& host, WeakLibOpacityGridDevice& device,
                              amrex::Arena* arena = amrex::The_Device_Arena())
{
  device.nPoints = host.nPoints;
  device.scale = host.scale;
  device.values.resize(host.values.size());
  if (!host.values.empty()) {
    amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                     host.values.begin(), host.values.end(),
                     device.values.begin());
  }
}

inline void CopyThermoStateToDevice(const WeakLibOpacityThermoState& host,
                                     WeakLibOpacityThermoStateDevice& device,
                                     amrex::Arena* arena = amrex::The_Device_Arena())
{
  device.dimensions = host.dimensions;
  device.scales = host.scales;

  for (int i = 0; i < 3; ++i) {
    const auto& hostAxis = host.axisStorage[i];
    auto& deviceAxis = device.axisStorage[i];
    deviceAxis.resize(hostAxis.size());
    if (!hostAxis.empty()) {
      amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                       hostAxis.begin(), hostAxis.end(),
                       deviceAxis.begin());
    }
    device.axes[i].grid = deviceAxis.data();
    device.axes[i].n = host.axes[i].n;
    device.axes[i].scale = host.axes[i].scale;
  }
}

} // namespace detail

inline WeakLibOpacityTableDevice MakeDeviceCopy(const WeakLibOpacityTable& host,
                                                 amrex::Arena* arena = amrex::The_Device_Arena())
{
  WeakLibOpacityTableDevice device{};

  // Copy grids
  detail::CopyGridToDevice(host.energyGrid, device.energyGrid, arena);
  detail::CopyGridToDevice(host.etaGrid, device.etaGrid, arena);
  detail::CopyThermoStateToDevice(host.thermoState, device.thermoState, arena);

  // Copy EmAb
  if (host.emAb.IsLoaded()) {
    device.emAb.nOpacities = host.emAb.nOpacities;
    device.emAb.dimensions = host.emAb.dimensions;
    device.emAb.offsets = host.emAb.offsets;
    device.emAb.parameters = host.emAb.parameters;
    device.emAb.layout = host.emAb.layout;

    const amrex::Array<int, 4> lo{{0, 0, 0, 0}};
    const amrex::Array<int, 4> hi{{host.emAb.dimensions[0] - 1, host.emAb.dimensions[1] - 1,
                                    host.emAb.dimensions[2] - 1, host.emAb.dimensions[3] - 1}};

    for (int i = 0; i < WeakLibEmAbTable::kNumSpecies; ++i) {
      if (host.emAb.opacities[i].size() > 0) {
        device.emAb.opacities[i].resize(lo, hi, arena);
        device.emAb.opacities[i].copy(host.emAb.opacities[i]);
      }
    }

    // Copy EC table if present
    if (host.emAb.ecTable.IsPresent()) {
      auto& hostEC = host.emAb.ecTable;
      auto& deviceEC = device.emAb.ecTable;

      deviceEC.nE = hostEC.nE;
      deviceEC.nRho = hostEC.nRho;
      deviceEC.nT = hostEC.nT;
      deviceEC.nYe = hostEC.nYe;
      deviceEC.rhoMin = hostEC.rhoMin;
      deviceEC.rhoMax = hostEC.rhoMax;
      deviceEC.tempMin = hostEC.tempMin;
      deviceEC.tempMax = hostEC.tempMax;
      deviceEC.yeMin = hostEC.yeMin;
      deviceEC.yeMax = hostEC.yeMax;
      deviceEC.specOffset = hostEC.specOffset;
      deviceEC.rateOffset = hostEC.rateOffset;

      // Copy axis values
      deviceEC.energyValues.resize(hostEC.energyValues.size());
      deviceEC.rhoValues.resize(hostEC.rhoValues.size());
      deviceEC.tempValues.resize(hostEC.tempValues.size());
      deviceEC.yeValues.resize(hostEC.yeValues.size());

      amrex::Gpu::copy(amrex::Gpu::hostToDevice, hostEC.energyValues.begin(), hostEC.energyValues.end(), deviceEC.energyValues.begin());
      amrex::Gpu::copy(amrex::Gpu::hostToDevice, hostEC.rhoValues.begin(), hostEC.rhoValues.end(), deviceEC.rhoValues.begin());
      amrex::Gpu::copy(amrex::Gpu::hostToDevice, hostEC.tempValues.begin(), hostEC.tempValues.end(), deviceEC.tempValues.begin());
      amrex::Gpu::copy(amrex::Gpu::hostToDevice, hostEC.yeValues.begin(), hostEC.yeValues.end(), deviceEC.yeValues.begin());

      // Copy spectrum and rate
      const amrex::Array<int, 4> specLo{{0, 0, 0, 0}};
      const amrex::Array<int, 4> specHi{{hostEC.nRho - 1, hostEC.nT - 1, hostEC.nYe - 1, hostEC.nE - 1}};
      deviceEC.spectrum.resize(specLo, specHi, arena);
      deviceEC.spectrum.copy(hostEC.spectrum);

      const amrex::Array<int, 3> rateLo{{0, 0, 0}};
      const amrex::Array<int, 3> rateHi{{hostEC.nRho - 1, hostEC.nT - 1, hostEC.nYe - 1}};
      deviceEC.rate.resize(rateLo, rateHi, arena);
      deviceEC.rate.copy(hostEC.rate);
    }
  }

  // Copy ScatIso
  if (host.scatIso.IsLoaded()) {
    device.scatIso.nOpacities = host.scatIso.nOpacities;
    device.scatIso.nMoments = host.scatIso.nMoments;
    device.scatIso.dimensions = host.scatIso.dimensions;
    device.scatIso.weak_magnetism_corrections = host.scatIso.weak_magnetism_corrections;
    device.scatIso.ion_ion_corrections = host.scatIso.ion_ion_corrections;
    device.scatIso.many_body_corrections = host.scatIso.many_body_corrections;
    device.scatIso.ga_strange = host.scatIso.ga_strange;
    device.scatIso.layout = host.scatIso.layout;

    const amrex::Array<int, 2> offLo{{0, 0}};
    const amrex::Array<int, 2> offHi{{host.scatIso.nOpacities - 1, host.scatIso.nMoments - 1}};
    device.scatIso.offsets.resize(offLo, offHi, arena);
    device.scatIso.offsets.copy(host.scatIso.offsets);

    // Kernels are stored as flat vectors
    for (int i = 0; i < WeakLibScatIsoTable::kNumSpecies; ++i) {
      if (!host.scatIso.kernels[i].empty()) {
        device.scatIso.kernels[i].resize(host.scatIso.kernels[i].size());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                         host.scatIso.kernels[i].begin(), host.scatIso.kernels[i].end(),
                         device.scatIso.kernels[i].begin());
      }
    }
  }

  // Copy ScatNES
  if (host.scatNES.IsLoaded()) {
    device.scatNES.nOpacities = host.scatNES.nOpacities;
    device.scatNES.nMoments = host.scatNES.nMoments;
    device.scatNES.dimensions = host.scatNES.dimensions;
    device.scatNES.NPS = host.scatNES.NPS;
    device.scatNES.layout = host.scatNES.layout;

    const amrex::Array<int, 2> offLo{{0, 0}};
    const amrex::Array<int, 2> offHi{{host.scatNES.nOpacities - 1, host.scatNES.nMoments - 1}};
    device.scatNES.offsets.resize(offLo, offHi, arena);
    device.scatNES.offsets.copy(host.scatNES.offsets);

    // Kernel is stored as flat vector
    if (!host.scatNES.kernel.empty()) {
      device.scatNES.kernel.resize(host.scatNES.kernel.size());
      amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                       host.scatNES.kernel.begin(), host.scatNES.kernel.end(),
                       device.scatNES.kernel.begin());
    }
  }

  // Copy ScatPair
  if (host.scatPair.IsLoaded()) {
    device.scatPair.nOpacities = host.scatPair.nOpacities;
    device.scatPair.nMoments = host.scatPair.nMoments;
    device.scatPair.dimensions = host.scatPair.dimensions;
    device.scatPair.layout = host.scatPair.layout;

    const amrex::Array<int, 2> offLo{{0, 0}};
    const amrex::Array<int, 2> offHi{{host.scatPair.nOpacities - 1, host.scatPair.nMoments - 1}};
    device.scatPair.offsets.resize(offLo, offHi, arena);
    device.scatPair.offsets.copy(host.scatPair.offsets);

    // Kernel is stored as flat vector
    if (!host.scatPair.kernel.empty()) {
      device.scatPair.kernel.resize(host.scatPair.kernel.size());
      amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                       host.scatPair.kernel.begin(), host.scatPair.kernel.end(),
                       device.scatPair.kernel.begin());
    }
  }

  // Copy ScatBrem
  if (host.scatBrem.IsLoaded()) {
    device.scatBrem.nOpacities = host.scatBrem.nOpacities;
    device.scatBrem.nMoments = host.scatBrem.nMoments;
    device.scatBrem.dimensions = host.scatBrem.dimensions;
    device.scatBrem.layout = host.scatBrem.layout;

    const amrex::Array<int, 2> offLo{{0, 0}};
    const amrex::Array<int, 2> offHi{{host.scatBrem.nOpacities - 1, host.scatBrem.nMoments - 1}};
    device.scatBrem.offsets.resize(offLo, offHi, arena);
    device.scatBrem.offsets.copy(host.scatBrem.offsets);

    // Kernel is stored as flat vector
    if (!host.scatBrem.kernel.empty()) {
      device.scatBrem.kernel.resize(host.scatBrem.kernel.size());
      amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                       host.scatBrem.kernel.begin(), host.scatBrem.kernel.end(),
                       device.scatBrem.kernel.begin());
    }
  }

  return device;
}

inline Hdf5LoadStatus LoadHdf5TableParallel(const std::string& filePath,
                                            Hdf5Table& output,
                                            const Hdf5LoadConfig& cfg = Hdf5LoadConfig{},
                                            int readerRank = amrex::ParallelDescriptor::IOProcessorNumber())
{
  const int nProcs = amrex::ParallelDescriptor::NProcs();
  if (nProcs <= 1) {
    return LoadHdf5Table(filePath, output, cfg);
  }

  int root = readerRank;
  if (root < 0 || root >= nProcs) {
    root = amrex::ParallelDescriptor::IOProcessorNumber();
  }

  const int myRank = amrex::ParallelDescriptor::MyProc();

  Hdf5Table localTable;
  Hdf5LoadStatus status = Hdf5LoadStatus::Success;
  if (myRank == root) {
    status = LoadHdf5Table(filePath, localTable, cfg);
  }

  int statusInt = static_cast<int>(status);
  amrex::ParallelDescriptor::Bcast(&statusInt, 1, root);
  status = static_cast<Hdf5LoadStatus>(statusInt);
  if (status != Hdf5LoadStatus::Success) {
    return status;
  }

  int header[6] = {0, 1, 1, 1, 1, 1};
  if (myRank == root) {
    header[0] = localTable.nd;
    for (int dim = 0; dim < 5; ++dim) {
      header[1 + dim] = localTable.extents[dim];
    }
  }
  amrex::ParallelDescriptor::Bcast(header, 6, root);

  const int nd = header[0];
  std::array<int, 5> extents{{1, 1, 1, 1, 1}};
  for (int dim = 0; dim < 5; ++dim) {
    extents[dim] = header[1 + dim];
  }

  int axisCounts[5] = {0, 0, 0, 0, 0};
  int axisScales[5] = {static_cast<int>(AxisScale::Linear),
                       static_cast<int>(AxisScale::Linear),
                       static_cast<int>(AxisScale::Linear),
                       static_cast<int>(AxisScale::Linear),
                       static_cast<int>(AxisScale::Linear)};

  if (myRank == root) {
    for (int dim = 0; dim < 5; ++dim) {
      axisCounts[dim] = localTable.axes[dim].n;
      axisScales[dim] = static_cast<int>(localTable.axes[dim].scale);
    }
  }

  amrex::ParallelDescriptor::Bcast(axisCounts, 5, root);
  amrex::ParallelDescriptor::Bcast(axisScales, 5, root);

  const std::size_t totalSize = detail::ComputeTotalSize(nd, extents);
  if (totalSize > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return Hdf5LoadStatus::IncompatibleDatasetExtent;
  }

  if (myRank != root) {
    output = Hdf5Table{};
    output.nd = nd;
    output.extents = extents;
    bool overflow = false;
    const amrex::Array<int, 4> lo{{0, 0, 0, 0}};
    const amrex::Array<int, 4> hi = detail::MakeHiArray(nd, output.extents, overflow);
    if (overflow || totalSize == 0) {
      return Hdf5LoadStatus::IncompatibleDatasetExtent;
    }
    output.values.resize(lo, hi, amrex::The_Pinned_Arena());
    output.layout = MakeLayout(output.extents.data(), output.nd);
    for (int dim = 0; dim < 5; ++dim) {
      amrex::Vector<double>& storage = output.axisStorage[dim];
      storage.resize(static_cast<std::size_t>(axisCounts[dim]));
      if (!storage.empty()) {
        output.axes[dim].grid = storage.data();
      } else {
        output.axes[dim].grid = nullptr;
      }
      output.axes[dim].n = axisCounts[dim];
      output.axes[dim].scale = static_cast<AxisScale>(axisScales[dim]);
    }
  }

  double* dataPtr = (myRank == root)
                        ? localTable.values.table().p
                        : output.values.table().p;
  if (totalSize > 0) {
    amrex::ParallelDescriptor::Bcast(dataPtr,
                                     static_cast<int>(totalSize),
                                     root);
  }

  for (int dim = 0; dim < 5; ++dim) {
    const int count = axisCounts[dim];
    if (count <= 0) {
      continue;
    }
    double* axisPtr = (myRank == root)
                          ? localTable.axisStorage[dim].data()
                          : output.axisStorage[dim].data();
    amrex::ParallelDescriptor::Bcast(axisPtr, count, root);
  }

  if (myRank == root) {
    output = std::move(localTable);
  }

  return status;
}

inline Hdf5LoadStatus LoadWeakLibEosTableFullParallel(
    const std::string& filePath,
    WeakLibEosTable& output,
    int readerRank = amrex::ParallelDescriptor::IOProcessorNumber())
{
  // Single-rank fallback
  const int nProcs = amrex::ParallelDescriptor::NProcs();
  if (nProcs <= 1) {
    return LoadWeakLibEosTableFull(filePath, output);
  }

  // Validate readerRank
  int root = readerRank;
  if (root < 0 || root >= nProcs) {
    root = amrex::ParallelDescriptor::IOProcessorNumber();
  }

  const int myRank = amrex::ParallelDescriptor::MyProc();

  // Root rank loads the table
  WeakLibEosTable localTable;
  Hdf5LoadStatus status = Hdf5LoadStatus::Success;
  if (myRank == root) {
    status = LoadWeakLibEosTableFull(filePath, localTable);
  }

  // Broadcast status
  int statusInt = static_cast<int>(status);
  amrex::ParallelDescriptor::Bcast(&statusInt, 1, root);
  status = static_cast<Hdf5LoadStatus>(statusInt);
  if (status != Hdf5LoadStatus::Success) {
    return status;
  }

  // Broadcast header: [nVariables, dims x3, axisScales x3]
  int header[7] = {0, 0, 0, 0, 0, 0, 0};
  if (myRank == root) {
    header[0] = localTable.nVariables;
    header[1] = localTable.dimensions[0];
    header[2] = localTable.dimensions[1];
    header[3] = localTable.dimensions[2];
    header[4] = static_cast<int>(localTable.axes[0].scale);
    header[5] = static_cast<int>(localTable.axes[1].scale);
    header[6] = static_cast<int>(localTable.axes[2].scale);
  }
  amrex::ParallelDescriptor::Bcast(header, 7, root);

  const int nVariables = header[0];
  std::array<int, 3> dimensions{{header[1], header[2], header[3]}};
  std::array<AxisScale, 3> axisScales{{
      static_cast<AxisScale>(header[4]),
      static_cast<AxisScale>(header[5]),
      static_cast<AxisScale>(header[6])
  }};

  // Broadcast axis counts
  int axisCounts[3] = {0, 0, 0};
  if (myRank == root) {
    for (int dim = 0; dim < 3; ++dim) {
      axisCounts[dim] = localTable.axes[dim].n;
    }
  }
  amrex::ParallelDescriptor::Bcast(axisCounts, 3, root);

  // Non-root: allocate base structure
  if (myRank != root) {
    output = WeakLibEosTable{};
    output.nVariables = nVariables;
    output.dimensions = dimensions;

    // Allocate axis storage
    for (int dim = 0; dim < 3; ++dim) {
      output.axisStorage[dim].resize(axisCounts[dim]);
      output.axes[dim].grid = output.axisStorage[dim].data();
      output.axes[dim].n = axisCounts[dim];
      output.axes[dim].scale = axisScales[dim];
    }

    // Allocate variable arrays
    const amrex::Array<int, 3> lo{{0, 0, 0}};
    const amrex::Array<int, 3> hi{{dimensions[0] - 1, dimensions[1] - 1, dimensions[2] - 1}};
    output.variables.resize(nVariables);
    for (int iVar = 0; iVar < nVariables; ++iVar) {
      output.variables[iVar].resize(lo, hi, amrex::The_Pinned_Arena());
    }

    // Allocate repaired mask
    output.repaired.resize(lo, hi, amrex::The_Pinned_Arena());

    // Allocate offset/string vectors
    output.offsets.resize(nVariables);
    output.variableNames.resize(nVariables);
    output.variableUnits.resize(nVariables);

    // Compute layout
    std::array<int, 5> extents5{{dimensions[0], dimensions[1], dimensions[2], 1, 1}};
    output.layout = MakeLayout(extents5.data(), 3);
  }

  // Get reference to working table
  WeakLibEosTable& table = (myRank == root) ? localTable : output;

  // Broadcast axis grids
  for (int dim = 0; dim < 3; ++dim) {
    if (axisCounts[dim] > 0) {
      amrex::ParallelDescriptor::Bcast(table.axisStorage[dim].data(), axisCounts[dim], root);
    }
  }

  // Broadcast string metadata
  detail::BcastStringArray(table.axisNames, root);
  detail::BcastStringArray(table.axisUnits, root);
  detail::BcastStringVector(table.variableNames, root);
  detail::BcastStringVector(table.variableUnits, root);

  // Broadcast offsets
  if (nVariables > 0) {
    amrex::ParallelDescriptor::Bcast(table.offsets.data(), nVariables, root);
  }

  // Broadcast each variable's 3D data
  const std::size_t varSize = static_cast<std::size_t>(dimensions[0]) *
                              static_cast<std::size_t>(dimensions[1]) *
                              static_cast<std::size_t>(dimensions[2]);
  if (varSize > 0) {
    for (int iVar = 0; iVar < nVariables; ++iVar) {
      amrex::ParallelDescriptor::Bcast(table.variables[iVar].table().p,
                                       static_cast<int>(varSize), root);
    }
  }

  // Broadcast repaired mask
  if (varSize > 0) {
    amrex::ParallelDescriptor::Bcast(table.repaired.table().p,
                                     static_cast<int>(varSize), root);
  }

  // Broadcast indices (15 integers as a block)
  static_assert(sizeof(WeakLibEosIndices) == 15 * sizeof(int),
                "WeakLibEosIndices must be 15 contiguous ints");
  amrex::ParallelDescriptor::Bcast(reinterpret_cast<int*>(&table.indices), 15, root);

  // Root moves its local table to output
  if (myRank == root) {
    output = std::move(localTable);
  }

  return status;
}

} // namespace WeakLibReader
