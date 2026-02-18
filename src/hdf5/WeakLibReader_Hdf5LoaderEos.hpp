#pragma once

#include "hdf5/WeakLibReader_Hdf5Types.hpp"
#include "hdf5/WeakLibReader_Hdf5LoaderDetail.hpp"

namespace WeakLibReader {
inline Hdf5LoadStatus LoadWeakLibEosTableFull(const std::string& filePath,
                                               WeakLibEosTable& output)
{
  WeakLibEosTable result;

  detail::ScopedHandle file(H5Fopen(filePath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
  if (!file.Valid()) {
    return Hdf5LoadStatus::FileOpenFailed;
  }

  detail::ScopedHandle thermoGroup(H5Gopen(file.Get(), "ThermoState", H5P_DEFAULT), H5Gclose);
  if (!thermoGroup.Valid()) {
    return Hdf5LoadStatus::DatasetOpenFailed;
  }

  int logInterp[3] = {0, 0, 0};
  if (!detail::ReadIntArray(thermoGroup.Get(), "LogInterp", logInterp, 3)) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }
  std::array<AxisScale, 3> scales{};
  for (int i = 0; i < 3; ++i) {
    scales[i] = (logInterp[i] == 1) ? AxisScale::Log10 : AxisScale::Linear;
  }

  // File order is [nRho, nT, nYe]
  int fileDims[3] = {0, 0, 0};
  if (!detail::ReadIntArray(thermoGroup.Get(), "Dimensions", fileDims, 3)) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }
  for (int i = 0; i < 3; ++i) {
    result.dimensions[i] = fileDims[i];
  }

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

  // Load axes: [Density, Temperature, Electron Fraction]
  const char* axisDatasetNames[3] = {"Density", "Temperature", "Electron Fraction"};
  for (int i = 0; i < 3; ++i) {
    Hdf5LoadStatus axisStatus = detail::LoadWeakLibAxis(
        thermoGroup.Get(), axisDatasetNames[i],
        result.dimensions[i], scales[i],
        result.axisStorage[i], result.axes[i]);
    if (axisStatus != Hdf5LoadStatus::Success) {
      return axisStatus;
    }
  }

  detail::ScopedHandle dvGroup(H5Gopen(file.Get(), "DependentVariables", H5P_DEFAULT), H5Gclose);
  if (!dvGroup.Valid()) {
    return Hdf5LoadStatus::DatasetOpenFailed;
  }

  if (!detail::ReadScalarInt(dvGroup.Get(), "nVariables", result.nVariables)) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }

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

  {
    std::array<int, 1> offsetDims{{result.nVariables}};
    if (!detail::ReadWeakLibArrayNd<double, 1>(dvGroup.Get(), "Offsets",
                                               result.offsets, offsetDims)) {
      return Hdf5LoadStatus::DatasetReadFailed;
    }
  }

  result.variables.resize(result.nVariables);
  for (int iVar = 0; iVar < result.nVariables; ++iVar) {
    const std::string& varName = result.variableNames[iVar];
    if (!detail::ReadWeakLibArrayNd<double, 3>(dvGroup.Get(), varName.c_str(),
                                   result.variables[iVar], result.dimensions)) {
      return Hdf5LoadStatus::DatasetReadFailed;
    }
  }

  if (!detail::ReadWeakLibArrayNd<int, 3>(dvGroup.Get(), "Repaired", result.repaired, result.dimensions)) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }

  // Read Fortran 1-based index mappings and normalize to 0-based for C++.
  {
    struct IndexEntry { const char* name; int WeakLibEosIndices::* field; };
    const IndexEntry entries[] = {
      {"iPressure",                  &WeakLibEosIndices::iPressure},
      {"iEntropyPerBaryon",          &WeakLibEosIndices::iEntropyPerBaryon},
      {"iInternalEnergyDensity",     &WeakLibEosIndices::iInternalEnergyDensity},
      {"iElectronChemicalPotential", &WeakLibEosIndices::iElectronChemicalPotential},
      {"iProtonChemicalPotential",   &WeakLibEosIndices::iProtonChemicalPotential},
      {"iNeutronChemicalPotential",  &WeakLibEosIndices::iNeutronChemicalPotential},
      {"iProtonMassFraction",        &WeakLibEosIndices::iProtonMassFraction},
      {"iNeutronMassFraction",       &WeakLibEosIndices::iNeutronMassFraction},
      {"iAlphaMassFraction",         &WeakLibEosIndices::iAlphaMassFraction},
      {"iHeavyMassFraction",         &WeakLibEosIndices::iHeavyMassFraction},
      {"iHeavyChargeNumber",         &WeakLibEosIndices::iHeavyChargeNumber},
      {"iHeavyMassNumber",           &WeakLibEosIndices::iHeavyMassNumber},
      {"iHeavyBindingEnergy",        &WeakLibEosIndices::iHeavyBindingEnergy},
      {"iThermalEnergy",             &WeakLibEosIndices::iThermalEnergy},
      {"iGamma1",                    &WeakLibEosIndices::iGamma1},
    };
    for (const auto& entry : entries) {
      int& idx = result.indices.*entry.field;
      if (!detail::ReadScalarInt(dvGroup.Get(), entry.name, idx)) {
        return Hdf5LoadStatus::DatasetReadFailed;
      }
      if (idx > 0) {
        --idx;
      }
    }
  }

  const std::array<int, 5> extents5{{result.dimensions[0], result.dimensions[1], result.dimensions[2], 1, 1}};
  result.layout = MakeLayout(extents5.data(), 3);

  output = std::move(result);
  return Hdf5LoadStatus::Success;
}

inline WeakLibEosTableDevice MakeDeviceCopy(const WeakLibEosTable& host)
{
  WeakLibEosTableDevice device{};
  device.nVariables = host.nVariables;
  device.dimensions = host.dimensions;
  device.layout = host.layout;
  device.indices = host.indices;
  device.offsets.resize(host.offsets.size());
  amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                   host.offsets.begin(), host.offsets.end(),
                   device.offsets.begin());

  for (int dim = 0; dim < 3; ++dim) {
    const auto& hostAxis = host.axisStorage[dim];
    auto& deviceAxis = device.axisStorage[dim];
    deviceAxis.resize(hostAxis.size());
    if (!hostAxis.empty()) {
      amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                       hostAxis.begin(), hostAxis.end(),
                       deviceAxis.begin());
    }
    device.axes[dim] = Axis{deviceAxis.data(), host.axes[dim].n, host.axes[dim].scale};
  }

  device.variables.resize(host.nVariables);
  for (int iVar = 0; iVar < host.nVariables; ++iVar) {
    device.variables[iVar].resize(host.variables[iVar].size());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                     host.variables[iVar].begin(), host.variables[iVar].end(),
                     device.variables[iVar].begin());
  }

  device.repaired.resize(host.repaired.size());
  amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                   host.repaired.begin(), host.repaired.end(),
                   device.repaired.begin());

  return device;
}

} // namespace WeakLibReader
