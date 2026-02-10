#pragma once

namespace WeakLibReader {
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
    if (!detail::ReadWeakLibArrayNd<double, 3>(dvGroup.Get(), varName.c_str(),
                                   result.variables[iVar], result.dimensions)) {
      return Hdf5LoadStatus::DatasetReadFailed;
    }
  }

  // Read Repaired mask
  if (!detail::ReadWeakLibArrayNd<int, 3>(dvGroup.Get(), "Repaired", result.repaired, result.dimensions)) {
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
  device.variables.resize(host.nVariables);
  for (int iVar = 0; iVar < host.nVariables; ++iVar) {
    device.variables[iVar].resize(host.variables[iVar].size());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                     host.variables[iVar].begin(), host.variables[iVar].end(),
                     device.variables[iVar].begin());
  }

  // Copy Repaired mask to device
  device.repaired.resize(host.repaired.size());
  amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                   host.repaired.begin(), host.repaired.end(),
                   device.repaired.begin());

  return device;
}

} // namespace WeakLibReader
