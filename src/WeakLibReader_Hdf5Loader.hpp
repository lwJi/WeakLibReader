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

inline Hdf5LoadStatus LoadWeakLibEosTable(const std::string& filePath,
                                          const std::string& variableName,
                                          Hdf5Table& output)
{
  Hdf5Table result;

  detail::ScopedHandle file(H5Fopen(filePath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
  if (!file.Valid()) {
    return Hdf5LoadStatus::FileOpenFailed;
  }

  // Open ThermoState group and read LogInterp
  detail::ScopedHandle thermoGroup(H5Gopen(file.Get(), "ThermoState", H5P_DEFAULT), H5Gclose);
  if (!thermoGroup.Valid()) {
    return Hdf5LoadStatus::DatasetOpenFailed;
  }

  int logInterp[3] = {0, 0, 0};
  if (!detail::ReadIntArray(thermoGroup.Get(), "LogInterp", logInterp, 3)) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }

  // Map LogInterp values to AxisScale (1=Log10, 0=Linear)
  // LogInterp order in file: [Density, Temperature, Ye]
  // After reversal: axis0=Density, axis1=Temperature, axis2=Ye
  const AxisScale scales[3] = {
    (logInterp[0] == 1) ? AxisScale::Log10 : AxisScale::Linear,  // Density -> axis0
    (logInterp[1] == 1) ? AxisScale::Log10 : AxisScale::Linear,  // Temperature -> axis1
    (logInterp[2] == 1) ? AxisScale::Log10 : AxisScale::Linear   // Ye -> axis2
  };

  // Open dependent variable dataset
  const std::string datasetPath = "DependentVariables/" + variableName;
  detail::ScopedHandle dataset(H5Dopen(file.Get(), datasetPath.c_str(), H5P_DEFAULT), H5Dclose);
  if (!dataset.Valid()) {
    return Hdf5LoadStatus::DatasetOpenFailed;
  }

  detail::ScopedHandle dataspace(H5Dget_space(dataset.Get()), H5Sclose);
  if (!dataspace.Valid()) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }

  const int rank = H5Sget_simple_extent_ndims(dataspace.Get());
  if (rank != 3) {
    return Hdf5LoadStatus::DatasetRankInvalid;
  }

  // Read dimensions in HDF5 (Fortran) order: (Ye, T, rho)
  std::array<hsize_t, 3> nativeDims{{0, 0, 0}};
  if (H5Sget_simple_extent_dims(dataspace.Get(), nativeDims.data(), nullptr) < 0) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }

  // Reverse to C order: axis0=rho, axis1=T, axis2=Ye
  result.nd = 3;
  std::array<int, 5> extents{{1, 1, 1, 1, 1}};
  for (int dim = 0; dim < 3; ++dim) {
    const hsize_t source = nativeDims[2 - dim];
    if (source == 0 || source > static_cast<hsize_t>(std::numeric_limits<int>::max())) {
      return Hdf5LoadStatus::IncompatibleDatasetExtent;
    }
    extents[dim] = static_cast<int>(source);
  }
  result.extents = extents;

  // Allocate and read data
  const std::size_t totalSize = detail::ComputeTotalSize(3, extents);
  if (totalSize == 0) {
    return Hdf5LoadStatus::IncompatibleDatasetExtent;
  }

  const amrex::Array<int, 4> lo{{0, 0, 0, 0}};
  bool extentOverflow = false;
  amrex::Array<int, 4> hi = detail::MakeHiArray(3, extents, extentOverflow);
  if (extentOverflow) {
    return Hdf5LoadStatus::IncompatibleDatasetExtent;
  }
  result.values.resize(lo, hi, amrex::The_Pinned_Arena());

  if (H5Dread(dataset.Get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
              result.values.table().p) < 0) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }

  // Load axes from ThermoState group
  // File order: Density(185), Temperature(81), Electron Fraction(30)
  // C order after reversal: axis0=Density(185), axis1=Temperature(81), axis2=Ye(30)
  Hdf5LoadStatus axisStatus;

  axisStatus = detail::LoadWeakLibAxis(thermoGroup.Get(), "Density",
                                       extents[0], scales[0], 0, result);
  if (axisStatus != Hdf5LoadStatus::Success) {
    return axisStatus;
  }

  axisStatus = detail::LoadWeakLibAxis(thermoGroup.Get(), "Temperature",
                                       extents[1], scales[1], 1, result);
  if (axisStatus != Hdf5LoadStatus::Success) {
    return axisStatus;
  }

  axisStatus = detail::LoadWeakLibAxis(thermoGroup.Get(), "Electron Fraction",
                                       extents[2], scales[2], 2, result);
  if (axisStatus != Hdf5LoadStatus::Success) {
    return axisStatus;
  }

  // Clear unused axes
  for (int dim = 3; dim < 5; ++dim) {
    result.axisStorage[dim].clear();
    result.axes[dim] = Axis{};
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
  result.scales[0] = (logInterp[0] == 1) ? AxisScale::Log10 : AxisScale::Linear;
  result.scales[1] = (logInterp[1] == 1) ? AxisScale::Log10 : AxisScale::Linear;
  result.scales[2] = (logInterp[2] == 1) ? AxisScale::Log10 : AxisScale::Linear;

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
                                       result.dimensions[0], result.scales[0], 0, tempTable);
  if (axisStatus != Hdf5LoadStatus::Success) return axisStatus;

  axisStatus = detail::LoadWeakLibAxis(thermoGroup.Get(), "Temperature",
                                       result.dimensions[1], result.scales[1], 1, tempTable);
  if (axisStatus != Hdf5LoadStatus::Success) return axisStatus;

  axisStatus = detail::LoadWeakLibAxis(thermoGroup.Get(), "Electron Fraction",
                                       result.dimensions[2], result.scales[2], 2, tempTable);
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

  // Read offsets
  result.offsets.resize(result.nVariables);
  {
    detail::ScopedHandle offsetsDs(H5Dopen(dvGroup.Get(), "Offsets", H5P_DEFAULT), H5Dclose);
    if (!offsetsDs.Valid()) {
      return Hdf5LoadStatus::DatasetOpenFailed;
    }
    if (H5Dread(offsetsDs.Get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                result.offsets.data()) < 0) {
      return Hdf5LoadStatus::DatasetReadFailed;
    }
  }

  // Read each variable's 3D data
  result.variables.resize(result.nVariables);
  const amrex::Array<int, 3> lo{{0, 0, 0}};
  const amrex::Array<int, 3> hi{{result.dimensions[0] - 1, result.dimensions[1] - 1, result.dimensions[2] - 1}};

  for (int iVar = 0; iVar < result.nVariables; ++iVar) {
    const std::string& varName = result.variableNames[iVar];
    detail::ScopedHandle varDs(H5Dopen(dvGroup.Get(), varName.c_str(), H5P_DEFAULT), H5Dclose);
    if (!varDs.Valid()) {
      return Hdf5LoadStatus::DatasetOpenFailed;
    }

    result.variables[iVar].resize(lo, hi, amrex::The_Pinned_Arena());
    if (H5Dread(varDs.Get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                result.variables[iVar].table().p) < 0) {
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

inline WeakLibEosTableDevice MakeDeviceCopy(const WeakLibEosTable& host,
                                             amrex::Arena* arena = amrex::The_Device_Arena())
{
  WeakLibEosTableDevice device{};
  device.nVariables = host.nVariables;
  device.dimensions = host.dimensions;
  device.scales = host.scales;
  device.layout = host.layout;
  device.indices = host.indices;

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

inline Hdf5LoadStatus LoadWeakLibEosTableParallel(
    const std::string& filePath,
    const std::string& variableName,
    Hdf5Table& output,
    int readerRank = amrex::ParallelDescriptor::IOProcessorNumber())
{
  // Single-rank fallback
  const int nProcs = amrex::ParallelDescriptor::NProcs();
  if (nProcs <= 1) {
    return LoadWeakLibEosTable(filePath, variableName, output);
  }

  // Validate readerRank
  int root = readerRank;
  if (root < 0 || root >= nProcs) {
    root = amrex::ParallelDescriptor::IOProcessorNumber();
  }

  const int myRank = amrex::ParallelDescriptor::MyProc();

  // Root rank loads the table
  Hdf5Table localTable;
  Hdf5LoadStatus status = Hdf5LoadStatus::Success;
  if (myRank == root) {
    status = LoadWeakLibEosTable(filePath, variableName, localTable);
  }

  // Broadcast status
  int statusInt = static_cast<int>(status);
  amrex::ParallelDescriptor::Bcast(&statusInt, 1, root);
  status = static_cast<Hdf5LoadStatus>(statusInt);
  if (status != Hdf5LoadStatus::Success) {
    return status;
  }

  // Broadcast header: [nd, extents x5]
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

  // Broadcast axis metadata
  int axisCounts[5] = {0, 0, 0, 0, 0};
  int axisScales[5] = {0, 0, 0, 0, 0};
  if (myRank == root) {
    for (int dim = 0; dim < 5; ++dim) {
      axisCounts[dim] = localTable.axes[dim].n;
      axisScales[dim] = static_cast<int>(localTable.axes[dim].scale);
    }
  }
  amrex::ParallelDescriptor::Bcast(axisCounts, 5, root);
  amrex::ParallelDescriptor::Bcast(axisScales, 5, root);

  // Compute total size and validate
  const std::size_t totalSize = detail::ComputeTotalSize(nd, extents);
  if (totalSize > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return Hdf5LoadStatus::IncompatibleDatasetExtent;
  }

  // Non-root ranks allocate storage
  if (myRank != root) {
    output = Hdf5Table{};
    output.nd = nd;
    output.extents = extents;
    bool overflow = false;
    const amrex::Array<int, 4> lo{{0, 0, 0, 0}};
    const amrex::Array<int, 4> hi = detail::MakeHiArray(nd, extents, overflow);
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

  // Broadcast values data
  double* dataPtr = (myRank == root)
                        ? localTable.values.table().p
                        : output.values.table().p;
  if (totalSize > 0) {
    amrex::ParallelDescriptor::Bcast(dataPtr, static_cast<int>(totalSize), root);
  }

  // Broadcast axis grids
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

  // Root moves its local table to output
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

  // Broadcast header: [nVariables, dims x3, scales x3]
  int header[7] = {0, 0, 0, 0, 0, 0, 0};
  if (myRank == root) {
    header[0] = localTable.nVariables;
    header[1] = localTable.dimensions[0];
    header[2] = localTable.dimensions[1];
    header[3] = localTable.dimensions[2];
    header[4] = static_cast<int>(localTable.scales[0]);
    header[5] = static_cast<int>(localTable.scales[1]);
    header[6] = static_cast<int>(localTable.scales[2]);
  }
  amrex::ParallelDescriptor::Bcast(header, 7, root);

  const int nVariables = header[0];
  std::array<int, 3> dimensions{{header[1], header[2], header[3]}};
  std::array<AxisScale, 3> scales{{
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
    output.scales = scales;

    // Allocate axis storage
    for (int dim = 0; dim < 3; ++dim) {
      output.axisStorage[dim].resize(axisCounts[dim]);
      output.axes[dim].grid = output.axisStorage[dim].data();
      output.axes[dim].n = axisCounts[dim];
      output.axes[dim].scale = scales[dim];
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
