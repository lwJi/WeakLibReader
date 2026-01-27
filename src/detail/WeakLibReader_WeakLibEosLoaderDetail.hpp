#pragma once

#include <AMReX_TableData.H>
#include <AMReX_Vector.H>

#include <array>
#include <vector>

#include "../WeakLibReader_Hdf5Types.hpp"
#include "WeakLibReader_Hdf5LoaderDetail.hpp"

#include <hdf5.h>

namespace WeakLibReader {
namespace detail {

// Dataset paths in WeakLib EOS HDF5 files
constexpr const char* kThermoStateGroup = "/ThermoState";
constexpr const char* kDependentVarsGroup = "/DependentVariables";
constexpr const char* kDensityDataset = "Density";
constexpr const char* kTemperatureDataset = "Temperature";
constexpr const char* kElectronFractionDataset = "Electron Fraction";
constexpr const char* kLogInterpDataset = "LogInterp";
constexpr const char* kOffsetsDataset = "Offsets";

// Variable name mapping (matches WeakLib HDF5 dataset names)
inline const char* EosVariableName(EosVariable var) noexcept
{
  switch (var) {
    case EosVariable::Pressure: return "Pressure";
    case EosVariable::EntropyPerBaryon: return "Entropy Per Baryon";
    case EosVariable::InternalEnergyDensity: return "Internal Energy Density";
    case EosVariable::ElectronChemicalPotential: return "Electron Chemical Potential";
    case EosVariable::ProtonChemicalPotential: return "Proton Chemical Potential";
    case EosVariable::NeutronChemicalPotential: return "Neutron Chemical Potential";
    case EosVariable::ProtonMassFraction: return "Proton Mass Fraction";
    case EosVariable::NeutronMassFraction: return "Neutron Mass Fraction";
    case EosVariable::AlphaMassFraction: return "Alpha Mass Fraction";
    case EosVariable::HeavyMassFraction: return "Heavy Mass Fraction";
    case EosVariable::HeavyChargeNumber: return "Heavy Charge Number";
    case EosVariable::HeavyMassNumber: return "Heavy Mass Number";
    case EosVariable::HeavyBindingEnergy: return "Heavy Binding Energy";
    case EosVariable::ThermalEnergy: return "Thermal Energy";
    case EosVariable::Gamma1: return "Gamma1";
    default: return "";
  }
}

// Read LogInterp array to determine axis scales
// Returns array of AxisScale values: [rho, T, Ye]
inline Hdf5LoadStatus ReadLogInterp(hid_t thermoGroup, std::array<AxisScale, 3>& scales)
{
  ScopedHandle dataset(H5Dopen(thermoGroup, kLogInterpDataset, H5P_DEFAULT), H5Dclose);
  if (!dataset.Valid()) {
    return Hdf5LoadStatus::LogInterpReadFailed;
  }

  ScopedHandle space(H5Dget_space(dataset.Get()), H5Sclose);
  if (!space.Valid()) {
    return Hdf5LoadStatus::LogInterpReadFailed;
  }

  hsize_t length = 0;
  if (H5Sget_simple_extent_dims(space.Get(), &length, nullptr) < 0 || length < 3) {
    return Hdf5LoadStatus::LogInterpReadFailed;
  }

  std::vector<int> logInterp(static_cast<std::size_t>(length));
  if (H5Dread(dataset.Get(), H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, logInterp.data()) < 0) {
    return Hdf5LoadStatus::LogInterpReadFailed;
  }

  // LogInterp[i] = 1 means Log10, 0 means Linear
  for (int i = 0; i < 3; ++i) {
    scales[i] = (logInterp[i] != 0) ? AxisScale::Log10 : AxisScale::Linear;
  }

  return Hdf5LoadStatus::Success;
}

// Read a single axis from the ThermoState group
inline Hdf5LoadStatus ReadEosAxis(hid_t thermoGroup,
                                   const char* datasetName,
                                   int expectedSize,
                                   AxisScale scale,
                                   amrex::Vector<double>& storage,
                                   Axis& axis)
{
  ScopedHandle dataset(H5Dopen(thermoGroup, datasetName, H5P_DEFAULT), H5Dclose);
  if (!dataset.Valid()) {
    return Hdf5LoadStatus::AxisDatasetOpenFailed;
  }

  ScopedHandle space(H5Dget_space(dataset.Get()), H5Sclose);
  if (!space.Valid()) {
    return Hdf5LoadStatus::AxisReadFailed;
  }

  const int rank = H5Sget_simple_extent_ndims(space.Get());
  if (rank != 1) {
    return Hdf5LoadStatus::AxisReadFailed;
  }

  hsize_t length = 0;
  if (H5Sget_simple_extent_dims(space.Get(), &length, nullptr) < 0) {
    return Hdf5LoadStatus::AxisReadFailed;
  }

  if (expectedSize > 0 && static_cast<int>(length) != expectedSize) {
    return Hdf5LoadStatus::AxisExtentMismatch;
  }

  storage.resize(static_cast<std::size_t>(length));
  if (H5Dread(dataset.Get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, storage.data()) < 0) {
    return Hdf5LoadStatus::AxisReadFailed;
  }

  if (!ValidateAxis(storage, scale)) {
    return Hdf5LoadStatus::AxisNotMonotone;
  }

  axis.grid = storage.data();
  axis.n = static_cast<int>(storage.size());
  axis.scale = scale;

  return Hdf5LoadStatus::Success;
}

// Load all axes from the ThermoState group
inline Hdf5LoadStatus LoadEosAxes(hid_t file, WeakLibEosTable& table)
{
  ScopedHandle thermoGroup(H5Gopen(file, kThermoStateGroup, H5P_DEFAULT), H5Gclose);
  if (!thermoGroup.Valid()) {
    return Hdf5LoadStatus::GroupOpenFailed;
  }

  // Read LogInterp to determine axis scales
  std::array<AxisScale, 3> scales{};
  Hdf5LoadStatus logInterpStatus = ReadLogInterp(thermoGroup.Get(), scales);
  if (logInterpStatus != Hdf5LoadStatus::Success) {
    return logInterpStatus;
  }

  // Read Density axis (axis 0) - determines rho extent
  Hdf5LoadStatus status = ReadEosAxis(thermoGroup.Get(), kDensityDataset, -1, scales[0],
                                       table.axisStorage[0], table.axes[0]);
  if (status != Hdf5LoadStatus::Success) {
    return status;
  }
  table.extents[0] = table.axes[0].n;

  // Read Temperature axis (axis 1) - determines T extent
  status = ReadEosAxis(thermoGroup.Get(), kTemperatureDataset, -1, scales[1],
                       table.axisStorage[1], table.axes[1]);
  if (status != Hdf5LoadStatus::Success) {
    return status;
  }
  table.extents[1] = table.axes[1].n;

  // Read Electron Fraction axis (axis 2) - determines Ye extent
  status = ReadEosAxis(thermoGroup.Get(), kElectronFractionDataset, -1, scales[2],
                       table.axisStorage[2], table.axes[2]);
  if (status != Hdf5LoadStatus::Success) {
    return status;
  }
  table.extents[2] = table.axes[2].n;

  return Hdf5LoadStatus::Success;
}

// Load offsets from /DependentVariables/Offsets
inline Hdf5LoadStatus LoadEosOffsets(hid_t file, std::vector<double>& offsets)
{
  ScopedHandle depVarsGroup(H5Gopen(file, kDependentVarsGroup, H5P_DEFAULT), H5Gclose);
  if (!depVarsGroup.Valid()) {
    return Hdf5LoadStatus::GroupOpenFailed;
  }

  ScopedHandle dataset(H5Dopen(depVarsGroup.Get(), kOffsetsDataset, H5P_DEFAULT), H5Dclose);
  if (!dataset.Valid()) {
    return Hdf5LoadStatus::OffsetsReadFailed;
  }

  ScopedHandle space(H5Dget_space(dataset.Get()), H5Sclose);
  if (!space.Valid()) {
    return Hdf5LoadStatus::OffsetsReadFailed;
  }

  hsize_t length = 0;
  if (H5Sget_simple_extent_dims(space.Get(), &length, nullptr) < 0) {
    return Hdf5LoadStatus::OffsetsReadFailed;
  }

  offsets.resize(static_cast<std::size_t>(length));
  if (H5Dread(dataset.Get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, offsets.data()) < 0) {
    return Hdf5LoadStatus::OffsetsReadFailed;
  }

  return Hdf5LoadStatus::Success;
}

// Load a single EOS variable from /DependentVariables/<name>
inline Hdf5LoadStatus LoadEosVariable(hid_t file,
                                       EosVariable var,
                                       const std::array<int, 3>& extents,
                                       amrex::TableData<double, 4>& output)
{
  ScopedHandle depVarsGroup(H5Gopen(file, kDependentVarsGroup, H5P_DEFAULT), H5Gclose);
  if (!depVarsGroup.Valid()) {
    return Hdf5LoadStatus::GroupOpenFailed;
  }

  const char* varName = EosVariableName(var);
  ScopedHandle dataset(H5Dopen(depVarsGroup.Get(), varName, H5P_DEFAULT), H5Dclose);
  if (!dataset.Valid()) {
    return Hdf5LoadStatus::DatasetOpenFailed;
  }

  ScopedHandle space(H5Dget_space(dataset.Get()), H5Sclose);
  if (!space.Valid()) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }

  const int rank = H5Sget_simple_extent_ndims(space.Get());
  if (rank != 3) {
    return Hdf5LoadStatus::DatasetRankInvalid;
  }

  // WeakLib stores data as (Ye, T, rho) in HDF5 file (Fortran-order)
  // Our C++ layout is (rho, T, Ye) so we expect reversed dims
  hsize_t dims[3] = {0, 0, 0};
  if (H5Sget_simple_extent_dims(space.Get(), dims, nullptr) < 0) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }

  // Verify shape: HDF5 stores (Ye, T, rho), so dims[0]=Ye, dims[1]=T, dims[2]=rho
  if (static_cast<int>(dims[0]) != extents[2] ||
      static_cast<int>(dims[1]) != extents[1] ||
      static_cast<int>(dims[2]) != extents[0]) {
    return Hdf5LoadStatus::IncompatibleDatasetExtent;
  }

  // Allocate output
  const amrex::Array<int, 4> lo{{0, 0, 0, 0}};
  const amrex::Array<int, 4> hi{{extents[0] - 1, extents[1] - 1, extents[2] - 1, 0}};
  output.resize(lo, hi, amrex::The_Pinned_Arena());

  // Read data
  if (H5Dread(dataset.Get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, output.table().p) < 0) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }

  return Hdf5LoadStatus::Success;
}

} // namespace detail
} // namespace WeakLibReader
