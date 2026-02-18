#pragma once

#include "hdf5/WeakLibReader_Hdf5Types.hpp"
#include "hdf5/WeakLibReader_Hdf5LoaderDetail.hpp"

namespace WeakLibReader {
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

  auto readArray = [&](const char* name, std::vector<double>& arr) -> bool {
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
  if (!ReadWeakLibArrayNd<double, 4>(group.Get(), "Spectrum", ecTable.spectrum, specDims)) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }

  // Read Rate (3D: [nRho, nT, nYe])
  std::array<int, 3> rateDims{{ecTable.nRho, ecTable.nT, ecTable.nYe}};
  if (!ReadWeakLibArrayNd<double, 3>(group.Get(), "Rate", ecTable.rate, rateDims)) {
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
  for (int i = 0; i < WeakLibEmAbTable::NumSpecies && i < emAb.nOpacities; ++i) {
    emAb.units[i] = unitVec[i];
  }

  // Read Offsets (1D: [nOpacities])
  {
    std::vector<double> offsetVec;
    std::array<int, 1> offsetDims{{emAb.nOpacities}};
    if (!detail::ReadWeakLibArrayNd<double, 1>(group.Get(), "Offsets", offsetVec, offsetDims)) {
      return Hdf5LoadStatus::DatasetReadFailed;
    }
    for (int i = 0; i < WeakLibEmAbTable::NumSpecies && i < emAb.nOpacities; ++i) {
      emAb.offsets[i] = offsetVec[i];
    }
  }

  // Set species names (hardcoded like Fortran)
  emAb.names[0] = "Electron Neutrino";
  emAb.names[1] = "Electron Antineutrino";

  // Read opacity data for each species
  for (int iSpecies = 0; iSpecies < WeakLibEmAbTable::NumSpecies && iSpecies < emAb.nOpacities; ++iSpecies) {
    if (!detail::ReadWeakLibArrayNd<double, 4>(group.Get(), emAb.names[iSpecies].c_str(),
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

} // namespace WeakLibReader
