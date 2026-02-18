#pragma once

#include "hdf5/WeakLibReader_Hdf5Types.hpp"
#include "hdf5/WeakLibReader_Hdf5LoaderDetail.hpp"
#include "hdf5/WeakLibReader_Hdf5LoaderOpacityEmAb.hpp"
#include "hdf5/WeakLibReader_Hdf5LoaderOpacityScat.hpp"

namespace WeakLibReader {
namespace detail {

inline Hdf5LoadStatus LoadWeakLibOpacityThermoState(hid_t file,
                                                     WeakLibOpacityThermoState& ts)
{
  ScopedHandle group(H5Gopen(file, "ThermoState", H5P_DEFAULT), H5Gclose);
  if (!group.Valid()) {
    return Hdf5LoadStatus::DatasetOpenFailed;
  }

  int logInterp[3] = {0, 0, 0};
  if (!ReadIntArray(group.Get(), "LogInterp", logInterp, 3)) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }
  for (int i = 0; i < 3; ++i) {
    ts.scales[i] = (logInterp[i] == 1) ? AxisScale::Log10 : AxisScale::Linear;
  }

  int dims[3] = {0, 0, 0};
  if (!ReadIntArray(group.Get(), "Dimensions", dims, 3)) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }
  for (int i = 0; i < 3; ++i) {
    ts.dimensions[i] = dims[i];
  }

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

  const char* axisDatasetNames[3] = {"Density", "Temperature", "Electron Fraction"};
  for (int i = 0; i < 3; ++i) {
    Hdf5LoadStatus status = LoadWeakLibAxis(
        group.Get(), axisDatasetNames[i],
        ts.dimensions[i], ts.scales[i],
        ts.axisStorage[i], ts.axes[i]);
    if (status != Hdf5LoadStatus::Success) {
      return status;
    }
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

  const bool hasEmAb = !fileEmAb.empty();
  const bool hasIso  = !fileIso.empty();
  const bool hasNES  = !fileNES.empty();
  const bool hasPair = !filePair.empty();
  const bool hasBrem = !fileBrem.empty();

  if (!hasEmAb && !hasIso && !hasNES && !hasPair && !hasBrem) {
    return Hdf5LoadStatus::FileOpenFailed;
  }

  // Use the first available file for reading shared grids (EnergyGrid, ThermoState).
  const std::string& firstFile = hasEmAb ? fileEmAb
                                : hasIso  ? fileIso
                                : hasNES  ? fileNES
                                : hasPair ? filePair
                                          : fileBrem;

  // Helper: open an HDF5 file and invoke a loader callback.
  auto openAndLoad = [](const std::string& path, auto&& loader) -> Hdf5LoadStatus {
    detail::ScopedHandle file(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
    if (!file.Valid()) {
      return Hdf5LoadStatus::FileOpenFailed;
    }
    return loader(file.Get());
  };

  // Load shared grids from the first available file.
  status = openAndLoad(firstFile, [&](hid_t fid) {
    Hdf5LoadStatus s = detail::LoadWeakLibOpacityGrid(fid, "EnergyGrid", result.energyGrid);
    if (s != Hdf5LoadStatus::Success) return s;
    return detail::LoadWeakLibOpacityThermoState(fid, result.thermoState);
  });
  if (status != Hdf5LoadStatus::Success) return status;

  // EtaGrid is only needed by NES/Pair.
  if (hasNES || hasPair) {
    const std::string& etaFile = hasNES ? fileNES : filePair;
    status = openAndLoad(etaFile, [&](hid_t fid) {
      return detail::LoadWeakLibOpacityGrid(fid, "EtaGrid", result.etaGrid);
    });
    if (status != Hdf5LoadStatus::Success) return status;
  }

  if (hasEmAb) {
    status = openAndLoad(fileEmAb, [&](hid_t fid) {
      return LoadWeakLibEmAbTable(fid, result.emAb, result.energyGrid, result.thermoState);
    });
    if (status != Hdf5LoadStatus::Success) return status;
  }

  if (hasIso) {
    status = openAndLoad(fileIso, [&](hid_t fid) {
      return LoadWeakLibScatIsoTable(fid, result.scatIso, result.energyGrid, result.thermoState);
    });
    if (status != Hdf5LoadStatus::Success) return status;
  }

  if (hasNES) {
    status = openAndLoad(fileNES, [&](hid_t fid) {
      return LoadWeakLibScatNESTable(fid, result.scatNES, result.energyGrid, result.etaGrid, result.thermoState);
    });
    if (status != Hdf5LoadStatus::Success) return status;
  }

  if (hasPair) {
    status = openAndLoad(filePair, [&](hid_t fid) {
      return LoadWeakLibScatPairTable(fid, result.scatPair, result.energyGrid, result.etaGrid, result.thermoState);
    });
    if (status != Hdf5LoadStatus::Success) return status;
  }

  if (hasBrem) {
    status = openAndLoad(fileBrem, [&](hid_t fid) {
      return LoadWeakLibScatBremTable(fid, result.scatBrem, result.energyGrid, result.thermoState);
    });
    if (status != Hdf5LoadStatus::Success) return status;
  }

  output = std::move(result);
  return Hdf5LoadStatus::Success;
}

} // namespace WeakLibReader
