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

  // Load axis data directly into thermoState storage
  Hdf5LoadStatus status;
  status = LoadWeakLibAxis(group.Get(), "Density",
                           ts.dimensions[0], ts.scales[0],
                           ts.axisStorage[0], ts.axes[0]);
  if (status != Hdf5LoadStatus::Success) return status;

  status = LoadWeakLibAxis(group.Get(), "Temperature",
                           ts.dimensions[1], ts.scales[1],
                           ts.axisStorage[1], ts.axes[1]);
  if (status != Hdf5LoadStatus::Success) return status;

  status = LoadWeakLibAxis(group.Get(), "Electron Fraction",
                           ts.dimensions[2], ts.scales[2],
                           ts.axisStorage[2], ts.axes[2]);
  if (status != Hdf5LoadStatus::Success) return status;

  return Hdf5LoadStatus::Success;
}


template<typename Loader>
inline Hdf5LoadStatus WithOpenedOpacityFile(const std::string& path,
                                            Loader&& loader)
{
  ScopedHandle file(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
  if (!file.Valid()) {
    return Hdf5LoadStatus::FileOpenFailed;
  }
  return loader(file.Get());
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

  // Load EnergyGrid and ThermoState from first file
  status = detail::WithOpenedOpacityFile(firstFile, [&](hid_t file) {
    Hdf5LoadStatus rc = detail::LoadWeakLibOpacityGrid(file, "EnergyGrid", result.energyGrid);
    if (rc != Hdf5LoadStatus::Success) {
      return rc;
    }
    return detail::LoadWeakLibOpacityThermoState(file, result.thermoState);
  });
  if (status != Hdf5LoadStatus::Success) {
    return status;
  }

  // Load EtaGrid if NES or Pair files are provided
  if (hasNES || hasPair) {
    const std::string etaFile = hasNES ? fileNES : filePair;
    status = detail::WithOpenedOpacityFile(etaFile, [&](hid_t file) {
      return detail::LoadWeakLibOpacityGrid(file, "EtaGrid", result.etaGrid);
    });
    if (status != Hdf5LoadStatus::Success) {
      return status;
    }
  }

  // Load EmAb
  if (hasEmAb) {
    status = detail::WithOpenedOpacityFile(fileEmAb, [&](hid_t file) {
      return LoadWeakLibEmAbTable(file, result.emAb, result.energyGrid, result.thermoState);
    });
    if (status != Hdf5LoadStatus::Success) {
      return status;
    }
  }

  // Load Iso
  if (hasIso) {
    status = detail::WithOpenedOpacityFile(fileIso, [&](hid_t file) {
      return LoadWeakLibScatIsoTable(file, result.scatIso, result.energyGrid, result.thermoState);
    });
    if (status != Hdf5LoadStatus::Success) {
      return status;
    }
  }

  // Load NES
  if (hasNES) {
    status = detail::WithOpenedOpacityFile(fileNES, [&](hid_t file) {
      return LoadWeakLibScatNESTable(file, result.scatNES, result.energyGrid, result.etaGrid, result.thermoState);
    });
    if (status != Hdf5LoadStatus::Success) {
      return status;
    }
  }

  // Load Pair
  if (hasPair) {
    status = detail::WithOpenedOpacityFile(filePair, [&](hid_t file) {
      return LoadWeakLibScatPairTable(file, result.scatPair, result.energyGrid, result.etaGrid, result.thermoState);
    });
    if (status != Hdf5LoadStatus::Success) {
      return status;
    }
  }

  // Load Brem
  if (hasBrem) {
    status = detail::WithOpenedOpacityFile(fileBrem, [&](hid_t file) {
      return LoadWeakLibScatBremTable(file, result.scatBrem, result.energyGrid, result.thermoState);
    });
    if (status != Hdf5LoadStatus::Success) {
      return status;
    }
  }

  output = std::move(result);
  return Hdf5LoadStatus::Success;
}

} // namespace WeakLibReader
