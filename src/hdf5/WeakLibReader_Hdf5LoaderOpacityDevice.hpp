#pragma once

#include "hdf5/WeakLibReader_Hdf5LoaderDetail.hpp"

namespace WeakLibReader {
namespace detail {

inline void CopyGridToDevice(const WeakLibOpacityGrid &host,
                             WeakLibOpacityGridDevice &device) {
  device.nPoints = host.nPoints;
  device.scale = host.scale;
  CopyVectorToDevice(host.values, device.values);
}

inline void CopyThermoStateToDevice(const WeakLibThermoState &host,
                                    WeakLibThermoStateDevice &device) {
  device.dimensions = host.dimensions;

  for (int i = 0; i < 3; ++i) {
    CopyVectorToDevice(host.axisStorage[i], device.axisStorage[i]);
    device.axes[i] =
        Axis{device.axisStorage[i].data(), host.axes[i].n, host.axes[i].scale};
  }
}

inline void CopyECTableToDevice(const WeakLibECTable &host,
                                WeakLibECTableDevice &device) {
  device.nE = host.nE;
  device.nRho = host.nRho;
  device.nT = host.nT;
  device.nYe = host.nYe;
  device.rhoMin = host.rhoMin;
  device.rhoMax = host.rhoMax;
  device.tempMin = host.tempMin;
  device.tempMax = host.tempMax;
  device.yeMin = host.yeMin;
  device.yeMax = host.yeMax;
  device.specOffset = host.specOffset;
  device.rateOffset = host.rateOffset;

  CopyVectorToDevice(host.energyValues, device.energyValues);
  CopyVectorToDevice(host.rhoValues, device.rhoValues);
  CopyVectorToDevice(host.tempValues, device.tempValues);
  CopyVectorToDevice(host.yeValues, device.yeValues);
  CopyVectorToDevice(host.spectrum, device.spectrum);
  CopyVectorToDevice(host.rate, device.rate);
}

inline void CopyEmAbToDevice(const WeakLibEmAbTable &host,
                             WeakLibEmAbTableDevice &device) {
  device.nOpacities = host.nOpacities;
  device.dimensions = host.dimensions;
  device.offsets = host.offsets;
  device.parameters = host.parameters;
  device.layout = host.layout;

  for (int i = 0; i < WeakLibEmAbTable::NumSpecies; ++i) {
    CopyVectorToDevice(host.opacities[i], device.opacities[i]);
  }

  if (host.ecTable.IsPresent()) {
    CopyECTableToDevice(host.ecTable, device.ecTable);
  }
}

inline void CopyScatIsoToDevice(const WeakLibScatIsoTable &host,
                                WeakLibScatIsoTableDevice &device) {
  device.nOpacities = host.nOpacities;
  device.nMoments = host.nMoments;
  device.dimensions = host.dimensions;
  device.weakMagnetismCorrections = host.weakMagnetismCorrections;
  device.ionIonCorrections = host.ionIonCorrections;
  device.manyBodyCorrections = host.manyBodyCorrections;
  device.gaStrange = host.gaStrange;
  device.layout = host.layout;

  device.offsets = host.offsets;
  for (int i = 0; i < WeakLibScatIsoTable::NumSpecies; ++i) {
    CopyVectorToDevice(host.kernels[i], device.kernels[i]);
  }
}

inline void CopyScatKernelToDevice(const WeakLibScatKernelTable &host,
                                   WeakLibScatKernelTableDevice &device) {
  device.nOpacities = host.nOpacities;
  device.nMoments = host.nMoments;
  device.dimensions = host.dimensions;
  device.nps = host.nps;
  device.layout = host.layout;

  device.offsets = host.offsets;
  CopyVectorToDevice(host.kernel, device.kernel);
}

} // namespace detail

inline WeakLibOpacityTableDevice
MakeDeviceCopy(const WeakLibOpacityTable &host) {
  WeakLibOpacityTableDevice device{};

  detail::CopyGridToDevice(host.energyGrid, device.energyGrid);
  detail::CopyGridToDevice(host.etaGrid, device.etaGrid);
  detail::CopyThermoStateToDevice(host.thermoState, device.thermoState);

  if (host.emAb.IsLoaded())
    detail::CopyEmAbToDevice(host.emAb, device.emAb);
  if (host.scatIso.IsLoaded())
    detail::CopyScatIsoToDevice(host.scatIso, device.scatIso);
  if (host.scatNES.IsLoaded())
    detail::CopyScatKernelToDevice(host.scatNES, device.scatNES);
  if (host.scatPair.IsLoaded())
    detail::CopyScatKernelToDevice(host.scatPair, device.scatPair);
  if (host.scatBrem.IsLoaded())
    detail::CopyScatKernelToDevice(host.scatBrem, device.scatBrem);

  return device;
}

} // namespace WeakLibReader
