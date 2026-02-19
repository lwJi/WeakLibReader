#pragma once

#include "hdf5/WeakLibReader_Hdf5Types.hpp"

namespace WeakLibReader {
namespace detail {

// Copy a host vector to a device vector (resize + copy).
template <typename T, typename HostVec, typename DeviceVec>
inline void CopyVectorToDevice(const HostVec& host, DeviceVec& device)
{
  device.resize(host.size());
  if (!host.empty()) {
    amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                     host.begin(), host.end(),
                     device.begin());
  }
}

inline void CopyGridToDevice(const WeakLibOpacityGrid& host, WeakLibOpacityGridDevice& device)
{
  device.nPoints = host.nPoints;
  device.scale = host.scale;
  CopyVectorToDevice<double>(host.values, device.values);
}

inline void CopyThermoStateToDevice(const WeakLibOpacityThermoState& host,
                                    WeakLibOpacityThermoStateDevice& device)
{
  device.dimensions = host.dimensions;

  for (int i = 0; i < 3; ++i) {
    CopyVectorToDevice<double>(host.axisStorage[i], device.axisStorage[i]);
    device.axes[i] = Axis{device.axisStorage[i].data(), host.axes[i].n, host.axes[i].scale};
  }
}

inline void CopyScatKernelToDevice(
    const WeakLibScatKernelTable& host,
    WeakLibScatKernelTableDevice& device)
{
  device.nOpacities = host.nOpacities;
  device.nMoments = host.nMoments;
  device.dimensions = host.dimensions;
  device.NPS = host.NPS;
  device.layout = host.layout;

  CopyVectorToDevice<double>(host.offsets, device.offsets);
  CopyVectorToDevice<double>(host.kernel, device.kernel);
}

} // namespace detail

inline WeakLibOpacityTableDevice MakeDeviceCopy(const WeakLibOpacityTable& host)
{
  WeakLibOpacityTableDevice device{};

  detail::CopyGridToDevice(host.energyGrid, device.energyGrid);
  detail::CopyGridToDevice(host.etaGrid, device.etaGrid);
  detail::CopyThermoStateToDevice(host.thermoState, device.thermoState);

  if (host.emAb.IsLoaded()) {
    device.emAb.nOpacities = host.emAb.nOpacities;
    device.emAb.dimensions = host.emAb.dimensions;
    device.emAb.offsets = host.emAb.offsets;
    device.emAb.parameters = host.emAb.parameters;
    device.emAb.layout = host.emAb.layout;

    for (int i = 0; i < WeakLibEmAbTable::NumSpecies; ++i) {
      detail::CopyVectorToDevice<double>(host.emAb.opacities[i], device.emAb.opacities[i]);
    }

    if (host.emAb.ecTable.IsPresent()) {
      const auto& hostEC = host.emAb.ecTable;
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

      detail::CopyVectorToDevice<double>(hostEC.energyValues, deviceEC.energyValues);
      detail::CopyVectorToDevice<double>(hostEC.rhoValues, deviceEC.rhoValues);
      detail::CopyVectorToDevice<double>(hostEC.tempValues, deviceEC.tempValues);
      detail::CopyVectorToDevice<double>(hostEC.yeValues, deviceEC.yeValues);
      detail::CopyVectorToDevice<double>(hostEC.spectrum, deviceEC.spectrum);
      detail::CopyVectorToDevice<double>(hostEC.rate, deviceEC.rate);
    }
  }

  if (host.scatIso.IsLoaded()) {
    device.scatIso.nOpacities = host.scatIso.nOpacities;
    device.scatIso.nMoments = host.scatIso.nMoments;
    device.scatIso.dimensions = host.scatIso.dimensions;
    device.scatIso.weak_magnetism_corrections = host.scatIso.weak_magnetism_corrections;
    device.scatIso.ion_ion_corrections = host.scatIso.ion_ion_corrections;
    device.scatIso.many_body_corrections = host.scatIso.many_body_corrections;
    device.scatIso.ga_strange = host.scatIso.ga_strange;
    device.scatIso.layout = host.scatIso.layout;

    detail::CopyVectorToDevice<double>(host.scatIso.offsets, device.scatIso.offsets);
    for (int i = 0; i < WeakLibScatIsoTable::NumSpecies; ++i) {
      detail::CopyVectorToDevice<double>(host.scatIso.kernels[i], device.scatIso.kernels[i]);
    }
  }

  if (host.scatNES.IsLoaded())  detail::CopyScatKernelToDevice(host.scatNES, device.scatNES);
  if (host.scatPair.IsLoaded()) detail::CopyScatKernelToDevice(host.scatPair, device.scatPair);
  if (host.scatBrem.IsLoaded()) detail::CopyScatKernelToDevice(host.scatBrem, device.scatBrem);

  return device;
}

} // namespace WeakLibReader
