#pragma once

#include "hdf5/WeakLibReader_Hdf5Types.hpp"

namespace WeakLibReader {
namespace detail {

inline void CopyGridToDevice(const WeakLibOpacityGrid& host, WeakLibOpacityGridDevice& device)
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
                                    WeakLibOpacityThermoStateDevice& device)
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

inline void CopyScatKernelToDevice(
    const WeakLibScatKernelTable& host,
    WeakLibScatKernelTableDevice& device)
{
  device.nOpacities = host.nOpacities;
  device.nMoments = host.nMoments;
  device.dimensions = host.dimensions;
  device.NPS = host.NPS;
  device.layout = host.layout;

  device.offsets.resize(host.offsets.size());
  amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                   host.offsets.begin(), host.offsets.end(),
                   device.offsets.begin());

  if (!host.kernel.empty()) {
    device.kernel.resize(host.kernel.size());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                     host.kernel.begin(), host.kernel.end(),
                     device.kernel.begin());
  }
}

} // namespace detail

inline WeakLibOpacityTableDevice MakeDeviceCopy(const WeakLibOpacityTable& host)
{
  WeakLibOpacityTableDevice device{};

  // Copy grids
  detail::CopyGridToDevice(host.energyGrid, device.energyGrid);
  detail::CopyGridToDevice(host.etaGrid, device.etaGrid);
  detail::CopyThermoStateToDevice(host.thermoState, device.thermoState);

  // Copy EmAb
  if (host.emAb.IsLoaded()) {
    device.emAb.nOpacities = host.emAb.nOpacities;
    device.emAb.dimensions = host.emAb.dimensions;
    device.emAb.offsets = host.emAb.offsets;
    device.emAb.parameters = host.emAb.parameters;
    device.emAb.layout = host.emAb.layout;

    for (int i = 0; i < WeakLibEmAbTable::NumSpecies; ++i) {
      if (host.emAb.opacities[i].size() > 0) {
        device.emAb.opacities[i].resize(host.emAb.opacities[i].size());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                         host.emAb.opacities[i].begin(), host.emAb.opacities[i].end(),
                         device.emAb.opacities[i].begin());
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
      deviceEC.spectrum.resize(hostEC.spectrum.size());
      amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                       hostEC.spectrum.begin(), hostEC.spectrum.end(),
                       deviceEC.spectrum.begin());

      deviceEC.rate.resize(hostEC.rate.size());
      amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                       hostEC.rate.begin(), hostEC.rate.end(),
                       deviceEC.rate.begin());
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

    device.scatIso.offsets.resize(host.scatIso.offsets.size());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                     host.scatIso.offsets.begin(), host.scatIso.offsets.end(),
                     device.scatIso.offsets.begin());

    // Kernels are stored as flat vectors
    for (int i = 0; i < WeakLibScatIsoTable::NumSpecies; ++i) {
      if (!host.scatIso.kernels[i].empty()) {
        device.scatIso.kernels[i].resize(host.scatIso.kernels[i].size());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                         host.scatIso.kernels[i].begin(), host.scatIso.kernels[i].end(),
                         device.scatIso.kernels[i].begin());
      }
    }
  }

  // Copy ScatNES / ScatPair / ScatBrem
  if (host.scatNES.IsLoaded()) detail::CopyScatKernelToDevice(host.scatNES, device.scatNES);
  if (host.scatPair.IsLoaded()) detail::CopyScatKernelToDevice(host.scatPair, device.scatPair);
  if (host.scatBrem.IsLoaded()) detail::CopyScatKernelToDevice(host.scatBrem, device.scatBrem);

  return device;
}

} // namespace WeakLibReader
