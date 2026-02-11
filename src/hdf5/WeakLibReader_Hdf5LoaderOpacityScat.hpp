#pragma once

#include "hdf5/WeakLibReader_Hdf5Types.hpp"
#include "hdf5/WeakLibReader_Hdf5LoaderDetail.hpp"

namespace WeakLibReader {
inline Hdf5LoadStatus LoadWeakLibScatIsoTable(hid_t file,
                                               WeakLibScatIsoTable& scatIso,
                                               const WeakLibOpacityGrid& energyGrid,
                                               const WeakLibOpacityThermoState& thermoState)
{
  if (!detail::GroupExists(file, "Scat_Iso_Kernels")) {
    return Hdf5LoadStatus::DatasetOpenFailed;
  }

  detail::ScopedHandle group(H5Gopen(file, "Scat_Iso_Kernels", H5P_DEFAULT), H5Gclose);
  if (!group.Valid()) {
    return Hdf5LoadStatus::DatasetOpenFailed;
  }

  // Read nOpacities and nMoments
  if (!detail::ReadScalarInt(group.Get(), "nOpacities", scatIso.nOpacities) ||
      !detail::ReadScalarInt(group.Get(), "nMoments", scatIso.nMoments)) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }

  // Set dimensions: [nE, nMom, nRho, nT, nYe]
  scatIso.dimensions[0] = energyGrid.nPoints;
  scatIso.dimensions[1] = scatIso.nMoments;
  scatIso.dimensions[2] = thermoState.dimensions[0];  // nRho
  scatIso.dimensions[3] = thermoState.dimensions[1];  // nT
  scatIso.dimensions[4] = thermoState.dimensions[2];  // nYe

  // Read Units
  std::vector<std::string> unitVec;
  if (detail::ReadStringArray(group.Get(), "Units", unitVec)) {
    for (int i = 0; i < WeakLibScatIsoTable::kNumSpecies && i < static_cast<int>(unitVec.size()); ++i) {
      scatIso.units[i] = unitVec[i];
    }
  }

  // Read Offsets (2D: [nOpacities, nMoments])
  std::array<int, 2> offsetDims{{scatIso.nOpacities, scatIso.nMoments}};
  if (!detail::ReadWeakLibArrayNd<double, 2>(group.Get(), "Offsets", scatIso.offsets, offsetDims)) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }

  // Read correction flags and ga_strange (optional, may not exist in legacy files)
  {
    detail::ScopedH5ErrorSuppressor suppress;

    detail::ReadScalarInt(group.Get(), "weak_magnetism_corr", scatIso.weak_magnetism_corrections);
    detail::ReadScalarInt(group.Get(), "ion_ion_corr", scatIso.ion_ion_corrections);
    detail::ReadScalarInt(group.Get(), "many_body_corr", scatIso.many_body_corrections);

    detail::ScopedHandle ds(H5Dopen(group.Get(), "ga_strange", H5P_DEFAULT), H5Dclose);
    if (ds.Valid()) {
      H5Dread(ds.Get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &scatIso.ga_strange);
    }
  }

  // Set species names
  scatIso.names[0] = "Electron Neutrino";
  scatIso.names[1] = "Electron Antineutrino";

  // Read kernel data for each species
  for (int iSpecies = 0; iSpecies < WeakLibScatIsoTable::kNumSpecies && iSpecies < scatIso.nOpacities; ++iSpecies) {
    if (!detail::ReadWeakLibArrayNd<double, 5>(group.Get(), scatIso.names[iSpecies].c_str(),
                                               scatIso.kernels[iSpecies], scatIso.dimensions)) {
      return Hdf5LoadStatus::DatasetReadFailed;
    }
  }

  // Compute layout
  scatIso.layout = MakeLayout(scatIso.dimensions.data(), 5);

  return Hdf5LoadStatus::Success;
}

/// Extract a contiguous 4D slice [nE, nRho, nT, nYe] from a 5D Iso kernel
/// at a fixed moment index.
///
/// The 5D kernel has dimensions [nE, nMom, nRho, nT, nYe] with column-major
/// strides [1, nE, nE*nMom, ...]. A slice at fixed iMom is not contiguous
/// because nMom interleaves the Rho stride.
///
/// @param kernel5d   Flat 5D kernel data (column-major ordered)
/// @param dims       5D dimensions: [nE, nMom, nRho, nT, nYe]
/// @param iMom       Moment index to extract (0-based)
/// @return           Contiguous 4D array [nE, nRho, nT, nYe]
inline amrex::Vector<double> ExtractIsoMomentSlice4D(
    const double* kernel5d,
    const std::array<int, 5>& dims,
    int iMom)
{
  const int nE   = dims[0];
  const int nMom = dims[1];
  const int nRho = dims[2];
  const int nT   = dims[3];
  const int nYe  = dims[4];

  const Layout layout5d = MakeLayout(dims.data(), 5);
  const std::size_t size4d =
      static_cast<std::size_t>(nE) * nRho * nT * nYe;
  amrex::Vector<double> result(size4d);

  const std::array<int, 4> dims4d{{nE, nRho, nT, nYe}};
  const Layout layout4d = MakeLayout(dims4d.data(), 4);

  for (int iYe = 0; iYe < nYe; ++iYe) {
    for (int iT = 0; iT < nT; ++iT) {
      for (int iRho = 0; iRho < nRho; ++iRho) {
        for (int iE = 0; iE < nE; ++iE) {
          const int src[5] = {iE, iMom, iRho, iT, iYe};
          const int dst[4] = {iE, iRho, iT, iYe};
          result[layout4d.Offset(dst)] = kernel5d[layout5d.Offset(src)];
        }
      }
    }
  }

  return result;
}

namespace detail {

inline Hdf5LoadStatus LoadScatKernelTable(
    hid_t file,
    const char* groupName,
    const char* kernelDatasetName,
    const std::array<int, 5>& dims,
    WeakLibScatKernelTable& table)
{
  if (!GroupExists(file, groupName)) {
    return Hdf5LoadStatus::DatasetOpenFailed;
  }

  ScopedHandle group(H5Gopen(file, groupName, H5P_DEFAULT), H5Gclose);
  if (!group.Valid()) {
    return Hdf5LoadStatus::DatasetOpenFailed;
  }

  // Read nOpacities and nMoments
  if (!ReadScalarInt(group.Get(), "nOpacities", table.nOpacities) ||
      !ReadScalarInt(group.Get(), "nMoments", table.nMoments)) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }

  table.dimensions = dims;
  table.dimensions[2] = table.nMoments;

  // Read Units
  std::vector<std::string> unitVec;
  if (ReadStringArray(group.Get(), "Units", unitVec) && !unitVec.empty()) {
    table.unit = unitVec[0];
  }

  // Read Offsets (2D: [nOpacities, nMoments] in C order)
  std::array<int, 2> offsetDims{{table.nOpacities, table.nMoments}};
  if (!ReadWeakLibArrayNd<double, 2>(group.Get(), "Offsets", table.offsets, offsetDims)) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }

  // Set kernel name and read kernel data
  table.name = kernelDatasetName;
  if (!ReadWeakLibArrayNd<double, 5>(group.Get(), kernelDatasetName, table.kernel, table.dimensions)) {
    return Hdf5LoadStatus::DatasetReadFailed;
  }

  // Compute layout
  table.layout = MakeLayout(table.dimensions.data(), 5);

  return Hdf5LoadStatus::Success;
}

} // namespace detail

inline Hdf5LoadStatus LoadWeakLibScatNESTable(hid_t file,
                                               WeakLibScatNESTable& scatNES,
                                               const WeakLibOpacityGrid& energyGrid,
                                               const WeakLibOpacityGrid& etaGrid,
                                               const WeakLibOpacityThermoState& thermoState)
{
  const std::array<int, 5> dims{{
      energyGrid.nPoints, energyGrid.nPoints, 0 /*placeholder*/,
      thermoState.dimensions[1], etaGrid.nPoints}};

  Hdf5LoadStatus status = detail::LoadScatKernelTable(
      file, "Scat_NES_Kernels", "Kernels", dims, scatNES);
  if (status != Hdf5LoadStatus::Success) {
    return status;
  }

  // Read NPS flag (optional — suppress HDF5 error if absent)
  {
    detail::ScopedH5ErrorSuppressor suppress;
    detail::ScopedHandle group(H5Gopen(file, "Scat_NES_Kernels", H5P_DEFAULT), H5Gclose);
    if (group.Valid()) {
      detail::ReadScalarInt(group.Get(), "NPS", scatNES.NPS);
    }
  }

  return Hdf5LoadStatus::Success;
}

inline Hdf5LoadStatus LoadWeakLibScatPairTable(hid_t file,
                                                WeakLibScatPairTable& scatPair,
                                                const WeakLibOpacityGrid& energyGrid,
                                                const WeakLibOpacityGrid& etaGrid,
                                                const WeakLibOpacityThermoState& thermoState)
{
  const std::array<int, 5> dims{{
      energyGrid.nPoints, energyGrid.nPoints, 0 /*placeholder*/,
      thermoState.dimensions[1], etaGrid.nPoints}};

  return detail::LoadScatKernelTable(
      file, "Scat_Pair_Kernels", "Kernels", dims, scatPair);
}

inline Hdf5LoadStatus LoadWeakLibScatBremTable(hid_t file,
                                                WeakLibScatBremTable& scatBrem,
                                                const WeakLibOpacityGrid& energyGrid,
                                                const WeakLibOpacityThermoState& thermoState)
{
  const std::array<int, 5> dims{{
      energyGrid.nPoints, energyGrid.nPoints, 0 /*placeholder*/,
      thermoState.dimensions[0], thermoState.dimensions[1]}};

  return detail::LoadScatKernelTable(
      file, "Scat_Brem_Kernels", "S_sigma", dims, scatBrem);
}

} // namespace WeakLibReader
