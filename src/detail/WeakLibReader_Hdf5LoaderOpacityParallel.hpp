#pragma once

namespace WeakLibReader {
namespace detail {

/// Broadcast a single std::string from root to all ranks.
inline void BcastString(std::string& s, int root)
{
  const int myRank = amrex::ParallelDescriptor::MyProc();
  int len = (myRank == root) ? static_cast<int>(s.size()) : 0;
  amrex::ParallelDescriptor::Bcast(&len, 1, root);
  if (len == 0) {
    if (myRank != root) { s.clear(); }
    return;
  }
  std::vector<char> buf(len);
  if (myRank == root) { std::memcpy(buf.data(), s.data(), len); }
  amrex::ParallelDescriptor::Bcast(buf.data(), len, root);
  if (myRank != root) { s.assign(buf.data(), len); }
}

/// Broadcast a WeakLibOpacityGrid from root to all ranks.
inline void BcastOpacityGrid(WeakLibOpacityGrid& grid, int root)
{
  const int myRank = amrex::ParallelDescriptor::MyProc();

  int header[2] = {0, 0};
  if (myRank == root) {
    header[0] = grid.nPoints;
    header[1] = static_cast<int>(grid.scale);
  }
  amrex::ParallelDescriptor::Bcast(header, 2, root);

  double doubles[3] = {0.0, 0.0, 0.0};
  if (myRank == root) {
    doubles[0] = grid.zoom;
    doubles[1] = grid.minValue;
    doubles[2] = grid.maxValue;
  }
  amrex::ParallelDescriptor::Bcast(doubles, 3, root);

  if (myRank != root) {
    grid.nPoints = header[0];
    grid.scale = static_cast<AxisScale>(header[1]);
    grid.zoom = doubles[0];
    grid.minValue = doubles[1];
    grid.maxValue = doubles[2];
    grid.values.resize(grid.nPoints);
  }

  if (grid.nPoints > 0) {
    amrex::ParallelDescriptor::Bcast(grid.values.data(), grid.nPoints, root);
  }

  BcastString(grid.name, root);
  BcastString(grid.unit, root);
}

/// Broadcast a WeakLibOpacityThermoState from root to all ranks.
inline void BcastThermoState(WeakLibOpacityThermoState& ts, int root)
{
  const int myRank = amrex::ParallelDescriptor::MyProc();

  int header[6] = {0, 0, 0, 0, 0, 0};
  if (myRank == root) {
    for (int i = 0; i < 3; ++i) {
      header[i] = ts.dimensions[i];
      header[3 + i] = static_cast<int>(ts.scales[i]);
    }
  }
  amrex::ParallelDescriptor::Bcast(header, 6, root);

  if (myRank != root) {
    for (int i = 0; i < 3; ++i) {
      ts.dimensions[i] = header[i];
      ts.scales[i] = static_cast<AxisScale>(header[3 + i]);
      ts.axisStorage[i].resize(ts.dimensions[i]);
    }
  }

  for (int i = 0; i < 3; ++i) {
    if (ts.dimensions[i] > 0) {
      amrex::ParallelDescriptor::Bcast(ts.axisStorage[i].data(), ts.dimensions[i], root);
    }
    ts.axes[i].grid = ts.axisStorage[i].data();
    ts.axes[i].n = ts.dimensions[i];
    ts.axes[i].scale = ts.scales[i];
  }

  BcastStringArray(ts.names, root);
  BcastStringArray(ts.units, root);
}

/// Broadcast a WeakLibECTable from root to all ranks.
inline void BcastECTable(WeakLibECTable& ec, int root)
{
  const int myRank = amrex::ParallelDescriptor::MyProc();

  int present = (myRank == root && ec.IsPresent()) ? 1 : 0;
  amrex::ParallelDescriptor::Bcast(&present, 1, root);
  if (present == 0) {
    if (myRank != root) { ec = WeakLibECTable{}; }
    return;
  }

  int header[4] = {0, 0, 0, 0};
  if (myRank == root) {
    header[0] = ec.nE;
    header[1] = ec.nRho;
    header[2] = ec.nT;
    header[3] = ec.nYe;
  }
  amrex::ParallelDescriptor::Bcast(header, 4, root);

  double doubles[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  if (myRank == root) {
    doubles[0] = ec.rhoMin;  doubles[1] = ec.rhoMax;
    doubles[2] = ec.tempMin; doubles[3] = ec.tempMax;
    doubles[4] = ec.yeMin;   doubles[5] = ec.yeMax;
    doubles[6] = ec.specOffset;
    doubles[7] = ec.rateOffset;
  }
  amrex::ParallelDescriptor::Bcast(doubles, 8, root);

  const int nE = header[0], nRho = header[1], nT = header[2], nYe = header[3];
  const auto specSize = static_cast<std::size_t>(nRho) * nT * nYe * nE;
  const auto rateSize = static_cast<std::size_t>(nRho) * nT * nYe;

  if (myRank != root) {
    ec.nE = nE; ec.nRho = nRho; ec.nT = nT; ec.nYe = nYe;
    ec.rhoMin = doubles[0];  ec.rhoMax = doubles[1];
    ec.tempMin = doubles[2]; ec.tempMax = doubles[3];
    ec.yeMin = doubles[4];   ec.yeMax = doubles[5];
    ec.specOffset = doubles[6];
    ec.rateOffset = doubles[7];
    ec.energyValues.resize(nE);
    ec.rhoValues.resize(nRho);
    ec.tempValues.resize(nT);
    ec.yeValues.resize(nYe);
    ec.spectrum.resize(specSize);
    ec.rate.resize(rateSize);
  }

  amrex::ParallelDescriptor::Bcast(ec.energyValues.data(), nE, root);
  amrex::ParallelDescriptor::Bcast(ec.rhoValues.data(), nRho, root);
  amrex::ParallelDescriptor::Bcast(ec.tempValues.data(), nT, root);
  amrex::ParallelDescriptor::Bcast(ec.yeValues.data(), nYe, root);
  if (specSize > 0) {
    amrex::ParallelDescriptor::Bcast(ec.spectrum.data(), static_cast<int>(specSize), root);
  }
  if (rateSize > 0) {
    amrex::ParallelDescriptor::Bcast(ec.rate.data(), static_cast<int>(rateSize), root);
  }

  BcastString(ec.unit, root);
}

/// Broadcast a WeakLibEmAbTable from root to all ranks.
inline void BcastEmAbTable(WeakLibEmAbTable& tab, int root)
{
  const int myRank = amrex::ParallelDescriptor::MyProc();

  int header[5] = {0, 0, 0, 0, 0};
  if (myRank == root) {
    header[0] = tab.nOpacities;
    for (int i = 0; i < 4; ++i) { header[1 + i] = tab.dimensions[i]; }
  }
  amrex::ParallelDescriptor::Bcast(header, 5, root);

  const int nOpacities = header[0];
  std::array<int, 4> dims{{header[1], header[2], header[3], header[4]}};
  const auto dataSize = static_cast<std::size_t>(dims[0]) * dims[1] * dims[2] * dims[3];

  // Broadcast parameters (7 ints)
  static_assert(sizeof(WeakLibEmAbParameters) == 7 * sizeof(int),
                "WeakLibEmAbParameters must be 7 contiguous ints");
  amrex::ParallelDescriptor::Bcast(
      reinterpret_cast<int*>(&tab.parameters), 7, root);

  // Broadcast offsets
  amrex::ParallelDescriptor::Bcast(tab.offsets.data(), 2, root);

  if (myRank != root) {
    tab.nOpacities = nOpacities;
    tab.dimensions = dims;
    for (int s = 0; s < nOpacities; ++s) { tab.opacities[s].resize(dataSize); }
    std::array<int, 5> ext5{{dims[0], dims[1], dims[2], dims[3], 1}};
    tab.layout = MakeLayout(ext5.data(), 4);
  }

  for (int s = 0; s < nOpacities; ++s) {
    if (dataSize > 0) {
      amrex::ParallelDescriptor::Bcast(tab.opacities[s].data(),
                                       static_cast<int>(dataSize), root);
    }
  }

  BcastStringArray(tab.names, root);
  BcastStringArray(tab.units, root);
  BcastECTable(tab.ecTable, root);
}

/// Broadcast a WeakLibScatIsoTable from root to all ranks.
inline void BcastScatIsoTable(WeakLibScatIsoTable& tab, int root)
{
  const int myRank = amrex::ParallelDescriptor::MyProc();

  int header[7] = {0, 0, 0, 0, 0, 0, 0};
  if (myRank == root) {
    header[0] = tab.nOpacities;
    header[1] = tab.nMoments;
    for (int i = 0; i < 5; ++i) { header[2 + i] = tab.dimensions[i]; }
  }
  amrex::ParallelDescriptor::Bcast(header, 7, root);

  const int nOpacities = header[0];
  const int nMoments = header[1];
  std::array<int, 5> dims;
  for (int i = 0; i < 5; ++i) { dims[i] = header[2 + i]; }
  const auto dataSize = static_cast<std::size_t>(dims[0]) * dims[1]
                         * dims[2] * dims[3] * dims[4];
  const auto offsetSize = static_cast<std::size_t>(nOpacities) * nMoments;

  int corrections[3] = {0, 0, 0};
  if (myRank == root) {
    corrections[0] = tab.weak_magnetism_corrections;
    corrections[1] = tab.ion_ion_corrections;
    corrections[2] = tab.many_body_corrections;
  }
  amrex::ParallelDescriptor::Bcast(corrections, 3, root);

  double gaStrange = (myRank == root) ? tab.ga_strange : 0.0;
  amrex::ParallelDescriptor::Bcast(&gaStrange, 1, root);

  if (myRank != root) {
    tab.nOpacities = nOpacities;
    tab.nMoments = nMoments;
    tab.dimensions = dims;
    tab.weak_magnetism_corrections = corrections[0];
    tab.ion_ion_corrections = corrections[1];
    tab.many_body_corrections = corrections[2];
    tab.ga_strange = gaStrange;
    tab.offsets.resize(offsetSize);
    for (int s = 0; s < nOpacities; ++s) { tab.kernels[s].resize(dataSize); }
    tab.layout = MakeLayout(dims.data(), 5);
  }

  if (offsetSize > 0) {
    amrex::ParallelDescriptor::Bcast(tab.offsets.data(),
                                     static_cast<int>(offsetSize), root);
  }
  for (int s = 0; s < nOpacities; ++s) {
    if (dataSize > 0) {
      amrex::ParallelDescriptor::Bcast(tab.kernels[s].data(),
                                       static_cast<int>(dataSize), root);
    }
  }

  BcastStringArray(tab.names, root);
  BcastStringArray(tab.units, root);
}

/// Broadcast a WeakLibScatKernelTable (NES/Pair/Brem) from root to all ranks.
inline void BcastScatKernelTable(WeakLibScatKernelTable& tab, int root)
{
  const int myRank = amrex::ParallelDescriptor::MyProc();

  int header[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  if (myRank == root) {
    header[0] = tab.nOpacities;
    header[1] = tab.nMoments;
    for (int i = 0; i < 5; ++i) { header[2 + i] = tab.dimensions[i]; }
    header[7] = tab.NPS;
  }
  amrex::ParallelDescriptor::Bcast(header, 8, root);

  const int nOpacities = header[0];
  const int nMoments = header[1];
  std::array<int, 5> dims;
  for (int i = 0; i < 5; ++i) { dims[i] = header[2 + i]; }
  const auto dataSize = static_cast<std::size_t>(dims[0]) * dims[1]
                         * dims[2] * dims[3] * dims[4];
  const auto offsetSize = static_cast<std::size_t>(nOpacities) * nMoments;

  if (myRank != root) {
    tab.nOpacities = nOpacities;
    tab.nMoments = nMoments;
    tab.dimensions = dims;
    tab.NPS = header[7];
    tab.offsets.resize(offsetSize);
    tab.kernel.resize(dataSize);
    tab.layout = MakeLayout(dims.data(), 5);
  }

  if (offsetSize > 0) {
    amrex::ParallelDescriptor::Bcast(tab.offsets.data(),
                                     static_cast<int>(offsetSize), root);
  }
  if (dataSize > 0) {
    amrex::ParallelDescriptor::Bcast(tab.kernel.data(),
                                     static_cast<int>(dataSize), root);
  }

  BcastString(tab.name, root);
  BcastString(tab.unit, root);
}

} // namespace detail

/// Load opacity tables in parallel: root reads HDF5, then broadcasts to all
/// ranks. Falls back to serial loader when running single-rank.
inline Hdf5LoadStatus LoadWeakLibOpacityTableFullParallel(
    WeakLibOpacityTable& output,
    const std::string& fileEmAb = "",
    const std::string& fileIso = "",
    const std::string& fileNES = "",
    const std::string& filePair = "",
    const std::string& fileBrem = "",
    int readerRank = amrex::ParallelDescriptor::IOProcessorNumber())
{
  // Single-rank fallback
  const int nProcs = amrex::ParallelDescriptor::NProcs();
  if (nProcs <= 1) {
    return LoadWeakLibOpacityTableFull(output, fileEmAb, fileIso,
                                       fileNES, filePair, fileBrem);
  }

  int root = readerRank;
  if (root < 0 || root >= nProcs) {
    root = amrex::ParallelDescriptor::IOProcessorNumber();
  }
  const int myRank = amrex::ParallelDescriptor::MyProc();

  // Root loads via serial
  WeakLibOpacityTable localTable;
  Hdf5LoadStatus status = Hdf5LoadStatus::Success;
  if (myRank == root) {
    status = LoadWeakLibOpacityTableFull(localTable, fileEmAb, fileIso,
                                          fileNES, filePair, fileBrem);
  }

  // Broadcast status
  int statusInt = static_cast<int>(status);
  amrex::ParallelDescriptor::Bcast(&statusInt, 1, root);
  status = static_cast<Hdf5LoadStatus>(statusInt);
  if (status != Hdf5LoadStatus::Success) {
    return status;
  }

  // Broadcast presence bitmask
  // [hasEmAb, hasIso, hasNES, hasPair, hasBrem, hasEtaGrid]
  int presence[6] = {0, 0, 0, 0, 0, 0};
  if (myRank == root) {
    presence[0] = localTable.HasEmAb() ? 1 : 0;
    presence[1] = localTable.HasScatIso() ? 1 : 0;
    presence[2] = localTable.HasScatNES() ? 1 : 0;
    presence[3] = localTable.HasScatPair() ? 1 : 0;
    presence[4] = localTable.HasScatBrem() ? 1 : 0;
    presence[5] = (localTable.etaGrid.nPoints > 0) ? 1 : 0;
  }
  amrex::ParallelDescriptor::Bcast(presence, 6, root);

  // Get reference to working table
  WeakLibOpacityTable& table = (myRank == root) ? localTable : output;

  // Broadcast shared grids (always present)
  detail::BcastOpacityGrid(table.energyGrid, root);
  detail::BcastThermoState(table.thermoState, root);

  // Broadcast etaGrid if present
  if (presence[5]) {
    detail::BcastOpacityGrid(table.etaGrid, root);
  }

  // Broadcast sub-tables conditionally
  if (presence[0]) { detail::BcastEmAbTable(table.emAb, root); }
  if (presence[1]) { detail::BcastScatIsoTable(table.scatIso, root); }
  if (presence[2]) { detail::BcastScatKernelTable(table.scatNES, root); }
  if (presence[3]) { detail::BcastScatKernelTable(table.scatPair, root); }
  if (presence[4]) { detail::BcastScatKernelTable(table.scatBrem, root); }

  // Root moves local table to output
  if (myRank == root) {
    output = std::move(localTable);
  }

  return status;
}

} // namespace WeakLibReader
