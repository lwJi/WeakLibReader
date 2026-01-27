#pragma once

#include <AMReX_Arena.H>
#include <AMReX_Array.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_ParallelDescriptor.H>

#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "WeakLibReader_Hdf5Types.hpp"
#include "detail/WeakLibReader_WeakLibEosLoaderDetail.hpp"

namespace WeakLibReader {

struct WeakLibEosLoadConfig {
  // Variables to load (empty = load all 15 variables)
  std::vector<EosVariable> variablesToLoad{};
};

inline Hdf5LoadStatus LoadWeakLibEosTable(const std::string& filePath,
                                           WeakLibEosTable& output,
                                           const WeakLibEosLoadConfig& cfg = WeakLibEosLoadConfig{})
{
  WeakLibEosTable result;

  detail::ScopedHandle file(H5Fopen(filePath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
  if (!file.Valid()) {
    return Hdf5LoadStatus::FileOpenFailed;
  }

  // Load axes from /ThermoState group
  const Hdf5LoadStatus axisStatus = detail::LoadEosAxes(file.Get(), result);
  if (axisStatus != Hdf5LoadStatus::Success) {
    return axisStatus;
  }

  // Load offsets from /DependentVariables/Offsets
  const Hdf5LoadStatus offsetsStatus = detail::LoadEosOffsets(file.Get(), result.offsets);
  if (offsetsStatus != Hdf5LoadStatus::Success) {
    return offsetsStatus;
  }

  // Determine which variables to load
  std::vector<EosVariable> varsToLoad;
  if (cfg.variablesToLoad.empty()) {
    varsToLoad.reserve(kEosVariableCount);
    for (int i = 0; i < kEosVariableCount; ++i) {
      varsToLoad.push_back(static_cast<EosVariable>(i));
    }
  } else {
    varsToLoad = cfg.variablesToLoad;
  }

  // Load each variable
  result.loadedVariables = varsToLoad;
  result.variables.resize(varsToLoad.size());
  for (std::size_t i = 0; i < varsToLoad.size(); ++i) {
    const Hdf5LoadStatus varStatus = detail::LoadEosVariable(
        file.Get(), varsToLoad[i], result.extents, result.variables[i]);
    if (varStatus != Hdf5LoadStatus::Success) {
      return varStatus;
    }
  }

  // Compute layout
  std::array<int, 5> extents5{{result.extents[0], result.extents[1], result.extents[2], 1, 1}};
  result.layout = MakeLayout(extents5.data(), result.nd);

  output = std::move(result);
  return Hdf5LoadStatus::Success;
}

inline WeakLibEosTableDevice MakeDeviceCopy(const WeakLibEosTable& host,
                                             amrex::Arena* arena = amrex::The_Device_Arena())
{
  WeakLibEosTableDevice device{};
  device.layout = host.layout;
  device.loadedVariables = host.loadedVariables;
  device.offsets = host.offsets;

  const amrex::Array<int, 4> lo{{0, 0, 0, 0}};
  const amrex::Array<int, 4> hi{{host.extents[0] - 1, host.extents[1] - 1, host.extents[2] - 1, 0}};

  // Copy all variables
  device.variables.resize(host.variables.size());
  for (std::size_t i = 0; i < host.variables.size(); ++i) {
    device.variables[i].resize(lo, hi, arena);
    device.variables[i].copy(host.variables[i]);
  }

  // Copy axes
  for (int dim = 0; dim < host.nd; ++dim) {
    const amrex::Vector<double>& hostAxis = host.axisStorage[dim];
    amrex::Gpu::DeviceVector<double>& deviceAxis = device.axisStorage[dim];
    deviceAxis.resize(hostAxis.size());
    if (!hostAxis.empty()) {
      amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                       hostAxis.begin(), hostAxis.end(),
                       deviceAxis.begin());
    }
    device.axes[dim].grid = deviceAxis.data();
    device.axes[dim].n = static_cast<int>(deviceAxis.size());
    device.axes[dim].scale = host.axes[dim].scale;
  }

  return device;
}

inline Hdf5LoadStatus LoadWeakLibEosTableParallel(
    const std::string& filePath,
    WeakLibEosTable& output,
    const WeakLibEosLoadConfig& cfg = WeakLibEosLoadConfig{},
    int readerRank = amrex::ParallelDescriptor::IOProcessorNumber())
{
  const int nProcs = amrex::ParallelDescriptor::NProcs();
  if (nProcs <= 1) {
    return LoadWeakLibEosTable(filePath, output, cfg);
  }

  int root = readerRank;
  if (root < 0 || root >= nProcs) {
    root = amrex::ParallelDescriptor::IOProcessorNumber();
  }

  const int myRank = amrex::ParallelDescriptor::MyProc();

  WeakLibEosTable localTable;
  Hdf5LoadStatus status = Hdf5LoadStatus::Success;
  if (myRank == root) {
    status = LoadWeakLibEosTable(filePath, localTable, cfg);
  }

  int statusInt = static_cast<int>(status);
  amrex::ParallelDescriptor::Bcast(&statusInt, 1, root);
  status = static_cast<Hdf5LoadStatus>(statusInt);
  if (status != Hdf5LoadStatus::Success) {
    return status;
  }

  // Broadcast header: nd, extents, numVariables, numOffsets
  int header[6] = {0, 1, 1, 1, 0, 0};
  if (myRank == root) {
    header[0] = localTable.nd;
    header[1] = localTable.extents[0];
    header[2] = localTable.extents[1];
    header[3] = localTable.extents[2];
    header[4] = static_cast<int>(localTable.loadedVariables.size());
    header[5] = static_cast<int>(localTable.offsets.size());
  }
  amrex::ParallelDescriptor::Bcast(header, 6, root);

  const int numVars = header[4];
  const int numOffsets = header[5];

  // Broadcast axis counts and scales
  int axisCounts[3] = {0, 0, 0};
  int axisScales[3] = {static_cast<int>(AxisScale::Linear),
                       static_cast<int>(AxisScale::Linear),
                       static_cast<int>(AxisScale::Linear)};
  if (myRank == root) {
    for (int dim = 0; dim < 3; ++dim) {
      axisCounts[dim] = localTable.axes[dim].n;
      axisScales[dim] = static_cast<int>(localTable.axes[dim].scale);
    }
  }
  amrex::ParallelDescriptor::Bcast(axisCounts, 3, root);
  amrex::ParallelDescriptor::Bcast(axisScales, 3, root);

  // Broadcast loaded variable indices
  std::vector<int> varIndices(static_cast<std::size_t>(numVars));
  if (myRank == root) {
    for (std::size_t i = 0; i < localTable.loadedVariables.size(); ++i) {
      varIndices[i] = static_cast<int>(localTable.loadedVariables[i]);
    }
  }
  if (numVars > 0) {
    amrex::ParallelDescriptor::Bcast(varIndices.data(), numVars, root);
  }

  // Allocate on non-root ranks
  const std::size_t totalSize = static_cast<std::size_t>(header[1]) *
                                static_cast<std::size_t>(header[2]) *
                                static_cast<std::size_t>(header[3]);

  if (myRank != root) {
    output = WeakLibEosTable{};
    output.extents = {{header[1], header[2], header[3]}};

    std::array<int, 5> extents5{{header[1], header[2], header[3], 1, 1}};
    output.layout = MakeLayout(extents5.data(), WeakLibEosTable::nd);

    // Allocate axes
    for (int dim = 0; dim < 3; ++dim) {
      amrex::Vector<double>& storage = output.axisStorage[dim];
      storage.resize(static_cast<std::size_t>(axisCounts[dim]));
      output.axes[dim].grid = storage.empty() ? nullptr : storage.data();
      output.axes[dim].n = axisCounts[dim];
      output.axes[dim].scale = static_cast<AxisScale>(axisScales[dim]);
    }

    // Allocate variables
    const auto numVarsSize = static_cast<std::size_t>(numVars);
    output.loadedVariables.resize(numVarsSize);
    output.variables.resize(numVarsSize);
    const amrex::Array<int, 4> lo{{0, 0, 0, 0}};
    const amrex::Array<int, 4> hi{{header[1] - 1, header[2] - 1, header[3] - 1, 0}};
    for (std::size_t i = 0; i < numVarsSize; ++i) {
      output.loadedVariables[i] = static_cast<EosVariable>(varIndices[i]);
      output.variables[i].resize(lo, hi, amrex::The_Pinned_Arena());
    }

    // Allocate offsets
    output.offsets.resize(static_cast<std::size_t>(numOffsets));
  }

  // Broadcast offsets
  if (numOffsets > 0) {
    double* offsetPtr = (myRank == root)
                            ? localTable.offsets.data()
                            : output.offsets.data();
    amrex::ParallelDescriptor::Bcast(offsetPtr, numOffsets, root);
  }

  // Broadcast axes
  for (int dim = 0; dim < 3; ++dim) {
    const int count = axisCounts[dim];
    if (count <= 0) continue;
    double* axisPtr = (myRank == root)
                          ? localTable.axisStorage[dim].data()
                          : output.axisStorage[dim].data();
    amrex::ParallelDescriptor::Bcast(axisPtr, count, root);
  }

  // Broadcast variable data
  for (std::size_t v = 0; v < static_cast<std::size_t>(numVars); ++v) {
    double* dataPtr = (myRank == root) ? localTable.variables[v].table().p
                                       : output.variables[v].table().p;
    if (totalSize > 0) {
      amrex::ParallelDescriptor::Bcast(dataPtr, static_cast<int>(totalSize), root);
    }
  }

  if (myRank == root) {
    output = std::move(localTable);
  }

  return status;
}

} // namespace WeakLibReader
