#pragma once

#include "hdf5/WeakLibReader_Hdf5LoaderEos.hpp"

namespace WeakLibReader {
inline Hdf5LoadStatus LoadWeakLibEosTableFullParallel(
    const std::string &filePath, WeakLibEosTable &output,
    int readerRank = amrex::ParallelDescriptor::IOProcessorNumber()) {
  const int nProcs = amrex::ParallelDescriptor::NProcs();
  if (nProcs <= 1) {
    return LoadWeakLibEosTableFull(filePath, output);
  }

  int root = readerRank;
  if (root < 0 || root >= nProcs) {
    root = amrex::ParallelDescriptor::IOProcessorNumber();
  }
  const int myRank = amrex::ParallelDescriptor::MyProc();

  WeakLibEosTable localTable;
  Hdf5LoadStatus status = Hdf5LoadStatus::Success;
  if (myRank == root) {
    status = LoadWeakLibEosTableFull(filePath, localTable);
  }

  int statusInt = static_cast<int>(status);
  amrex::ParallelDescriptor::Bcast(&statusInt, 1, root);
  status = static_cast<Hdf5LoadStatus>(statusInt);
  if (status != Hdf5LoadStatus::Success) {
    return status;
  }

  // Header: [nVariables, dims x3, axisScales x3]
  int header[7] = {0, 0, 0, 0, 0, 0, 0};
  if (myRank == root) {
    header[0] = localTable.nVariables;
    for (int i = 0; i < 3; ++i) {
      header[1 + i] = localTable.dimensions[i];
      header[4 + i] = static_cast<int>(localTable.axes[i].scale);
    }
  }
  amrex::ParallelDescriptor::Bcast(header, 7, root);

  const int nVariables = header[0];
  const std::array<int, 3> dimensions{{header[1], header[2], header[3]}};

  int axisCounts[3] = {0, 0, 0};
  if (myRank == root) {
    for (int dim = 0; dim < 3; ++dim) {
      axisCounts[dim] = localTable.axes[dim].n;
    }
  }
  amrex::ParallelDescriptor::Bcast(axisCounts, 3, root);

  if (myRank != root) {
    output = WeakLibEosTable{};
    output.nVariables = nVariables;
    output.dimensions = dimensions;

    for (int dim = 0; dim < 3; ++dim) {
      output.axisStorage[dim].resize(axisCounts[dim]);
      output.axes[dim] = Axis{output.axisStorage[dim].data(), axisCounts[dim],
                              static_cast<AxisScale>(header[4 + dim])};
    }

    const std::size_t varSize = static_cast<std::size_t>(dimensions[0]) *
                                static_cast<std::size_t>(dimensions[1]) *
                                static_cast<std::size_t>(dimensions[2]);
    output.variables.resize(nVariables);
    for (int iVar = 0; iVar < nVariables; ++iVar) {
      output.variables[iVar].resize(varSize);
    }
    output.repaired.resize(varSize);
    output.offsets.resize(nVariables);
    output.variableNames.resize(nVariables);
    output.variableUnits.resize(nVariables);

    const std::array<int, 5> extents5{
        {dimensions[0], dimensions[1], dimensions[2], 1, 1}};
    output.layout = MakeLayout(extents5.data(), 3);
  }

  WeakLibEosTable &table = (myRank == root) ? localTable : output;

  for (int dim = 0; dim < 3; ++dim) {
    if (axisCounts[dim] > 0) {
      amrex::ParallelDescriptor::Bcast(table.axisStorage[dim].data(),
                                       axisCounts[dim], root);
    }
  }

  detail::BcastStringArray(table.axisNames, root);
  detail::BcastStringArray(table.axisUnits, root);
  detail::BcastStringVector(table.variableNames, root);
  detail::BcastStringVector(table.variableUnits, root);

  if (nVariables > 0) {
    amrex::ParallelDescriptor::Bcast(table.offsets.data(), nVariables, root);
  }

  const std::size_t varSize = static_cast<std::size_t>(dimensions[0]) *
                              static_cast<std::size_t>(dimensions[1]) *
                              static_cast<std::size_t>(dimensions[2]);
  if (varSize > 0) {
    for (int iVar = 0; iVar < nVariables; ++iVar) {
      amrex::ParallelDescriptor::Bcast(table.variables[iVar].data(),
                                       static_cast<int>(varSize), root);
    }
    amrex::ParallelDescriptor::Bcast(table.repaired.data(),
                                     static_cast<int>(varSize), root);
  }

  static_assert(sizeof(WeakLibEosIndices) == 15 * sizeof(int),
                "WeakLibEosIndices must be 15 contiguous ints");
  amrex::ParallelDescriptor::Bcast(reinterpret_cast<int *>(&table.indices), 15,
                                   root);

  if (myRank == root) {
    output = std::move(localTable);
  }

  return status;
}

} // namespace WeakLibReader
