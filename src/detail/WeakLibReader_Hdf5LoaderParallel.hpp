#pragma once

namespace WeakLibReader {
inline Hdf5LoadStatus LoadHdf5TableParallel(const std::string& filePath,
                                            Hdf5Table& output,
                                            const Hdf5LoadConfig& cfg = Hdf5LoadConfig{},
                                            int readerRank = amrex::ParallelDescriptor::IOProcessorNumber())
{
  const int nProcs = amrex::ParallelDescriptor::NProcs();
  if (nProcs <= 1) {
    return LoadHdf5Table(filePath, output, cfg);
  }

  int root = readerRank;
  if (root < 0 || root >= nProcs) {
    root = amrex::ParallelDescriptor::IOProcessorNumber();
  }

  const int myRank = amrex::ParallelDescriptor::MyProc();

  Hdf5Table localTable;
  Hdf5LoadStatus status = Hdf5LoadStatus::Success;
  if (myRank == root) {
    status = LoadHdf5Table(filePath, localTable, cfg);
  }

  int statusInt = static_cast<int>(status);
  amrex::ParallelDescriptor::Bcast(&statusInt, 1, root);
  status = static_cast<Hdf5LoadStatus>(statusInt);
  if (status != Hdf5LoadStatus::Success) {
    return status;
  }

  int header[6] = {0, 1, 1, 1, 1, 1};
  if (myRank == root) {
    header[0] = localTable.nd;
    for (int dim = 0; dim < 5; ++dim) {
      header[1 + dim] = localTable.extents[dim];
    }
  }
  amrex::ParallelDescriptor::Bcast(header, 6, root);

  const int nd = header[0];
  std::array<int, 5> extents{{1, 1, 1, 1, 1}};
  for (int dim = 0; dim < 5; ++dim) {
    extents[dim] = header[1 + dim];
  }

  int axisCounts[5] = {0, 0, 0, 0, 0};
  int axisScales[5] = {static_cast<int>(AxisScale::Linear),
                       static_cast<int>(AxisScale::Linear),
                       static_cast<int>(AxisScale::Linear),
                       static_cast<int>(AxisScale::Linear),
                       static_cast<int>(AxisScale::Linear)};

  if (myRank == root) {
    for (int dim = 0; dim < 5; ++dim) {
      axisCounts[dim] = localTable.axes[dim].n;
      axisScales[dim] = static_cast<int>(localTable.axes[dim].scale);
    }
  }

  amrex::ParallelDescriptor::Bcast(axisCounts, 5, root);
  amrex::ParallelDescriptor::Bcast(axisScales, 5, root);

  const std::size_t totalSize = detail::ComputeTotalSize(nd, extents);
  if (totalSize > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return Hdf5LoadStatus::IncompatibleDatasetExtent;
  }

  if (myRank != root) {
    output = Hdf5Table{};
    output.nd = nd;
    output.extents = extents;
    if (totalSize == 0) {
      return Hdf5LoadStatus::IncompatibleDatasetExtent;
    }
    output.values.resize(totalSize);
    output.layout = MakeLayout(output.extents.data(), output.nd);
    for (int dim = 0; dim < 5; ++dim) {
      amrex::Vector<double>& storage = output.axisStorage[dim];
      storage.resize(static_cast<std::size_t>(axisCounts[dim]));
      if (!storage.empty()) {
        output.axes[dim].grid = storage.data();
      } else {
        output.axes[dim].grid = nullptr;
      }
      output.axes[dim].n = axisCounts[dim];
      output.axes[dim].scale = static_cast<AxisScale>(axisScales[dim]);
    }
  }

  double* dataPtr = (myRank == root)
                        ? localTable.values.data()
                        : output.values.data();
  if (totalSize > 0) {
    amrex::ParallelDescriptor::Bcast(dataPtr,
                                     static_cast<int>(totalSize),
                                     root);
  }

  for (int dim = 0; dim < 5; ++dim) {
    const int count = axisCounts[dim];
    if (count <= 0) {
      continue;
    }
    double* axisPtr = (myRank == root)
                          ? localTable.axisStorage[dim].data()
                          : output.axisStorage[dim].data();
    amrex::ParallelDescriptor::Bcast(axisPtr, count, root);
  }

  if (myRank == root) {
    output = std::move(localTable);
  }

  return status;
}

inline Hdf5LoadStatus LoadWeakLibEosTableFullParallel(
    const std::string& filePath,
    WeakLibEosTable& output,
    int readerRank = amrex::ParallelDescriptor::IOProcessorNumber())
{
  // Single-rank fallback
  const int nProcs = amrex::ParallelDescriptor::NProcs();
  if (nProcs <= 1) {
    return LoadWeakLibEosTableFull(filePath, output);
  }

  // Validate readerRank
  int root = readerRank;
  if (root < 0 || root >= nProcs) {
    root = amrex::ParallelDescriptor::IOProcessorNumber();
  }

  const int myRank = amrex::ParallelDescriptor::MyProc();

  // Root rank loads the table
  WeakLibEosTable localTable;
  Hdf5LoadStatus status = Hdf5LoadStatus::Success;
  if (myRank == root) {
    status = LoadWeakLibEosTableFull(filePath, localTable);
  }

  // Broadcast status
  int statusInt = static_cast<int>(status);
  amrex::ParallelDescriptor::Bcast(&statusInt, 1, root);
  status = static_cast<Hdf5LoadStatus>(statusInt);
  if (status != Hdf5LoadStatus::Success) {
    return status;
  }

  // Broadcast header: [nVariables, dims x3, axisScales x3]
  int header[7] = {0, 0, 0, 0, 0, 0, 0};
  if (myRank == root) {
    header[0] = localTable.nVariables;
    header[1] = localTable.dimensions[0];
    header[2] = localTable.dimensions[1];
    header[3] = localTable.dimensions[2];
    header[4] = static_cast<int>(localTable.axes[0].scale);
    header[5] = static_cast<int>(localTable.axes[1].scale);
    header[6] = static_cast<int>(localTable.axes[2].scale);
  }
  amrex::ParallelDescriptor::Bcast(header, 7, root);

  const int nVariables = header[0];
  std::array<int, 3> dimensions{{header[1], header[2], header[3]}};
  std::array<AxisScale, 3> axisScales{{
      static_cast<AxisScale>(header[4]),
      static_cast<AxisScale>(header[5]),
      static_cast<AxisScale>(header[6])
  }};

  // Broadcast axis counts
  int axisCounts[3] = {0, 0, 0};
  if (myRank == root) {
    for (int dim = 0; dim < 3; ++dim) {
      axisCounts[dim] = localTable.axes[dim].n;
    }
  }
  amrex::ParallelDescriptor::Bcast(axisCounts, 3, root);

  // Non-root: allocate base structure
  if (myRank != root) {
    output = WeakLibEosTable{};
    output.nVariables = nVariables;
    output.dimensions = dimensions;

    // Allocate axis storage
    for (int dim = 0; dim < 3; ++dim) {
      output.axisStorage[dim].resize(axisCounts[dim]);
      output.axes[dim].grid = output.axisStorage[dim].data();
      output.axes[dim].n = axisCounts[dim];
      output.axes[dim].scale = axisScales[dim];
    }

    // Allocate variable arrays
    const std::size_t varSize = static_cast<std::size_t>(dimensions[0]) *
                                static_cast<std::size_t>(dimensions[1]) *
                                static_cast<std::size_t>(dimensions[2]);
    output.variables.resize(nVariables);
    for (int iVar = 0; iVar < nVariables; ++iVar) {
      output.variables[iVar].resize(varSize);
    }

    // Allocate repaired mask
    output.repaired.resize(varSize);

    // Allocate offset/string vectors
    output.offsets.resize(nVariables);
    output.variableNames.resize(nVariables);
    output.variableUnits.resize(nVariables);

    // Compute layout
    std::array<int, 5> extents5{{dimensions[0], dimensions[1], dimensions[2], 1, 1}};
    output.layout = MakeLayout(extents5.data(), 3);
  }

  // Get reference to working table
  WeakLibEosTable& table = (myRank == root) ? localTable : output;

  // Broadcast axis grids
  for (int dim = 0; dim < 3; ++dim) {
    if (axisCounts[dim] > 0) {
      amrex::ParallelDescriptor::Bcast(table.axisStorage[dim].data(), axisCounts[dim], root);
    }
  }

  // Broadcast string metadata
  detail::BcastStringArray(table.axisNames, root);
  detail::BcastStringArray(table.axisUnits, root);
  detail::BcastStringVector(table.variableNames, root);
  detail::BcastStringVector(table.variableUnits, root);

  // Broadcast offsets
  if (nVariables > 0) {
    amrex::ParallelDescriptor::Bcast(table.offsets.data(), nVariables, root);
  }

  // Broadcast each variable's 3D data
  const std::size_t varSize = static_cast<std::size_t>(dimensions[0]) *
                              static_cast<std::size_t>(dimensions[1]) *
                              static_cast<std::size_t>(dimensions[2]);
  if (varSize > 0) {
    for (int iVar = 0; iVar < nVariables; ++iVar) {
      amrex::ParallelDescriptor::Bcast(table.variables[iVar].data(),
                                       static_cast<int>(varSize), root);
    }
  }

  // Broadcast repaired mask
  if (varSize > 0) {
    amrex::ParallelDescriptor::Bcast(table.repaired.data(),
                                     static_cast<int>(varSize), root);
  }

  // Broadcast indices (15 integers as a block)
  static_assert(sizeof(WeakLibEosIndices) == 15 * sizeof(int),
                "WeakLibEosIndices must be 15 contiguous ints");
  amrex::ParallelDescriptor::Bcast(reinterpret_cast<int*>(&table.indices), 15, root);

  // Root moves its local table to output
  if (myRank == root) {
    output = std::move(localTable);
  }

  return status;
}

} // namespace WeakLibReader
