// Placeholder: AMReX CUDA interpolation demo
//
// This file provides a template for using WeakLibReader with AMReX GPU kernels.
// Full implementation will be completed in Phase 3 (see AGENTS.md).

#include <WeakLibReader.hpp>
#include <Hdf5Loader.hpp>

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_PlotFileUtil.H>

#include <iostream>
#include <string>

// TODO: Implement full CUDA kernel example
// This is a conceptual outline showing the usage pattern

namespace {

void RunInterpolationDemo(const std::string& hdf5Path)
{
  using namespace WeakLibReader;

  // Step 1: Load HDF5 table on host
  Hdf5Table hostTable;
  const auto status = LoadHdf5Table(hdf5Path, hostTable);
  if (status != Hdf5LoadStatus::Success) {
    std::cerr << "Failed to load HDF5 table" << std::endl;
    return;
  }

  std::cout << "Loaded " << hostTable.nd << "D table with extents: ";
  for (int dim = 0; dim < hostTable.nd; ++dim) {
    std::cout << hostTable.extents[dim];
    if (dim < hostTable.nd - 1) {
      std::cout << " x ";
    }
  }
  std::cout << std::endl;

  // Step 2: Copy to device memory
  auto deviceTable = MakeDeviceCopy(hostTable);

  // Step 3: Example AMReX MultiFab interpolation (conceptual)
  // Uncomment and adapt when implementing:
  //
  // amrex::Box domain(amrex::IntVect(0), amrex::IntVect(63));
  // amrex::BoxArray ba(domain);
  // ba.maxSize(32);
  // amrex::DistributionMapping dm(ba);
  // amrex::MultiFab mf(ba, dm, 1, 0);
  //
  // auto view = deviceTable.View();
  //
  // for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
  //   const amrex::Box& box = mfi.validbox();
  //   amrex::Array4<amrex::Real> const& fab = mf.array(mfi);
  //
  //   amrex::ParallelFor(box,
  //   [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
  //     // Map (i,j,k) to physical coordinates
  //     double coords[5] = {
  //       /* map i to axis 0 coordinate */,
  //       /* map j to axis 1 coordinate */,
  //       /* map k to axis 2 coordinate */,
  //       0.0, 0.0
  //     };
  //
  //     InterpConfig cfg;
  //     const double value = InterpLinearND(
  //       view.data, view.layout, view.axes, coords, cfg, view.nd
  //     );
  //
  //     fab(i, j, k) = value;
  //   });
  // }

  std::cout << "Demo complete (placeholder - see interp_demo.cpp for TODO items)" << std::endl;
}

} // anonymous namespace

int main(int argc, char* argv[])
{
  amrex::Initialize(argc, argv);

  {
    if (argc < 2) {
      std::cout << "Usage: " << argv[0] << " <path-to-hdf5-table>" << std::endl;
      std::cout << "Note: This is a placeholder demo (see examples/amrex/README.md)" << std::endl;
    } else {
      RunInterpolationDemo(argv[1]);
    }
  }

  amrex::Finalize();
  return 0;
}
